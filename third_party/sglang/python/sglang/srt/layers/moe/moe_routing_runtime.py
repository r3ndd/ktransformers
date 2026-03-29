from __future__ import annotations

from contextlib import contextmanager
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from sglang.srt.managers.moe_routing_config import MoeRoutingConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


def _softmax_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    ex = torch.exp(x)
    den = ex.sum(dim=-1, keepdim=True)
    return ex / den


@dataclass
class _DecodeLayerState:
    window: Optional[deque] = None
    ema: Optional[torch.Tensor] = None
    short_ema: Optional[torch.Tensor] = None
    long_ema: Optional[torch.Tensor] = None


@dataclass
class _PrefillLayerState:
    window: Optional[deque] = None
    full_sum: Optional[torch.Tensor] = None
    full_count: int = 0


class _RequestRoutingState:
    def __init__(self, layer_count: int):
        self.decode_layers: List[_DecodeLayerState] = [_DecodeLayerState() for _ in range(layer_count)]
        self.prefill_layers: List[_PrefillLayerState] = [_PrefillLayerState() for _ in range(layer_count)]


class MoeRoutingRuntime:
    """Runtime request-scoped routing transforms for MoE router logits."""

    SHORT_BETA = 0.5
    LONG_BETA = 0.05

    def __init__(self) -> None:
        self._states: Dict[str, _RequestRoutingState] = {}

    def _get_state(self, rid: str, layer_count: int) -> _RequestRoutingState:
        s = self._states.get(rid)
        if s is None:
            s = _RequestRoutingState(layer_count=layer_count)
            self._states[rid] = s
        elif layer_count > len(s.decode_layers):
            grow = layer_count - len(s.decode_layers)
            s.decode_layers.extend(_DecodeLayerState() for _ in range(grow))
            s.prefill_layers.extend(_PrefillLayerState() for _ in range(grow))
        return s

    def cleanup_active(self, active_rids: Sequence[str]) -> None:
        active = set(active_rids)
        stale = [rid for rid in self._states.keys() if rid not in active]
        for rid in stale:
            self._states.pop(rid, None)

    def apply(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        configs = getattr(forward_batch, "moe_routing_configs", None)
        rids = getattr(forward_batch, "rids", None)
        if not configs or not rids:
            return router_logits

        if forward_batch.forward_mode == ForwardMode.IDLE:
            return router_logits

        out = router_logits
        for i, cfg in enumerate(configs):
            if cfg is None:
                continue

            if forward_batch.forward_mode.is_decode():
                if i >= out.shape[0]:
                    break
                out_i = self._apply_decode(out[i], cfg, rid=rids[i], layer_id=layer_id)
                if out_i is not out[i]:
                    if out is router_logits:
                        out = router_logits.clone()
                    out[i] = out_i
                continue

            if forward_batch.forward_mode.is_prefill():
                if (
                    forward_batch.extend_start_loc is None
                    or forward_batch.extend_seq_lens is None
                    or i >= len(forward_batch.extend_start_loc)
                    or i >= len(forward_batch.extend_seq_lens)
                ):
                    continue
                start = int(forward_batch.extend_start_loc[i].item())
                length = int(forward_batch.extend_seq_lens[i].item())
                end = min(start + length, out.shape[0])
                if start < 0 or start >= end:
                    continue
                out_seg = self._apply_prefill_segment(
                    out[start:end],
                    cfg,
                    rid=rids[i],
                    layer_id=layer_id,
                )
                if out_seg is not out[start:end]:
                    if out is router_logits:
                        out = router_logits.clone()
                    out[start:end] = out_seg

        self.cleanup_active(rids)
        return out

    def _apply_prefill_segment(
        self,
        logits_segment: torch.Tensor,
        cfg: MoeRoutingConfig,
        *,
        rid: str,
        layer_id: int,
    ) -> torch.Tensor:
        prefill = cfg.prefill
        state = self._get_state(rid, layer_count=max(layer_id + 1, 1))
        ls = state.prefill_layers[layer_id]

        if prefill.scheme == "prefill_full_mean":
            seg_sum = logits_segment.sum(dim=0)
            seg_count = int(logits_segment.shape[0])
            if ls.full_sum is None:
                ls.full_sum = seg_sum.detach().clone()
                ls.full_count = seg_count
            else:
                ls.full_sum = ls.full_sum + seg_sum.detach()
                ls.full_count += seg_count
            mean_vec = ls.full_sum / float(max(ls.full_count, 1))
            return mean_vec.unsqueeze(0).expand_as(logits_segment)

        if prefill.scheme == "prefill_block_mean":
            window_size = int(prefill.params.get("window_size", 64))
            if window_size <= 1:
                return logits_segment
            if ls.window is None or ls.window.maxlen != max(window_size - 1, 1):
                ls.window = deque(maxlen=max(window_size - 1, 1))
            out_rows: List[torch.Tensor] = []
            for row in logits_segment:
                if not ls.window:
                    out_rows.append(row)
                else:
                    acc = row.clone()
                    for prev in ls.window:
                        acc.add_(prev)
                    acc.mul_(1.0 / float(len(ls.window) + 1))
                    out_rows.append(acc)
                ls.window.append(row.detach().clone())
            return torch.stack(out_rows, dim=0)

        return logits_segment

    def _apply_decode(
        self,
        logits_row: torch.Tensor,
        cfg: MoeRoutingConfig,
        *,
        rid: str,
        layer_id: int,
    ) -> torch.Tensor:
        decode = cfg.decode
        state = self._get_state(rid, layer_count=max(layer_id + 1, 1))
        if layer_id >= len(state.decode_layers):
            state.decode_layers.extend(_DecodeLayerState() for _ in range(layer_id + 1 - len(state.decode_layers)))
        ls = state.decode_layers[layer_id]

        if decode.scheme == "sliding_window_score_averaging":
            window_size = int(decode.params.get("window_size", 1))
            if window_size <= 1:
                return logits_row
            if ls.window is None or ls.window.maxlen != max(window_size - 1, 1):
                ls.window = deque(maxlen=max(window_size - 1, 1))
            if not ls.window:
                ls.window.append(logits_row.detach().clone())
                return logits_row
            acc = logits_row.clone()
            for prev in ls.window:
                acc.add_(prev)
            acc.mul_(1.0 / float(len(ls.window) + 1))
            ls.window.append(logits_row.detach().clone())
            return acc

        if decode.scheme == "ema_score_averaging":
            beta = float(decode.params.get("ema_beta", 0.3))
            if ls.ema is None:
                ls.ema = logits_row.detach().clone()
                return logits_row
            smoothed = beta * logits_row + (1.0 - beta) * ls.ema
            ls.ema = smoothed.detach().clone()
            return smoothed

        if decode.scheme == "two_timescale_ema":
            lam = float(decode.params.get("mix_lambda", 0.2))
            if ls.short_ema is None or ls.long_ema is None:
                ls.short_ema = logits_row.detach().clone()
                ls.long_ema = logits_row.detach().clone()
                return logits_row
            short = self.SHORT_BETA * logits_row + (1.0 - self.SHORT_BETA) * ls.short_ema
            long = self.LONG_BETA * logits_row + (1.0 - self.LONG_BETA) * ls.long_ema
            ls.short_ema = short.detach().clone()
            ls.long_ema = long.detach().clone()
            return lam * short + (1.0 - lam) * long

        if decode.scheme == "two_timescale_softmax":
            lam = float(decode.params.get("mix_lambda", 0.2))
            rho = max(0.0, float(decode.params.get("rho", 1.0)))
            x = _softmax_tensor(rho * logits_row)
            if ls.short_ema is None or ls.long_ema is None:
                ls.short_ema = x.detach().clone()
                ls.long_ema = x.detach().clone()
                return x
            short = self.SHORT_BETA * x + (1.0 - self.SHORT_BETA) * ls.short_ema
            long = self.LONG_BETA * x + (1.0 - self.LONG_BETA) * ls.long_ema
            ls.short_ema = short.detach().clone()
            ls.long_ema = long.detach().clone()
            return lam * short + (1.0 - lam) * long

        return logits_row


_GLOBAL_MOE_ROUTING_RUNTIME: Optional[MoeRoutingRuntime] = None
_CURRENT_FORWARD_BATCH: Optional[ForwardBatch] = None


def get_global_moe_routing_runtime() -> MoeRoutingRuntime:
    global _GLOBAL_MOE_ROUTING_RUNTIME
    if _GLOBAL_MOE_ROUTING_RUNTIME is None:
        _GLOBAL_MOE_ROUTING_RUNTIME = MoeRoutingRuntime()
    return _GLOBAL_MOE_ROUTING_RUNTIME


@contextmanager
def set_current_forward_batch(forward_batch: ForwardBatch):
    global _CURRENT_FORWARD_BATCH
    prev = _CURRENT_FORWARD_BATCH
    _CURRENT_FORWARD_BATCH = forward_batch
    try:
        yield
    finally:
        _CURRENT_FORWARD_BATCH = prev


def get_current_forward_batch() -> Optional[ForwardBatch]:
    return _CURRENT_FORWARD_BATCH
