from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


_PREFILL_SCHEMES = {"prefill_block_mean", "prefill_full_mean"}
_DECODE_SCHEMES = {
    "sliding_window_score_averaging",
    "ema_score_averaging",
    "two_timescale_ema",
    "two_timescale_softmax",
}


@dataclass(frozen=True)
class RoutingSchemeConfig:
    scheme: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class MoeRoutingConfig:
    prefill: RoutingSchemeConfig
    decode: RoutingSchemeConfig
    scope: str = "request"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prefill": {"scheme": self.prefill.scheme, "params": dict(self.prefill.params)},
            "decode": {"scheme": self.decode.scheme, "params": dict(self.decode.params)},
            "scope": self.scope,
        }


def _normalize_int(value: Any, *, min_value: int, name: str) -> int:
    iv = int(value)
    if iv < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    return iv


def _normalize_float_in_open01(value: Any, *, name: str) -> float:
    fv = float(value)
    if fv <= 0.0 or fv >= 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return fv


def _normalize_float_in_0_1_closed(value: Any, *, name: str) -> float:
    fv = float(value)
    if fv <= 0.0 or fv > 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return fv


def _normalize_non_negative_float(value: Any, *, name: str) -> float:
    fv = float(value)
    if fv < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return fv


def _normalize_prefill(entry: Dict[str, Any]) -> RoutingSchemeConfig:
    scheme = str(entry.get("scheme", "prefill_block_mean"))
    if scheme not in _PREFILL_SCHEMES:
        raise ValueError(f"unsupported prefill scheme: {scheme}")

    params = entry.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("prefill.params must be an object")

    out: Dict[str, Any] = {}
    if scheme == "prefill_block_mean":
        out["window_size"] = _normalize_int(params.get("window_size", 64), min_value=1, name="window_size")
    elif scheme == "prefill_full_mean":
        out = {}
    return RoutingSchemeConfig(scheme=scheme, params=out)


def _normalize_decode(entry: Dict[str, Any]) -> RoutingSchemeConfig:
    scheme = str(entry.get("scheme", "sliding_window_score_averaging"))
    if scheme not in _DECODE_SCHEMES:
        raise ValueError(f"unsupported decode scheme: {scheme}")

    params = entry.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("decode.params must be an object")

    out: Dict[str, Any] = {}
    if scheme == "sliding_window_score_averaging":
        out["window_size"] = _normalize_int(params.get("window_size", 1), min_value=1, name="window_size")
    elif scheme == "ema_score_averaging":
        out["ema_beta"] = _normalize_float_in_0_1_closed(params.get("ema_beta", 0.3), name="ema_beta")
    elif scheme == "two_timescale_ema":
        out["mix_lambda"] = _normalize_float_in_open01(params.get("mix_lambda", 0.2), name="mix_lambda")
    elif scheme == "two_timescale_softmax":
        out["mix_lambda"] = _normalize_float_in_open01(params.get("mix_lambda", 0.2), name="mix_lambda")
        out["rho"] = _normalize_non_negative_float(params.get("rho", 1.0), name="rho")
    return RoutingSchemeConfig(scheme=scheme, params=out)


def parse_moe_routing_config(custom_params: Any) -> Tuple[Optional[MoeRoutingConfig], Optional[str]]:
    """Parse custom_params.moe_routing with fallback semantics.

    Returns (config, warning). If config is None, caller should use baseline routing.
    """
    if not isinstance(custom_params, dict):
        return None, None

    raw = custom_params.get("moe_routing")
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "moe_routing must be an object; falling back to baseline"

    try:
        prefill = _normalize_prefill(raw.get("prefill") or {})
        decode = _normalize_decode(raw.get("decode") or {})
        scope = str(raw.get("scope", "request"))
        if scope != "request":
            raise ValueError("scope must be 'request'")
        return MoeRoutingConfig(prefill=prefill, decode=decode, scope=scope), None
    except Exception as e:
        return None, f"invalid moe_routing config ({e}); falling back to baseline"


def build_moe_routing_signature(config: Optional[MoeRoutingConfig]) -> Optional[str]:
    if config is None:
        return None
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"moe-routing:{h}"
