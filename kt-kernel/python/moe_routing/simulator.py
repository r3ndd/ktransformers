from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from .routing_schemes import BaseRoutingScheme
    from .token_indexing import add_absolute_token_position
except ImportError:
    from routing_schemes import BaseRoutingScheme
    from token_indexing import add_absolute_token_position


@dataclass(slots=True)
class TokenLayerEvent:
    context_id: str
    absolute_token_position: int
    layer_id: int
    baseline_experts: list[int]
    all_scores: list[float]


def _build_token_layer_events(traces: pd.DataFrame) -> list[TokenLayerEvent]:
    required = {
        "context_id",
        "layer_id",
        "token_position",
        "expert_ids",
        "expert_scores_all",
    }
    missing = sorted(required - set(traces.columns))
    if missing:
        raise ValueError(f"simulate_routing_scheme requires columns {sorted(required)}; missing {missing}")

    traces = add_absolute_token_position(traces)
    traces = traces.sort_values(["context_id", "absolute_token_position", "layer_id"]).reset_index(drop=True)

    events: list[TokenLayerEvent] = []
    for row in traces.itertuples(index=False):
        context_id = str(getattr(row, "context_id"))
        absolute_token_position = int(getattr(row, "absolute_token_position"))
        layer_id = int(getattr(row, "layer_id"))
        baseline_experts = [int(x) for x in list(getattr(row, "expert_ids"))]
        all_scores = [float(x) for x in list(getattr(row, "expert_scores_all"))]

        if len(all_scores) == 0:
            raise ValueError("expert_scores_all must be non-empty per row")
        if len(baseline_experts) == 0:
            raise ValueError("expert_ids must be non-empty per row")
        if any(e < 0 or e >= len(all_scores) for e in baseline_experts):
            raise ValueError("expert_ids must be valid indices into expert_scores_all")

        events.append(
            TokenLayerEvent(
                context_id=context_id,
                absolute_token_position=absolute_token_position,
                layer_id=layer_id,
                baseline_experts=baseline_experts,
                all_scores=all_scores,
            )
        )

    return events


def _outside_penalty(alpha: float) -> float:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1 inclusive; got {alpha}")
    if alpha >= 1.0:
        return 0.0
    if alpha <= 0.0:
        return float("inf")
    return (1.0 - alpha) / alpha


def _select_routed_experts(scores: list[float], baseline_k: int, pool: set[int], alpha: float) -> list[int]:
    if baseline_k <= 0:
        return []

    penalty = _outside_penalty(alpha)
    adjusted: list[tuple[float, int]] = []
    for expert_id, score in enumerate(scores):
        if penalty == 0.0:
            adj = score
        elif expert_id in pool:
            adj = score
        elif penalty == float("inf"):
            adj = float("-inf")
        else:
            adj = score - penalty
        adjusted.append((adj, expert_id))

    adjusted.sort(key=lambda x: (-x[0], x[1]))
    return [expert_id for _, expert_id in adjusted[:baseline_k]]


def simulate_routing_scheme(traces: pd.DataFrame, scheme: BaseRoutingScheme, alpha: float) -> dict[str, float]:
    events = _build_token_layer_events(traces)
    if not events:
        return {
            "hit_rate": 0.0,
            "ssd_fetches_per_token": 0.0,
            "quality_degradation": 1.0,
            "speedup_ratio": 1.0,
            "token_count": 0.0,
            "experts_per_token": 0.0,
        }

    hit_rate_sum = 0.0
    quality_ratio_sum = 0.0
    ssd_fetches_sum = 0.0
    event_count = 0

    token_keys = {(e.context_id, e.absolute_token_position) for e in events}
    token_count = len(token_keys)

    grouped: dict[tuple[str, int], list[TokenLayerEvent]] = {}
    for e in events:
        grouped.setdefault((e.context_id, e.absolute_token_position), []).append(e)

    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        token_events = sorted(grouped[key], key=lambda x: x.layer_id)
        for e in token_events:
            baseline_set = set(e.baseline_experts)
            k = len(e.baseline_experts)
            pool = scheme.get_pool(e.layer_id)
            routed = _select_routed_experts(e.all_scores, baseline_k=k, pool=pool, alpha=alpha)
            routed_set = set(routed)

            hit_rate_sum += len(routed_set & baseline_set) / float(k)
            ssd_fetches_sum += float(len(routed_set - pool))

            baseline_score_avg = sum(e.all_scores[idx] for idx in e.baseline_experts) / float(k)
            routed_score_avg = sum(e.all_scores[idx] for idx in routed) / float(k)
            if baseline_score_avg == 0.0:
                quality_ratio = 1.0 if routed_score_avg == 0.0 else 0.0
            else:
                quality_ratio = routed_score_avg / baseline_score_avg
            quality_ratio_sum += quality_ratio

            scheme.observe(e.layer_id, routed)
            event_count += 1

        scheme.end_token()

    hit_rate = hit_rate_sum / float(event_count) if event_count else 0.0
    quality_degradation = quality_ratio_sum / float(event_count) if event_count else 1.0
    ssd_fetches_per_token = ssd_fetches_sum / float(token_count) if token_count else 0.0

    experts_per_token = float(len(events[0].baseline_experts))
    speedup_ratio = (1.0 + (1.0 - hit_rate) * experts_per_token) / (1.0 + ssd_fetches_per_token)

    return {
        "hit_rate": hit_rate,
        "ssd_fetches_per_token": ssd_fetches_per_token,
        "quality_degradation": quality_degradation,
        "speedup_ratio": speedup_ratio,
        "token_count": float(token_count),
        "experts_per_token": experts_per_token,
    }
