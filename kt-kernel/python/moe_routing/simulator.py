from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

try:
    from .cache_policies import PerLayerLRUPolicy
    from .routing_schemes import BaseRoutingScheme
    from .token_indexing import add_absolute_token_position
except ImportError:
    from cache_policies import PerLayerLRUPolicy
    from routing_schemes import BaseRoutingScheme
    from token_indexing import add_absolute_token_position


BASE_SECONDS_PER_TOKEN_NO_SSD = 0.1
EXTRA_SECONDS_PER_SSD_FETCH = 0.0015


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
                context_id=str(getattr(row, "context_id")),
                absolute_token_position=int(getattr(row, "absolute_token_position")),
                layer_id=int(getattr(row, "layer_id")),
                baseline_experts=baseline_experts,
                all_scores=all_scores,
            )
        )

    return events


def _topk_experts(scores: list[float], k: int) -> list[int]:
    pairs = [(float(s), i) for i, s in enumerate(scores)]
    pairs.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in pairs[:k]]


def _softmax_probs(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exps = [math.exp(float(s) - float(max_score)) for s in scores]
    denom = sum(exps)
    if denom == 0.0:
        return [0.0 for _ in scores]
    return [v / denom for v in exps]


def simulate_routing_scheme(
    traces: pd.DataFrame,
    scheme: BaseRoutingScheme,
    capacity_per_layer: int = 25,
) -> dict[str, float]:
    events = _build_token_layer_events(traces)
    if not events:
        return {
            "hit_rate": 0.0,
            "ssd_fetches_per_token": 0.0,
            "baseline_overlap": 0.0,
            "quality_degradation": 1.0,
            "speedup_ratio": 1.0,
            "quality_speed_score": 1.0,
            "token_count": 0.0,
            "experts_per_token": 0.0,
            "baseline_ssd_fetches_per_token": 0.0,
        }

    if capacity_per_layer < 0:
        raise ValueError(f"capacity_per_layer must be >= 0; got {capacity_per_layer}")

    scheme_cache = PerLayerLRUPolicy(capacity_per_layer=capacity_per_layer)
    baseline_cache = PerLayerLRUPolicy(capacity_per_layer=capacity_per_layer)

    token_keys = {(e.context_id, e.absolute_token_position) for e in events}
    token_count = len(token_keys)

    grouped: dict[tuple[str, int], list[TokenLayerEvent]] = {}
    for e in events:
        grouped.setdefault((e.context_id, e.absolute_token_position), []).append(e)

    baseline_overlap_sum = 0.0
    quality_ratio_sum = 0.0
    step_count = 0
    hit_rate_sum = 0.0
    ssd_fetches_total = 0.0
    baseline_ssd_fetches_total = 0.0

    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        token_events = sorted(grouped[key], key=lambda x: x.layer_id)
        for e in token_events:
            k = len(e.baseline_experts)
            smoothed = scheme.smooth_scores(e.layer_id, e.all_scores)
            chosen = _topk_experts(smoothed, k)

            baseline_set = set(e.baseline_experts)
            chosen_set = set(chosen)
            baseline_overlap_sum += len(chosen_set & baseline_set) / float(k)

            prev_scheme_fastmem = scheme_cache.cached_for_layer(e.layer_id)
            prev_baseline_fastmem = baseline_cache.cached_for_layer(e.layer_id)
            hit_rate_sum += len(chosen_set & prev_scheme_fastmem) / float(k)
            ssd_fetches_total += float(len(chosen_set - prev_scheme_fastmem))
            baseline_ssd_fetches_total += float(len(baseline_set - prev_baseline_fastmem))

            probs = _softmax_probs(e.all_scores)
            baseline_score_avg = sum(probs[idx] for idx in e.baseline_experts) / float(k)
            chosen_score_avg = sum(probs[idx] for idx in chosen) / float(k)
            quality_ratio = (
                1.0
                if baseline_score_avg == 0.0 and chosen_score_avg == 0.0
                else ((chosen_score_avg / baseline_score_avg) if baseline_score_avg != 0.0 else 0.0)
            )
            quality_ratio_sum += quality_ratio
            step_count += 1

            scheme.observe(e.layer_id, e.all_scores)
            scheme_cache.observe([(e.layer_id, expert_id) for expert_id in chosen])
            baseline_cache.observe([(e.layer_id, expert_id) for expert_id in e.baseline_experts])

        scheme.end_token()

    hit_rate = hit_rate_sum / float(step_count) if step_count else 0.0
    ssd_fetches_per_token = ssd_fetches_total / float(token_count) if token_count else 0.0
    baseline_ssd_fetches_per_token = baseline_ssd_fetches_total / float(token_count) if token_count else 0.0
    baseline_overlap = baseline_overlap_sum / float(step_count) if step_count else 0.0
    quality_degradation = quality_ratio_sum / float(step_count) if step_count else 1.0

    extra_seconds_per_token = ssd_fetches_per_token * EXTRA_SECONDS_PER_SSD_FETCH
    baseline_extra_seconds_per_token = baseline_ssd_fetches_per_token * EXTRA_SECONDS_PER_SSD_FETCH
    speedup_ratio = (BASE_SECONDS_PER_TOKEN_NO_SSD + baseline_extra_seconds_per_token) / (
        BASE_SECONDS_PER_TOKEN_NO_SSD + extra_seconds_per_token
    )
    quality_speed_score = quality_degradation * speedup_ratio
    experts_per_token = float(len(events[0].baseline_experts))

    return {
        "hit_rate": hit_rate,
        "ssd_fetches_per_token": ssd_fetches_per_token,
        "baseline_overlap": baseline_overlap,
        "quality_degradation": quality_degradation,
        "speedup_ratio": speedup_ratio,
        "quality_speed_score": quality_speed_score,
        "token_count": float(token_count),
        "experts_per_token": experts_per_token,
        "baseline_ssd_fetches_per_token": baseline_ssd_fetches_per_token,
    }
