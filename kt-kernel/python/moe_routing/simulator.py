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


def _build_baseline_fastmem_sets(events: list[TokenLayerEvent]) -> dict[tuple[str, int, int], set[int]]:
    out: dict[tuple[str, int, int], set[int]] = {}
    for e in events:
        prev_key = (e.context_id, e.absolute_token_position - 1, e.layer_id)
        out[(e.context_id, e.absolute_token_position, e.layer_id)] = set(
            out.get(prev_key, set()) | set(e.baseline_experts)
        )
    return out


def _build_scheme_fastmem_sets(
    chosen_events: list[tuple[str, int, int, list[int]]],
) -> dict[tuple[str, int, int], set[int]]:
    out: dict[tuple[str, int, int], set[int]] = {}
    for context_id, abs_pos, layer_id, chosen in chosen_events:
        prev_key = (context_id, abs_pos - 1, layer_id)
        out[(context_id, abs_pos, layer_id)] = set(out.get(prev_key, set()) | set(chosen))
    return out


def simulate_routing_scheme(traces: pd.DataFrame, scheme: BaseRoutingScheme) -> dict[str, float]:
    events = _build_token_layer_events(traces)
    if not events:
        return {
            "hit_rate": 0.0,
            "ssd_fetches_per_token": 0.0,
            "baseline_overlap": 0.0,
            "quality_degradation": 1.0,
            "speedup_ratio": 1.0,
            "token_count": 0.0,
            "experts_per_token": 0.0,
            "baseline_ssd_fetches_per_token": 0.0,
        }

    token_keys = {(e.context_id, e.absolute_token_position) for e in events}
    token_count = len(token_keys)

    grouped: dict[tuple[str, int], list[TokenLayerEvent]] = {}
    for e in events:
        grouped.setdefault((e.context_id, e.absolute_token_position), []).append(e)

    chosen_records: list[tuple[str, int, int, list[int]]] = []
    baseline_overlap_sum = 0.0
    quality_ratio_sum = 0.0
    step_count = 0

    for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
        token_events = sorted(grouped[key], key=lambda x: x.layer_id)
        for e in token_events:
            k = len(e.baseline_experts)
            smoothed = scheme.smooth_scores(e.layer_id, e.all_scores)
            chosen = _topk_experts(smoothed, k)
            chosen_records.append((e.context_id, e.absolute_token_position, e.layer_id, chosen))

            baseline_set = set(e.baseline_experts)
            chosen_set = set(chosen)
            baseline_overlap_sum += len(chosen_set & baseline_set) / float(k)

            baseline_score_avg = sum(e.all_scores[idx] for idx in e.baseline_experts) / float(k)
            chosen_score_avg = sum(e.all_scores[idx] for idx in chosen) / float(k)
            quality_ratio = (
                1.0
                if baseline_score_avg == 0.0 and chosen_score_avg == 0.0
                else ((chosen_score_avg / baseline_score_avg) if baseline_score_avg != 0.0 else 0.0)
            )
            quality_ratio_sum += quality_ratio
            step_count += 1

            scheme.observe(e.layer_id, e.all_scores)

        scheme.end_token()

    scheme_fastmem_sets = _build_scheme_fastmem_sets(chosen_records)
    baseline_fastmem_sets = _build_baseline_fastmem_sets(events)

    hit_rate_sum = 0.0
    ssd_fetches_total = 0.0
    baseline_ssd_fetches_total = 0.0

    event_lookup: dict[tuple[str, int, int], TokenLayerEvent] = {
        (e.context_id, e.absolute_token_position, e.layer_id): e for e in events
    }

    for context_id, abs_pos, layer_id, chosen in chosen_records:
        key_now = (context_id, abs_pos, layer_id)
        key_prev = (context_id, abs_pos - 1, layer_id)

        prev_scheme_fastmem = scheme_fastmem_sets.get(key_prev, set())
        prev_baseline_fastmem = baseline_fastmem_sets.get(key_prev, set())

        chosen_set = set(chosen)
        baseline_set = set(event_lookup[key_now].baseline_experts)
        k = len(chosen)

        hit_rate_sum += len(chosen_set & prev_scheme_fastmem) / float(k)
        ssd_fetches_total += float(len(chosen_set - prev_scheme_fastmem))
        baseline_ssd_fetches_total += float(len(baseline_set - prev_baseline_fastmem))

    hit_rate = hit_rate_sum / float(step_count) if step_count else 0.0
    ssd_fetches_per_token = ssd_fetches_total / float(token_count) if token_count else 0.0
    baseline_ssd_fetches_per_token = baseline_ssd_fetches_total / float(token_count) if token_count else 0.0
    baseline_overlap = baseline_overlap_sum / float(step_count) if step_count else 0.0
    quality_degradation = quality_ratio_sum / float(step_count) if step_count else 1.0

    speedup_ratio = (1.0 + baseline_ssd_fetches_per_token) / (1.0 + ssd_fetches_per_token)
    experts_per_token = float(len(events[0].baseline_experts))

    return {
        "hit_rate": hit_rate,
        "ssd_fetches_per_token": ssd_fetches_per_token,
        "baseline_overlap": baseline_overlap,
        "quality_degradation": quality_degradation,
        "speedup_ratio": speedup_ratio,
        "token_count": float(token_count),
        "experts_per_token": experts_per_token,
        "baseline_ssd_fetches_per_token": baseline_ssd_fetches_per_token,
    }
