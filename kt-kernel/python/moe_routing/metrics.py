from __future__ import annotations

import math
from collections import Counter
from collections import deque

import pandas as pd

try:
    from .token_indexing import add_absolute_token_position
except ImportError:
    from token_indexing import add_absolute_token_position


def temporal_reuse_curve(traces: pd.DataFrame, max_distance: int = 64) -> dict[int, float]:
    traces = add_absolute_token_position(traces)
    traces = traces.sort_values(["context_id", "layer_id", "absolute_token_position"]).reset_index(drop=True)
    curve: dict[int, float] = {}
    for d in range(1, max_distance + 1):
        hits = 0
        total = 0
        for _, g in traces.groupby(["context_id", "layer_id"]):
            for i in range(len(g) - d):
                a = set(g.iloc[i]["expert_ids"])
                b = set(g.iloc[i + d]["expert_ids"])
                hits += len(a & b)
                total += len(b)
        curve[d] = (hits / total) if total else 0.0
    return curve


def expert_entropy_by_layer(traces: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for layer_id, g in traces.groupby("layer_id"):
        counts = Counter()
        for ids in g["expert_ids"]:
            counts.update(ids)
        total = sum(counts.values())
        if total == 0:
            out[int(layer_id)] = 0.0
            continue
        probs = [c / total for c in counts.values()]
        out[int(layer_id)] = -sum(p * math.log2(p) for p in probs if p > 0)
    return out


def previous_token_reuse_curve(traces: pd.DataFrame) -> dict[int, float]:
    """Per-step reuse against the immediate previous token.

    Returns a curve indexed by token step (1..n-1), where n is the number of
    absolute tokens in the context. Each point is averaged across layers and
    contexts available in the input dataframe.
    """
    traces = add_absolute_token_position(traces)
    traces = traces.sort_values(["context_id", "layer_id", "absolute_token_position"]).reset_index(drop=True)

    by_step_hits: dict[int, int] = {}
    by_step_total: dict[int, int] = {}

    for _, g in traces.groupby(["context_id", "layer_id"]):
        if len(g) < 2:
            continue
        prev_by_pos: dict[int, set[int]] = {}
        for _, row in g.iterrows():
            pos = int(row["absolute_token_position"])
            ids = set(row["expert_ids"])
            prev = prev_by_pos.get(pos - 1)
            if prev is not None:
                step = pos
                by_step_hits[step] = by_step_hits.get(step, 0) + len(prev & ids)
                by_step_total[step] = by_step_total.get(step, 0) + len(ids)
            prev_by_pos[pos] = ids

    steps = sorted(by_step_total.keys())
    return {step: (by_step_hits.get(step, 0) / by_step_total[step]) for step in steps if by_step_total[step] > 0}


def sliding_window_hit_rate(
    traces: pd.DataFrame, window_sizes: list[int] | tuple[int, ...] = (4, 8, 16, 32, 64)
) -> dict[int, float]:
    """Compute layer-qualified sliding-window hit rates by window size.

    For each context and layer, each token's expert set is compared against the
    union of experts seen in the previous ``window_size`` tokens in that same
    context/layer stream. The returned hit rate is:

        total_hits / total_requested

    where hits are experts already present in the history window.
    """
    traces = add_absolute_token_position(traces)
    traces = traces.sort_values(["context_id", "layer_id", "absolute_token_position"]).reset_index(drop=True)

    out: dict[int, float] = {}
    for window_size in window_sizes:
        if int(window_size) <= 0:
            continue

        total_hits = 0
        total_requested = 0

        for _, g in traces.groupby(["context_id", "layer_id"]):
            history: deque[set[int]] = deque(maxlen=int(window_size))
            for ids in g["expert_ids"]:
                current = set(ids)
                cached: set[int] = set()
                for prev in history:
                    cached.update(prev)

                total_hits += len(current & cached)
                total_requested += len(current)

                history.append(current)

        out[int(window_size)] = (total_hits / total_requested) if total_requested else 0.0

    return out


def context_switch_churn(traces: pd.DataFrame) -> float:
    """Average per-step expert churn between consecutive tokens.

    Churn at a step is measured per context/layer stream as:

        1 - (|prev ∩ curr| / |curr|)

    This is the fraction of current-token experts that were not present in the
    immediate previous token.
    """
    traces = add_absolute_token_position(traces)
    traces = traces.sort_values(["context_id", "layer_id", "absolute_token_position"]).reset_index(drop=True)

    churn_sum = 0.0
    churn_count = 0

    for _, g in traces.groupby(["context_id", "layer_id"]):
        prev_ids: set[int] | None = None
        for ids in g["expert_ids"]:
            current = set(ids)
            if prev_ids is not None and len(current) > 0:
                reuse = len(prev_ids & current) / len(current)
                churn_sum += 1.0 - reuse
                churn_count += 1
            prev_ids = current

    return (churn_sum / churn_count) if churn_count else 0.0
