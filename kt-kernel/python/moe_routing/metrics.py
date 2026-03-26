from __future__ import annotations

import math
from collections import Counter

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
