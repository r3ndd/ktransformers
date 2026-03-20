from __future__ import annotations

import math
from collections import Counter

import pandas as pd


def temporal_reuse_curve(traces: pd.DataFrame, max_distance: int = 64) -> dict[int, float]:
    traces = traces.sort_values(["layer_id", "token_position"]).reset_index(drop=True)
    curve: dict[int, float] = {}
    for d in range(1, max_distance + 1):
        hits = 0
        total = 0
        for _, g in traces.groupby("layer_id"):
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
