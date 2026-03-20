from __future__ import annotations

import pandas as pd

try:
    from .cache_policies import BasePolicy
except ImportError:
    from cache_policies import BasePolicy


def estimate_quality_proxy(hit_rate: float, partial_hit_rate: float, alpha: float) -> float:
    constrained_penalty = (1.0 - partial_hit_rate) * (1.0 - alpha)
    return max(0.0, min(1.0, constrained_penalty))


def simulate_policy(traces: pd.DataFrame, policy: BasePolicy, alpha: float) -> dict[str, float]:
    assert 0.0 <= alpha <= 1.0
    total_tokens = 0
    full_hits = 0
    cached_needed = 0
    total_needed = 0
    simulated_fetches = 0

    for _, row in traces.iterrows():
        needed = set(row["expert_ids"])
        cache = policy.cached()
        hit = needed & cache

        total_tokens += 1
        total_needed += len(needed)
        cached_needed += len(hit)
        if len(hit) == len(needed):
            full_hits += 1
        simulated_fetches += len(needed - cache)

        policy.observe(list(needed))

    hit_rate = full_hits / total_tokens if total_tokens else 0.0
    partial_hit_rate = cached_needed / total_needed if total_needed else 0.0
    return {
        "hit_rate": hit_rate,
        "partial_hit_rate": partial_hit_rate,
        "simulated_ssd_fetches": float(simulated_fetches),
        "quality_proxy_degradation": estimate_quality_proxy(hit_rate, partial_hit_rate, alpha),
    }
