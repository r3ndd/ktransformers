from __future__ import annotations

import pandas as pd

try:
    from .cache_policies import BasePolicy
except ImportError:
    from cache_policies import BasePolicy


def estimate_quality_proxy(hit_rate: float, partial_hit_rate: float, alpha: float) -> float:
    constrained_penalty = (1.0 - partial_hit_rate) * (1.0 - alpha)
    return max(0.0, min(1.0, constrained_penalty))


def _build_token_needed_experts(traces: pd.DataFrame) -> list[set[tuple[int, int]]]:
    required_columns = {"context_id", "token_position", "layer_id", "expert_ids"}
    missing = sorted(required_columns - set(traces.columns))
    if missing:
        raise ValueError(f"simulate_policy requires columns {sorted(required_columns)}; missing {missing}")

    token_groups: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for context_id, token_position, layer_id, expert_ids in traces[
        ["context_id", "token_position", "layer_id", "expert_ids"]
    ].itertuples(index=False, name=None):
        token_key = (context_id, token_position)

        needed = token_groups.setdefault(token_key, set())
        for expert_id in expert_ids:
            needed.add((layer_id, expert_id))

    return list(token_groups.values())


def simulate_policy(traces: pd.DataFrame, policy: BasePolicy, alpha: float) -> dict[str, float]:
    """Simulate a cache policy on routing traces.

    Uses layer-qualified expert tuples (layer_id, expert_id) to avoid cross-layer
    collisions where the same expert ID in different layers would be incorrectly
    treated as the same expert.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1 inclusive; got {alpha}")

    token_needed_sets = _build_token_needed_experts(traces)

    token_count = 0
    full_hits = 0
    partial_hit_sum = 0.0
    total_misses = 0

    for needed in token_needed_sets:
        cache = policy.cached()
        hits = len(needed & cache)
        needed_count = len(needed)
        misses = len(needed - cache)

        token_count += 1
        if needed_count == 0:
            partial_hit_sum += 1.0
            full_hits += 1
        else:
            partial_hit_sum += hits / needed_count
            if hits == needed_count:
                full_hits += 1
        total_misses += misses

        # Pass layer-qualified experts to policy after token evaluation
        policy.observe(list(needed))

    full_hit_rate = full_hits / token_count if token_count else 0.0
    partial_hit_rate = partial_hit_sum / token_count if token_count else 0.0
    return {
        "hit_rate": full_hit_rate,
        "full_hit_rate": full_hit_rate,
        "partial_hit_rate": partial_hit_rate,
        "simulated_ssd_fetches": float(total_misses),
        "avg_misses_per_token": float(total_misses / token_count) if token_count else 0.0,
        "quality_proxy_degradation": estimate_quality_proxy(full_hit_rate, partial_hit_rate, alpha),
        "token_count": float(token_count),
    }
