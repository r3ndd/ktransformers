import os
import sys

import pandas as pd

_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)

from moe_routing.routing_schemes import SlidingWindowAdaptivePoolRouting
from moe_routing.simulator import simulate_routing_scheme


def test_alpha_one_recovers_baseline_with_large_pool():
    df = pd.DataFrame(
        {
            "token_id": [0, 1],
            "context_id": ["c", "c"],
            "token_position": [0, 0],
            "layer_id": [0, 0],
            "expert_ids": [[1, 2], [1, 2]],
            "expert_weights": [[0.9, 0.8], [0.9, 0.8]],
            "expert_scores_all": [[0.0, 0.9, 0.8, 0.1], [0.0, 0.9, 0.8, 0.1]],
        }
    )
    scheme = SlidingWindowAdaptivePoolRouting(window_size=1, pool_size_per_layer=4, update_interval_tokens=1)
    out = simulate_routing_scheme(df, scheme, alpha=1.0)
    assert out["hit_rate"] == 1.0
    assert out["quality_degradation"] == 1.0


def test_hard_pool_with_empty_pool_can_miss_baseline():
    df = pd.DataFrame(
        {
            "token_id": [0],
            "context_id": ["c"],
            "token_position": [0],
            "layer_id": [0],
            "expert_ids": [[2, 3]],
            "expert_weights": [[0.9, 0.8]],
            "expert_scores_all": [[0.95, 0.9, 0.8, 0.7]],
        }
    )
    scheme = SlidingWindowAdaptivePoolRouting(window_size=1, pool_size_per_layer=0, update_interval_tokens=1)
    out = simulate_routing_scheme(df, scheme, alpha=0.0)
    assert out["hit_rate"] == 0.0
    assert out["ssd_fetches_per_token"] == 2.0


def test_speedup_ratio_formula_matches_spec():
    df = pd.DataFrame(
        {
            "token_id": [0],
            "context_id": ["c"],
            "token_position": [0],
            "layer_id": [0],
            "expert_ids": [[1, 2]],
            "expert_weights": [[0.9, 0.8]],
            "expert_scores_all": [[0.95, 0.9, 0.8, 0.7]],
        }
    )
    scheme = SlidingWindowAdaptivePoolRouting(window_size=1, pool_size_per_layer=0, update_interval_tokens=1)
    out = simulate_routing_scheme(df, scheme, alpha=0.0)
    expected = (1.0 + (1.0 - out["hit_rate"]) * out["experts_per_token"]) / (1.0 + out["ssd_fetches_per_token"])
    assert abs(out["speedup_ratio"] - expected) < 1e-12


def test_requires_full_score_column():
    df = pd.DataFrame(
        {
            "context_id": ["c"],
            "token_position": [0],
            "layer_id": [0],
            "expert_ids": [[1, 2]],
        }
    )
    scheme = SlidingWindowAdaptivePoolRouting(window_size=1, pool_size_per_layer=2, update_interval_tokens=1)
    try:
        simulate_routing_scheme(df, scheme, alpha=0.5)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "expert_scores_all" in str(exc)
