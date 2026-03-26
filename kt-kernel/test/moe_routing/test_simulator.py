import os
import sys

import pandas as pd

_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)

from moe_routing.routing_schemes import SlidingWindowScoreAveragingRouting
from moe_routing.simulator import simulate_routing_scheme


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token_id": [0, 1],
            "context_id": ["c", "c"],
            "token_position": [0, 1],
            "layer_id": [0, 0],
            "expert_ids": [[1, 2], [1, 2]],
            "expert_weights": [[0.9, 0.8], [0.9, 0.8]],
            "expert_scores_all": [[0.0, 0.9, 0.8, 0.1], [0.0, 0.9, 0.8, 0.1]],
        }
    )


def test_window_1_matches_baseline_overlap():
    out = simulate_routing_scheme(_base_df(), SlidingWindowScoreAveragingRouting(window_size=1))
    assert out["baseline_overlap"] == 1.0
    assert out["quality_degradation"] == 1.0


def test_hit_rate_and_ssd_fetches_use_previous_cache_state():
    out = simulate_routing_scheme(_base_df(), SlidingWindowScoreAveragingRouting(window_size=1))
    # token 0 cache miss all 2; token 1 cache hit all 2
    assert out["hit_rate"] == 0.5
    assert out["ssd_fetches_per_token"] == 1.0


def test_speedup_ratio_uses_baseline_fetches_formula():
    out = simulate_routing_scheme(_base_df(), SlidingWindowScoreAveragingRouting(window_size=1))
    expected = (1.0 + out["baseline_ssd_fetches_per_token"]) / (1.0 + out["ssd_fetches_per_token"])
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
    try:
        simulate_routing_scheme(df, SlidingWindowScoreAveragingRouting(window_size=1))
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "expert_scores_all" in str(exc)
