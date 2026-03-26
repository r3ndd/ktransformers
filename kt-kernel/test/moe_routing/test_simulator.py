import os
import sys

import pandas as pd

_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)

from moe_routing.routing_schemes import (
    EMAScoreAveragingRouting,
    SlidingWindowScoreAveragingRouting,
    TwoTimescaleSoftmaxRouting,
    TwoTimescaleEMARouting,
)
from moe_routing.simulator import (
    BASE_SECONDS_PER_TOKEN_NO_SSD,
    EXTRA_SECONDS_PER_SSD_FETCH,
    simulate_routing_scheme,
)


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
    expected = (BASE_SECONDS_PER_TOKEN_NO_SSD + out["baseline_ssd_fetches_per_token"] * EXTRA_SECONDS_PER_SSD_FETCH) / (
        BASE_SECONDS_PER_TOKEN_NO_SSD + out["ssd_fetches_per_token"] * EXTRA_SECONDS_PER_SSD_FETCH
    )
    assert abs(out["speedup_ratio"] - expected) < 1e-12
    assert abs(out["quality_speed_score"] - (out["quality_degradation"] * out["speedup_ratio"])) < 1e-12


def test_quality_degradation_uses_softmax_probabilities_and_is_bounded_by_one():
    df = pd.DataFrame(
        {
            "token_id": [0, 1],
            "context_id": ["c", "c"],
            "token_position": [0, 1],
            "layer_id": [0, 0],
            "expert_ids": [[0], [1]],
            "expert_weights": [[1.0], [1.0]],
            # token 1 baseline prefers expert 1, but W=2 smoothing ties 0 and 1,
            # then top-k tie-break selects expert 0 (lower index)
            "expert_scores_all": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
        }
    )

    # W=2 mixes previous token and current token scores.
    out = simulate_routing_scheme(df, SlidingWindowScoreAveragingRouting(window_size=2))
    assert out["baseline_overlap"] < 1.0
    assert out["quality_degradation"] <= 1.0


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


def test_ema_beta_one_matches_baseline_overlap():
    out = simulate_routing_scheme(_base_df(), EMAScoreAveragingRouting(ema_beta=1.0))
    assert out["baseline_overlap"] == 1.0


def test_ema_invalid_beta_raises():
    for bad in [0.0, -0.1, 1.1]:
        try:
            EMAScoreAveragingRouting(ema_beta=bad)
            assert False, f"Expected ValueError for ema_beta={bad}"
        except ValueError:
            pass


def test_two_timescale_invalid_lambda_raises():
    for bad in [0.0, 1.0, -0.1, 1.1]:
        try:
            TwoTimescaleEMARouting(mix_lambda=bad)
            assert False, f"Expected ValueError for mix_lambda={bad}"
        except ValueError:
            pass


def test_two_timescale_includes_mix_lambda_and_runs():
    out = simulate_routing_scheme(_base_df(), TwoTimescaleEMARouting(mix_lambda=0.5))
    assert "hit_rate" in out
    assert "quality_degradation" in out


def test_two_timescale_plus_current_runs_with_rho_zero_and_positive():
    out0 = simulate_routing_scheme(_base_df(), TwoTimescaleSoftmaxRouting(mix_lambda=0.3, rho=0.0))
    out1 = simulate_routing_scheme(_base_df(), TwoTimescaleSoftmaxRouting(mix_lambda=0.3, rho=1.0))
    assert "baseline_overlap" in out0
    assert "quality_degradation" in out1


def test_two_timescale_long_beta_is_updated_to_point_zero_five():
    scheme = TwoTimescaleEMARouting(mix_lambda=0.5)
    assert abs(scheme.LONG_BETA - 0.05) < 1e-12
