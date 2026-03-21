import os
import sys

# Add python directory to path and import directly
_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)
# Add moe_routing directory to path for simulator imports
_moe_routing_dir = os.path.join(_python_dir, "moe_routing")
sys.path.insert(0, _moe_routing_dir)

# Import directly from the module files
import importlib.util

# Import cache_policies
spec = importlib.util.spec_from_file_location(
    "cache_policies", os.path.join(_python_dir, "moe_routing", "cache_policies.py")
)
assert spec is not None
assert spec.loader is not None
cache_policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_policies)
SlidingWindowPolicy = cache_policies.SlidingWindowPolicy

# Import simulator
spec_sim = importlib.util.spec_from_file_location("simulator", os.path.join(_python_dir, "moe_routing", "simulator.py"))
assert spec_sim is not None
assert spec_sim.loader is not None
simulator = importlib.util.module_from_spec(spec_sim)
spec_sim.loader.exec_module(simulator)
simulate_policy = simulator.simulate_policy

import pandas as pd


def test_alpha_one_is_unconstrained():
    df = pd.DataFrame(
        {
            "context_id": [0],
            "token_position": [0],
            "layer_id": [0],
            "expert_ids": [[1, 2, 3, 4, 5, 6]],
            "expert_weights": [[1, 1, 1, 1, 1, 1]],
        }
    )
    res = simulate_policy(df, SlidingWindowPolicy(capacity=2, window_size=1), alpha=1.0)
    assert res["quality_proxy_degradation"] == 0.0


def test_alpha_zero_hard_constraint_increases_degradation():
    df = pd.DataFrame(
        {
            "context_id": [0],
            "token_position": [0],
            "layer_id": [0],
            "expert_ids": [[1, 2, 3, 4, 5, 6]],
            "expert_weights": [[1, 1, 1, 1, 1, 1]],
        }
    )
    res = simulate_policy(df, SlidingWindowPolicy(capacity=0, window_size=1), alpha=0.0)
    assert res["partial_hit_rate"] == 0.0


def test_cross_layer_collision_regression():
    """
    Regression test: experts with the same ID in different layers should NOT collide.

    Before the fix, expert 1 in layer 0 and expert 1 in layer 1 would be treated
    as the same expert, causing incorrect cache hit calculations. After the fix,
    each (layer_id, expert_id) pair is treated as a unique expert key.
    """
    # Simulate: layer 0 needs expert 1, layer 1 also needs expert 1
    # These are DIFFERENT experts (same ID, different layers)
    df = pd.DataFrame(
        {
            "context_id": [0, 0],
            "token_position": [0, 1],
            "layer_id": [0, 1],
            "expert_ids": [[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11]],
            "expert_weights": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        }
    )

    # Policy with capacity 1, window size 2
    policy = SlidingWindowPolicy(capacity=1, window_size=2)

    res = simulate_policy(df, policy, alpha=0.5)

    # With layer-qualified keys:
    # - Token 1 (layer 0): needs (0,1), (0,2), ... cache miss, fetches all
    # - Token 2 (layer 1): needs (1,1), (1,7), ...
    #   (1,1) is different from (0,1), so cache miss on (1,1)
    # Partial hit rate should be 0 since no layer-qualified expert is cached
    assert (
        res["partial_hit_rate"] == 0.0
    ), f"Cross-layer collision detected: partial_hit_rate={res['partial_hit_rate']}, expected 0.0"


def test_same_layer_expert_cache_hit():
    """
    Test that experts in the same layer are properly cached.
    """
    df = pd.DataFrame(
        {
            "context_id": [0, 0],
            "token_position": [0, 1],
            "layer_id": [0, 0],  # Same layer
            "expert_ids": [[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 9, 10]],  # Some overlap
            "expert_weights": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        }
    )

    # Policy with capacity large enough to hold all experts from first token
    policy = SlidingWindowPolicy(capacity=10, window_size=2)

    res = simulate_policy(df, policy, alpha=0.5)

    # Token 1 (layer 0): needs 6 experts, cache is empty, 0 hits
    # After observe, cache has all 6 experts from token 1
    # Token 2 (layer 0): needs 6 experts (1,2,7,8,9,10), cache has (1,2,3,4,5,6)
    # Hits: (0,1), (0,2) = 2 hits out of 6 needed
    # Total partial hits: 0 + 2 = 2
    # Total needed: 6 + 6 = 12
    # partial_hit_rate = 2/12 = 0.167
    expected_rate = 2 / 12
    assert res["partial_hit_rate"] > 0.0, f"Same-layer cache hit failed: partial_hit_rate={res['partial_hit_rate']}"
    assert (
        abs(res["partial_hit_rate"] - expected_rate) < 0.001
    ), f"Expected ~{expected_rate:.3f} partial_hit_rate, got {res['partial_hit_rate']}"


def test_layer_qualified_keys_in_cache():
    """
    Verify that the cache contains layer-qualified tuples.
    """
    policy = SlidingWindowPolicy(capacity=10, window_size=2)

    df = pd.DataFrame(
        {
            "context_id": [0, 0],
            "token_position": [0, 1],
            "layer_id": [0, 1],
            "expert_ids": [[1, 2], [3, 4]],
            "expert_weights": [[1, 1], [1, 1]],
        }
    )

    simulate_policy(df, policy, alpha=0.5)

    cached = policy.cached()

    # All cached items should be tuples of (layer_id, expert_id)
    for item in cached:
        assert isinstance(item, tuple), f"Cached item {item} is not a tuple"
        assert len(item) == 2, f"Cached item {item} should have 2 elements"
        assert isinstance(item[0], int), f"Layer ID {item[0]} should be int"
        assert isinstance(item[1], int), f"Expert ID {item[1]} should be int"


def test_token_grouping_merges_layers_for_same_token_key():
    df = pd.DataFrame(
        {
            "context_id": [0, 0],
            "token_position": [5, 5],
            "layer_id": [0, 1],
            "expert_ids": [[1], [2]],
            "expert_weights": [[1], [1]],
        }
    )

    res = simulate_policy(df, SlidingWindowPolicy(capacity=10, window_size=2), alpha=0.5)

    assert res["token_count"] == 1.0
    assert res["simulated_ssd_fetches"] == 2.0
    assert res["avg_misses_per_token"] == 2.0


def test_partial_hit_rate_is_average_of_per_token_partials():
    df = pd.DataFrame(
        {
            "context_id": [0, 0, 0],
            "token_position": [0, 1, 2],
            "layer_id": [0, 0, 0],
            "expert_ids": [[1], [1, 2, 3], [3]],
            "expert_weights": [[1], [1, 1, 1], [1]],
        }
    )

    res = simulate_policy(df, SlidingWindowPolicy(capacity=1, window_size=1), alpha=0.5)

    # Per-token partials: 0.0, 1/3, 0.0 => average 1/9
    assert abs(res["partial_hit_rate"] - (1 / 9)) < 1e-9


def test_full_hit_requires_all_needed_experts_for_token():
    df = pd.DataFrame(
        {
            "context_id": [0, 0, 0, 0],
            "token_position": [0, 0, 1, 1],
            "layer_id": [0, 1, 0, 1],
            "expert_ids": [[1], [2], [1], [3]],
            "expert_weights": [[1], [1], [1], [1]],
        }
    )

    res = simulate_policy(df, SlidingWindowPolicy(capacity=2, window_size=1), alpha=0.5)

    # Token 0: miss both; Token 1: only (0,1) hit and (1,3) miss => no full hit
    assert res["full_hit_rate"] == 0.0
    assert res["hit_rate"] == res["full_hit_rate"]


def test_token_cannot_self_hit_before_post_token_observe():
    df = pd.DataFrame(
        {
            "context_id": [0, 0],
            "token_position": [7, 7],
            "layer_id": [0, 1],
            "expert_ids": [[10], [11]],
            "expert_weights": [[1], [1]],
        }
    )

    res = simulate_policy(df, SlidingWindowPolicy(capacity=10, window_size=2), alpha=0.5)

    assert res["token_count"] == 1.0
    assert res["partial_hit_rate"] == 0.0
    assert res["full_hit_rate"] == 0.0
    assert res["simulated_ssd_fetches"] == 2.0


def test_simulate_policy_raises_clear_error_for_missing_required_columns():
    df = pd.DataFrame({"layer_id": [0], "expert_ids": [[1]]})

    try:
        simulate_policy(df, SlidingWindowPolicy(capacity=10, window_size=2), alpha=0.5)
        assert False, "Expected ValueError for missing columns"
    except ValueError as exc:
        msg = str(exc)
        assert "context_id" in msg
        assert "token_position" in msg
