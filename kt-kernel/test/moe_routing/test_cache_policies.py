import os
import sys
from collections import Counter

# Add python directory to path and import directly
_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)

# Import directly from the module file
import importlib.util

spec = importlib.util.spec_from_file_location(
    "cache_policies", os.path.join(_python_dir, "moe_routing", "cache_policies.py")
)
cache_policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_policies)

# Import all classes and functions
SlidingWindowPolicy = cache_policies.SlidingWindowPolicy
FixedHotPolicy = cache_policies.FixedHotPolicy
BaselinePolicy = cache_policies.BaselinePolicy
LayerAwareSlidingWindowPolicy = cache_policies.LayerAwareSlidingWindowPolicy
build_hotset = cache_policies.build_hotset
ExpertKey = cache_policies.ExpertKey
QualifiedExpertId = cache_policies.QualifiedExpertId


# =============================================================================
# Basic Regression Tests (from original test file)
# =============================================================================


def test_sliding_window_retains_recent_experts():
    """Original regression test - basic sliding window behavior."""
    p = SlidingWindowPolicy(capacity=4, window_size=2)
    p.observe([(0, 1), (0, 2)])  # Use qualified identities
    p.observe([(0, 3), (0, 4)])
    assert p.cached() == {(0, 1), (0, 2), (0, 3), (0, 4)}
    p.observe([(0, 5), (0, 6)])
    assert (0, 1) not in p.cached()
    assert (0, 5) in p.cached()


# =============================================================================
# Qualified Identity Tests (layer_id, expert_id)
# =============================================================================


def test_qualified_identity_types():
    """Test that ExpertKey type alias works correctly."""
    # ExpertKey should be tuple[int, int]
    key: ExpertKey = (0, 5)
    assert isinstance(key, tuple)
    assert len(key) == 2
    assert key[0] == 0  # layer_id
    assert key[1] == 5  # expert_id


def test_sliding_window_with_qualified_identities():
    """Test SlidingWindowPolicy with qualified (layer_id, expert_id) tuples."""
    p = SlidingWindowPolicy(capacity=6, window_size=3)

    # Observe experts from different layers
    p.observe([(0, 1), (0, 2), (1, 1), (1, 3)])
    p.observe([(0, 3), (1, 2), (2, 1)])
    p.observe([(0, 4), (2, 2), (2, 3)])

    cached = p.cached()
    assert len(cached) <= 6

    # Check specific experts are cached (from first observations due to sorting)
    # With 10 experts total and capacity=6, sorted order takes first 6:
    # (0,1), (0,2), (0,3), (0,4), (1,1), (1,2) are cached
    assert (0, 1) in cached  # From first observation
    assert (0, 4) in cached  # From last observation, but early in sort order


def test_sliding_window_respects_capacity_with_qualified():
    """Test that SlidingWindowPolicy respects capacity with qualified identities."""
    p = SlidingWindowPolicy(capacity=3, window_size=2)

    # Add more experts than capacity
    p.observe([(0, 1), (0, 2), (0, 3), (0, 4)])
    p.observe([(0, 5), (0, 6)])

    cached = p.cached()
    assert len(cached) <= 3


def test_sliding_window_empty_input():
    """Test SlidingWindowPolicy with empty input."""
    p = SlidingWindowPolicy(capacity=4, window_size=2)
    p.observe([])
    assert p.cached() == set()
    p.observe([(0, 1)])
    assert p.cached() == {(0, 1)}


def test_sliding_window_reset():
    """Test SlidingWindowPolicy reset functionality."""
    p = SlidingWindowPolicy(capacity=4, window_size=2)
    p.observe([(0, 1), (0, 2)])
    p.observe([(0, 3), (0, 4)])
    assert len(p.cached()) == 4

    p.reset()
    assert p.cached() == set()
    assert len(p.window) == 0


# =============================================================================
# FixedHotPolicy Tests
# =============================================================================


def test_fixed_hot_policy_with_qualified():
    """Test FixedHotPolicy with qualified identities."""
    hot_experts = [(0, 1), (0, 5), (1, 2), (2, 3)]
    p = FixedHotPolicy(hot_experts)

    # Observe should not change cached set
    p.observe([(0, 10), (1, 20)])
    assert p.cached() == set(hot_experts)

    p.observe([])
    assert p.cached() == set(hot_experts)


def test_fixed_hot_policy_deterministic():
    """Test that FixedHotPolicy returns consistent results."""
    hot_experts = [(0, 1), (1, 2), (2, 3)]
    p = FixedHotPolicy(hot_experts)

    # Multiple calls should return same result
    assert p.cached() == p.cached()
    assert p.cached() == {(0, 1), (1, 2), (2, 3)}


def test_fixed_hot_policy_reset():
    """Test FixedHotPolicy reset functionality."""
    p = FixedHotPolicy([(0, 1), (0, 2)])
    p.reset()
    # Reset should not affect FixedHotPolicy
    assert p.cached() == {(0, 1), (0, 2)}


# =============================================================================
# BaselinePolicy Tests
# =============================================================================


def test_baseline_policy():
    """Test BaselinePolicy caches nothing."""
    p = BaselinePolicy()

    p.observe([(0, 1), (0, 2), (1, 3)])
    assert p.cached() == set()

    p.observe([])
    assert p.cached() == set()

    p.reset()
    assert p.cached() == set()


# =============================================================================
# LayerAwareSlidingWindowPolicy Tests
# =============================================================================


def test_layer_aware_policy_basic():
    """Test LayerAwareSlidingWindowPolicy basic functionality."""
    p = LayerAwareSlidingWindowPolicy(capacity=10, window_size=2, num_layers=3)

    # Observe experts from different layers
    p.observe([(0, 1), (0, 2), (1, 1), (1, 2), (2, 1)])

    cached = p.cached()
    assert (0, 1) in cached
    assert (0, 2) in cached
    assert (1, 1) in cached
    assert (1, 2) in cached
    assert (2, 1) in cached


def test_layer_aware_policy_per_layer_tracking():
    """Test LayerAwareSlidingWindowPolicy tracks per-layer experts."""
    p = LayerAwareSlidingWindowPolicy(capacity=10, window_size=2, num_layers=3)

    p.observe([(0, 1), (0, 2), (1, 5), (2, 10)])
    p.observe([(0, 3), (1, 6), (2, 11)])

    # Check per-layer cached experts
    assert p.cached_for_layer(0) == {1, 2, 3}
    assert p.cached_for_layer(1) == {5, 6}
    assert p.cached_for_layer(2) == {10, 11}


def test_layer_aware_policy_window_expires():
    """Test that old observations expire from the window."""
    p = LayerAwareSlidingWindowPolicy(capacity=10, window_size=2, num_layers=2)

    p.observe([(0, 1), (1, 1)])
    p.observe([(0, 2), (1, 2)])
    assert (0, 1) in p.cached()

    # Third observation pushes first out
    p.observe([(0, 3), (1, 3)])
    assert (0, 1) not in p.cached()
    assert (0, 2) in p.cached()
    assert (0, 3) in p.cached()


def test_layer_aware_policy_capacity_limit():
    """Test LayerAwareSlidingWindowPolicy respects capacity."""
    p = LayerAwareSlidingWindowPolicy(capacity=4, window_size=3, num_layers=2)

    # Add more experts than capacity across layers
    p.observe([(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)])
    p.observe([(0, 4), (1, 4)])

    cached = p.cached()
    assert len(cached) <= 4


def test_layer_aware_policy_empty_input():
    """Test LayerAwareSlidingWindowPolicy with empty input."""
    p = LayerAwareSlidingWindowPolicy(capacity=4, window_size=2, num_layers=2)

    p.observe([])
    assert p.cached() == set()

    p.observe([(0, 1)])
    assert p.cached() == {(0, 1)}


def test_layer_aware_policy_reset():
    """Test LayerAwareSlidingWindowPolicy reset functionality."""
    p = LayerAwareSlidingWindowPolicy(capacity=10, window_size=2, num_layers=2)

    p.observe([(0, 1), (0, 2), (1, 1)])
    assert len(p.cached()) > 0

    p.reset()
    assert p.cached() == set()
    assert len(p.cached_for_layer(0)) == 0
    assert len(p.cached_for_layer(1)) == 0


def test_layer_aware_policy_invalid_layer():
    """Test LayerAwareSlidingWindowPolicy raises KeyError for layer_id out of range."""
    p = LayerAwareSlidingWindowPolicy(capacity=10, window_size=2, num_layers=2)

    # Layer 5 is out of range (only layers 0-1 are valid), should raise KeyError
    try:
        p.observe([(0, 1), (5, 1)])
        assert False, "Expected KeyError for invalid layer_id"
    except KeyError:
        pass  # Expected behavior

    # Valid observation should work
    p.observe([(0, 1)])
    assert (0, 1) in p.cached()


# =============================================================================
# build_hotset Tests
# =============================================================================


def test_build_hotset_with_qualified():
    """Test build_hotset with qualified identities."""
    freq = Counter(
        {
            (0, 1): 100,
            (0, 5): 80,
            (1, 2): 60,
            (1, 3): 40,
            (2, 1): 20,
        }
    )

    hotset = build_hotset(freq, pool_size=3)
    assert len(hotset) == 3
    assert hotset[0] == (0, 1)  # Most frequent
    assert hotset[1] == (0, 5)  # Second most frequent
    assert hotset[2] == (1, 2)  # Third most frequent


def test_build_hotset_pool_size_larger_than_freq():
    """Test build_hotset when pool_size > number of experts."""
    freq = Counter({(0, 1): 10, (0, 2): 5})

    hotset = build_hotset(freq, pool_size=10)
    assert len(hotset) == 2
    assert (0, 1) in hotset
    assert (0, 2) in hotset


def test_build_hotset_empty_freq():
    """Test build_hotset with empty frequency counter."""
    freq: Counter[QualifiedExpertId] = Counter()
    hotset = build_hotset(freq, pool_size=5)
    assert hotset == []


def test_build_hotset_pool_size_zero():
    """Test build_hotset with pool_size=0."""
    freq = Counter({(0, 1): 100, (0, 2): 50})
    hotset = build_hotset(freq, pool_size=0)
    assert hotset == []


# =============================================================================
# Integration Tests
# =============================================================================


def test_policy_integration_scenario():
    """Test a realistic scenario with multiple policies."""
    # Simulate routing data
    routing_data = [
        [(0, 1), (0, 2), (1, 1), (1, 3)],
        [(0, 1), (0, 3), (1, 2), (1, 3)],
        [(0, 2), (0, 4), (1, 1), (1, 4)],
    ]

    # Use SlidingWindowPolicy
    policy = SlidingWindowPolicy(capacity=6, window_size=2)
    for experts in routing_data:
        policy.observe(experts)

    cached = policy.cached()
    assert len(cached) <= 6

    # Build hotset from frequency
    freq: Counter[QualifiedExpertId] = Counter()
    for experts in routing_data:
        freq.update(experts)

    hotset = build_hotset(freq, pool_size=4)
    assert len(hotset) == 4

    # Use FixedHotPolicy with the hotset
    hot_policy = FixedHotPolicy(hotset)
    assert hot_policy.cached() == set(hotset)


def test_cross_layer_expert_distinction():
    """Test that same expert_id in different layers are distinct."""
    p = SlidingWindowPolicy(capacity=10, window_size=2)

    # Same expert_id but different layers
    p.observe([(0, 5), (1, 5), (2, 5)])
    cached = p.cached()

    # All three should be cached (they're different qualified identities)
    assert (0, 5) in cached
    assert (1, 5) in cached
    assert (2, 5) in cached


# =============================================================================
# Type Export Tests
# =============================================================================


def test_type_aliases_available():
    """Test that all type aliases are exported."""
    assert hasattr(cache_policies, "ExpertKey")
    assert hasattr(cache_policies, "ExpertId")
    assert hasattr(cache_policies, "QualifiedExpertId")
    assert hasattr(cache_policies, "ExpertSet")
    assert hasattr(cache_policies, "ExpertList")


def test_classes_exported():
    """Test that all policy classes are exported."""
    assert hasattr(cache_policies, "BasePolicy")
    assert hasattr(cache_policies, "BaselinePolicy")
    assert hasattr(cache_policies, "SlidingWindowPolicy")
    assert hasattr(cache_policies, "FixedHotPolicy")
    assert hasattr(cache_policies, "LayerAwareSlidingWindowPolicy")
    assert hasattr(cache_policies, "build_hotset")
