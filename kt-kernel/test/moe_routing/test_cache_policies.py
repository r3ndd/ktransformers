import os
import sys

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
SlidingWindowPolicy = cache_policies.SlidingWindowPolicy


def test_sliding_window_retains_recent_experts():
    p = SlidingWindowPolicy(capacity=4, window_size=2)
    p.observe([1, 2])
    p.observe([3, 4])
    assert p.cached() == {1, 2, 3, 4}
    p.observe([5, 6])
    assert 1 not in p.cached()
    assert 5 in p.cached()
