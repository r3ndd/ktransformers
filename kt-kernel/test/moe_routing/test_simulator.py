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
cache_policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_policies)
SlidingWindowPolicy = cache_policies.SlidingWindowPolicy

# Import simulator
spec_sim = importlib.util.spec_from_file_location("simulator", os.path.join(_python_dir, "moe_routing", "simulator.py"))
simulator = importlib.util.module_from_spec(spec_sim)
spec_sim.loader.exec_module(simulator)
simulate_policy = simulator.simulate_policy

import pandas as pd


def test_alpha_one_is_unconstrained():
    df = pd.DataFrame({"layer_id": [0], "expert_ids": [[1, 2, 3, 4, 5, 6]], "expert_weights": [[1, 1, 1, 1, 1, 1]]})
    res = simulate_policy(df, SlidingWindowPolicy(capacity=2, window_size=1), alpha=1.0)
    assert res["quality_proxy_degradation"] == 0.0


def test_alpha_zero_hard_constraint_increases_degradation():
    df = pd.DataFrame({"layer_id": [0], "expert_ids": [[1, 2, 3, 4, 5, 6]], "expert_weights": [[1, 1, 1, 1, 1, 1]]})
    res = simulate_policy(df, SlidingWindowPolicy(capacity=0, window_size=1), alpha=0.0)
    assert res["partial_hit_rate"] == 0.0
