# Layer-Qualified Simulation Cache Implementation Plan

**Goal:** Fix MoE routing simulation cache identity so cache hits are computed on `(layer_id, expert_id)` instead of raw `expert_id`, eliminating cross-layer collisions and restoring correct `partial_hit_rate` semantics.

**Architecture:** The implementation keeps the simulator pipeline structure but upgrades the cache-policy contract to layer-qualified identities. Simulator data flow will construct qualified experts per row and pass them through `cached()/observe()` consistently. Compatibility is preserved at CLI/output level while explicitly documenting metric interpretation changes.

**Design:** `/root/ktransformers/thoughts/shared/designs/2026-03-20-layer-qualified-simulation-cache-design.md`

---

## Dependency Graph

```text
Batch 1 (parallel): 1.1, 1.2 [foundation - policy+simulator core]
Batch 2 (parallel): 2.1, 2.2 [tests+cli compatibility - depends on batch 1]
Batch 3 (parallel): 3.1 [docs/notes verification - depends on batch 2]
```

---

## Batch 1: Foundation (parallel - 2 implementers)

All tasks in this batch are independent and can run simultaneously.

### Task 1.1: Update cache policy interfaces to layer-qualified identity
**File:** `kt-kernel/python/moe_routing/cache_policies.py`  
**Test:** `kt-kernel/test/moe_routing/test_cache_policies.py`  
**Depends:** none

Design requires layer-qualified cache identity. I’m implementing this as a typed `QualifiedExpert` tuple alias and policy interface update (`observe/cached`) so every cache operation is layer-aware.

```python
# kt-kernel/test/moe_routing/test_cache_policies.py
import os
import sys
import importlib.util

_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)

spec = importlib.util.spec_from_file_location(
    "cache_policies", os.path.join(_python_dir, "moe_routing", "cache_policies.py")
)
cache_policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_policies)

SlidingWindowPolicy = cache_policies.SlidingWindowPolicy
FixedHotPolicy = cache_policies.FixedHotPolicy


def test_sliding_window_retains_recent_qualified_experts():
    p = SlidingWindowPolicy(capacity=4, window_size=2)
    p.observe([(0, 1), (0, 2)])
    p.observe([(1, 1), (1, 2)])
    assert p.cached() == {(0, 1), (0, 2), (1, 1), (1, 2)}

    p.observe([(2, 5), (2, 6)])
    cached = p.cached()
    assert (0, 1) not in cached
    assert (2, 5) in cached


def test_fixed_hot_policy_preserves_layer_qualification():
    p = FixedHotPolicy(hot_experts=[(0, 7), (3, 7)])
    assert p.cached() == {(0, 7), (3, 7)}
```

```python
# kt-kernel/python/moe_routing/cache_policies.py
from __future__ import annotations

from collections import Counter, deque
from typing import Iterable

QualifiedExpert = tuple[int, int]  # (layer_id, expert_id)


class BasePolicy:
    def observe(self, experts: Iterable[QualifiedExpert]) -> None:
        raise NotImplementedError

    def cached(self) -> set[QualifiedExpert]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class BaselinePolicy(BasePolicy):
    def observe(self, experts: Iterable[QualifiedExpert]) -> None:
        return

    def cached(self) -> set[QualifiedExpert]:
        return set()


class SlidingWindowPolicy(BasePolicy):
    def __init__(self, capacity: int, window_size: int):
        self.capacity = capacity
        self.window: deque[set[QualifiedExpert]] = deque(maxlen=window_size)

    def observe(self, experts: Iterable[QualifiedExpert]) -> None:
        self.window.append(set(experts))

    def cached(self) -> set[QualifiedExpert]:
        merged: set[QualifiedExpert] = set()
        for s in self.window:
            merged.update(s)
        if len(merged) <= self.capacity:
            return merged
        return set(list(merged)[: self.capacity])


class FixedHotPolicy(BasePolicy):
    def __init__(self, hot_experts: list[QualifiedExpert]):
        self._hot: set[QualifiedExpert] = set(hot_experts)

    def observe(self, experts: Iterable[QualifiedExpert]) -> None:
        return

    def cached(self) -> set[QualifiedExpert]:
        return set(self._hot)


def build_hotset(freq: Counter[QualifiedExpert], pool_size: int) -> list[QualifiedExpert]:
    return [x for x, _ in freq.most_common(pool_size)]
```

**Verify:** `pytest -q kt-kernel/test/moe_routing/test_cache_policies.py`
**Commit:** `fix(moe-routing): qualify cache policy identity by layer`

### Task 1.2: Update simulator data flow to pass layer-qualified experts
**File:** `kt-kernel/python/moe_routing/simulator.py`  
**Test:** `kt-kernel/test/moe_routing/test_simulator.py`  
**Depends:** none

Design requires full needed/cache/observe path to include `layer_id`. I’m implementing row-local qualification in simulator so every hit/miss uses `(layer_id, expert_id)`.

```python
# kt-kernel/test/moe_routing/test_simulator.py
import os
import sys
import importlib.util
import pandas as pd

_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)
_moe_routing_dir = os.path.join(_python_dir, "moe_routing")
sys.path.insert(0, _moe_routing_dir)

spec = importlib.util.spec_from_file_location(
    "cache_policies", os.path.join(_python_dir, "moe_routing", "cache_policies.py")
)
cache_policies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_policies)
SlidingWindowPolicy = cache_policies.SlidingWindowPolicy
FixedHotPolicy = cache_policies.FixedHotPolicy

spec_sim = importlib.util.spec_from_file_location(
    "simulator", os.path.join(_python_dir, "moe_routing", "simulator.py")
)
simulator = importlib.util.module_from_spec(spec_sim)
spec_sim.loader.exec_module(simulator)
simulate_policy = simulator.simulate_policy


def test_alpha_one_is_unconstrained():
    df = pd.DataFrame({
        "layer_id": [0],
        "expert_ids": [[1, 2, 3, 4, 5, 6]],
        "expert_weights": [[1, 1, 1, 1, 1, 1]],
    })
    res = simulate_policy(df, SlidingWindowPolicy(capacity=2, window_size=1), alpha=1.0)
    assert res["quality_proxy_degradation"] == 0.0


def test_alpha_zero_hard_constraint_increases_degradation():
    df = pd.DataFrame({
        "layer_id": [0],
        "expert_ids": [[1, 2, 3, 4, 5, 6]],
        "expert_weights": [[1, 1, 1, 1, 1, 1]],
    })
    res = simulate_policy(df, SlidingWindowPolicy(capacity=0, window_size=1), alpha=0.0)
    assert res["partial_hit_rate"] == 0.0


def test_cross_layer_id_collision_no_longer_counts_as_hit():
    df = pd.DataFrame({
        "layer_id": [0, 1],
        "expert_ids": [[7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12]],
        "expert_weights": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
    })
    policy = SlidingWindowPolicy(capacity=6, window_size=1)
    res = simulate_policy(df, policy, alpha=1.0)
    assert res["hit_rate"] == 0.0
    assert res["partial_hit_rate"] == 0.0


def test_fixed_hot_layer_specific_identity_behavior():
    df = pd.DataFrame({
        "layer_id": [0, 1],
        "expert_ids": [[7, 1, 2, 3, 4, 5], [7, 1, 2, 3, 4, 5]],
        "expert_weights": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
    })
    policy = FixedHotPolicy(hot_experts=[(0, 7), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
    res = simulate_policy(df, policy, alpha=1.0)
    assert res["partial_hit_rate"] == 0.5
```

```python
# kt-kernel/python/moe_routing/simulator.py
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
        layer_id = int(row["layer_id"])
        needed = {(layer_id, int(expert_id)) for expert_id in row["expert_ids"]}
        cache = policy.cached()
        hit = needed & cache

        total_tokens += 1
        total_needed += len(needed)
        cached_needed += len(hit)
        if len(hit) == len(needed):
            full_hits += 1
        simulated_fetches += len(needed - cache)

        policy.observe(needed)

    hit_rate = full_hits / total_tokens if total_tokens else 0.0
    partial_hit_rate = cached_needed / total_needed if total_needed else 0.0
    return {
        "hit_rate": hit_rate,
        "partial_hit_rate": partial_hit_rate,
        "simulated_ssd_fetches": float(simulated_fetches),
        "quality_proxy_degradation": estimate_quality_proxy(hit_rate, partial_hit_rate, alpha),
    }
```

**Verify:** `pytest -q kt-kernel/test/moe_routing/test_simulator.py`
**Commit:** `fix(moe-routing): use layer-qualified identity in simulator replay`

---

## Batch 2: Compatibility + Regression Coverage (parallel - 2 implementers)

All tasks in this batch depend on Batch 1.

### Task 2.1: Ensure simulate CLI remains compatible with updated policy contract
**File:** `kt-kernel/python/moe_routing/simulate.py`  
**Test:** `kt-kernel/test/moe_routing/test_simulate_cli.py`  
**Depends:** 1.1, 1.2

Design requires output compatibility and interpretation notes. I’m keeping CLI JSON schema stable and adding a compatibility metadata flag to make metric semantics explicit.

```python
# kt-kernel/test/moe_routing/test_simulate_cli.py
from pathlib import Path
import json

import pandas as pd

from moe_routing.simulate import run_simulation


def test_run_simulation_writes_results(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    pd.DataFrame(
        {
            "layer_id": [0],
            "expert_ids": [[1, 2, 3, 4, 5, 6]],
            "expert_weights": [[1, 1, 1, 1, 1, 1]],
        }
    ).to_parquet(in_file)
    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)
    result = json.loads((out_dir / "results.json").read_text())
    assert len(result["runs"]) > 0
    assert result["cache_identity"] == "layer_qualified"
```

```python
# kt-kernel/python/moe_routing/simulate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .cache_policies import SlidingWindowPolicy
from .simulator import simulate_policy


def run_simulation(trace_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = pd.read_parquet(trace_file)

    runs = []
    for window in [8, 16, 32, 64]:
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            policy = SlidingWindowPolicy(capacity=32, window_size=window)
            m = simulate_policy(traces, policy, alpha=alpha)
            m.update({"policy": "sliding_window", "window_size": window, "alpha": alpha})
            runs.append(m)

    payload = {
        "cache_identity": "layer_qualified",
        "metric_note": "partial_hit_rate counts only layer-qualified cache matches",
        "runs": runs,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    try:
        import matplotlib.pyplot as plt

        xs = [r["partial_hit_rate"] for r in runs]
        ys = [1.0 - r["quality_proxy_degradation"] for r in runs]
        plt.figure(figsize=(6, 5))
        plt.scatter(xs, ys)
        plt.xlabel("Partial Hit Rate")
        plt.ylabel("Quality Proxy (1 - degradation)")
        plt.title("Cache Tradeoff Frontier")
        plt.tight_layout()
        plt.savefig(output_dir / "tradeoff_curves.png")
        plt.close()
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser("moe-routing-simulate")
    p.add_argument("--trace-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run_simulation(args.trace_file, args.output_dir)


if __name__ == "__main__":
    main()
```

**Verify:** `pytest -q kt-kernel/test/moe_routing/test_simulate_cli.py`
**Commit:** `fix(moe-routing): annotate layer-qualified cache semantics in simulation output`

### Task 2.2: Add focused regression test for cross-layer collision semantics
**File:** `kt-kernel/test/moe_routing/test_simulator.py`  
**Test:** `kt-kernel/test/moe_routing/test_simulator.py`  
**Depends:** 1.2

This task is a dedicated handoff checkpoint to ensure the cross-layer regression remains present even if other tests are refactored.

```python
# Add this test function in kt-kernel/test/moe_routing/test_simulator.py
def test_regression_cross_layer_same_expert_ids_do_not_collide_in_cache():
    df = pd.DataFrame({
        "layer_id": [5, 6],
        "expert_ids": [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
        "expert_weights": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
    })
    policy = SlidingWindowPolicy(capacity=6, window_size=1)
    res = simulate_policy(df, policy, alpha=1.0)

    # Before fix, this could be >0 due to raw expert_id collisions.
    assert res["partial_hit_rate"] == 0.0
    assert res["hit_rate"] == 0.0
```

**Verify:** `pytest -q kt-kernel/test/moe_routing/test_simulator.py -k cross_layer`
**Commit:** `test(moe-routing): add cross-layer cache identity regression`

---

## Batch 3: Final verification + interpretation notes (sequential)

### Task 3.1: Run full simulation-path verification and capture interpretation guidance
**File:** `thoughts/shared/plans/2026-03-20-layer-qualified-simulation-cache.md`  
**Test:** none  
**Depends:** 2.1, 2.2

Document final operator guidance:
- `partial_hit_rate` may decrease compared with historical runs because collisions are now removed.
- `simulated_ssd_fetches` may increase correspondingly; this is expected and more accurate.
- Compare only against runs produced after this fix (or runs that explicitly declare `cache_identity=layer_qualified`).

**Verify commands:**

```bash
pytest -q kt-kernel/test/moe_routing/test_cache_policies.py
pytest -q kt-kernel/test/moe_routing/test_simulator.py
pytest -q kt-kernel/test/moe_routing/test_simulate_cli.py
pytest -q kt-kernel/test/moe_routing
python -m moe_routing.simulate --trace-file /tmp/sample_trace.parquet --output-dir /tmp/moe_sim_out
python - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/moe_sim_out/results.json')
if p.exists():
    data = json.loads(p.read_text())
    print(data.get('cache_identity'), data.get('metric_note'))
PY
```

**Commit:** `docs(moe-routing): add layer-qualified simulation cache implementation plan`

---

## Compatibility Decisions Summary

1. **Policy interface updates:** required and breaking at Python type level inside simulation internals; accepted due to scope confinement.
2. **Baseline policy:** no semantic change; still yields empty cache, now typed as qualified identities.
3. **Fixed-hot policy:** explicitly layer-qualified hotsets are canonical to avoid ambiguity.
4. **Simulator data flow:** always constructs qualified identities from row `layer_id` + `expert_ids`.
5. **CLI/output compatibility:** keys remain stable; additional metadata clarifies new metric semantics.

## Executor Handoff Checklist

- [ ] Update `cache_policies.py` to `QualifiedExpert` identity end-to-end.
- [ ] Update `simulator.py` to build/use `(layer_id, expert_id)` in needed/cache/observe.
- [ ] Keep baseline/fixed-hot behavior compatible under layer-qualified identity.
- [ ] Add/retain regression test proving no cross-layer ID collision hits.
- [ ] Keep `simulate.py` result keys stable; add semantic metadata note.
- [ ] Run full test commands and confirm pass.
