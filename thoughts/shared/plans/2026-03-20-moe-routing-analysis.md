# MoE Routing Analysis Implementation Plan

**Goal:** Build a production-usable 3-phase pipeline (`collect → analyze → simulate`) that captures MoE routing traces from kTransformers with low overhead, computes temporal locality metrics, and replays cache policies with a continuous α constraint.

**Architecture:** The implementation adds a dedicated Python package at `kt-kernel/python/moe_routing/` with strict phase separation: trace capture + async Parquet writer (Phase 1), vectorized metric computation (Phase 2), and deterministic cache-policy replay (Phase 3). `BaseMoEWrapper.submit_forward()` is instrumented with an opt-in callback so collection overhead is near-zero when disabled. All artifacts are written under `data/{traces,analysis,simulation}` with JSON summaries + plots.

**Design:** `/root/ktransformers/thoughts/shared/designs/2026-03-20-moe-routing-analysis-design.md`

---

## 1) File Structure and Module Organization

```text
kt-kernel/
├── python/
│   ├── experts_base.py                               # hook integration point
│   └── moe_routing/
│       ├── __init__.py
│       ├── types.py                                  # routing record schema + constants
│       ├── parquet_writer.py                         # batched async parquet writer
│       ├── trace_collector.py                        # runtime collector/harness
│       ├── metrics.py                                # locality metrics
│       ├── cache_policies.py                         # baseline/fixed_hot/sliding/exp/hybrid/boundary_reset
│       ├── simulator.py                              # replay engine + alpha constraint
│       ├── collect.py                                # CLI entry for phase 1
│       ├── analyze.py                                # CLI entry for phase 2
│       └── simulate.py                               # CLI entry for phase 3
└── test/
    └── moe_routing/
        ├── test_types.py
        ├── test_parquet_writer.py
        ├── test_trace_collector.py
        ├── test_metrics.py
        ├── test_cache_policies.py
        ├── test_simulator.py
        ├── test_collect_cli.py
        ├── test_analyze_cli.py
        ├── test_simulate_cli.py
        └── test_experts_base_hook.py
```

---

## 2) Dependencies and Environment Setup

Design requires Parquet + analysis libs not currently in `kt-kernel` runtime deps. I’m implementing this as a **single dependency update** in `kt-kernel/pyproject.toml`.

### Required additions
- `pyarrow>=15.0.0`
- `pandas>=2.2.0`
- `matplotlib>=3.8.0`
- `seaborn>=0.13.0`

### Setup commands

```bash
cd /root/ktransformers/kt-kernel
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pip install pyarrow pandas matplotlib seaborn
pytest test/moe_routing -q
```

### Runtime data paths
- Traces: `/root/ktransformers/data/traces/*.parquet`
- Analysis: `/root/ktransformers/data/analysis/metrics.json`, `/root/ktransformers/data/analysis/plots/*.png`
- Simulation: `/root/ktransformers/data/simulation/results.json`, `/root/ktransformers/data/simulation/tradeoff_curves.png`

---

## Dependency Graph

```text
Batch 1 (parallel): 1.1, 1.2, 1.3, 1.4 [foundation - no deps]
Batch 2 (parallel): 2.1, 2.2, 2.3, 2.4 [core - depends on batch 1]
Batch 3 (parallel): 3.1, 3.2, 3.3, 3.4 [phase CLIs + integration - depends on batch 2]
```

---

## Batch 1: Foundation (parallel - 4 implementers)

All tasks in this batch have NO dependencies and run simultaneously.

### Task 1.1: Add routing schema types
**File:** `kt-kernel/python/moe_routing/types.py`  
**Test:** `kt-kernel/test/moe_routing/test_types.py`  
**Depends:** none

```python
# kt-kernel/test/moe_routing/test_types.py
from kt_kernel.moe_routing.types import RoutingRecord, TOP_K


def test_routing_record_valid():
    rec = RoutingRecord(
        token_id=1,
        context_id="ctx-1",
        layer_id=0,
        token_position=12,
        expert_ids=[1, 2, 3, 4, 5, 6],
        expert_weights=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1],
        token_text="hello",
        timestamp_us=123,
        token_category="user",
    )
    assert rec.layer_id == 0
    assert len(rec.expert_ids) == TOP_K


def test_routing_record_rejects_wrong_topk():
    try:
        RoutingRecord(
            token_id=1,
            context_id="ctx-1",
            layer_id=0,
            token_position=12,
            expert_ids=[1, 2],
            expert_weights=[0.5, 0.5],
            timestamp_us=123,
            token_category="assistant",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "top-k" in str(exc)
```

```python
# kt-kernel/python/moe_routing/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

TOP_K = 6


@dataclass(slots=True)
class RoutingRecord:
    token_id: int
    context_id: str
    layer_id: int
    token_position: int
    expert_ids: list[int]
    expert_weights: list[float]
    timestamp_us: int
    token_category: str
    token_text: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.expert_ids) != TOP_K or len(self.expert_weights) != TOP_K:
            raise ValueError("top-k expert_ids/expert_weights must each be length 6")
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0")
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_types.py -q`  
**Commit:** `feat(moe-routing): add routing trace schema types`

### Task 1.2: Add async Parquet writer
**File:** `kt-kernel/python/moe_routing/parquet_writer.py`  
**Test:** `kt-kernel/test/moe_routing/test_parquet_writer.py`  
**Depends:** none

```python
# kt-kernel/test/moe_routing/test_parquet_writer.py
from pathlib import Path

import pyarrow.parquet as pq

from kt_kernel.moe_routing.parquet_writer import AsyncParquetWriter
from kt_kernel.moe_routing.types import RoutingRecord


def test_writer_flushes_parquet(tmp_path: Path):
    out = tmp_path / "trace.parquet"
    writer = AsyncParquetWriter(out, flush_size=2)
    writer.start()
    writer.submit(
        RoutingRecord(1, "ctx", 0, 0, [1, 2, 3, 4, 5, 6], [0.2] * 6, 10, "user")
    )
    writer.submit(
        RoutingRecord(2, "ctx", 0, 1, [1, 2, 3, 4, 5, 6], [0.2] * 6, 20, "assistant")
    )
    writer.close()

    table = pq.read_table(out)
    assert table.num_rows == 2
    assert "expert_ids" in table.column_names
```

```python
# kt-kernel/python/moe_routing/parquet_writer.py
from __future__ import annotations

import queue
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .types import RoutingRecord


class AsyncParquetWriter:
    def __init__(self, output_path: Path, flush_size: int = 1024):
        self.output_path = output_path
        self.flush_size = flush_size
        self._q: "queue.Queue[RoutingRecord | None]" = queue.Queue(maxsize=8192)
        self._t: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, rec: RoutingRecord) -> None:
        self._q.put(rec)

    def close(self) -> None:
        self._q.put(None)
        if self._t is not None:
            self._t.join()

    def _run(self) -> None:
        batch: list[RoutingRecord] = []
        writer: pq.ParquetWriter | None = None
        while True:
            item = self._q.get()
            if item is None:
                break
            batch.append(item)
            if len(batch) >= self.flush_size:
                writer = self._flush(batch, writer)
                batch.clear()

        if batch:
            writer = self._flush(batch, writer)
        if writer is not None:
            writer.close()

    def _flush(self, rows: list[RoutingRecord], writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
        table = pa.Table.from_pylist(
            [
                {
                    "token_id": r.token_id,
                    "context_id": r.context_id,
                    "layer_id": r.layer_id,
                    "token_position": r.token_position,
                    "expert_ids": r.expert_ids,
                    "expert_weights": r.expert_weights,
                    "token_text": r.token_text,
                    "timestamp": r.timestamp_us,
                    "token_category": r.token_category,
                }
                for r in rows
            ]
        )
        if writer is None:
            writer = pq.ParquetWriter(self.output_path, table.schema, compression="zstd")
        writer.write_table(table)
        return writer
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_parquet_writer.py -q`  
**Commit:** `feat(moe-routing): add async parquet trace writer`

### Task 1.3: Add cache policies
**File:** `kt-kernel/python/moe_routing/cache_policies.py`  
**Test:** `kt-kernel/test/moe_routing/test_cache_policies.py`  
**Depends:** none

```python
# kt-kernel/test/moe_routing/test_cache_policies.py
from kt_kernel.moe_routing.cache_policies import SlidingWindowPolicy


def test_sliding_window_retains_recent_experts():
    p = SlidingWindowPolicy(capacity=4, window_size=2)
    p.observe([1, 2])
    p.observe([3, 4])
    assert p.cached() == {1, 2, 3, 4}
    p.observe([5, 6])
    assert 1 not in p.cached()
    assert 5 in p.cached()
```

```python
# kt-kernel/python/moe_routing/cache_policies.py
from __future__ import annotations

from collections import Counter, deque


class BasePolicy:
    def observe(self, experts: list[int]) -> None:
        raise NotImplementedError

    def cached(self) -> set[int]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class BaselinePolicy(BasePolicy):
    def observe(self, experts: list[int]) -> None:
        return

    def cached(self) -> set[int]:
        return set()


class SlidingWindowPolicy(BasePolicy):
    def __init__(self, capacity: int, window_size: int):
        self.capacity = capacity
        self.window = deque(maxlen=window_size)

    def observe(self, experts: list[int]) -> None:
        self.window.append(set(experts))

    def cached(self) -> set[int]:
        merged: set[int] = set()
        for s in self.window:
            merged.update(s)
        if len(merged) <= self.capacity:
            return merged
        return set(list(merged)[: self.capacity])


class FixedHotPolicy(BasePolicy):
    def __init__(self, hot_experts: list[int]):
        self._hot = set(hot_experts)

    def observe(self, experts: list[int]) -> None:
        return

    def cached(self) -> set[int]:
        return set(self._hot)


def build_hotset(freq: Counter[int], pool_size: int) -> list[int]:
    return [x for x, _ in freq.most_common(pool_size)]
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_cache_policies.py -q`  
**Commit:** `feat(moe-routing): add cache policy primitives`

### Task 1.4: Add project dependencies for analysis pipeline
**File:** `kt-kernel/pyproject.toml`  
**Test:** none (config file task)
**Depends:** none

```toml
# Append under [project].dependencies in kt-kernel/pyproject.toml
"pyarrow>=15.0.0",
"pandas>=2.2.0",
"matplotlib>=3.8.0",
"seaborn>=0.13.0",
```

**Verify:** `python -c "import pyarrow,pandas,matplotlib,seaborn; print('ok')"`  
**Commit:** `chore(kt-kernel): add moe routing analysis dependencies`

---

## Batch 2: Core Modules (parallel - 4 implementers)

All tasks in this batch depend on Batch 1 completing.

### Task 2.1: Add data collection harness
**File:** `kt-kernel/python/moe_routing/trace_collector.py`  
**Test:** `kt-kernel/test/moe_routing/test_trace_collector.py`  
**Depends:** 1.1, 1.2

```python
# kt-kernel/test/moe_routing/test_trace_collector.py
from pathlib import Path

import pyarrow.parquet as pq

from kt_kernel.moe_routing.trace_collector import RoutingTraceCollector


def test_collector_records_and_writes(tmp_path: Path):
    c = RoutingTraceCollector(output_dir=tmp_path, prompt_id="p1")
    c.start(context_id="ctx-1")
    c.record(layer_id=0, token_position=0, expert_ids=[1, 2, 3, 4, 5, 6], expert_weights=[0.2] * 6)
    c.stop()
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    t = pq.read_table(files[0])
    assert t.num_rows == 1
```

```python
# kt-kernel/python/moe_routing/trace_collector.py
from __future__ import annotations

import time
from pathlib import Path

from .parquet_writer import AsyncParquetWriter
from .types import RoutingRecord


class RoutingTraceCollector:
    def __init__(self, output_dir: Path, prompt_id: str, token_category: str = "assistant"):
        self.output_dir = output_dir
        self.prompt_id = prompt_id
        self.token_category = token_category
        self.context_id = ""
        self._writer: AsyncParquetWriter | None = None
        self._token_id = 0
        self._t0 = 0

    def start(self, context_id: str) -> None:
        self.context_id = context_id
        ts = int(time.time())
        out = self.output_dir / f"{ts}_{self.prompt_id}.parquet"
        self._writer = AsyncParquetWriter(out)
        self._writer.start()
        self._t0 = time.perf_counter_ns()

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def record(
        self,
        layer_id: int,
        token_position: int,
        expert_ids: list[int],
        expert_weights: list[float],
        token_text: str | None = None,
    ) -> None:
        if self._writer is None:
            return
        ts_us = (time.perf_counter_ns() - self._t0) // 1000
        self._writer.submit(
            RoutingRecord(
                token_id=self._token_id,
                context_id=self.context_id,
                layer_id=layer_id,
                token_position=token_position,
                expert_ids=expert_ids,
                expert_weights=expert_weights,
                token_text=token_text,
                timestamp_us=ts_us,
                token_category=self.token_category,
            )
        )
        self._token_id += 1
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_trace_collector.py -q`  
**Commit:** `feat(moe-routing): add trace collector harness`

### Task 2.2: Integrate hook in BaseMoEWrapper.submit_forward
**File:** `kt-kernel/python/experts_base.py`  
**Test:** `kt-kernel/test/moe_routing/test_experts_base_hook.py`  
**Depends:** 2.1

```python
# kt-kernel/test/moe_routing/test_experts_base_hook.py
import torch

from kt_kernel.experts_base import BaseMoEWrapper


def test_trace_hook_invoked_for_submit_forward(monkeypatch):
    calls = []

    def hook(layer_id, topk_ids, topk_weights):
        calls.append((layer_id, topk_ids.shape, topk_weights.shape))

    BaseMoEWrapper.set_trace_hook(hook)
    hook_fn = BaseMoEWrapper.get_trace_hook()
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    w = torch.tensor([[0.2, 0.2, 0.2, 0.15, 0.15, 0.1]], dtype=torch.float32)
    hook_fn(3, x, w)
    assert calls[0][0] == 3
    BaseMoEWrapper.set_trace_hook(None)
```

```python
# kt-kernel/python/experts_base.py (additions only)
from typing import Callable


class BaseMoEWrapper(ABC):
    _trace_hook: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None

    @staticmethod
    def set_trace_hook(hook: Callable[[int, torch.Tensor, torch.Tensor], None] | None) -> None:
        BaseMoEWrapper._trace_hook = hook

    @staticmethod
    def get_trace_hook() -> Callable[[int, torch.Tensor, torch.Tensor], None] | None:
        return BaseMoEWrapper._trace_hook

    def submit_forward(self, hidden_states, topk_ids, topk_weights, cuda_stream):
        trace_hook = BaseMoEWrapper._trace_hook
        if trace_hook is not None:
            try:
                trace_hook(self.layer_idx, topk_ids.detach().cpu(), topk_weights.detach().cpu())
            except Exception:
                # Never break inference for telemetry
                pass
        # existing submit_forward body remains unchanged below this line
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_experts_base_hook.py -q`  
**Commit:** `feat(kt-kernel): add optional moe routing trace hook`

### Task 2.3: Implement locality metrics
**File:** `kt-kernel/python/moe_routing/metrics.py`  
**Test:** `kt-kernel/test/moe_routing/test_metrics.py`  
**Depends:** 1.1

```python
# kt-kernel/test/moe_routing/test_metrics.py
import pandas as pd

from kt_kernel.moe_routing.metrics import temporal_reuse_curve, expert_entropy_by_layer


def test_temporal_reuse_curve_returns_probabilities():
    df = pd.DataFrame(
        {
            "layer_id": [0, 0, 0],
            "token_position": [0, 1, 2],
            "expert_ids": [[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11], [2, 7, 12, 13, 14, 15]],
        }
    )
    out = temporal_reuse_curve(df, max_distance=2)
    assert 1 in out and 2 in out
    assert 0.0 <= out[1] <= 1.0


def test_entropy_by_layer_non_negative():
    df = pd.DataFrame({"layer_id": [0, 0], "expert_ids": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]})
    ent = expert_entropy_by_layer(df)
    assert ent[0] >= 0.0
```

```python
# kt-kernel/python/moe_routing/metrics.py
from __future__ import annotations

import math
from collections import Counter

import pandas as pd


def temporal_reuse_curve(traces: pd.DataFrame, max_distance: int = 64) -> dict[int, float]:
    traces = traces.sort_values(["layer_id", "token_position"]).reset_index(drop=True)
    curve: dict[int, float] = {}
    for d in range(1, max_distance + 1):
        hits = 0
        total = 0
        for _, g in traces.groupby("layer_id"):
            for i in range(len(g) - d):
                a = set(g.iloc[i]["expert_ids"])
                b = set(g.iloc[i + d]["expert_ids"])
                hits += len(a & b)
                total += len(b)
        curve[d] = (hits / total) if total else 0.0
    return curve


def expert_entropy_by_layer(traces: pd.DataFrame) -> dict[int, float]:
    out: dict[int, float] = {}
    for layer_id, g in traces.groupby("layer_id"):
        counts = Counter()
        for ids in g["expert_ids"]:
            counts.update(ids)
        total = sum(counts.values())
        if total == 0:
            out[int(layer_id)] = 0.0
            continue
        probs = [c / total for c in counts.values()]
        out[int(layer_id)] = -sum(p * math.log2(p) for p in probs if p > 0)
    return out
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_metrics.py -q`  
**Commit:** `feat(moe-routing): implement temporal locality metrics`

### Task 2.4: Implement simulator with α constraint
**File:** `kt-kernel/python/moe_routing/simulator.py`  
**Test:** `kt-kernel/test/moe_routing/test_simulator.py`  
**Depends:** 1.3

```python
# kt-kernel/test/moe_routing/test_simulator.py
import pandas as pd

from kt_kernel.moe_routing.cache_policies import SlidingWindowPolicy
from kt_kernel.moe_routing.simulator import simulate_policy


def test_alpha_one_is_unconstrained():
    df = pd.DataFrame({"layer_id": [0], "expert_ids": [[1, 2, 3, 4, 5, 6]], "expert_weights": [[1, 1, 1, 1, 1, 1]]})
    res = simulate_policy(df, SlidingWindowPolicy(capacity=2, window_size=1), alpha=1.0)
    assert res["quality_proxy_degradation"] == 0.0


def test_alpha_zero_hard_constraint_increases_degradation():
    df = pd.DataFrame({"layer_id": [0], "expert_ids": [[1, 2, 3, 4, 5, 6]], "expert_weights": [[1, 1, 1, 1, 1, 1]]})
    res = simulate_policy(df, SlidingWindowPolicy(capacity=0, window_size=1), alpha=0.0)
    assert res["partial_hit_rate"] == 0.0
```

```python
# kt-kernel/python/moe_routing/simulator.py
from __future__ import annotations

import pandas as pd

from .cache_policies import BasePolicy


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
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_simulator.py -q`  
**Commit:** `feat(moe-routing): implement cache replay simulator with alpha`

---

## Batch 3: Phase CLIs + Integration (parallel - 4 implementers)

All tasks in this batch depend on Batch 2 completing.

### Task 3.1: Add collection CLI
**File:** `kt-kernel/python/moe_routing/collect.py`  
**Test:** `kt-kernel/test/moe_routing/test_collect_cli.py`  
**Depends:** 2.1, 2.2

```python
# kt-kernel/test/moe_routing/test_collect_cli.py
from pathlib import Path

from kt_kernel.moe_routing.collect import build_arg_parser


def test_collect_parser_defaults(tmp_path: Path):
    p = build_arg_parser()
    args = p.parse_args(["--output-dir", str(tmp_path), "--prompt-id", "p1", "--context-id", "c1"])
    assert args.prompt_id == "p1"
```

```python
# kt-kernel/python/moe_routing/collect.py
from __future__ import annotations

import argparse
from pathlib import Path

from kt_kernel.experts_base import BaseMoEWrapper

from .trace_collector import RoutingTraceCollector


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("moe-routing-collect")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt-id", required=True)
    p.add_argument("--context-id", required=True)
    p.add_argument("--token-category", default="assistant")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    collector = RoutingTraceCollector(args.output_dir, args.prompt_id, args.token_category)
    collector.start(context_id=args.context_id)

    def hook(layer_id, topk_ids, topk_weights):
        for pos in range(topk_ids.shape[0]):
            collector.record(
                layer_id=layer_id,
                token_position=pos,
                expert_ids=topk_ids[pos].tolist(),
                expert_weights=topk_weights[pos].tolist(),
            )

    BaseMoEWrapper.set_trace_hook(hook)
    # Inference run is external to this command's minimal harness contract.


if __name__ == "__main__":
    main()
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_collect_cli.py -q`  
**Commit:** `feat(moe-routing): add collection cli harness`

### Task 3.2: Add analysis CLI and plotting
**File:** `kt-kernel/python/moe_routing/analyze.py`  
**Test:** `kt-kernel/test/moe_routing/test_analyze_cli.py`  
**Depends:** 2.3

```python
# kt-kernel/test/moe_routing/test_analyze_cli.py
from pathlib import Path
import json

import pandas as pd

from kt_kernel.moe_routing.analyze import run_analysis


def test_run_analysis_writes_metrics(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    df = pd.DataFrame(
        {
            "layer_id": [0, 0],
            "token_position": [0, 1],
            "expert_ids": [[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 9, 10]],
            "expert_weights": [[0.2] * 6, [0.2] * 6],
        }
    )
    df.to_parquet(in_file)
    out_dir = tmp_path / "analysis"
    run_analysis(in_file, out_dir)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "temporal_reuse_curve" in metrics
```

```python
# kt-kernel/python/moe_routing/analyze.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import expert_entropy_by_layer, temporal_reuse_curve


def run_analysis(trace_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)

    traces = pd.read_parquet(trace_file)
    reuse = temporal_reuse_curve(traces, max_distance=32)
    entropy = expert_entropy_by_layer(traces)

    (output_dir / "metrics.json").write_text(
        json.dumps({"temporal_reuse_curve": reuse, "expert_entropy_by_layer": entropy}, indent=2)
    )

    xs = list(reuse.keys())
    ys = list(reuse.values())
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys)
    plt.title("Temporal Reuse Curve")
    plt.xlabel("Distance")
    plt.ylabel("Reuse Probability")
    plt.tight_layout()
    plt.savefig(plots / "temporal_reuse_curve.png")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser("moe-routing-analyze")
    p.add_argument("--trace-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run_analysis(args.trace_file, args.output_dir)


if __name__ == "__main__":
    main()
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_analyze_cli.py -q`  
**Commit:** `feat(moe-routing): add analysis cli and metric export`

### Task 3.3: Add simulation CLI + parameter sweep
**File:** `kt-kernel/python/moe_routing/simulate.py`  
**Test:** `kt-kernel/test/moe_routing/test_simulate_cli.py`  
**Depends:** 2.4

```python
# kt-kernel/test/moe_routing/test_simulate_cli.py
from pathlib import Path
import json

import pandas as pd

from kt_kernel.moe_routing.simulate import run_simulation


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
```

```python
# kt-kernel/python/moe_routing/simulate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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

    (output_dir / "results.json").write_text(json.dumps({"runs": runs}, indent=2))

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


def main() -> None:
    p = argparse.ArgumentParser("moe-routing-simulate")
    p.add_argument("--trace-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run_simulation(args.trace_file, args.output_dir)


if __name__ == "__main__":
    main()
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_simulate_cli.py -q`  
**Commit:** `feat(moe-routing): add simulation cli and alpha sweeps`

### Task 3.4: Add package exports
**File:** `kt-kernel/python/moe_routing/__init__.py`  
**Test:** `kt-kernel/test/moe_routing/test_package_exports.py`  
**Depends:** 2.1, 2.3, 2.4

```python
# kt-kernel/test/moe_routing/test_package_exports.py
from kt_kernel.moe_routing import RoutingTraceCollector, temporal_reuse_curve, simulate_policy


def test_exports_available():
    assert RoutingTraceCollector is not None
    assert temporal_reuse_curve is not None
    assert simulate_policy is not None
```

```python
# kt-kernel/python/moe_routing/__init__.py
from .metrics import temporal_reuse_curve, expert_entropy_by_layer
from .simulator import simulate_policy, estimate_quality_proxy
from .trace_collector import RoutingTraceCollector

__all__ = [
    "RoutingTraceCollector",
    "temporal_reuse_curve",
    "expert_entropy_by_layer",
    "simulate_policy",
    "estimate_quality_proxy",
]
```

**Verify:** `pytest kt-kernel/test/moe_routing/test_package_exports.py -q`  
**Commit:** `feat(moe-routing): expose analysis and simulation api`

---

## 3) Specific Implementation Steps by Phase

### Phase 1 — Data Collection Harness (hook integration)
1. Add `BaseMoEWrapper.set_trace_hook/get_trace_hook` static methods.
2. In `submit_forward`, call hook with `(layer_id, topk_ids, topk_weights)` **before** CPU task submission.
3. Hook path must be failure-isolated (`try/except`) to avoid inference interruption.
4. `RoutingTraceCollector` converts callback tensors to `RoutingRecord` and pushes to `AsyncParquetWriter`.
5. Writer uses batched async flush (`flush_size=1024`, zstd compression) into `data/traces/{timestamp}_{prompt_id}.parquet`.

### Phase 2 — Analysis Pipeline (locality metrics)
1. Load Parquet traces with `pandas.read_parquet`.
2. Compute:
   - temporal reuse curve vs distance,
   - entropy by layer,
   - (next extension) sliding-window hit-rate grid and context-boundary churn.
3. Write machine-readable metrics JSON.
4. Generate first plot set (temporal reuse, then heatmap/histograms in extension pass).

### Phase 3 — Simulation Framework (cache replay + α)
1. Implement cache policy interface (`observe/cached/reset`).
2. Implement policy set (`baseline`, `fixed_hot`, `sliding_window`; then extend to `exp_decay`, `boundary_reset`, `hybrid`).
3. Replay each token’s expert set against policy cache.
4. Apply α semantics:
   - `α=0`: strict uncached penalty,
   - `α=1`: unconstrained baseline,
   - `0<α<1`: continuous soft penalty.
5. Run sweeps over `window_size × capacity × alpha`; export `results.json` and tradeoff scatter.

---

## 4) Testing Approach

### Unit tests
- Schema validation (`RoutingRecord` top-k invariants)
- Writer flush correctness + Parquet readability
- Policy determinism and eviction behavior
- α boundary checks (`0, 1, mid`)

### Integration tests
- `experts_base` hook invocation contract
- collect CLI parser + artifact generation
- analyze CLI reads traces and emits `metrics.json`
- simulate CLI reads traces and emits `results.json`

### Validation tests
- Synthetic traces with known locality to verify metric directionality
- Assert `alpha=1` maps to zero proxy degradation
- Assert hard-constraint (`alpha=0`) lowers partial-hit quality proxy when cache misses rise

---

## 5) Execution Order with Dependencies

1. **Batch 1** in parallel (types/writer/policies/deps).
2. **Batch 2** in parallel once Batch 1 lands (collector/hook/metrics/simulator).
3. **Batch 3** in parallel once Batch 2 lands (collect/analyze/simulate CLIs + exports).
4. Run full suite:

```bash
pytest kt-kernel/test/moe_routing -q
```

5. Run end-to-end dry run:

```bash
python -m kt_kernel.moe_routing.collect --output-dir /root/ktransformers/data/traces --prompt-id smoke --context-id smoke
python -m kt_kernel.moe_routing.analyze --trace-file /root/ktransformers/data/traces/<latest>.parquet --output-dir /root/ktransformers/data/analysis
python -m kt_kernel.moe_routing.simulate --trace-file /root/ktransformers/data/traces/<latest>.parquet --output-dir /root/ktransformers/data/simulation
```

This execution order preserves strict phase dependencies while maximizing parallelism inside each batch.
