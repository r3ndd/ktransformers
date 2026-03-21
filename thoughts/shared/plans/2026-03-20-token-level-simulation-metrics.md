# Token-Level Across-Layers Simulation Metrics Implementation Plan

**Goal:** Move MoE routing simulation from row-level metrics to token-level across-layers semantics with fixed cache capacity `156` and no capacity sweep.

**Architecture:** Token groups are constructed from trace rows using `(context_id, token_position)`, then replayed as one token-wide required expert set containing layer-qualified expert identities. Metrics are computed per token (partial/full/misses), cache is updated once per token, and outputs are emitted with explicit token-level semantic metadata.

**Design:** `/root/ktransformers/thoughts/shared/designs/2026-03-20-token-level-simulation-metrics-design.md`

---

## Dependency Graph

```text
Batch 1 (parallel): 1.1, 1.2 [core semantics]
Batch 2 (parallel): 2.1, 2.2 [CLI schema + regression tests]
Batch 3 (sequential): 3.1 [full verification]
```

---

## Batch 1: Core semantics (parallel)

### Task 1.1: Implement token-level across-layers simulator
- [ ] **File:** `kt-kernel/python/moe_routing/simulator.py`
- [ ] **Depends:** none

#### Required edits
- Add internal grouping helper to build token groups from `(context_id, token_position)`.
- For each token group, build `needed_experts` as `set[(layer_id, expert_id)]` across all rows/layers.
- Compute token metrics before cache update:
  - `partial_hit_token = hits/needed`
  - `full_hit_token = 1 if all needed experts cached else 0`
  - `misses_token = len(needed - cache_before)`
- Update cache with `policy.observe(list(needed_experts))` **after** token evaluation.
- Return metrics:
  - `partial_hit_rate` (primary; average token partial)
  - `full_hit_rate`
  - `hit_rate` (alias of `full_hit_rate`)
  - `simulated_ssd_fetches`
  - `avg_misses_per_token`
  - `quality_proxy_degradation`
  - `token_count`
- Validate required columns and raise clear `ValueError` when absent.

#### Verify
- [ ] `pytest -q kt-kernel/test/moe_routing/test_simulator.py`

#### Commit suggestion
- `feat(moe-routing): compute token-level across-layer simulation metrics`

---

### Task 1.2: Add simulator token-level unit coverage
- [ ] **File:** `kt-kernel/test/moe_routing/test_simulator.py`
- [ ] **Depends:** none

#### Required tests
- Add deterministic test for token grouping:
  - two layers same `(context_id, token_position)` become one simulated token.
- Add test asserting primary metric semantics:
  - `partial_hit_rate` equals average of per-token partials, not global row ratio.
- Add test for full-hit semantics:
  - token full hit only when all across-layer experts are cached.
- Add cold-start/update semantics test:
  - token cannot self-hit before its own post-token observe update.
- Keep/extend existing cross-layer collision regression.

#### Verify
- [ ] `pytest -q kt-kernel/test/moe_routing/test_simulator.py -k "token or cross_layer or full_hit"`

#### Commit suggestion
- `test(moe-routing): add token-level simulator semantic coverage`

---

## Batch 2: CLI schema + regression (parallel)

### Task 2.1: Enforce fixed capacity=156 and token-level output schema
- [ ] **File:** `kt-kernel/python/moe_routing/simulate.py`
- [ ] **Depends:** 1.1

#### Required edits
- Remove cache-capacity sweep logic.
- Lock simulation cache capacity to `156` for all runs.
- Keep policy/alpha loops if needed, but no capacity variant dimension.
- Emit top-level metadata:
  - `metric_level: "token_across_layers"`
  - `cache_identity: "layer_qualified"`
  - `cache_capacity: 156`
  - `token_grouping_key: ["context_id", "token_position"]`
  - `deprecated_fields: ["hit_rate"]`
- Preserve existing `runs` structure for minimal compatibility.

#### Verify
- [ ] `pytest -q kt-kernel/test/moe_routing/test_simulate_cli.py`

#### Commit suggestion
- `feat(moe-routing): fix simulation capacity to 156 with token-level schema`

---

### Task 2.2: Update CLI tests for schema and fixed capacity
- [ ] **File:** `kt-kernel/test/moe_routing/test_simulate_cli.py`
- [ ] **Depends:** 2.1

#### Required tests
- Assert `results.json` includes:
  - `metric_level == "token_across_layers"`
  - `cache_identity == "layer_qualified"`
  - `cache_capacity == 156`
  - `token_grouping_key == ["context_id", "token_position"]`
- Assert run entries include new fields:
  - `full_hit_rate`, `avg_misses_per_token`, `token_count`
- Assert `hit_rate == full_hit_rate` alias behavior.
- Assert no capacity-sweep fields/axes are present.

#### Verify
- [ ] `pytest -q kt-kernel/test/moe_routing/test_simulate_cli.py`

#### Commit suggestion
- `test(moe-routing): validate token-level result schema and fixed capacity`

---

## Batch 3: Final verification (sequential)

### Task 3.1: End-to-end validation pass
- [ ] **File:** `thoughts/shared/plans/2026-03-20-token-level-simulation-metrics.md` (execution log section update optional)
- [ ] **Depends:** 1.1, 1.2, 2.1, 2.2

#### Verification checklist
- [ ] `pytest -q kt-kernel/test/moe_routing/test_simulator.py`
- [ ] `pytest -q kt-kernel/test/moe_routing/test_simulate_cli.py`
- [ ] `pytest -q kt-kernel/test/moe_routing`
- [ ] Run one sample simulation and inspect JSON:

```bash
python -m moe_routing.simulate --trace-file /tmp/sample_trace.parquet --output-dir /tmp/moe_sim_token
python - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/moe_sim_token/results.json')
d = json.loads(p.read_text())
assert d['metric_level'] == 'token_across_layers'
assert d['cache_capacity'] == 156
assert d['token_grouping_key'] == ['context_id', 'token_position']
print('ok')
PY
```

#### Commit suggestion
- `chore(moe-routing): verify token-level simulation metrics rollout`

---

## Executor-Ready Checklist

- [ ] Implement token grouping by `(context_id, token_position)` in simulator.
- [ ] Compute per-token partial/full/miss metrics across all layers.
- [ ] Update cache only after per-token metric computation.
- [ ] Emit `full_hit_rate`, `avg_misses_per_token`, `token_count`, and `hit_rate` alias.
- [ ] Lock cache capacity to `156` in simulation runner.
- [ ] Remove any cache-capacity sweep behavior from output.
- [ ] Add token-level semantic metadata to `results.json`.
- [ ] Update simulator and CLI tests for new semantics.
- [ ] Run full moe_routing test suite and one sample CLI replay.
