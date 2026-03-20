---
date: 2026-03-20
topic: "Layer-Qualified Cache Identity for MoE Routing Simulation"
status: validated
---

# Layer-Qualified Simulation Cache Design

## Problem Statement

The MoE routing simulator currently keys cache identity by raw `expert_id` only. This causes false cache hits when different layers select experts with the same local expert index (for example, layer 3 expert 7 and layer 18 expert 7). As a result, simulation metrics—especially `partial_hit_rate`—are inflated and do not reflect actual cross-layer expert residency behavior.

The approved fix is to make cache identity layer-qualified: expert identity in simulation must be represented as `(layer_id, expert_id)`.

## Constraints

- Scope is **simulation path only** under `kt-kernel/python/moe_routing/`.
- No behavioral change is required for real inference paths.
- CLI surface should remain backward-compatible unless change is necessary for correctness.
- Existing policies (`baseline`, `fixed_hot`, `sliding_window`) must remain available.
- Existing output schema should be preserved where possible; semantic interpretation can be updated.

## Approach

Design requires eliminating cross-layer collisions in cache keys. I’m implementing this by introducing a canonical simulation identity tuple:

```python
QualifiedExpert = tuple[int, int]  # (layer_id, expert_id)
```

All cache-facing interfaces in simulation are updated to consume and return qualified identities. The simulator converts each trace row into a set of qualified experts before querying policy cache state and before calling `observe`.

For compatibility:
- `BaselinePolicy` remains stateless and layer-agnostic in behavior.
- `FixedHotPolicy` gains explicit support for qualified hotsets while preserving compatibility with current usage through normalization at construction.

## Architecture

Three modules are updated:

1. **Policy Interface (`cache_policies.py`)**
   - Base policy methods become layer-qualified.
   - Policy internals store `QualifiedExpert` values.

2. **Simulator Core (`simulator.py`)**
   - Constructs `needed` as `set[(layer_id, expert_id)]` per row.
   - Computes hits/misses using qualified identity.
   - Calls `policy.observe(needed)` with qualified identities.

3. **Simulation Runner + Tests (`simulate.py`, `test_*`)**
   - Ensure compatible policy creation and metrics output.
   - Add regression coverage for cross-layer ID collisions.

## Components

### 1) `QualifiedExpert` identity alias
- Defined in `cache_policies.py` (or a local alias in simulator if minimal change preferred).
- Used consistently in policy interfaces and simulator logic.

### 2) Policy interface contract
- `observe(experts: list[QualifiedExpert] | set[QualifiedExpert]) -> None`
- `cached() -> set[QualifiedExpert]`

### 3) Simulator identity builder
- For each row: `needed = {(int(row.layer_id), int(expert_id)) for expert_id in row.expert_ids}`

### 4) Compatibility adapters
- `FixedHotPolicy` constructor normalizes either:
  - qualified entries `(layer_id, expert_id)` (preferred), or
  - legacy unqualified `int` entries by pairing with an explicit layer context when provided by caller.
- In this fix scope, CLI simulation path will pass qualified hotsets where needed and avoid ambiguous unqualified hotset usage.

## Data Flow

```text
trace row (layer_id, expert_ids)
    ↓
qualified needed set: {(layer_id, e1), (layer_id, e2), ...}
    ↓
policy.cached() returns set[(layer, expert)]
    ↓
hit = needed ∩ cached
miss = needed - cached
    ↓
metrics accumulate (hit_rate, partial_hit_rate, simulated_ssd_fetches)
    ↓
policy.observe(needed)
```

This removes false positives caused by raw `expert_id` overlap across layers.

## Error Handling

- Validate tuple shapes where policies accept externally supplied hotsets.
- Raise `ValueError` for malformed qualified identities.
- Keep simulator alpha validation (`0.0 <= alpha <= 1.0`) unchanged.
- Continue best-effort plot generation behavior in CLI (`results.json` remains primary artifact).

## Testing Strategy

1. **Unit policy tests**
   - `SlidingWindowPolicy` caches qualified identities correctly.
   - `FixedHotPolicy` returns qualified identities without collapsing layers.

2. **Simulator correctness tests**
   - Existing alpha-boundary tests continue to pass.
   - New regression test: two rows with identical `expert_ids` but different `layer_id` must **not** produce full hit due to cross-layer collision.

3. **CLI/result compatibility checks**
   - `simulate.py` still writes `results.json` with same keys (`hit_rate`, `partial_hit_rate`, `simulated_ssd_fetches`, `quality_proxy_degradation`).
   - Interpretive note: `partial_hit_rate` is expected to decrease relative to pre-fix for traces with cross-layer index overlap.

## Open Questions

1. Should we add an explicit metadata field in `results.json` such as `cache_identity: "layer_qualified"` to prevent confusion in historical comparisons?
2. Should `build_hotset` expose a separate helper for qualified hotset generation from per-layer frequencies to avoid accidental unqualified usage?
3. Do we want a one-time migration note in docs warning that historical simulation numbers are not directly comparable?

---

*Status: Validated*
