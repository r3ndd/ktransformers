---
date: 2026-03-20
topic: "Token-Level Across-Layers Simulation Metrics (Fixed Capacity)"
status: validated
---

# Token-Level Across-Layers Simulation Metrics Design

## Problem Statement

Current simulation metrics are effectively row-level (one trace row = one layer/token fragment), which overstates fidelity for token-level decisions and makes cross-layer behavior hard to interpret. We need to move to **token-level across all layers** semantics while preserving layer-qualified expert identity.

Required behavior:
- Replace row-level metrics with token-level metrics aggregated across all layers.
- Primary metric is **average per-token partial hit rate across all layers**.
- Full hit means a token is a hit only when **all required experts across all layers** are cached.
- Use fixed cache capacity only: **156** (no capacity sweep).
- Keep expert identity as `(layer_id, expert_id)`.

## Constraints

- Scope: `kt-kernel/python/moe_routing/` simulation and related tests.
- Trace schema is unchanged and already includes: `context_id`, `token_position`, `layer_id`, `expert_ids`, `timestamp`, `token_category`, `token_id`.
- Minimal backward compatibility: prioritize semantic clarity over preserving ambiguous legacy behavior.
- Existing policy interface (layer-qualified expert keys) remains in use.
- No cache-capacity parameter sweep in simulation outputs.

## Approach

Design requires token-level aggregation. I am implementing this as:

1. **Token grouping phase**: group rows into logical tokens using existing schema.
2. **Token-level replay phase**: evaluate cache hits/misses against each token’s full across-layer expert set.
3. **Post-token cache update**: update cache only after a token is evaluated (token-granularity cold-start/update semantics).

### Token grouping strategy (existing trace schema)

Primary grouping key:
`(context_id, token_position)`

Within a group:
- Aggregate all rows from different `layer_id` values.
- Build token required expert set as:
  `needed_token = set((layer_id, expert_id) for each row/layer expert)`
- Deduplicate repeated experts in same layer.

Ordering:
- Token groups are processed in deterministic order by first-seen `timestamp`, then `token_position`.

Rationale:
- `context_id + token_position` is the most stable token-level key available in current schema.
- `token_id` is row-level in collector and cannot represent one token across layers.

## Architecture

1. **Token grouping utility (simulator path)**
   - Input: trace DataFrame rows.
   - Output: ordered token groups with per-token needed expert sets.

2. **Token-level simulator core**
   - For each token group, compute:
     - token partial hit ratio
     - token full-hit boolean
     - token miss count (simulated fetches)
   - Update policy once per token group after computing all metrics for that token.

3. **Simulation runner output formatter (`simulate.py`)**
   - Enforces fixed capacity=156.
   - Removes capacity sweep output shape.
   - Emits explicit metric semantics metadata.

## Components

### 1) Token grouping component

- Location: `kt-kernel/python/moe_routing/simulator.py` (internal helper).
- Function behavior:
  - Validate required columns: `context_id`, `token_position`, `layer_id`, `expert_ids`.
  - Group rows into token units.
  - Produce ordered iterable of token objects:
    - `context_id`
    - `token_position`
    - `needed_experts: set[(layer_id, expert_id)]`

### 2) Token-level metric computation component

For each token:
- `cache_before = policy.cached()`
- `hit = needed_experts ∩ cache_before`
- `partial_hit_token = len(hit) / len(needed_experts)` (0 when empty needed set)
- `full_hit_token = (len(hit) == len(needed_experts) and len(needed_experts) > 0)`
- `misses_token = len(needed_experts - cache_before)`
- `policy.observe(list(needed_experts))` (after evaluation)

Aggregate outputs:
- `partial_hit_rate` = mean of `partial_hit_token` across tokens (**primary metric**)
- `full_hit_rate` = mean of `full_hit_token` across tokens
- `simulated_ssd_fetches` = sum of `misses_token`
- `avg_misses_per_token` = mean of `misses_token`
- retain `quality_proxy_degradation` computed from token-level partial/full rates

### 3) Fixed-capacity simulation runner component

- Location: `kt-kernel/python/moe_routing/simulate.py`
- Capacity locked to `156`.
- No capacity sweep in output.
- If policy variants are run, they all share `cache_capacity=156`.

## Data Flow

```text
parquet trace rows
  ↓
group by (context_id, token_position)
  ↓
token needed set: {(layer, expert), ... across all layers}
  ↓
cache lookup (before token update)
  ↓
token metrics: partial/full/misses
  ↓
policy.observe(token needed set)  # post-token update
  ↓
aggregate token-level metrics
  ↓
results.json (token-level semantics metadata)
```

## Error Handling

- Missing required trace columns → `ValueError` with explicit missing column names.
- Non-list/invalid `expert_ids` rows → skip row with warning counter and include `dropped_rows` in metadata.
- Empty token groups after filtering → metrics default to 0.0 and emit `token_count=0`.
- Preserve existing `alpha` validation (`0.0 <= alpha <= 1.0`).

## Output Metrics and Result Schema Updates

New canonical semantics fields:
- `metric_level: "token_across_layers"`
- `cache_identity: "layer_qualified"`
- `cache_capacity: 156`
- `token_grouping_key: ["context_id", "token_position"]`

Run metrics (per run/policy):
- `partial_hit_rate` (primary, token-average partial)
- `full_hit_rate` (token full-hit rate)
- `simulated_ssd_fetches`
- `avg_misses_per_token`
- `quality_proxy_degradation`
- `token_count`

Backward-compat field strategy (minimal):
- Keep `hit_rate` as alias of `full_hit_rate` for one release window.
- Add `deprecated_fields: ["hit_rate"]` metadata note.
- Do not emit legacy row-level metrics in same payload to avoid semantic confusion.

Example shape:

```json
{
  "metric_level": "token_across_layers",
  "cache_identity": "layer_qualified",
  "cache_capacity": 156,
  "token_grouping_key": ["context_id", "token_position"],
  "runs": [
    {
      "policy": "sliding_window",
      "window_size": 16,
      "alpha": 0.5,
      "partial_hit_rate": 0.71,
      "full_hit_rate": 0.42,
      "hit_rate": 0.42,
      "simulated_ssd_fetches": 12345,
      "avg_misses_per_token": 4.18,
      "quality_proxy_degradation": 0.145,
      "token_count": 2950
    }
  ],
  "deprecated_fields": ["hit_rate"]
}
```

## Backward Compatibility Strategy

Minimal and clarity-first:

1. Preserve top-level `runs` list and keep `hit_rate` alias.
2. Explicitly mark semantics via `metric_level` and `token_grouping_key`.
3. Remove capacity sweep shape; all results are fixed-capacity 156.
4. Consumers relying on row-level assumptions must migrate; this is intentional.

## Testing Strategy

### Unit tests

1. Token grouping correctness
   - rows for same `(context_id, token_position)` across multiple `layer_id` become one token group.
2. Token partial metric
   - verify token-average partial differs from global-row ratio in crafted fixture.
3. Token full-hit metric
   - full hit only when all across-layer experts are cached.
4. Post-token update semantics
   - same token cannot self-warm within evaluation.

### Integration tests

1. `simulate.py` writes results with:
   - `cache_capacity = 156`
   - no capacity sweep fields
   - token-level metadata fields present
2. End-to-end replay on sample parquet with two tokens, two layers, deterministic assertions.

### Regression tests

- Cross-layer same `expert_id` collision regression remains impossible due to layer-qualified keys.
- Token-position collision scenario across contexts validates grouping isolation by `context_id`.

## Success Criteria

1. Primary metric equals expected token-average partial hit on deterministic fixtures.
2. Full-hit metric marks hit only for tokens with complete across-layer coverage.
3. All simulation runs in output use `cache_capacity=156`.
4. No capacity sweep artifacts remain in result schema.
5. Existing tests pass plus new token-level tests pass.

## Risks / Limitations and Mitigations

1. **`token_position` semantics risk**
   - Risk: token positions may reset/reuse unexpectedly depending on collection mode.
   - Mitigation: grouping key includes `context_id`; deterministic ordering also uses `timestamp`.
   - Additional mitigation: emit `token_grouping_key` metadata and document expected collector behavior.

2. **Batched/in-flight collection ambiguity**
   - Risk: if collector mixes contexts without stable `context_id`, grouping may merge unrelated rows.
   - Mitigation: fail fast when `context_id` missing/empty beyond threshold; include dropped/invalid counters.

3. **Semantic discontinuity with historical numbers**
   - Risk: metrics are not directly comparable to row-level history.
   - Mitigation: explicit `metric_level` and deprecation marker in payload.

## Open Questions

1. Should we hard-error (instead of warning) on duplicate `(context_id, token_position, layer_id)` rows?
2. Should `hit_rate` alias be removed immediately or after one release cycle?
3. Do we want optional strict mode requiring monotonic token positions per context?

---

*Status: Validated*
