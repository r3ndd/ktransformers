---
session: ses_2f24
updated: 2026-03-21T00:02:30.767Z
---

# Session Summary

## Goal
Design and finalize a validated documentation package for updating MoE routing simulation metrics to token-level-across-layers semantics with fixed cache capacity 156 and no capacity sweep.

## Constraints & Preferences
- Replace row-level metrics with token-level metrics across all layers.
- Primary metric: average per-token partial hit rate across all layers.
- Full hit: token is a hit only if all needed experts across all layers are cached.
- Fixed cache capacity only: 156 (no cache-size sweep).
- Preserve layer-qualified expert identity semantics `(layer_id, expert_id)`.
- Include in design: token grouping strategy, token-granularity cold-start/update semantics, output/result schema changes, minimal backward compatibility, risks/mitigations (especially `token_position`), testing strategy/success criteria.
- Deliverables required:
  - `thoughts/shared/designs/2026-03-20-token-level-simulation-metrics-design.md` with specific sections and status `validated`.
  - `thoughts/shared/plans/2026-03-20-token-level-simulation-metrics.md` with executor-ready checklist.
- Prioritize clarity over broad backward compatibility.
- If possible, commit docs and report hash.

## Progress
### Done
- [x] Scanned repository structure and located relevant MoE routing modules and existing design/plan artifacts.
- [x] Reviewed current simulation and policy behavior in:
  - `simulate_policy` in `/root/ktransformers/kt-kernel/python/moe_routing/simulator.py`
  - `run_simulation` in `/root/ktransformers/kt-kernel/python/moe_routing/simulate.py`
  - policy interfaces in `/root/ktransformers/kt-kernel/python/moe_routing/cache_policies.py`
- [x] Reviewed trace schema sources to ground token grouping decisions:
  - `RoutingRecord` in `/root/ktransformers/kt-kernel/python/moe_routing/types.py`
  - collection/writer paths in `collect.py`, `trace_collector.py`, `parquet_writer.py`
- [x] Reviewed relevant tests to align proposed semantics with current validation surface:
  - `/root/ktransformers/kt-kernel/test/moe_routing/test_simulator.py`
  - `/root/ktransformers/kt-kernel/test/moe_routing/test_simulate_cli.py`
- [x] Created and finalized validated design doc:
  - `/root/ktransformers/thoughts/shared/designs/2026-03-20-token-level-simulation-metrics-design.md`
  - Includes required sections: Problem Statement, Constraints, Approach, Architecture, Components, Data Flow, Error Handling, Testing Strategy, Open Questions.
  - Status set to `validated`.
- [x] Created implementation plan with executable checklist:
  - `/root/ktransformers/thoughts/shared/plans/2026-03-20-token-level-simulation-metrics.md`
  - Includes dependency graph, batched tasks, verification commands, and executor-ready checklist.
- [x] Reported completion summary, created paths, and noted no commit hash (not committed).

### In Progress
- [ ] No code implementation changes were made yet; only design/plan docs were produced.

### Blocked
- (none)

## Key Decisions
- **Use token grouping key `(context_id, token_position)`**: This is the most stable token-level key available in current trace schema while preserving multi-context separation.
- **Keep layer-qualified identity `(layer_id, expert_id)` throughout**: Prevents cross-layer collisions and preserves correctness of cache-hit semantics.
- **Compute metrics at token granularity with post-token cache update**: Enforces clear cold-start semantics and avoids self-warming within the same token.
- **Primary metric = average per-token partial hit rate**: Matches user requirement and avoids row-level dilution.
- **Define full-hit at token level across all layers**: A token only fully hits when all required across-layer experts are cached.
- **Fix cache capacity to 156 and remove capacity sweep from simulation outputs**: Directly aligns with user constraint and simplifies result interpretation.
- **Minimal backward compatibility via `hit_rate` alias to `full_hit_rate`**: Preserves short-term consumer compatibility while making new semantics explicit via metadata.
- **Add explicit schema metadata (`metric_level`, `cache_capacity`, `token_grouping_key`)**: Prevents ambiguity when comparing with historical row-level results.

## Next Steps
1. Implement token-grouping and token-level replay logic in `/root/ktransformers/kt-kernel/python/moe_routing/simulator.py` (`simulate_policy` + helper).
2. Update `/root/ktransformers/kt-kernel/python/moe_routing/simulate.py` (`run_simulation`) to enforce fixed capacity `156` and remove capacity sweep output shape.
3. Add/adjust tests in:
   - `/root/ktransformers/kt-kernel/test/moe_routing/test_simulator.py` for token grouping, per-token partial/full semantics, and post-token update behavior.
   - `/root/ktransformers/kt-kernel/test/moe_routing/test_simulate_cli.py` for fixed-capacity schema assertions.
4. Run full moe_routing test suite and verify output schema fields.
5. Optionally commit doc artifacts (and implementation later), then report commit hash.

## Critical Context
- Current simulator (`simulate_policy`) still iterates row-by-row, not token-grouped; this is the semantic gap the new design addresses.
- Current CLI simulation (`run_simulation`) still performs a window/alpha loop and currently uses capacity `32` in code; design requires fixed `156` with no capacity sweep.
- `RoutingRecord` schema contains `context_id`, `token_position`, `layer_id`, `expert_ids`, enabling token-across-layer aggregation without schema migration.
- Existing code already uses layer-qualified cache identity in policy/simulator paths, which is compatible with proposed token-level aggregation.
- No `.mindmodel/` project guidance was found; decisions were based on existing repository patterns and current tests.
- No runtime/test errors occurred during this session; work was documentation-only.

## File Operations
### Read
- `/root/ktransformers/kt-kernel/python/moe_routing/analyze.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/cache_policies.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/collect.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/metrics.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/parquet_writer.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/simulate.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/simulator.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/trace_collector.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/types.py`
- `/root/ktransformers/kt-kernel/test/moe_routing/test_simulate_cli.py`
- `/root/ktransformers/kt-kernel/test/moe_routing/test_simulator.py`
- `/root/ktransformers/thoughts/shared/designs/2026-03-20-layer-qualified-simulation-cache-design.md`
- `/root/ktransformers/thoughts/shared/designs/2026-03-20-moe-routing-analysis-design.md`
- `/root/ktransformers/thoughts/shared/plans/2026-03-20-layer-qualified-simulation-cache.md`

### Modified
- `/root/ktransformers/thoughts/shared/designs/2026-03-20-token-level-simulation-metrics-design.md`
- `/root/ktransformers/thoughts/shared/plans/2026-03-20-token-level-simulation-metrics.md`
