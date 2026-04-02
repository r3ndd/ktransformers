# GPU/CPU/SSD Promotion Roadmap (Steps 1, 2, 3)

## Context

We have implemented tier telemetry (step 3), so per-request hit/load/promotion counters are now visible in benchmark outputs.

Remaining work is to make tiering behavior enforce real placement and measurable SSD pressure:

1. Wire tier manager decisions into real runtime residency updates.
2. Add a true cold-load path for experts outside GPU/CPU hot tiers.
3. Control and report OS page-cache effects for reproducible SSD-vs-RAM benchmarking.

## Goals

- Make `kt_expert_gpu_cache_capacity` and `kt_expert_cpu_cache_capacity` enforce actual hot residency.
- Ensure experts outside hot tiers are treated as SSD-resident and only promoted on access.
- Preserve correctness and avoid deadlocks/regressions in TP rank 0 CPU/GPU parallel path.
- Produce benchmark evidence that separates compute changes from storage/cache effects.

## Step 0: GGUF vs FP weights
- Discuss with the user whether this can be weight-agnostic, or if certain weight types (e.g. GGUF) may inherently be harder to offload to the SSD.

## Step 1 Plan: Runtime Residency Enforcement

### Design

- Treat `ExpertTierResidencyManager` as authoritative state for current tier ownership.
- Integrate manager outputs with existing dynamic expert remap helpers:
  - `update_gpu_expert_mappings(...)`
  - `update_kt_wrapper_masks(...)`
  - GPU weight copy/remap path in `kt_ep_wrapper.py`
- Trigger updates at bounded cadence (not every token) using a request-local or interval-based policy.

### High-level work

- Add a promotion decision interface in `expert_tier_cache.py` (e.g. return changed expert sets, not just stats).
- Add an update controller in `kt_ep_wrapper.py` to:
  - detect GPU set changes,
  - remap logical<->GPU indices,
  - refresh masks/tables atomically,
  - synchronize streams/events safely before switching mappings.
- Keep CPU path valid during transitions (no dropped experts, no stale index mapping).

### Deliverables

- Runtime path where tier manager can change active GPU expert set.
- Stable mapping updates under prefill/decode without correctness regressions.
- New counters: number of mapping updates and experts remapped per request.

## Step 2 Plan: True SSD Cold-load Path

### Design

- Distinguish three residency states explicitly for non-GPU experts:
  - hot CPU cache (RAM),
  - cold SSD-backed (not RAM-hot),
  - in-flight promotion.
- On miss to cold SSD tier, force a materialization/loading path before expert compute.

### High-level work

- Introduce an expert-weight lifecycle abstraction in kt-kernel wrappers:
  - load/unload hooks per expert or expert block,
  - eviction callbacks when CPU cache overflows,
  - optional async prefetch queue for predicted next experts.
- Gate CPU expert execution on residency readiness:
  - if SSD miss: load, then execute,
  - if CPU hit: execute directly.
- Add backpressure/timeout policy to avoid queue blowups under repeated misses.

### Deliverables

- Functional cold-load behavior where SSD misses increase latency and appear in counters.
- Capacity enforcement such that CPU cache size materially changes hit rate.
- Clear separation of promotion cost vs compute cost in telemetry.

## Step 3 Plan: OS Cache Control and Benchmark Protocol

### Design

- Introduce benchmark modes to reduce page-cache confounding:
  - warm-cache mode (steady-state),
  - cold-cache mode (best-effort cache drop/isolation),
  - mixed mode (controlled warmup then measured runs).
- Always report cache mode and cache-control actions in output metadata.

### High-level work

- Add benchmark options and metadata fields in `run_real_routing_benchmark.py`:
  - cache mode, pre-run conditioning steps, and host details.
- Add optional pre-run cache conditioning script/hooks (requires elevated permissions where needed).
- Add repeated-run protocol (N>=3) with variance reporting for tier metrics and TPS.

### Deliverables

- Reproducible benchmark recipe documenting how cache was controlled.
- Results with confidence bands/variance for throughput and tier hit/load deltas.
- Guidance for interpreting SSD-load metrics under warm vs cold conditions.

## Risks and Mitigations

- **Risk:** Mapping updates race with in-flight kernels.
  - **Mitigation:** stream/event barriers, swap-at-safe-point policy.
- **Risk:** Cold-load overhead destabilizes latency.
  - **Mitigation:** bounded async prefetch and admission control.
- **Risk:** OS cache controls are platform/permission dependent.
  - **Mitigation:** best-effort controls + explicit metadata and fallback modes.

## Validation Strategy

- Unit tests:
  - tier state transitions,
  - mapping update correctness,
  - cold-load and eviction behavior.
- Integration tests:
  - deterministic prompt runs with fixed token budgets,
  - compare outputs before/after tier enforcement.
- Benchmark acceptance:
  - tier deltas respond to cache capacities,
  - cold-cache mode shows higher SSD loads and latency than warm-cache mode,
  - no regressions in correctness metrics.

## Suggested Execution Order

1. Step 1 minimal end-to-end mapping updates (no async prefetch yet).
2. Step 2 synchronous cold-load path with correctness focus.
3. Step 2 performance pass (prefetch, batching, backpressure).
4. Step 3 benchmark protocol and reporting hardening.
