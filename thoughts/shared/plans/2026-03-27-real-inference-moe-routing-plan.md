# Real-Inference MoE Routing Plan (Prompt Suite + API-Specified Scheme)

## Context

The offline MoE routing simulation pipeline is in place and producing stable results:
- Schemes: `sliding_window_score_averaging`, `ema_score_averaging`, `two_timescale_ema`, `two_timescale_softmax`
- Inputs: full per-layer expert score vectors (`expert_scores_all`)
- Existing metrics: hit rate, baseline overlap, quality proxy, speed proxy

Next step: run these schemes during real inference on the prompt suite, with routing configuration passed per API request.

---

## Goals

1. Enable per-request routing scheme selection in real serving.
2. Keep runtime overhead low enough for meaningful throughput/latency comparisons.
3. Benchmark against baseline on the prompt suite.
4. Add explicit expert-weight residency controls across GPU VRAM, CPU RAM, and SSD for reproducible experiments.
5. Keep KV/radix behavior unchanged while explicitly budgeting memory headroom for KV cache.
6. Support separate routing behavior for prefill vs decode in one request.

---

## High-Level Design

### 1) API-level routing configuration

Use existing OpenAI-compatible request extension field `custom_params` to carry routing policy:

```json
{
  "custom_params": {
    "moe_routing": {
      "prefill": {
        "scheme": "prefill_block_mean",
        "params": { "window_size": 64 }
      },
      "decode": {
        "scheme": "ema_score_averaging",
        "params": { "ema_beta": 0.3 }
      },
      "scope": "request"
    }
  }
}
```

Why:
- Fits existing request plumbing and avoids protocol churn.
- Prefill/decode schemes and parameters travel with prompt + sampling arguments.
- Allows mixed experiment runs from one server process.

### 2) Request-to-runtime plumbing

Plumb `custom_params.moe_routing` from API request -> tokenized request -> scheduler request -> forward batch metadata.

Validation at ingress:
- Prefill and decode schemes must be independently validated against supported identifiers.
- Prefill and decode parameters must be type/range-checked.
- Invalid config should fall back to baseline and log a warning (not fail hard).

### 3) Apply smoothing at top-k selection boundary

Integrate routing transforms immediately before expert top-k selection (router logits/score path), preserving existing kernels where possible:
- Keep baseline fast path untouched.
- Apply transform only when request has `moe_routing` enabled.
- Maintain per-request state for decode stateful schemes (window/EMA history).
- For prefill, apply vectorized per-layer transforms over the prefill token block (not token-serial state updates).

### 4) Expert-weight residency manager (new)

Add a dedicated expert residency subsystem for MoE expert weights only:
- Core/dense model weights remain GPU-resident (no layer-wise offloading of core model blocks).
- Expert weights are managed across three tiers: GPU VRAM, CPU RAM, SSD.
- Tier manager supports two capacity modes:
  - `global`: one shared expert cache across all layers.
  - `layerwise`: isolated cache partitions per MoE layer.
- Initial eviction policy: LRU.

Memory budgeting order:
1. Reserve GPU memory for core weights.
2. Reserve GPU memory for KV cache headroom.
3. Use remaining GPU memory for GPU expert cache.
4. Use configured CPU RAM budget for CPU expert cache.
5. SSD acts as backing tier for non-resident experts.

Failure policy:
- If `core + KV reservation` cannot be satisfied, fail fast at startup with explicit diagnostics.

### 5) Batch scheduling policy for efficiency

Initial strategy:
- Group requests by routing signature (scheme + parameter hash) using routing-key scheduling.
- Avoid mixed-scheme batches in first implementation.
- This minimizes branchy per-token/per-row routing logic in hot kernels and simplifies state ownership.

---

## Phased Implementation Plan

## Phase 1: Schema + Validation
- Define routing payload schema (`prefill` + `decode`, each with `scheme + params`).
- Add ingress validation and normalized internal representation.
- Add explicit baseline behavior when routing payload absent.

Deliverables:
- Parser/validator.
- Unit tests for valid/invalid payloads and fallback behavior.

## Phase 2: Expert Tier Cache + Budgeting
- Implement expert tier cache manager for GPU/CPU/SSD expert residency.
- Implement cache modes: `global` and `layerwise`.
- Implement LRU eviction and promotion/demotion between tiers.
- Add memory budgeting checks and fail-fast validation for core + KV reservation.

Deliverables:
- Expert residency manager.
- Startup validation and diagnostics for memory budgeting.
- Tier metrics counters (GPU hits, CPU hits, SSD loads, bytes, per-layer latency).

## Phase 3: Runtime State Management (Routing)
- Add request-scoped routing state containers:
  - Decode: sliding window buffers / EMA accumulators per layer.
  - Prefill: per-layer vectorized temporary buffers for block/full averaging transforms.
- Lifecycle hooks:
  - initialize on request admission.
  - update on each token step.
  - cleanup on completion/cancel/abort.

Deliverables:
- State manager + tests for reset/isolation across requests.

## Phase 4: Top-k Integration
- Hook transformed scores into top-k expert selection path.
- Ensure no overhead for baseline/no-routing requests.
- Confirm compatibility with existing grouped top-k and relevant MoE paths.
- Use forward mode to select routing path:
  - prefill path for `EXTEND`/prefill tokens,
  - decode path for `DECODE`.

Deliverables:
- Integration tests proving selected experts differ only when routing enabled.
- Golden-path parity tests for baseline mode.

## Phase 5: Prompt Suite Real-Inference Runner
- Rework suite runner to keep one server alive for all prompts.
- Send per-prompt prefill + decode routing configuration in request payload.
- Record outputs + routed experts + timing counters.

Deliverables:
- Runner script updates.
- Structured experiment output per scheme/param.

## Phase 6: Benchmark + Analysis
Collect:
- Throughput: tokens/sec.
- Latency: TTFT, ITL, end-to-end.
- Routing behavior: overlap with baseline, routed-expert distributions.
- Quality proxy: existing similarity/degradation metrics.
- Tier behavior: GPU/CPU/SSD expert hit/miss/load metrics.

Deliverables:
- Comparable result tables/plots across schemes and parameters.
- Regression checks vs simulation expectations.

## Phase 7: Hardening
- Add robust fallback on runtime exceptions in routing path.
- Add metric tags: scheme, parameter hash, prompt_id, category.
- Add operator-facing logs for enabled scheme per request.
- Add operator-facing logs for expert tier cache mode/capacity and miss promotions.

Deliverables:
- Ops checklist and failure mode documentation.

---

## Efficiency Guidance

- Keep one long-lived server for entire suite (avoid per-prompt cold starts).
- Warmup pass before timed runs.
- Prefer homogeneous batches by routing signature.
- Retain baseline fast path exactly as-is when no routing policy present.
- Start with deterministic settings for A/B validity (seeded sampling and controlled parameters).
- For single-request runs, disable confounding scheduler features where needed and keep routing logic isolated.

---

## Default Caching Policy in Current Stack

## Prefix/KV behavior (serving layer)
- Prefix cache (RadixAttention) is enabled unless explicitly disabled.
- Default radix eviction policy is `lru`.
- Hierarchical cache is disabled by default.
- If hierarchical cache enabled:
  - host cache sizing defaults via ratio (`hicache_ratio=2.0`) unless absolute size override (`hicache_size`).
  - default write policy is `write_through`.
  - optional storage backend can provide L3 tier.

## MoE simulation cache behavior (analysis layer)
- Current simulation uses per-layer LRU with fixed capacity (`capacity_per_layer=25`) for locality accounting.
- This is a simulation model, not the live serving cache implementation.

## Plan Decisions for This Workstream
- Keep KV cache and radix/prefix cache behavior unchanged.
- Explicitly reserve memory budget for KV cache before assigning GPU/CPU memory to expert caches.
- Do not use general layer-wise offloading for core model blocks in this experiment plan.

---

## MoE Offloading Systems: Current Reality

## Expert offload (KT MoE path)
- GPU expert residency controlled via:
  - `--kt-num-gpu-experts` (per-layer count), or
  - `--kt-gpu-experts-ratio` (fraction across MoE experts, overrides count).
- Placement strategies available:
  - `frequency`, `front-loading`, `uniform`, `random`.
- CPU expert compute controlled by `--kt-cpuinfer` and CPU weight path (`--kt-weight-path`).
- Full-layer GPU fallback for large prefill controlled by `--kt-gpu-prefill-token-threshold`.
- Optional dynamic expert update can adapt GPU expert set from runtime activation statistics.

## Planned shift for experiments
- Replace fixed static-only GPU expert placement as the primary mechanism with hierarchical expert-tier residency:
  - GPU expert cache (hot set),
  - CPU expert cache (warm set),
  - SSD expert backing store (cold set).
- Support cache scope selection:
  - global across all layers,
  - layerwise isolated per layer.

## General layer offload mechanisms
- Offloader V1: CPU offload budget (`--cpu-offload-gb`).
- Offloader V2: grouped layer offload controls:
  - `--offload-group-size`
  - `--offload-num-in-group`
  - `--offload-prefetch-step`
  - `--offload-mode` (`cpu`, `shm_cpu`, `sharded_gpu`).

## SSD/off-disk tier
- For KV cache, hierarchical storage backends provide disk/distributed tiers (for example, `file`, `mooncake`).
- For expert weights in the KT path, the CPU weight path naturally supports an on-disk source, but an explicit "GPU/CPU/SSD expert cache ratio policy" is not yet first-class.

---

## Manual Offloading Test Plan (Hardware-Independent Stress Testing)

## Objective
Control expert residency directly through tier capacities, rather than indirect placement-only knobs.

## Proposed controls
1. GPU expert cache capacity (absolute count or bytes).
2. CPU expert cache capacity (absolute count or bytes).
3. Cache mode: `global` or `layerwise`.
4. Eviction policy: `lru` (v1).
5. Optional admission/ranking policy (future extension).

## Required counters
- GPU hit
- CPU RAM hit
- SSD load count/bytes
- promotion/demotion counts
- per-layer added latency

This yields controlled experiments that are less coupled to a single hardware profile.

---

## Experiment Matrix (Initial)

Baseline:
- No routing transform.

Routing schemes:
- Prefill schemes:
  - `prefill_block_mean`: `window_size` in {32, 64, 128}
  - `prefill_full_mean`: whole prefill span per layer
- Decode schemes:
  - Sliding window: W in {1, 4, 16, 64}
  - EMA: beta in {0.9, 0.7, 0.5, 0.3, 0.1, 0.05}
  - Two-timescale EMA: lambda in {0.1, 0.2, 0.3, 0.4}
  - Two-timescale softmax: lambda=0.2, rho in {0.25, 1, 4, 16, 64, 256, 1024}

Expert tier cache settings:
- mode in {`global`, `layerwise`}
- GPU cache budget sweep
- CPU cache budget sweep

For each:
- fixed prompt suite
- fixed seed/sampling controls
- collect runtime + routing + quality proxy + tier cache metrics
- compare against simulation ranking trends

---

## Risks and Mitigations

- Kernel-path overhead from conditional routing logic:
  - mitigate via strict baseline fast path and homogeneous batching.
- State leakage across requests:
  - mitigate with explicit per-request state lifecycle tests.
- Metric ambiguity between KV cache and expert locality:
  - separate and label metrics by subsystem (KV vs expert routing).
- Dynamic expert update can confound routing-scheme effects:
  - benchmark with dynamic update both OFF and ON.
- Memory-budget ambiguity between KV and expert tiers:
  - enforce explicit reservation order and fail-fast checks.
- Prefill/decode scheme interaction complexity:
  - validate each path independently, then combine in matrix runs.

---

## Definition of Done

1. Per-request prefill + decode routing schemes are accepted in the API and visible in logs/metrics.
2. Baseline path unaffected when routing config absent.
3. Prompt suite can run all scheme sweeps on a single live server process.
4. Expert tier cache supports GPU/CPU/SSD with `global` and `layerwise` modes.
5. Result artifacts include throughput/latency + routed-expert traces + quality proxies + tier cache counters.
6. KV/radix behavior remains unchanged, and KV memory reservation is explicitly documented and validated.
7. Clear recommendation for production default configuration (or baseline) based on measured frontier.
