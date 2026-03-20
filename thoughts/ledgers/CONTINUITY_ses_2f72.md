---
session: ses_2f72
updated: 2026-03-20T21:35:00Z
---

# Session Summary

## Goal
Get DeepSeek-V2-Lite running via `kt run`/SGLang with GPU+CPU experts and produce valid MoE routing parquet traces in `data/traces` from real generation traffic, then run analysis/simulation.

## Constraints & Preferences
- Avoid legacy `archive/ktransformers.local_chat` / `KTransformersOps` path.
- Use SGLang runtime with CPU offload (`--kt-method AMXINT4`).
- Keep existing trace-hook integration (`BaseMoEWrapper.set_trace_hook` / `_trace_hook`) intact.
- Prefer minimal, targeted changes.

## Progress

### Done
- [x] Confirmed deterministic-mode requirement with DeepSeek + RL target and switched attention backend to `triton` for compatibility.
- [x] Fixed AMX weight path usage for AMXINT4 runtime (`DeepSeek-V2-Lite-Chat-AMXINT4` instead of GGUF directory).
- [x] Fixed trace hook to be CUDA-graph capture-safe in `BaseMoEWrapper.submit_forward` by skipping CPU telemetry work during stream capture.
- [x] Added env support for explicit output file via `KT_MOE_ROUTING_TRACE_FILE` and wired it through `RoutingTraceCollector.start(output_path=...)`.
- [x] Added graceful collector shutdown on SIGINT/SIGTERM (in addition to `atexit`) so parquet footer is written reliably.
- [x] Replaced outdated `scripts/run_collection.sh` implementation (which used removed `run_inference_with_collection`) with current SGLang/HTTP flow.
- [x] Updated collection script shutdown to signal the full process group (`setsid` + `kill -INT -- -PID`) so worker processes exit cleanly and traces finalize.
- [x] Produced valid live trace from real chat request:
  - `data/traces/live_capture.parquet` (validated with `pyarrow`, 1924 rows)
- [x] Ran analysis and simulation successfully on live trace:
  - `data/analysis/metrics.json`
  - `data/simulation/results.json`
- [x] Captured headline outputs:
  - `reuse@1 = 0.2620302072356867`
  - `entropy_layers = 26`
  - `sim_runs = 20`
  - `best_partial_hit_rate = 0.7197678447678447`
- [x] Removed stale/corrupted trace artifacts from working set and kept valid outputs for commit.
- [x] Created branch `feature/moe-trace-collection-fixes` and committed work:
  - commit `8e38468`
  - message: `[feat](moe-routing): add live trace collection and analysis pipeline`

### In Progress
- [ ] None for implementation scope requested.

### Blocked / Known Environment Issues
- Fake `nvcc` wrapper at `/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/bin/nvcc` still breaks non-deterministic rope JIT path (`ptxas` unknown option `-generate-dependencies-with-compile`).
- Runtime emits warnings about older `numexpr`/`bottleneck` versions; does not block trace/analysis JSON generation.
- Matplotlib import environment is inconsistent; plotting is now best-effort and non-blocking in analyze/simulate.

## Key Decisions
- Use deterministic RL mode (`--rl-on-policy-target fsdp`) to bypass non-deterministic rope JIT path impacted by fake `nvcc`.
- Keep hook contract unchanged and make telemetry non-invasive/safe under CUDA graph capture.
- Prioritize valid parquet + JSON metrics artifacts over plotting reliability in this environment.

## Next Steps
1. Push branch `feature/moe-trace-collection-fixes` when requested.
2. Open PR summarizing runtime fixes, trace collection flow, and produced artifacts.
3. Optional follow-up: improve model output quality/regression checks (responses currently low quality despite successful execution).
4. Optional follow-up: pin/repair plotting stack if PNG outputs are required in CI.

## Critical Context
- Deterministic mode with DeepSeek requires compatible attention backend (`triton` or `fa3`), not `flashinfer`.
- AMXINT4 method must use converted AMX layer directory, not GGUF path.
- Parquet corruption was caused by incomplete shutdown of spawned scheduler/worker processes; process-group SIGINT resolved it.

## File Operations (this continuation)

### Modified
- `/root/ktransformers/kt-kernel/python/experts_base.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/analyze.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/simulate.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/trace_collector.py`
- `/root/ktransformers/scripts/run_collection.sh`

### Added (tracked in commit)
- `/root/ktransformers/kt-kernel/python/moe_routing/__init__.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/cache_policies.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/collect.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/metrics.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/parquet_writer.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/simulator.py`
- `/root/ktransformers/kt-kernel/python/moe_routing/types.py`
- `/root/ktransformers/data/traces/live_capture.parquet`
- `/root/ktransformers/data/analysis/metrics.json`
- `/root/ktransformers/data/simulation/results.json`

### Removed (cleanup)
- corrupted trace artifacts from prior runs (not part of final tracked output)
