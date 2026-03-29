# MoE Routing Analysis - Quick Start Guide

This directory contains an end-to-end pipeline for collecting, analyzing, and simulating MoE expert routing traces on **Qwen3.5-35B-A3B**.

This document reflects the **current implementation**. A short roadmap section near the end lists items that are still planned.

Terminology used here is aligned with the real-inference plan:
- `prefill` means extend/prompt processing.
- `decode` means autoregressive token generation.
- `expert-tier cache` means expert-weight residency across GPU/CPU/SSD tiers.
- `cache accounting` in this document refers to the **simulation model** (not live serving cache internals).

## System Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Collection    │────▶│    Analysis     │────▶│   Simulation    │
│  (inference +   │     │ (locality metrics│     │ (routing scheme │
│   trace capture)│     │  + summaries)    │     │   tradeoffs)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Hardware Requirements

- **RAM**: 64GB+ (128GB recommended)
- **VRAM**: 16GB+ (24GB comfortable)
- **Storage**: ~30GB+ for models and traces

## Quick Start

Run the full pipeline:

```bash
cd /root/ktransformers
./scripts/run_full_pipeline.sh
```

`run_full_pipeline.sh` performs:
1. Model setup/download checks
2. Trace collection
3. Analysis
4. Simulation

Notes:
- It uses Hugging Face CLI command `hf download`.
- If trace files already exist, the script may prompt to skip collection.

## Manual Phase Execution

### Phase 1: Data Collection

```bash
python3 scripts/run_collection.py
```

Current behavior:
- Loads prompts from `data/prompt_suite.json` (currently 17 prompts).
- Runs one server session per prompt.
- Writes per-prompt outputs/logs plus per-prompt session parquet traces.
- Records full per-layer expert score vectors in `expert_scores_all` when enabled (float16 list).
- Aggregates successful traces into:
  - `data/traces/live_capture.parquet`

### Phase 2: Analysis

```bash
python -m kt_kernel.moe_routing.analyze \
  --trace-file data/traces/live_capture.parquet \
  --output-dir data/analysis
```

Current metrics:
- `temporal_reuse_curve` (distance capped to available tokens, max 64)
- `previous_token_reuse_curve` (reuse vs immediate previous token)
- `sliding_window_hit_rate` (window sweep currently 4/8/16/32/64)
- `context_switch_churn`
- `expert_entropy_by_layer`

Implementation details:
- Adds robust `absolute_token_position` derived from trace order and microbatch boundaries.
- Computes per-context metrics and writes them under `data/analysis/contexts/`.
- Computes overall averages using context alignment to minimum token count.

Outputs:
- `data/analysis/metrics.json`
- `data/analysis/contexts/*.json`
- `data/analysis/plots/temporal_reuse_curve.png` (best-effort plotting)

### Phase 3: Simulation

```bash
python -m kt_kernel.moe_routing.simulate \
  --trace-file data/traces/live_capture.parquet \
  --output-dir data/simulation
```

Current simulation scope (decode-style scoring transforms):
- Schemes:
  - `sliding_window_score_averaging`
  - `ema_score_averaging`
  - `two_timescale_ema`
  - `two_timescale_softmax`
- Parameter sweeps:
  - sliding window: `window_size` in `[1, 4, 16, 64]`
  - EMA: `ema_beta` in `[0.9, 0.7, 0.5, 0.3, 0.1, 0.05]`
  - two-timescale EMA: `mix_lambda` in `[0.1, 0.2, 0.3, 0.4]`
  - two-timescale softmax: fixed `mix_lambda=0.2`, `rho` in `[0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]`
- Baseline-equivalent reference remains `window_size=1` for sliding-window.

Current simulation metrics:
- `hit_rate`
- `ssd_fetches_per_token`
- `baseline_overlap`
- `quality_degradation`
- `speedup_ratio`
- `quality_speed_score`
- `baseline_ssd_fetches_per_token`

Simulation notes:
- Expert cache accounting uses per-layer LRU with `capacity_per_layer=25` (~1000 total expert slots for 40 layers).
- `quality_degradation` uses softmax-probability mass ratio (chosen vs baseline) over token-layer steps.
- `speedup_ratio` uses a timing model:
  - `speedup_ratio = (0.1 + baseline_extra_seconds_per_token) / (0.1 + extra_seconds_per_token)`
  - with `extra_seconds_per_token = ssd_fetches_per_token * 0.0015`.

Outputs:
- `data/simulation/results.json`
- `data/simulation/contexts/*.json`
- `data/simulation/tradeoff_curves.png` (best-effort plotting)

## Key Artifacts to Inspect

1. `data/analysis/metrics.json`
   - Global aligned averages and context token counts.
2. `data/analysis/contexts/*.json`
   - Per-context locality behavior.
3. `data/simulation/results.json`
   - Aggregated routing-scheme tradeoffs and context inclusion counts.
4. `data/simulation/tradeoff_curves.png`
   - Hit-rate vs quality-degradation scatter.

## Architecture Details

### Qwen3.5-35B-A3B MoE Characteristics

| Property | Value |
|----------|-------|
| Total params | ~35B |
| Active params | ~3B |
| Experts | 256 |
| Experts per token | 8 (top-8 routing) |
| Layers | 40 |
| Hidden size | 2048 |
| MoE intermediate size | 512 |

### Important Trace Indexing Detail

The trace hook emits batch-local `token_position`. For batched decoding this repeats across microbatches. The pipeline derives `absolute_token_position` so analysis/simulation operate on true per-context token order.

## Extending the Analysis Today

### Add or modify prompts

Edit `data/prompt_suite.json` and add entries to the `prompts` array.

### Add new routing schemes

Implement scheme classes in `kt-kernel/python/moe_routing/routing_schemes.py`, then wire them into `kt-kernel/python/moe_routing/simulate.py`.

### Layer-level inspection

Use `expert_entropy_by_layer` values from:
- `data/analysis/metrics.json`
- `data/analysis/contexts/*.json`

## Planned / Not Yet Implemented

The following items are **not yet wired into the current CLI pipeline**:

1. Runtime benchmarking:
   - Direct tokens/sec evaluation on hardware for selected scheme/parameter settings.
2. Additional analysis visualizations:
   - richer multi-scheme frontier plots and per-scheme dashboards
3. Auto-tuning / adaptive sweep selection:
   - narrowing parameter ranges from previous runs
4. Prefill/decode split routing in real serving:
   - separate prefill and decode scheme configuration in request payload
5. Expert-tier cache controls in real serving:
   - explicit GPU/CPU/SSD expert residency controls with global/layerwise cache modes

## Troubleshooting

### Model download fails

```bash
mkdir -p models/Qwen3.5-35B-A3B
hf download "Qwen/Qwen3.5-35B-A3B" --local-dir models/Qwen3.5-35B-A3B

mkdir -p models/Qwen3.5-35B-A3B-GGUF-Q4_K_M
hf download \
  "unsloth/Qwen3.5-35B-A3B-GGUF" \
  "Qwen3.5-35B-A3B-Q4_K_M.gguf" \
  --local-dir models/Qwen3.5-35B-A3B-GGUF-Q4_K_M
```

### Out of memory

Tune collection settings in `scripts/run_collection.py` (for example `COLLECTION_MAX_TOKENS` / `MAX_TOKENS`) and server token limits.

### Slow first run

Initial runs may compile CUDA kernels and warm dependencies; later runs are usually faster.

## Next Steps

1. Benchmark runtime impact on consumer hardware.
2. Add richer comparative plots across all four schemes.
3. Add auto-tuned sweep presets from prior run results.

## File Structure

```
.
├── scripts/
│   ├── run_collection.py
│   ├── run_full_pipeline.sh
│   └── sanity_check.py
├── data/
│   ├── prompt_suite.json
│   ├── traces/
│   ├── analysis/
│   └── simulation/
└── kt-kernel/python/moe_routing/
```

## Citation

If you use this system in research, cite kTransformers:

```bibtex
@inproceedings{10.1145/3731569.3764843,
  title = {KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models},
  author = {Chen, Hongtao and Xie, Weiyu and Zhang, Boxin and Tang, Jingqi and Wang, Jiahao and Dong, Jianwei and Chen, Shaoyuan and Yuan, Ziwei and Lin, Chen and Qiu, Chengyu and Zhu, Yuening and Ou, Qingliang and Liao, Jiaqi and Chen, Xianglin and Ai, Zhiyuan and Wu, Yongwei and Zhang, Mingxing},
  booktitle = {Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles},
  year = {2025}
}
```
