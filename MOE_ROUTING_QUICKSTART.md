# MoE Routing Analysis - Quick Start Guide

This directory contains a complete system for analyzing MoE expert routing patterns in **Qwen3.5-35B-A3B**, designed to validate smoothed routing hypotheses for consumer hardware.

## System Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Collection    │────▶│    Analysis     │────▶│   Simulation    │
│  (inference +   │     │ (locality metrics│     │ (cache policy   │
│   trace capture)│     │  + visualizations)│    │   tradeoffs)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Hardware Requirements

- **RAM**: 64GB+ (128GB recommended for comfort)
- **VRAM**: 16GB+ (24GB comfortable)
- **Storage**: ~30GB for model + traces

## Quick Start

### 1. Setup (One-time)

```bash
cd /root/ktransformers
./scripts/run_full_pipeline.sh
```

The setup phase will automatically download:
- Qwen3.5-35B-A3B base model (~70GB)
- Qwen3.5-35B-A3B-GGUF-Q4_K_M weights (~20GB)
- Model config files from HuggingFace

### 2. Run Full Pipeline

```bash
./scripts/run_full_pipeline.sh
```

This executes all three phases automatically.

## Manual Phase Execution

### Phase 1: Data Collection

```bash
python3 scripts/run_collection.py
```

Processes 17 diverse prompts across categories:
- **coding**: Algorithmic problems, debugging, API design
- **reasoning**: Math, logic puzzles, argument analysis
- **creative**: Story writing, poetry
- **factual**: Explanations, comparisons
- **multi_turn**: Simulated conversations
- **mixed**: Combined tasks (coding + explanation)
- **edge**: Ambiguous, short, nonsensical inputs

**Output**: `data/traces/{prompt_id}.parquet`

### Phase 2: Analysis

```bash
python -m kt_kernel.moe_routing.analyze \
    --trace-file data/traces/*.parquet \
    --output-dir data/analysis
```

**Computes**:
- Temporal reuse curve: P(expert at t+n | expert at t)
- Sliding window hit rates for various window sizes
- Expert entropy by layer
- Context switch churn

**Output**:
- `data/analysis/metrics.json` - Raw metrics
- `data/analysis/plots/` - Visualization plots

### Phase 3: Simulation

```bash
python -m kt_kernel.moe_routing.simulate \
    --trace-file data/traces/*.parquet \
    --output-dir data/simulation
```

**Simulates**:
- Baseline (no caching)
- Fixed hot pool (most frequent experts)
- Sliding window (recent experts)
- Exponential decay (weighted recency)
- **Alpha constraint**: Continuous parameter from hard (α=0) to soft (α=1) constraints

**Output**:
- `data/simulation/results.json` - Hit rates, fetch counts
- `data/simulation/tradeoff_curves.png` - Quality vs speed Pareto frontier

## Key Results to Examine

After running the pipeline, check these files:

### 1. Temporal Reuse Curve
`data/analysis/plots/temporal_reuse_curve.png`

**What to look for**: High reuse probability at small n (e.g., >60% at n=10). This validates the core assumption that experts exhibit temporal locality.

### 2. Sliding Window Hit Rate
`data/analysis/plots/sliding_window_hit_rate.png`

**What to look for**: >70% hit rate with 32-expert window. This indicates a manageable working set size.

### 3. Tradeoff Curves
`data/simulation/tradeoff_curves.png`

**What to look for**: Soft constraints (α=0.5) achieving >80% hit rate with minimal quality degradation. This shows the viability of smoothed routing.

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

### Constraint Parameter (α)

The simulator implements a continuous constraint:

- **α = 0**: Hard constraint - router can ONLY select from cached experts
- **α = 0.5**: Soft constraint - uncached experts' scores multiplied by 0.5
- **α = 1**: No constraint - router selects from all experts (baseline)

This allows exploring the full spectrum from aggressive caching to unconstrained routing.

## Extending the Analysis

### Add Custom Prompts

Edit `data/prompt_suite.json` and add prompts to the `prompts` array:

```json
{
  "id": "your_custom_001",
  "category": "your_category",
  "description": "Brief description",
  "prompt": "Your prompt text here..."
}
```

### Test Different Cache Policies

Modify `kt-kernel/python/moe_routing/cache_policies.py` to add new policies, then re-run simulation.

### Analyze Specific Layers

The analysis pipeline computes per-layer metrics. Check `data/analysis/metrics.json` for layer-wise breakdowns.

## Troubleshooting

### Model Download Fails

```bash
# Manual download using huggingface-cli
mkdir -p models/Qwen3.5-35B-A3B
huggingface-cli download "Qwen/Qwen3.5-35B-A3B" --local-dir models/Qwen3.5-35B-A3B

mkdir -p models/Qwen3.5-35B-A3B-GGUF-Q4_K_M
huggingface-cli download \
    "unsloth/Qwen3.5-35B-A3B-GGUF" \
    "Qwen3.5-35B-A3B-Q4_K_M.gguf" \
    --local-dir models/Qwen3.5-35B-A3B-GGUF-Q4_K_M
```

### Out of Memory

Reduce `MAX_TOKENS` in `scripts/run_collection.py` (default: 500) or adjust `max_total_tokens` in the server configuration.

### Slow Collection

This is expected - first run compiles CUDA kernels. Subsequent runs are faster.

## Success Criteria

Your smoothed routing hypothesis is viable if:

1. **Temporal locality**: >60% expert reuse at distance 10
2. **Bounded working set**: 32-expert window captures >70% of needs
3. **Graceful degradation**: α=0.5 achieves >80% hit rate with <10% quality loss
4. **Layer heterogeneity**: Different layers show different optimal cache sizes

## Next Steps

After validating on Qwen3.5-35B-A3B:

1. **Scale up**: Test on larger models like Kimi-K2.5 or Qwen3.5-235B-A22B (requires more RAM)
2. **Implement**: Build the actual smoothed routing engine in kTransformers
3. **Benchmark**: Measure real-world tokens/sec improvement on consumer hardware

## File Structure

```
.
├── scripts/
│   ├── run_collection.py            # Phase 1: Data collection
│   ├── run_full_pipeline.sh         # All phases (setup → collection → analysis → simulation)
│   └── sanity_check.py              # Quick validation test
├── data/
│   ├── prompt_suite.json            # Test prompts
│   ├── traces/                      # Raw routing traces
│   ├── analysis/                    # Metrics + plots
│   └── simulation/                  # Policy simulation results
└── kt-kernel/python/moe_routing/    # Analysis system code
```

## Citation

If you use this analysis system in research, cite kTransformers:

```bibtex
@inproceedings{10.1145/3731569.3764843,
  title = {KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models},
  author = {Chen, Hongtao and Xie, Weiyu and Zhang, Boxin and Tang, Jingqi and Wang, Jiahao and Dong, Jianwei and Chen, Shaoyuan and Yuan, Ziwei and Lin, Chen and Qiu, Chengyu and Zhu, Yuening and Ou, Qingliang and Liao, Jiaqi and Chen, Xianglin and Ai, Zhiyuan and Wu, Yongwei and Zhang, Mingxing},
  booktitle = {Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles},
  year = {2025}
}
```
