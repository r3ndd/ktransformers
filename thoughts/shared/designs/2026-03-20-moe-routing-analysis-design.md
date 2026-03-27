---
date: 2026-03-20
topic: "MoE Routing Analysis System for Smoothed Routing Validation"
status: draft
---

# MoE Routing Analysis System Design

## Problem Statement

We need to validate whether **smoothed MoE routing** (sliding-window expert caching with boundary-aware resets) can make trillion-parameter MoE models viable on consumer hardware (24GB VRAM + 32-128GB RAM + SSD offload).

The core hypothesis: Expert selection exhibits sufficient **temporal locality** that constraining routing to a recently-active subset introduces acceptable quality degradation while dramatically reducing SSD thrashing.

## Constraints

- **Target model for data collection**: DeepSeek-V2-Lite (14B total, ~2B active, 64 experts, top-6 routing, 26 layers)
- **Hardware**: 128GB RAM + 16GB VRAM available for collection
- **Collection overhead**: Must not significantly impact inference speed
- **Analysis scope**: Focus on locality metrics and cache policy simulation, not end-to-end quality evaluation
- **Simulation fidelity**: Must support both hard constraints (route only to cached experts) and soft constraints (down-weight uncached experts via continuous parameter)

## Approach

We'll build a **three-stage pipeline**:

1. **Collection**: Capture per-token expert routing decisions from kTransformers inference
2. **Analysis**: Compute locality metrics, expert usage patterns, layer-wise behavior
3. **Simulation**: Replay traces through configurable cache policies to estimate hit rates and quality proxies

**Why this approach**:
- DeepSeek-V2-Lite runs fast on available hardware, enabling large-scale data collection
- MoE routing dynamics (expert locality, temporal correlation) scale across model sizes
- Simulation allows testing many policies without re-running inference
- Conservative validation: if smoothed routing fails on smaller model, it won't work on larger

## Architecture

### Data Collection Harness

**Integration point**: Hook into `BaseMoEWrapper.submit_forward()` to intercept routing decisions before execution.

**Captured data per token**:
- `expert_ids`: [6] - selected expert indices (DeepSeek-V2-Lite uses top-6)
- `expert_weights`: [6] - selection weights/scores
- `layer_id`: int - which MoE layer (0-25)
- `token_position`: int - position in sequence
- `context_id`: str - prompt/conversation identifier
- `token_category`: str - user/assistant/system

**Storage format**: Parquet with compression, batched async writes to minimize overhead.

**Schema**:
```
token_id: int64
context_id: string
layer_id: int16
token_position: int32
expert_ids: list<int16>  # length 6
expert_weights: list<float32>
token_text: string  # optional, for debugging
timestamp: int64  # relative microseconds
```

### Analysis Pipeline

**Locality Metrics**:
- **Temporal Reuse Curve**: P(expert at t+n | expert at t) vs n for various n
- **Sliding Window Hit Rate**: For window size W, fraction of needed experts present in cache
- **Expert Entropy**: H(expert distribution) by layer and globally
- **Layer-wise Diversity**: Unique expert count per layer over sliding windows
- **Context Switch Detection**: Expert churn at conversation boundaries

**Visualization Outputs**:
- Heatmaps: expert usage frequency by layer and position
- Curves: hit rate vs window size for various cache capacities
- Histograms: expert selection weight distributions

### Cache Policy Simulator

**Cache model**: Per-layer LRU cache with configurable capacity (number of experts)

**Policies to evaluate**:

| Policy | Description | Parameters |
|--------|-------------|------------|
| `baseline` | No caching; all fetches from SSD | — |
| `fixed_hot` | Most frequent experts always resident | pool_size |
| `sliding_window` | Cache experts from last W tokens | window_size |
| `exp_decay` | Weight recent tokens exponentially | decay_rate |
| `boundary_reset` | Clear dynamic cache at turn boundaries | reset_trigger |
| `hybrid` | Fixed base + dynamic overflow | base_size, overflow_budget |

**Constraint model**:

The simulator supports a **continuous constraint parameter** `α ∈ [0, 1]`:

- `α = 0`: Hard constraint - router can only select from cached experts
- `α = 1`: No constraint - router selects from all experts (baseline)
- `0 < α < 1`: Soft constraint - uncached experts' scores are multiplied by `α`

This allows exploring the full tradeoff spectrum from "strict caching" to "unconstrained routing."

**Simulator outputs**:
- Cache hit rate (% of tokens where all needed experts were cached)
- Partial hit rate (% of needed experts that were cached)
- Simulated SSD fetches (unique experts loaded per token)
- Quality proxy: Estimated routing quality degradation from constrained selection

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Collection                                            │
│  DeepSeek-V2-Lite + kTransformers + Custom Hook                 │
│  └── Prompt Suite (coding, reasoning, multi-turn, etc.)        │
│       └── Routing Traces (Parquet files)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Analysis                                              │
│  Load Traces → Compute Metrics → Export Summaries               │
│  └── Locality metrics, expert entropy, layer patterns          │
│       └── Visualization outputs (plots, heatmaps)               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Simulation                                            │
│  Load Traces → Replay with Cache Policies                      │
│  └── Parameter sweep: window sizes, cache capacities, α        │
│       └── Tradeoff curves (hit rate vs quality proxy)           │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Collection Module (`collect.py`)

**Responsibilities**:
- Integrate with kTransformers inference
- Capture routing decisions with minimal overhead
- Write to Parquet in batches
- Track context boundaries (user/assistant turns)

**Output**: `traces/{timestamp}_{prompt_id}.parquet`

### 2. Analysis Module (`analyze.py`)

**Responsibilities**:
- Load routing traces
- Compute locality metrics
- Generate visualizations
- Export summary statistics for simulation

**Key functions**:
- `temporal_reuse_curve(traces, max_distance)` → reuse probability vs token distance
- `sliding_window_hit_rate(traces, window_size, cache_capacity)` → hit rate
- `expert_entropy_by_layer(traces)` → entropy per layer
- `context_switch_churn(traces)` → expert turnover at boundaries

**Output**: `analysis/metrics.json`, `analysis/plots/*.png`

### 3. Simulation Module (`simulate.py`)

**Responsibilities**:
- Replay traces through cache policies
- Simulate hard and soft constraints via α parameter
- Generate tradeoff data for policy comparison

**Key functions**:
- `simulate_policy(traces, policy, cache_capacity, alpha)` → hit rates, fetches
- `parameter_sweep(traces, policy_space)` → Pareto frontier data
- `estimate_quality_proxy(hit_rate, partial_hit_rate, alpha)` → quality degradation estimate

**Output**: `simulation/results.json`, `simulation/tradeoff_curves.png`

## Prompt Suite for Data Collection

To ensure diverse expert usage patterns, collect across these categories:

| Category | Example Prompts | Expected Expert Pattern |
|----------|-----------------|------------------------|
| `coding` | LeetCode problems, debugging | Structured reasoning, syntax experts |
| `reasoning` | Math word problems, logic | Multi-step deduction, arithmetic |
| `creative` | Story writing, poetry | Style, narrative, diverse vocabulary |
| `factual` | Trivia, explanations | Knowledge retrieval, fact lookup |
| `multi_turn` | Iterative debugging, refinement | Context-dependent switching |
| `mixed` | Coding with explanation, tool use | Mode transitions, boundary effects |

**Collection volume target**: 10,000+ tokens per category for statistical significance.

## Success Criteria

The smoothed routing hypothesis is viable if analysis reveals:

1. **High Temporal Locality**: >60% of experts at position t+10 were active in tokens t..t+9
2. **Bounded Working Set**: A 32-expert sliding window captures >70% of needed experts
3. **Context Coherence**: Expert sets remain stable within conversations, shift at boundaries
4. **Layer Heterogeneity**: Different layers exhibit different optimal cache sizes (justifies per-layer policies)
5. **Graceful Degradation**: Soft constraint (α = 0.5) achieves >80% hit rate with <10% quality proxy degradation

## Testing Strategy

### Unit Tests
- Trace loading and validation
- Metric computation correctness
- Cache policy deterministic behavior
- Constraint parameter boundary conditions

### Integration Tests
- End-to-end: collect → analyze → simulate pipeline
- Verify simulation replay matches collected traces
- Cross-validate hit rate computations

### Validation Tests
- Run simulator on synthetic traces with known properties
- Verify locality metrics match expected distributions
- Check that α=1 produces baseline behavior (no constraint)

## Error Handling Strategy

| Failure Mode | Mitigation |
|--------------|------------|
| Collection overhead too high | Reduce write frequency, increase batch size |
| Trace file corruption | Write checksums, keep backup batches |
| Insufficient data diversity | Expand prompt suite, track coverage metrics |
| Simulation divergence | Validate against analytical ground truth |

## Open Questions (for Planner)

1. **Hook implementation**: Specific integration point in kTransformers (wrapper vs patch)
2. **Storage optimization**: Chunking strategy for long traces (>1M tokens)
3. **Simulation acceleration**: Vectorized replay for large parameter sweeps
4. **Quality proxy validation**: How to estimate routing quality without running full inference

## Dependencies

- `pyarrow`: Parquet I/O
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`/`seaborn`: Visualization
- `ktransformers`: Inference engine (external)

## Output Artifacts

1. **Design Document**: `thoughts/shared/designs/2026-03-20-moe-routing-analysis-design.md` (this file)
2. **Implementation Plan**: `thoughts/shared/plans/2026-03-20-moe-routing-analysis.md` (to be created)
3. **Routing Traces**: `data/traces/*.parquet`
4. **Analysis Results**: `data/analysis/metrics.json`, `data/analysis/plots/`
5. **Simulation Results**: `data/simulation/results.json`, `data/simulation/tradeoff_curves.png`

## Next Steps

1. Validate design with user
2. Spawn Planner agent to create implementation plan
3. Execute plan via Executor agent
4. Validate results against success criteria

---

*Design created: 2026-03-20*
*Status: Draft - awaiting validation*
