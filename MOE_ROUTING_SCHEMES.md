# MoE Routing Schemes

This document defines a simplified routing-simulation design:

- Routing schemes operate on **full per-layer expert score vectors** (`expert_scores_all`), not expert pools.
- Each scheme transforms scores, then routing is always **top-k** from transformed scores.
- Baseline routing does not need a separate scheme: it is already in trace `expert_ids` and corresponds to `W=1` for score averaging.

## Core idea

For each token and layer:

1. Read current full score vector `s_t` (length = number of experts, e.g. 256).
2. Compute transformed/smoothed score vector `s'_t` using scheme state.
3. Select routed experts as `topk(s'_t)`.

The purpose is to test whether smoothing routing decisions improves locality/cache behavior while preserving quality.

## Scheme 1: Sliding-window score averaging

This is the first implemented scheme.

- **Definition**: for window size `W`,
  - `s'_t = mean(s_{t-W+1}, ..., s_{t-1}, s_t)` over available history in same context/layer.
- **Equivalent baseline**: `W=1` gives `s'_t = s_t`, i.e. original routing behavior.
- **Parameter**:
  - `window_size` (`W`), swept in simulation.

Implementation note:

- History is maintained per layer and per context token order.
- Scores are full vectors from parquet `expert_scores_all`.

## Scheme 2 (planned): EMA score averaging

Alternative smoothing with exponential decay:

- `s'_t = beta * s_t + (1-beta) * s'_{t-1}`
- Parameter: `ema_beta`.

This keeps the same framework: transform full scores, then top-k.

## Scheme 3 (planned): Two-timescale averaging

Combine short and long horizon smoothing:

- `s'_t = lambda * short_window_avg + (1-lambda) * long_window_avg`
- Parameters: `short_W`, `long_W`, `lambda`.

Again, no discrete pool logic; routing is still top-k on transformed full scores.

## Metrics (current simulation)

For simulated chosen experts vs baseline trace experts:

- `hit_rate`: percentage of chosen experts already cached from the previous token state.
- `ssd_fetches_per_token`: average over tokens of sum across layers of uncached chosen experts.
- `baseline_overlap`: percentage of chosen experts that overlap baseline chosen experts.
- `quality_degradation`: ratio of average score of chosen experts to average score of baseline experts.
- `speedup_ratio`: `(1 + baseline_ssd_fetches_per_token) / (1 + ssd_fetches_per_token)`.

`baseline_ssd_fetches_per_token` is computed from baseline trace routing under the same cache accounting rule.

## Practical sweep recommendation

For the sliding-window scheme:

- Sweep `window_size` over `[1, 2, 4, 8, 16, 32, 64]`.
- Start analysis with:
  - `baseline_overlap` vs `ssd_fetches_per_token`
  - `quality_degradation` vs `speedup_ratio`

Interpretation:

- Larger `W` generally increases smoothing and can improve cache locality.
- Too large `W` can lag adaptation and reduce baseline overlap/quality.
