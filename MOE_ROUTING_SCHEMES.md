# MoE Routing Schemes

This document defines the current routing-simulation design:

- Routing schemes operate on **full per-layer expert score vectors** (`expert_scores_all`), not expert pools.
- Each scheme transforms scores, then routing is always **top-k** from transformed scores.
- Baseline routing does not need a separate scheme: it is already in trace `expert_ids` and corresponds to `W=1` for score averaging.

Real-inference benchmarking now uses the same scheme family and sweep set via
`scripts/run_real_routing_benchmark.py`, with per-request config passed in
`custom_params.moe_routing`.

Terminology:
- This document describes **simulation-time score transforms** and does not yet implement separate prefill/decode runtime paths.
- `cache accounting` here means simulation accounting for expert locality, not the live serving cache implementation.

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

## Scheme 2: EMA score averaging

Alternative smoothing with exponential decay (implemented):

- `s'_t = beta * s_t + (1-beta) * s'_{t-1}`
- Parameter: `ema_beta`.

Current sweep:

- `ema_beta` in `[0.9, 0.7, 0.5, 0.3, 0.1, 0.05]`.

This keeps the same framework: transform full scores, then top-k.

## Scheme 3: Two-timescale EMA averaging

Combine short and long horizon EMA smoothing (implemented):

- `s'_t = lambda * short_window_avg + (1-lambda) * long_window_avg`
- `short_t = EMA(beta=0.5)`
- `long_t = EMA(beta=0.05)`
- Parameter sweep: `mix_lambda` in `[0.1, 0.2, 0.3, 0.4]`.

Again, no discrete pool logic; routing is still top-k on transformed full scores.

## Scheme 4: Two-timescale softmax

This is a scaled-softmax-input version of scheme 3:

- `x_t = softmax(rho * s_t)`
- `short_t = EMA_beta_0.5(x_t)`
- `long_t = EMA_beta_0.05(x_t)`
- `s'_t = mix_lambda * short_t + (1-mix_lambda) * long_t`

where:

- `s_t` is current token full expert score vector (logits)
- `rho` scales logits before softmax (`rho >= 0` in implementation)

Current sweep:

- fixed `mix_lambda = 0.2`
- `rho` in `[0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]`

## Planned prefill/decode split (real inference)

To align with the real-inference plan, runtime routing will support separate policies per request phase:
- Prefill (extend) policy examples:
  - `prefill_block_mean` with fixed `window_size` (for example, 64)
  - `prefill_full_mean` across the full prefill span per layer
- Decode policy examples:
  - Existing decode-oriented schemes in this document (`sliding_window_score_averaging`, `ema_score_averaging`, `two_timescale_ema`, `two_timescale_softmax`)

The prefill policies above are planned for real serving; they are not yet part of the current simulation CLI.

## Metrics (current simulation)

For simulated chosen experts vs baseline trace experts:

- `hit_rate`: percentage of chosen experts already cached from the previous token state.
- `ssd_fetches_per_token`: average over tokens of sum across layers of uncached chosen experts.
- `baseline_overlap`: percentage of chosen experts that overlap baseline chosen experts.
- `quality_degradation`: ratio of average softmax probability mass of chosen experts to baseline experts.
- `speedup_ratio`: `(0.1 + baseline_extra_seconds_per_token) / (0.1 + extra_seconds_per_token)` where extra seconds are derived from SSD fetch counts.
- `quality_speed_score`: `quality_degradation * speedup_ratio`.

`baseline_ssd_fetches_per_token` is computed from baseline trace routing under the same cache accounting rule.

Expert cache accounting currently uses per-layer LRU with fixed `capacity_per_layer=25` (about 1000 expert slots total for 40 layers).

## Metrics (real-inference benchmark)

The real benchmark reports per-run and aggregated metrics per scheme/parameter set:

- Runtime:
  - `elapsed_seconds`
  - `tokens_per_second`
  - `speedup_ratio` (relative to baseline tokens/sec on the same prompt)
- Quality proxy (text-level against baseline output on same prompt and seed):
  - `quality_similarity` (difflib sequence ratio)
  - `quality_jaccard` (token-set Jaccard)
  - `quality_degradation = 1 - quality_similarity`
- Combined score:
  - `quality_speed_score = quality_similarity * speedup_ratio`

In addition, generated text is persisted for every run so quality can be
inspected qualitatively and via downstream metrics.

## Practical sweep recommendation

Current sweeps:

- Scheme 1 (`sliding_window_score_averaging`): `window_size` in `[1, 4, 16, 64]`
- Scheme 2 (`ema_score_averaging`): `ema_beta` in `[0.9, 0.7, 0.5, 0.3, 0.1, 0.05]`
- Scheme 3 (`two_timescale_ema`): `mix_lambda` in `[0.1, 0.2, 0.3, 0.4]`
- Scheme 4 (`two_timescale_softmax`): fixed `mix_lambda=0.2`, `rho` in `[0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]`

Suggested analysis views:

- Start analysis with:
  - `baseline_overlap` vs `ssd_fetches_per_token`
  - `quality_degradation` vs `speedup_ratio`

Interpretation:

- Larger `W` generally increases smoothing and can improve cache locality.
- Too large `W` can lag adaptation and reduce baseline overlap/quality.
