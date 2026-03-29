from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .routing_schemes import (
    EMAScoreAveragingRouting,
    PrefillBlockMeanRouting,
    PrefillFullMeanRouting,
    SlidingWindowScoreAveragingRouting,
    TwoTimescaleSoftmaxRouting,
    TwoTimescaleEMARouting,
)
from .simulator import simulate_routing_scheme
from .token_indexing import add_absolute_token_position


def _safe_context_filename(context_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(context_id))
    return safe or "context"


def _run_grid(traces: pd.DataFrame) -> list[dict]:
    token_count = int(traces["absolute_token_position"].nunique()) if len(traces) > 0 else 0
    runs: list[dict] = []
    if token_count == 0:
        return runs

    for window_size in [1, 4, 16, 64]:
        if window_size > token_count:
            continue
        scheme = SlidingWindowScoreAveragingRouting(window_size=window_size)
        m = simulate_routing_scheme(traces, scheme=scheme)
        run = {
            "scheme": "sliding_window_score_averaging",
            "window_size": window_size,
            **m,
        }
        runs.append(run)

    for window_size in [32, 64, 128]:
        scheme = PrefillBlockMeanRouting(window_size=window_size)
        m = simulate_routing_scheme(traces, scheme=scheme)
        run = {
            "scheme": "prefill_block_mean",
            "window_size": window_size,
            **m,
        }
        runs.append(run)

    scheme = PrefillFullMeanRouting()
    m = simulate_routing_scheme(traces, scheme=scheme)
    runs.append(
        {
            "scheme": "prefill_full_mean",
            **m,
        }
    )

    for ema_beta in [0.9, 0.7, 0.5, 0.3, 0.1, 0.05]:
        scheme = EMAScoreAveragingRouting(ema_beta=ema_beta)
        m = simulate_routing_scheme(traces, scheme=scheme)
        run = {
            "scheme": "ema_score_averaging",
            "ema_beta": ema_beta,
            **m,
        }
        runs.append(run)

    for mix_lambda in [0.1, 0.2, 0.3, 0.4]:
        scheme = TwoTimescaleEMARouting(mix_lambda=mix_lambda)
        m = simulate_routing_scheme(traces, scheme=scheme)
        run = {
            "scheme": "two_timescale_ema",
            "mix_lambda": mix_lambda,
            **m,
        }
        runs.append(run)

    for rho in [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]:
        scheme = TwoTimescaleSoftmaxRouting(mix_lambda=0.2, rho=rho)
        m = simulate_routing_scheme(traces, scheme=scheme)
        run = {
            "scheme": "two_timescale_softmax",
            "mix_lambda": 0.2,
            "rho": rho,
            **m,
        }
        runs.append(run)
    return runs


def _average_runs(per_context_runs: list[list[dict]]) -> list[dict]:
    if not per_context_runs:
        return []

    buckets: dict[tuple[str, float, float], list[dict]] = {}
    for runs in per_context_runs:
        for run in runs:
            scheme = str(run["scheme"])
            if scheme == "sliding_window_score_averaging":
                key = (scheme, float(int(run["window_size"])), -1.0)
            elif scheme == "prefill_block_mean":
                key = (scheme, float(int(run["window_size"])), -1.0)
            elif scheme == "prefill_full_mean":
                key = (scheme, -1.0, -1.0)
            elif scheme == "ema_score_averaging":
                key = (scheme, float(run["ema_beta"]), -1.0)
            elif scheme == "two_timescale_ema":
                key = (scheme, float(run["mix_lambda"]), -1.0)
            elif scheme == "two_timescale_softmax":
                key = (scheme, float(run["mix_lambda"]), float(run["rho"]))
            else:
                raise ValueError(f"Unknown scheme in runs: {scheme}")
            buckets.setdefault(key, []).append(run)

    averaged: list[dict] = []

    def _sort_key(k: tuple[str, float, float]) -> tuple[int, float, float]:
        if k[0] == "sliding_window_score_averaging":
            scheme_order = 0
        elif k[0] == "prefill_block_mean":
            scheme_order = 1
        elif k[0] == "prefill_full_mean":
            scheme_order = 2
        elif k[0] == "ema_score_averaging":
            scheme_order = 3
        elif k[0] == "two_timescale_ema":
            scheme_order = 4
        elif k[0] == "two_timescale_softmax":
            scheme_order = 5
        else:
            scheme_order = 99
        return (scheme_order, k[1], k[2])

    for key in sorted(buckets.keys(), key=_sort_key):
        rows = buckets[key]
        if key[0] == "sliding_window_score_averaging":
            base = {
                "scheme": key[0],
                "window_size": int(key[1]),
                "contexts_included": float(len(rows)),
            }
        elif key[0] == "prefill_block_mean":
            base = {
                "scheme": key[0],
                "window_size": int(key[1]),
                "contexts_included": float(len(rows)),
            }
        elif key[0] == "prefill_full_mean":
            base = {
                "scheme": key[0],
                "contexts_included": float(len(rows)),
            }
        elif key[0] == "ema_score_averaging":
            base = {
                "scheme": key[0],
                "ema_beta": float(key[1]),
                "contexts_included": float(len(rows)),
            }
        elif key[0] == "two_timescale_ema":
            base = {
                "scheme": key[0],
                "mix_lambda": float(key[1]),
                "contexts_included": float(len(rows)),
            }
        else:
            base = {
                "scheme": key[0],
                "mix_lambda": float(key[1]),
                "rho": float(key[2]),
                "contexts_included": float(len(rows)),
            }
        numeric_keys = [
            "hit_rate",
            "ssd_fetches_per_token",
            "baseline_overlap",
            "quality_degradation",
            "speedup_ratio",
            "quality_speed_score",
            "token_count",
            "experts_per_token",
            "baseline_ssd_fetches_per_token",
        ]
        for k in numeric_keys:
            base[k] = float(sum(r[k] for r in rows) / len(rows))
        averaged.append(base)

    return averaged


def run_simulation(trace_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    context_results_dir = output_dir / "contexts"
    context_results_dir.mkdir(exist_ok=True)

    traces = pd.read_parquet(trace_file)
    if "context_id" not in traces.columns:
        traces = traces.copy()
        traces["context_id"] = "default"

    traces = add_absolute_token_position(traces)

    context_token_counts: dict[str, int] = {}
    for context_id, g in traces.groupby("context_id", sort=False):
        context_token_counts[str(context_id)] = int(g["absolute_token_position"].nunique())

    per_context_runs: list[list[dict]] = []
    for context_id, g in traces.groupby("context_id", sort=False):
        context_traces = g.copy()
        context_traces = (
            context_traces.sort_values(["token_id"], kind="stable")
            if "token_id" in context_traces.columns
            else context_traces
        )

        runs = _run_grid(context_traces)
        per_context_runs.append(runs)

        context_doc = {
            "context_id": context_id,
            "runs": runs,
            "scheme": "mixed",
            "token_grouping_key": ["context_id", "absolute_token_position", "layer_id"],
            "context_token_count": context_token_counts[str(context_id)],
            "scheme_candidates": {
                "sliding_window_score_averaging": {"window_size": [1, 4, 16, 64]},
                "prefill_block_mean": {"window_size": [32, 64, 128]},
                "prefill_full_mean": {},
                "ema_score_averaging": {"ema_beta": [0.9, 0.7, 0.5, 0.3, 0.1, 0.05]},
                "two_timescale_ema": {"mix_lambda": [0.1, 0.2, 0.3, 0.4]},
                "two_timescale_softmax": {
                    "mix_lambda": [0.2],
                    "rho": [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0],
                },
            },
            "metric_note": "hit_rate is percentage of chosen experts already in previous-token fast-memory state",
        }
        context_file = context_results_dir / f"{_safe_context_filename(str(context_id))}.json"
        context_file.write_text(json.dumps(context_doc, indent=2))

    runs = _average_runs(per_context_runs)

    result_doc = {
        "runs": runs,
        "scheme": "mixed",
        "token_grouping_key": ["context_id", "absolute_token_position", "layer_id"],
        "context_count": len(context_token_counts),
        "context_token_counts": context_token_counts,
        "per_context_dir": str(context_results_dir),
        "scheme_candidates": {
            "sliding_window_score_averaging": {"window_size": [1, 4, 16, 64]},
            "prefill_block_mean": {"window_size": [32, 64, 128]},
            "prefill_full_mean": {},
            "ema_score_averaging": {"ema_beta": [0.9, 0.7, 0.5, 0.3, 0.1, 0.05]},
            "two_timescale_ema": {"mix_lambda": [0.1, 0.2, 0.3, 0.4]},
            "two_timescale_softmax": {
                "mix_lambda": [0.2],
                "rho": [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0],
            },
        },
        "averaging_note": "Average metrics are computed per parameter set over contexts that included that parameter set.",
        "metric_note": "baseline_overlap is chosen-vs-baseline overlap averaged across token-layer steps.",
    }
    (output_dir / "results.json").write_text(json.dumps(result_doc, indent=2))

    try:
        import matplotlib.pyplot as plt

        xs = [r["hit_rate"] for r in runs]
        ys = [r["quality_degradation"] for r in runs]
        plt.figure(figsize=(6, 5))
        plt.scatter(xs, ys)
        plt.xlabel("Hit Rate")
        plt.ylabel("Quality Degradation Ratio")
        plt.title("Routing Scheme Tradeoff Frontier")
        plt.tight_layout()
        plt.savefig(output_dir / "tradeoff_curves.png")
        plt.close()
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser("moe-routing-simulate")
    p.add_argument("--trace-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run_simulation(args.trace_file, args.output_dir)


if __name__ == "__main__":
    main()
