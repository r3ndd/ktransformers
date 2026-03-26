from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .routing_schemes import SlidingWindowAdaptivePoolRouting
from .simulator import simulate_routing_scheme
from .token_indexing import add_absolute_token_position


def _safe_context_filename(context_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(context_id))
    return safe or "context"


def _run_grid(
    traces: pd.DataFrame,
    pool_size_per_layer: int,
    update_interval_tokens: int,
) -> list[dict]:
    token_count = int(traces["absolute_token_position"].nunique()) if len(traces) > 0 else 0
    runs: list[dict] = []
    if token_count == 0:
        return runs

    for window_size in [4, 8, 16, 32, 64]:
        if window_size > token_count:
            continue
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            scheme = SlidingWindowAdaptivePoolRouting(
                window_size=window_size,
                pool_size_per_layer=pool_size_per_layer,
                update_interval_tokens=update_interval_tokens,
            )
            m = simulate_routing_scheme(traces, scheme=scheme, alpha=alpha)
            run = {
                "scheme": "sliding_window_adaptive_pool",
                "window_size": window_size,
                "pool_size_per_layer": pool_size_per_layer,
                "update_interval_tokens": update_interval_tokens,
                "alpha": alpha,
                **m,
            }
            runs.append(run)
    return runs


def _average_runs(per_context_runs: list[list[dict]]) -> list[dict]:
    if not per_context_runs:
        return []

    buckets: dict[tuple[str, int, int, int, float], list[dict]] = {}
    for runs in per_context_runs:
        for run in runs:
            key = (
                str(run["scheme"]),
                int(run["window_size"]),
                int(run["pool_size_per_layer"]),
                int(run["update_interval_tokens"]),
                float(run["alpha"]),
            )
            buckets.setdefault(key, []).append(run)

    averaged: list[dict] = []
    for key in sorted(buckets.keys(), key=lambda k: (k[1], k[4])):
        rows = buckets[key]
        base = {
            "scheme": key[0],
            "window_size": key[1],
            "pool_size_per_layer": key[2],
            "update_interval_tokens": key[3],
            "alpha": key[4],
            "contexts_included": float(len(rows)),
        }
        numeric_keys = [
            "hit_rate",
            "ssd_fetches_per_token",
            "quality_degradation",
            "speedup_ratio",
            "token_count",
            "experts_per_token",
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

    pool_size_per_layer = 32
    update_interval_tokens = 1

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

        runs = _run_grid(
            context_traces,
            pool_size_per_layer=pool_size_per_layer,
            update_interval_tokens=update_interval_tokens,
        )
        per_context_runs.append(runs)

        context_doc = {
            "context_id": context_id,
            "runs": runs,
            "scheme": "sliding_window_adaptive_pool",
            "token_grouping_key": ["context_id", "absolute_token_position", "layer_id"],
            "context_token_count": context_token_counts[str(context_id)],
            "pool_size_per_layer": pool_size_per_layer,
            "update_interval_tokens": update_interval_tokens,
            "window_candidates": [4, 8, 16, 32, 64],
            "metric_note": "hit_rate is routed-vs-baseline expert overlap averaged across token-layer steps",
        }
        context_file = context_results_dir / f"{_safe_context_filename(str(context_id))}.json"
        context_file.write_text(json.dumps(context_doc, indent=2))

    runs = _average_runs(per_context_runs)

    result_doc = {
        "runs": runs,
        "scheme": "sliding_window_adaptive_pool",
        "token_grouping_key": ["context_id", "absolute_token_position", "layer_id"],
        "context_count": len(context_token_counts),
        "context_token_counts": context_token_counts,
        "per_context_dir": str(context_results_dir),
        "averaging_note": "Average metrics are computed per parameter set over contexts that included that parameter set.",
        "metric_note": "quality_degradation is mean(routed_score_avg / baseline_score_avg) over token-layer steps.",
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
