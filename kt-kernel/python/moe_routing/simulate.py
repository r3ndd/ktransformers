from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .cache_policies import SlidingWindowPolicy
from .simulator import simulate_policy
from .token_indexing import add_absolute_token_position


def _safe_context_filename(context_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(context_id))
    return safe or "context"


def _run_grid(traces: pd.DataFrame, capacity: int) -> list[dict]:
    token_count = int(traces["absolute_token_position"].nunique()) if len(traces) > 0 else 0
    runs = []
    for window in [4, 8, 16, 32, 64]:
        if window > token_count:
            continue
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            policy = SlidingWindowPolicy(capacity=capacity, window_size=window)
            m = simulate_policy(traces, policy, alpha=alpha)
            run = {
                "policy": "sliding_window",
                "window_size": window,
                "alpha": alpha,
                **m,
            }
            runs.append(run)
    return runs


def _average_runs(per_context_runs: list[list[dict]]) -> list[dict]:
    if not per_context_runs:
        return []

    buckets: dict[tuple[str, int, float], list[dict]] = {}
    for runs in per_context_runs:
        for run in runs:
            key = (str(run["policy"]), int(run["window_size"]), float(run["alpha"]))
            buckets.setdefault(key, []).append(run)

    averaged: list[dict] = []
    for key in sorted(buckets.keys(), key=lambda k: (k[1], k[2])):
        rows = buckets[key]
        base = {
            "policy": key[0],
            "window_size": key[1],
            "alpha": key[2],
            "contexts_included": float(len(rows)),
        }
        numeric_keys = [
            "partial_hit_rate",
            "simulated_ssd_fetches",
            "avg_misses_per_token",
            "quality_proxy_degradation",
            "token_count",
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

    fixed_capacity = 1024

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

        runs = _run_grid(context_traces, fixed_capacity)
        per_context_runs.append(runs)

        context_doc = {
            "context_id": context_id,
            "runs": runs,
            "metric_level": "token_across_layers",
            "cache_identity": "layer_qualified",
            "cache_capacity": fixed_capacity,
            "token_grouping_key": ["context_id", "absolute_token_position"],
            "context_token_count": context_token_counts[str(context_id)],
            "skipped_window_sizes": [w for w in [4, 8, 16, 32, 64] if w > context_token_counts[str(context_id)]],
            "metric_note": "partial_hit_rate is average per-token partial hit across all layers",
        }
        context_file = context_results_dir / f"{_safe_context_filename(str(context_id))}.json"
        context_file.write_text(json.dumps(context_doc, indent=2))

    runs = _average_runs(per_context_runs)

    result_doc = {
        "runs": runs,
        "metric_level": "token_across_layers",
        "cache_identity": "layer_qualified",
        "cache_capacity": fixed_capacity,
        "token_grouping_key": ["context_id", "absolute_token_position"],
        "context_count": len(context_token_counts),
        "context_token_counts": context_token_counts,
        "per_context_dir": str(context_results_dir),
        "averaging_note": "Average simulation metrics are computed per parameter set by averaging over contexts that had enough tokens to run that parameter set.",
        "metric_note": "partial_hit_rate is average per-token partial hit across all layers",
    }
    (output_dir / "results.json").write_text(json.dumps(result_doc, indent=2))

    try:
        import matplotlib.pyplot as plt

        xs = [r["partial_hit_rate"] for r in runs]
        ys = [1.0 - r["quality_proxy_degradation"] for r in runs]
        plt.figure(figsize=(6, 5))
        plt.scatter(xs, ys)
        plt.xlabel("Partial Hit Rate")
        plt.ylabel("Quality Proxy (1 - degradation)")
        plt.title("Cache Tradeoff Frontier")
        plt.tight_layout()
        plt.savefig(output_dir / "tradeoff_curves.png")
        plt.close()
    except Exception:
        # JSON results are primary; plotting is best-effort.
        pass


def main() -> None:
    p = argparse.ArgumentParser("moe-routing-simulate")
    p.add_argument("--trace-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run_simulation(args.trace_file, args.output_dir)


if __name__ == "__main__":
    main()
