from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .metrics import (
    context_switch_churn,
    expert_entropy_by_layer,
    previous_token_reuse_curve,
    sliding_window_hit_rate,
    temporal_reuse_curve,
)
from .token_indexing import add_absolute_token_position


def _safe_context_filename(context_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(context_id))
    return safe or "context"


def _average_numeric_dicts(dicts: list[dict[int, float]]) -> dict[int, float]:
    if not dicts:
        return {}
    all_keys = sorted({k for d in dicts for k in d.keys()})
    out: dict[int, float] = {}
    for k in all_keys:
        vals = [d[k] for d in dicts if k in d]
        out[k] = float(sum(vals) / len(vals)) if vals else 0.0
    return out


def run_analysis(trace_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)
    context_metrics_dir = output_dir / "contexts"
    context_metrics_dir.mkdir(exist_ok=True)

    traces = pd.read_parquet(trace_file)
    if "context_id" not in traces.columns:
        traces = traces.copy()
        traces["context_id"] = "default"

    traces = add_absolute_token_position(traces)

    context_payloads: list[dict] = []
    context_reuse_curves: list[dict[int, float]] = []
    context_sliding_window_maps: list[dict[int, float]] = []
    context_entropy_maps: list[dict[int, float]] = []
    context_churn_values: list[float] = []
    context_token_counts: dict[str, int] = {}

    for context_id, g in traces.groupby("context_id", sort=False):
        token_count = int(g["absolute_token_position"].nunique())
        max_distance = min(64, max(0, token_count - 1))

        reuse = temporal_reuse_curve(g, max_distance=max_distance)
        prev_reuse = previous_token_reuse_curve(g)
        sliding_hits = sliding_window_hit_rate(g)
        entropy = expert_entropy_by_layer(g)
        churn = context_switch_churn(g)

        context_reuse_curves.append(reuse)
        context_sliding_window_maps.append(sliding_hits)
        context_entropy_maps.append(entropy)
        context_churn_values.append(churn)
        context_token_counts[str(context_id)] = token_count

        payload = {
            "context_id": context_id,
            "token_count": token_count,
            "temporal_reuse_curve": reuse,
            "previous_token_reuse_curve": prev_reuse,
            "sliding_window_hit_rate": sliding_hits,
            "expert_entropy_by_layer": entropy,
            "context_switch_churn": churn,
        }
        context_payloads.append(payload)

        context_file = context_metrics_dir / f"{_safe_context_filename(str(context_id))}.json"
        context_file.write_text(json.dumps(payload, indent=2))

    if context_token_counts:
        min_tokens = min(context_token_counts.values())
    else:
        min_tokens = 0

    aligned_reuse_curves: list[dict[int, float]] = []
    aligned_prev_reuse_curves: list[dict[int, float]] = []
    aligned_sliding_window_maps: list[dict[int, float]] = []
    aligned_entropy_maps: list[dict[int, float]] = []
    aligned_churn_values: list[float] = []
    if min_tokens > 0:
        for context_id, g in traces.groupby("context_id", sort=False):
            aligned = g[g["absolute_token_position"] < min_tokens]
            aligned_reuse = temporal_reuse_curve(aligned, max_distance=min(64, max(0, min_tokens - 1)))
            aligned_prev_reuse = previous_token_reuse_curve(aligned)
            aligned_sliding_hits = sliding_window_hit_rate(aligned)
            aligned_entropy = expert_entropy_by_layer(aligned)
            aligned_churn = context_switch_churn(aligned)
            aligned_reuse_curves.append(aligned_reuse)
            aligned_prev_reuse_curves.append(aligned_prev_reuse)
            aligned_sliding_window_maps.append(aligned_sliding_hits)
            aligned_entropy_maps.append(aligned_entropy)
            aligned_churn_values.append(aligned_churn)

    reuse = _average_numeric_dicts(aligned_reuse_curves)
    prev_reuse = _average_numeric_dicts(aligned_prev_reuse_curves)
    sliding_hits = _average_numeric_dicts(aligned_sliding_window_maps)
    entropy = _average_numeric_dicts(aligned_entropy_maps)
    churn = (float(sum(aligned_churn_values) / len(aligned_churn_values))) if aligned_churn_values else 0.0

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "temporal_reuse_curve": reuse,
                "previous_token_reuse_curve": prev_reuse,
                "sliding_window_hit_rate": sliding_hits,
                "expert_entropy_by_layer": entropy,
                "context_switch_churn": churn,
                "context_count": len(context_token_counts),
                "aligned_token_count": min_tokens,
                "averaging_note": "Overall metrics average per-context metrics over the first N tokens, where N is the minimum token count across contexts.",
                "per_context_dir": str(context_metrics_dir),
                "context_token_counts": context_token_counts,
            },
            indent=2,
        )
    )

    try:
        import matplotlib.pyplot as plt

        xs = list(reuse.keys())
        ys = list(reuse.values())
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys)
        plt.title("Temporal Reuse Curve")
        plt.xlabel("Distance")
        plt.ylabel("Reuse Probability")
        plt.tight_layout()
        plt.savefig(plots / "temporal_reuse_curve.png")
        plt.close()
    except Exception:
        # Metrics JSON is the primary artifact; plotting is best-effort.
        pass


def main() -> None:
    p = argparse.ArgumentParser("moe-routing-analyze")
    p.add_argument("--trace-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    run_analysis(args.trace_file, args.output_dir)


if __name__ == "__main__":
    main()
