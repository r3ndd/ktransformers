from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metrics import expert_entropy_by_layer, temporal_reuse_curve


def run_analysis(trace_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = output_dir / "plots"
    plots.mkdir(exist_ok=True)

    traces = pd.read_parquet(trace_file)
    reuse = temporal_reuse_curve(traces, max_distance=32)
    entropy = expert_entropy_by_layer(traces)

    (output_dir / "metrics.json").write_text(
        json.dumps({"temporal_reuse_curve": reuse, "expert_entropy_by_layer": entropy}, indent=2)
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
