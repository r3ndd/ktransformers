from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .cache_policies import SlidingWindowPolicy
from .simulator import simulate_policy


def run_simulation(trace_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = pd.read_parquet(trace_file)

    fixed_capacity = 156
    runs = []
    for window in [8, 16, 32, 64]:
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            policy = SlidingWindowPolicy(capacity=fixed_capacity, window_size=window)
            m = simulate_policy(traces, policy, alpha=alpha)
            run = {
                "policy": "sliding_window",
                "window_size": window,
                "alpha": alpha,
                **m,
            }
            runs.append(run)

    result_doc = {
        "runs": runs,
        "metric_level": "token_across_layers",
        "cache_identity": "layer_qualified",
        "cache_capacity": fixed_capacity,
        "token_grouping_key": ["context_id", "token_position"],
        "deprecated_fields": ["hit_rate"],
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
