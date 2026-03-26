from pathlib import Path
import json
import os
import sys

import pandas as pd

_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)

from moe_routing.simulate import run_simulation


def _sample_df() -> pd.DataFrame:
    rows = []
    tid = 0
    for tok in range(6):
        rows.append(
            {
                "token_id": tid,
                "context_id": "c0",
                "token_position": tok,
                "layer_id": 0,
                "expert_ids": [1, 2],
                "expert_weights": [0.9, 0.8],
                "expert_scores_all": [0.0, 0.9, 0.8, 0.1],
            }
        )
        tid += 1
    return pd.DataFrame(rows)


def test_run_simulation_writes_routing_scheme_results(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    _sample_df().to_parquet(in_file)

    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)

    result = json.loads((out_dir / "results.json").read_text())
    assert len(result["runs"]) > 0
    assert result["scheme"] == "sliding_window_score_averaging"
    assert result["token_grouping_key"] == ["context_id", "absolute_token_position", "layer_id"]

    run = result["runs"][0]
    assert run["scheme"] == "sliding_window_score_averaging"
    assert "window_size" in run
    assert "hit_rate" in run
    assert "ssd_fetches_per_token" in run
    assert "baseline_overlap" in run
    assert "quality_degradation" in run
    assert "speedup_ratio" in run
    assert "token_count" in run
    assert "contexts_included" in run
    assert "alpha" not in run


def test_run_simulation_includes_window_1_baseline_equivalent(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    _sample_df().to_parquet(in_file)

    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)
    result = json.loads((out_dir / "results.json").read_text())

    run_w1 = next(r for r in result["runs"] if int(r["window_size"]) == 1)
    assert run_w1["baseline_overlap"] == 1.0


def test_group_average_uses_available_contexts_per_parameter_set(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"

    short_rows = []
    long_rows = []

    for tok in range(4):
        short_rows.append(
            {
                "token_id": tok,
                "context_id": "short",
                "token_position": tok,
                "layer_id": 0,
                "expert_ids": [1, 2],
                "expert_weights": [0.9, 0.8],
                "expert_scores_all": [0.0, 0.9, 0.8, 0.1],
            }
        )

    for tok in range(8):
        long_rows.append(
            {
                "token_id": tok,
                "context_id": "long",
                "token_position": tok,
                "layer_id": 0,
                "expert_ids": [1, 2],
                "expert_weights": [0.9, 0.8],
                "expert_scores_all": [0.0, 0.9, 0.8, 0.1],
            }
        )

    pd.DataFrame(short_rows + long_rows).to_parquet(in_file)

    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)
    result = json.loads((out_dir / "results.json").read_text())

    run_w4 = next(r for r in result["runs"] if int(r["window_size"]) == 4)
    run_w8 = next(r for r in result["runs"] if int(r["window_size"]) == 8)
    assert run_w4["contexts_included"] == 2.0
    assert run_w8["contexts_included"] == 1.0
