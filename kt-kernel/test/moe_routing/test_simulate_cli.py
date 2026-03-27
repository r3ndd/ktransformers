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
    assert result["scheme"] == "mixed"
    assert result["token_grouping_key"] == ["context_id", "absolute_token_position", "layer_id"]
    assert "scheme_candidates" in result

    run = result["runs"][0]
    assert run["scheme"] in {
        "sliding_window_score_averaging",
        "ema_score_averaging",
        "two_timescale_ema",
        "two_timescale_softmax",
    }
    if run["scheme"] == "sliding_window_score_averaging":
        assert "window_size" in run
    if run["scheme"] == "ema_score_averaging":
        assert "ema_beta" in run
    if run["scheme"] == "two_timescale_ema":
        assert "mix_lambda" in run
    if run["scheme"] == "two_timescale_softmax":
        assert "mix_lambda" in run
        assert "rho" in run
    assert "hit_rate" in run
    assert "ssd_fetches_per_token" in run
    assert "baseline_overlap" in run
    assert "quality_degradation" in run
    assert "speedup_ratio" in run
    assert "quality_speed_score" in run
    assert "token_count" in run
    assert "contexts_included" in run
    assert "alpha" not in run


def test_run_simulation_includes_window_1_baseline_equivalent(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    _sample_df().to_parquet(in_file)

    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)
    result = json.loads((out_dir / "results.json").read_text())

    run_w1 = next(
        r for r in result["runs"] if r["scheme"] == "sliding_window_score_averaging" and int(r["window_size"]) == 1
    )
    assert run_w1["baseline_overlap"] == 1.0

    run_tt = next(r for r in result["runs"] if r["scheme"] == "two_timescale_ema" and float(r["mix_lambda"]) == 0.4)
    assert "baseline_overlap" in run_tt

    run_tt_curr = next(
        r
        for r in result["runs"]
        if r["scheme"] == "two_timescale_softmax" and float(r["mix_lambda"]) == 0.2 and float(r["rho"]) == 0.25
    )
    assert "baseline_overlap" in run_tt_curr


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

    for tok in range(20):
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

    run_w4 = next(
        r for r in result["runs"] if r["scheme"] == "sliding_window_score_averaging" and int(r["window_size"]) == 4
    )
    run_w16 = next(
        r for r in result["runs"] if r["scheme"] == "sliding_window_score_averaging" and int(r["window_size"]) == 16
    )
    assert run_w4["contexts_included"] == 2.0
    assert run_w16["contexts_included"] == 1.0
