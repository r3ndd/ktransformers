from pathlib import Path
import json
import os
import sys

import pandas as pd

# Add python directory to path for local test execution/import resolution
_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)

from moe_routing.simulate import run_simulation


def test_run_simulation_writes_results(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    pd.DataFrame(
        {
            "token_id": [0, 1, 2, 3],
            "context_id": [0, 0, 0, 0],
            "token_position": [0, 1, 2, 3],
            "layer_id": [0, 0, 0, 0],
            "expert_ids": [[1, 2, 3, 4, 5, 6]] * 4,
            "expert_weights": [[1, 1, 1, 1, 1, 1]] * 4,
        }
    ).to_parquet(in_file)
    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)
    result = json.loads((out_dir / "results.json").read_text())
    assert len(result["runs"]) > 0

    assert result["metric_level"] == "token_across_layers"
    assert result["cache_identity"] == "layer_qualified"
    assert result["cache_capacity"] == 1024
    assert result["token_grouping_key"] == ["context_id", "absolute_token_position"]

    run = result["runs"][0]
    assert "partial_hit_rate" in run
    assert "avg_misses_per_token" in run
    assert "token_count" in run
    assert "hit_rate" not in run
    assert "full_hit_rate" not in run

    assert "capacity" not in run
    assert "cache_capacities" not in result
    assert "capacity_sweep" not in result
    assert "contexts_included" in run
    assert (out_dir / "contexts" / "0.json").exists()


def test_run_simulation_writes_per_context_and_averages_aligned_tokens(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"

    # context a: 3 absolute tokens (single-token microbatches)
    # context b: 2 absolute tokens
    df = pd.DataFrame(
        {
            "token_id": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
            "context_id": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b"],
            "layer_id": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "token_position": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "expert_ids": [[1], [2], [1], [2], [1], [2], [1], [2], [1], [2]],
            "expert_weights": [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]],
        }
    )
    df.to_parquet(in_file)

    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)

    result = json.loads((out_dir / "results.json").read_text())
    assert result["context_count"] == 2
    assert (out_dir / "contexts" / "a.json").exists()
    assert (out_dir / "contexts" / "b.json").exists()


def test_run_simulation_skips_window_size_larger_than_context_token_count(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    # One context, only 2 tokens -> all configured windows should be skipped
    df = pd.DataFrame(
        {
            "token_id": [0, 1, 2, 3],
            "context_id": ["tiny", "tiny", "tiny", "tiny"],
            "layer_id": [0, 1, 0, 1],
            "token_position": [0, 0, 0, 0],
            "expert_ids": [[1], [2], [1], [2]],
            "expert_weights": [[1], [1], [1], [1]],
        }
    )
    df.to_parquet(in_file)

    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)

    ctx = json.loads((out_dir / "contexts" / "tiny.json").read_text())
    windows = sorted({int(r["window_size"]) for r in ctx["runs"]})
    assert windows == []


def test_group_average_uses_available_contexts_per_parameter_set(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    # context short: 2 tokens; context long: 8 tokens
    short_rows = [
        {
            "token_id": 0,
            "context_id": "short",
            "layer_id": 0,
            "token_position": 0,
            "expert_ids": [1],
            "expert_weights": [1],
        },
        {
            "token_id": 1,
            "context_id": "short",
            "layer_id": 1,
            "token_position": 0,
            "expert_ids": [2],
            "expert_weights": [1],
        },
        {
            "token_id": 2,
            "context_id": "short",
            "layer_id": 0,
            "token_position": 0,
            "expert_ids": [1],
            "expert_weights": [1],
        },
        {
            "token_id": 3,
            "context_id": "short",
            "layer_id": 1,
            "token_position": 0,
            "expert_ids": [3],
            "expert_weights": [1],
        },
    ]
    long_rows = []
    tid = 0
    for _ in range(8):
        long_rows.append(
            {
                "token_id": tid,
                "context_id": "long",
                "layer_id": 0,
                "token_position": 0,
                "expert_ids": [1],
                "expert_weights": [1],
            }
        )
        tid += 1
        long_rows.append(
            {
                "token_id": tid,
                "context_id": "long",
                "layer_id": 1,
                "token_position": 0,
                "expert_ids": [2],
                "expert_weights": [1],
            }
        )
        tid += 1

    df = pd.DataFrame(short_rows + long_rows)
    df.to_parquet(in_file)

    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)
    result = json.loads((out_dir / "results.json").read_text())

    run_w4 = next(r for r in result["runs"] if int(r["window_size"]) == 4 and float(r["alpha"]) == 0.0)
    run_w8 = next(r for r in result["runs"] if int(r["window_size"]) == 8 and float(r["alpha"]) == 0.0)
    assert run_w4["contexts_included"] == 1.0
    assert run_w8["contexts_included"] == 1.0
