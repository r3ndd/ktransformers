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
            "context_id": [0],
            "token_position": [0],
            "layer_id": [0],
            "expert_ids": [[1, 2, 3, 4, 5, 6]],
            "expert_weights": [[1, 1, 1, 1, 1, 1]],
        }
    ).to_parquet(in_file)
    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)
    result = json.loads((out_dir / "results.json").read_text())
    assert len(result["runs"]) > 0

    assert result["metric_level"] == "token_across_layers"
    assert result["cache_identity"] == "layer_qualified"
    assert result["cache_capacity"] == 156
    assert result["token_grouping_key"] == ["context_id", "token_position"]

    run = result["runs"][0]
    assert "partial_hit_rate" in run
    assert "avg_misses_per_token" in run
    assert "token_count" in run
    assert "hit_rate" not in run
    assert "full_hit_rate" not in run

    assert "capacity" not in run
    assert "cache_capacities" not in result
    assert "capacity_sweep" not in result
