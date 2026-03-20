from pathlib import Path
import json

import pandas as pd

from moe_routing.simulate import run_simulation


def test_run_simulation_writes_results(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    pd.DataFrame(
        {
            "layer_id": [0],
            "expert_ids": [[1, 2, 3, 4, 5, 6]],
            "expert_weights": [[1, 1, 1, 1, 1, 1]],
        }
    ).to_parquet(in_file)
    out_dir = tmp_path / "sim"
    run_simulation(in_file, out_dir)
    result = json.loads((out_dir / "results.json").read_text())
    assert len(result["runs"]) > 0
    # Task 2.1: assert compatibility metadata for stable CLI schema
    assert result["cache_identity"] == "layer_qualified"
    assert "partial_hit_rate counts layer-qualified matches" in result["metric_note"]
