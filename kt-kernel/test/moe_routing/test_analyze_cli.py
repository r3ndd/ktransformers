import sys
import os

# Add the python directory to the path so moe_routing can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from pathlib import Path
import json

import pandas as pd

from moe_routing.analyze import run_analysis


def test_run_analysis_writes_metrics(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    df = pd.DataFrame(
        {
            "layer_id": [0, 0],
            "token_position": [0, 1],
            "expert_ids": [[1, 2, 3, 4, 5, 6], [1, 2, 7, 8, 9, 10]],
            "expert_weights": [[0.2] * 6, [0.2] * 6],
        }
    )
    df.to_parquet(in_file)
    out_dir = tmp_path / "analysis"
    run_analysis(in_file, out_dir)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "temporal_reuse_curve" in metrics
