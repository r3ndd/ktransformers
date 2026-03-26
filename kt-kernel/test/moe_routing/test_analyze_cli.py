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
    assert "previous_token_reuse_curve" in metrics
    assert "aligned_token_count" in metrics
    assert (out_dir / "contexts" / "default.json").exists()


def test_run_analysis_writes_one_file_per_context_and_aligns_by_min_tokens(tmp_path: Path):
    in_file = tmp_path / "trace.parquet"
    # context a has 3 absolute tokens, context b has 2; overall should align to 2
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

    out_dir = tmp_path / "analysis"
    run_analysis(in_file, out_dir)

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["aligned_token_count"] == 2
    assert metrics["context_count"] == 2
    assert "previous_token_reuse_curve" in metrics
    assert (out_dir / "contexts" / "a.json").exists()
    assert (out_dir / "contexts" / "b.json").exists()
