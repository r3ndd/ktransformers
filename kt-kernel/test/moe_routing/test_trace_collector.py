from pathlib import Path

import pyarrow.parquet as pq

from moe_routing.trace_collector import RoutingTraceCollector


def test_collector_records_and_writes(tmp_path: Path):
    c = RoutingTraceCollector(output_dir=tmp_path, prompt_id="p1")
    c.start(context_id="ctx-1")
    c.record(
        layer_id=0,
        token_position=0,
        expert_ids=[1, 2, 3, 4, 5, 6],
        expert_weights=[0.2] * 6,
        expert_scores_all=[0.1, 0.2, 0.3],
    )
    c.stop()
    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
    t = pq.read_table(files[0])
    assert t.num_rows == 1
    assert "expert_scores_all" in t.column_names
