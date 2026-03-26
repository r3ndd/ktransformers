from pathlib import Path

import pyarrow.parquet as pq

from kt_kernel.moe_routing.parquet_writer import AsyncParquetWriter
from kt_kernel.moe_routing.types import RoutingRecord


def test_writer_flushes_parquet(tmp_path: Path):
    out = tmp_path / "trace.parquet"
    writer = AsyncParquetWriter(out, flush_size=2)
    writer.start()
    writer.submit(RoutingRecord(1, "ctx", 0, 0, [1, 2, 3, 4, 5, 6], [0.2] * 6, 10, "user"))
    writer.submit(
        RoutingRecord(
            2,
            "ctx",
            0,
            1,
            [1, 2, 3, 4, 5, 6],
            [0.2] * 6,
            20,
            "assistant",
            expert_scores_all=[0.1, 0.2, 0.3],
        )
    )
    writer.close()

    table = pq.read_table(out)
    assert table.num_rows == 2
    assert "expert_ids" in table.column_names
    assert "expert_scores_all" in table.column_names
    assert str(table.schema.field("expert_scores_all").type) == "list<element: halffloat>"
