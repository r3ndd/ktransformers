from __future__ import annotations

import queue
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .types import RoutingRecord


class AsyncParquetWriter:
    def __init__(self, output_path: Path, flush_size: int = 1024):
        self.output_path = output_path
        self.flush_size = flush_size
        self._q: "queue.Queue[RoutingRecord | None]" = queue.Queue(maxsize=8192)
        self._t: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, rec: RoutingRecord) -> None:
        self._q.put(rec)

    def close(self) -> None:
        self._q.put(None)
        if self._t is not None:
            self._t.join()

    def _run(self) -> None:
        batch: list[RoutingRecord] = []
        writer: pq.ParquetWriter | None = None
        while True:
            item = self._q.get()
            if item is None:
                break
            batch.append(item)
            if len(batch) >= self.flush_size:
                writer = self._flush(batch, writer)
                batch.clear()

        if batch:
            writer = self._flush(batch, writer)
        if writer is not None:
            writer.close()

    def _flush(self, rows: list[RoutingRecord], writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
        data = {
            "token_id": pa.array([r.token_id for r in rows], type=pa.int64()),
            "context_id": pa.array([r.context_id for r in rows], type=pa.string()),
            "layer_id": pa.array([r.layer_id for r in rows], type=pa.int32()),
            "token_position": pa.array([r.token_position for r in rows], type=pa.int32()),
            "expert_ids": pa.array([r.expert_ids for r in rows], type=pa.list_(pa.int32())),
            "expert_weights": pa.array([r.expert_weights for r in rows], type=pa.list_(pa.float32())),
            "token_text": pa.array([r.token_text for r in rows], type=pa.string()),
            "timestamp": pa.array([r.timestamp_us for r in rows], type=pa.int64()),
            "token_category": pa.array([r.token_category for r in rows], type=pa.string()),
            "expert_scores_all": pa.array(
                [r.expert_scores_all for r in rows],
                type=pa.list_(pa.float16()),
            ),
        }

        table = pa.table(data)
        if writer is None:
            writer = pq.ParquetWriter(self.output_path, table.schema, compression="zstd")
        writer.write_table(table)
        return writer
