from __future__ import annotations

import time
from pathlib import Path

from .parquet_writer import AsyncParquetWriter
from .types import RoutingRecord


class RoutingTraceCollector:
    def __init__(self, output_dir: Path, prompt_id: str, token_category: str = "assistant"):
        self.output_dir = output_dir
        self.prompt_id = prompt_id
        self.token_category = token_category
        self.context_id = ""
        self._writer: AsyncParquetWriter | None = None
        self._token_id = 0
        self._t0 = 0

    def start(self, context_id: str, output_path: Path | None = None) -> None:
        self.context_id = context_id
        if output_path is None:
            ts = int(time.time())
            out = self.output_dir / f"{ts}_{self.prompt_id}.parquet"
        else:
            out = output_path
        self._writer = AsyncParquetWriter(out)
        self._writer.start()
        self._t0 = time.perf_counter_ns()

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def record(
        self,
        layer_id: int,
        token_position: int,
        expert_ids: list[int],
        expert_weights: list[float],
        expert_scores_all: list[float] | None = None,
        token_text: str | None = None,
    ) -> None:
        if self._writer is None:
            return
        ts_us = (time.perf_counter_ns() - self._t0) // 1000
        self._writer.submit(
            RoutingRecord(
                token_id=self._token_id,
                context_id=self.context_id,
                layer_id=layer_id,
                token_position=token_position,
                expert_ids=expert_ids,
                expert_weights=expert_weights,
                timestamp_us=ts_us,
                token_category=self.token_category,
                expert_scores_all=expert_scores_all,
                token_text=token_text,
            )
        )
        self._token_id += 1
