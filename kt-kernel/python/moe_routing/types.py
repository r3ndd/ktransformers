from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

TOP_K = 6


@dataclass(slots=True)
class RoutingRecord:
    token_id: int
    context_id: str
    layer_id: int
    token_position: int
    expert_ids: list[int]
    expert_weights: list[float]
    timestamp_us: int
    token_category: str
    token_text: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.expert_ids) != TOP_K or len(self.expert_weights) != TOP_K:
            raise ValueError("top-k expert_ids/expert_weights must each be length 6")
        if self.layer_id < 0:
            raise ValueError("layer_id must be >= 0")
