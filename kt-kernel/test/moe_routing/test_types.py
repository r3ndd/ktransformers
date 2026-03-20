import sys
import os

# Add python directory to path for development testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from moe_routing.types import RoutingRecord, TOP_K


def test_routing_record_valid():
    rec = RoutingRecord(
        token_id=1,
        context_id="ctx-1",
        layer_id=0,
        token_position=12,
        expert_ids=[1, 2, 3, 4, 5, 6],
        expert_weights=[0.2, 0.2, 0.2, 0.15, 0.15, 0.1],
        token_text="hello",
        timestamp_us=123,
        token_category="user",
    )
    assert rec.layer_id == 0
    assert len(rec.expert_ids) == TOP_K


def test_routing_record_rejects_wrong_topk():
    try:
        RoutingRecord(
            token_id=1,
            context_id="ctx-1",
            layer_id=0,
            token_position=12,
            expert_ids=[1, 2],
            expert_weights=[0.5, 0.5],
            timestamp_us=123,
            token_category="assistant",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "top-k" in str(exc)
