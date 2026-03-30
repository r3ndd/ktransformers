import unittest

import torch

from sglang.srt.layers.moe.moe_routing_runtime import MoeRoutingRuntime
from sglang.srt.managers.moe_routing_config import parse_moe_routing_config
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _DummyBatch:
    def __init__(self, mode, cfgs, rids, extend_start_loc=None, extend_seq_lens=None):
        self.forward_mode = mode
        self.moe_routing_configs = cfgs
        self.rids = rids
        self.extend_start_loc = extend_start_loc
        self.extend_seq_lens = extend_seq_lens


class TestMoeRoutingRuntime(unittest.TestCase):
    def test_decode_ema_stateful(self):
        cfg, _ = parse_moe_routing_config(
            {
                "moe_routing": {
                    "prefill": {"scheme": "prefill_block_mean", "params": {"window_size": 64}},
                    "decode": {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.5}},
                    "scope": "request",
                }
            }
        )
        rt = MoeRoutingRuntime()
        b = _DummyBatch(ForwardMode.DECODE, [cfg], ["r1"])
        x1 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        y1 = rt.apply(b, layer_id=0, router_logits=x1)
        self.assertTrue(torch.allclose(x1, y1))

        x2 = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        y2 = rt.apply(b, layer_id=0, router_logits=x2)
        self.assertFalse(torch.allclose(x2, y2))

    def test_prefill_block_mean_segment(self):
        cfg, _ = parse_moe_routing_config(
            {
                "moe_routing": {
                    "prefill": {"scheme": "prefill_block_mean", "params": {"window_size": 2}},
                    "decode": {"scheme": "sliding_window_score_averaging", "params": {"window_size": 1}},
                    "scope": "request",
                }
            }
        )
        rt = MoeRoutingRuntime()
        b = _DummyBatch(
            ForwardMode.EXTEND,
            [cfg],
            ["r1"],
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([2], dtype=torch.int32),
        )
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        y = rt.apply(b, layer_id=0, router_logits=x)
        self.assertTrue(torch.allclose(y[0], x[0]))
        self.assertFalse(torch.allclose(y[1], x[1]))

    def test_prefill_block_mean_window_one_identity(self):
        cfg, _ = parse_moe_routing_config(
            {
                "moe_routing": {
                    "prefill": {"scheme": "prefill_block_mean", "params": {"window_size": 1}},
                    "decode": {"scheme": "sliding_window_score_averaging", "params": {"window_size": 1}},
                    "scope": "request",
                }
            }
        )
        rt = MoeRoutingRuntime()
        b = _DummyBatch(
            ForwardMode.EXTEND,
            [cfg],
            ["r1"],
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([4], dtype=torch.int32),
        )
        x = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.4, 0.6],
                [0.6, 0.4],
            ],
            dtype=torch.float32,
        )
        y = rt.apply(b, layer_id=0, router_logits=x)
        self.assertTrue(torch.allclose(x, y))

    def test_prefill_block_mean_chunked_equivalence(self):
        cfg, _ = parse_moe_routing_config(
            {
                "moe_routing": {
                    "prefill": {"scheme": "prefill_block_mean", "params": {"window_size": 4}},
                    "decode": {"scheme": "sliding_window_score_averaging", "params": {"window_size": 1}},
                    "scope": "request",
                }
            }
        )
        x = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
                [5.0, 6.0, 7.0],
                [6.0, 7.0, 8.0],
            ],
            dtype=torch.float32,
        )

        rt_full = MoeRoutingRuntime()
        b_full = _DummyBatch(
            ForwardMode.EXTEND,
            [cfg],
            ["r1"],
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([x.shape[0]], dtype=torch.int32),
        )
        y_full = rt_full.apply(b_full, layer_id=0, router_logits=x)

        rt_chunked = MoeRoutingRuntime()
        b_chunk_1 = _DummyBatch(
            ForwardMode.EXTEND,
            [cfg],
            ["r1"],
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([2], dtype=torch.int32),
        )
        b_chunk_2 = _DummyBatch(
            ForwardMode.EXTEND,
            [cfg],
            ["r1"],
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_seq_lens=torch.tensor([4], dtype=torch.int32),
        )
        y_chunk_1 = rt_chunked.apply(b_chunk_1, layer_id=0, router_logits=x[:2])
        y_chunk_2 = rt_chunked.apply(b_chunk_2, layer_id=0, router_logits=x[2:])
        y_chunked = torch.cat([y_chunk_1, y_chunk_2], dim=0)

        self.assertTrue(torch.allclose(y_full, y_chunked, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
