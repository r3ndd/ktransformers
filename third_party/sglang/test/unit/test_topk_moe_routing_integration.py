import unittest

import torch

from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.moe.moe_routing_runtime import set_current_forward_batch
from sglang.srt.managers.moe_routing_config import parse_moe_routing_config
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _Req:
    def __init__(self):
        class _S:
            custom_params = {}

        self.sampling_params = _S()


class _Batch:
    def __init__(self, cfg):
        self.forward_mode = ForwardMode.DECODE
        self.moe_routing_configs = [cfg]
        self.rids = ["r1"]
        self.request_custom_params = [{"__req__": _Req()}]


class TestTopkMoeRoutingIntegration(unittest.TestCase):
    def test_select_experts_runs_with_runtime_transform(self):
        cfg, _ = parse_moe_routing_config(
            {
                "moe_routing": {
                    "prefill": {"scheme": "prefill_block_mean", "params": {"window_size": 64}},
                    "decode": {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.5}},
                    "scope": "request",
                }
            }
        )
        batch = _Batch(cfg)
        hidden_states = torch.randn(1, 8, dtype=torch.float32)
        router_logits = torch.tensor([[1.0, 0.2, 0.1, 0.0]], dtype=torch.float32)
        topk_cfg = TopKConfig(top_k=2)
        topk_cfg.torch_native = True

        with set_current_forward_batch(batch):
            out = select_experts(hidden_states, router_logits, topk_cfg, layer_id=0)
        self.assertEqual(tuple(out.topk_ids.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
