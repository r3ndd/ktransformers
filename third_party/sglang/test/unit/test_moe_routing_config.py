import unittest

from sglang.srt.managers.moe_routing_config import (
    MoeRoutingConfig,
    build_moe_routing_signature,
    parse_moe_routing_config,
)


class TestMoeRoutingConfig(unittest.TestCase):
    def test_parse_valid(self):
        cfg, warn = parse_moe_routing_config(
            {
                "moe_routing": {
                    "prefill": {"scheme": "prefill_block_mean", "params": {"window_size": 64}},
                    "decode": {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.3}},
                    "scope": "request",
                }
            }
        )
        self.assertIsNone(warn)
        self.assertIsInstance(cfg, MoeRoutingConfig)
        self.assertEqual(cfg.prefill.scheme, "prefill_block_mean")
        self.assertEqual(cfg.prefill.params["window_size"], 64)
        self.assertEqual(cfg.decode.scheme, "ema_score_averaging")
        self.assertAlmostEqual(cfg.decode.params["ema_beta"], 0.3)

    def test_parse_invalid_fallback(self):
        cfg, warn = parse_moe_routing_config(
            {
                "moe_routing": {
                    "prefill": {"scheme": "bad_prefill", "params": {}},
                    "decode": {"scheme": "ema_score_averaging", "params": {"ema_beta": 0.3}},
                    "scope": "request",
                }
            }
        )
        self.assertIsNone(cfg)
        self.assertIsNotNone(warn)

    def test_parse_absent(self):
        cfg, warn = parse_moe_routing_config({"foo": 1})
        self.assertIsNone(cfg)
        self.assertIsNone(warn)

    def test_signature_stable(self):
        cfg, _ = parse_moe_routing_config(
            {
                "moe_routing": {
                    "prefill": {"scheme": "prefill_full_mean", "params": {}},
                    "decode": {"scheme": "sliding_window_score_averaging", "params": {"window_size": 4}},
                    "scope": "request",
                }
            }
        )
        s1 = build_moe_routing_signature(cfg)
        s2 = build_moe_routing_signature(cfg)
        self.assertEqual(s1, s2)
        self.assertTrue(s1.startswith("moe-routing:"))


if __name__ == "__main__":
    unittest.main()
