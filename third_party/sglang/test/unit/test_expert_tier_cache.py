import unittest

from sglang.srt.layers.moe.expert_tier_cache import ExpertTierResidencyManager


class TestExpertTierResidencyManager(unittest.TestCase):
    def test_layerwise_counts_hits_and_loads(self):
        m = ExpertTierResidencyManager(
            num_layers=2,
            num_experts=8,
            cache_mode="layerwise",
            gpu_capacity=1,
            cpu_capacity=2,
        )
        m.observe_experts(0, [1])
        self.assertEqual(m.stats.ssd_loads, 1)
        self.assertEqual(m.stats.gpu_hits, 0)

        m.observe_experts(0, [1])
        self.assertEqual(m.stats.gpu_hits, 1)

        m.observe_experts(0, [2])
        self.assertGreaterEqual(m.stats.demotions_from_gpu, 1)

    def test_global_mode_shared(self):
        m = ExpertTierResidencyManager(
            num_layers=4,
            num_experts=16,
            cache_mode="global",
            gpu_capacity=2,
            cpu_capacity=4,
        )
        m.observe_experts(0, [3])
        self.assertEqual(m.stats.ssd_loads, 1)
        m.observe_experts(2, [3])
        self.assertGreaterEqual(m.stats.gpu_hits + m.stats.cpu_hits, 1)

    def test_budget_validation(self):
        with self.assertRaises(ValueError):
            ExpertTierResidencyManager.validate_budget_or_raise(
                gpu_total_bytes=10,
                core_weights_bytes=8,
                kv_reservation_bytes=4,
            )


if __name__ == "__main__":
    unittest.main()
