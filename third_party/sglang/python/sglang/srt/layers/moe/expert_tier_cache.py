from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ExpertTierCacheStats:
    gpu_hits: int = 0
    cpu_hits: int = 0
    ssd_loads: int = 0
    promotions_to_gpu: int = 0
    promotions_to_cpu: int = 0
    demotions_from_gpu: int = 0
    demotions_from_cpu: int = 0


class _LRUSet:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self._d: OrderedDict[int, None] = OrderedDict()

    def __contains__(self, item: int) -> bool:
        return item in self._d

    def touch(self, item: int) -> None:
        if item in self._d:
            self._d.move_to_end(item)
            return
        self._d[item] = None

    def pop_lru(self) -> Optional[int]:
        if not self._d:
            return None
        k, _ = self._d.popitem(last=False)
        return k

    def ensure_capacity(self) -> List[int]:
        evicted: List[int] = []
        while self.capacity >= 0 and len(self._d) > self.capacity:
            v = self.pop_lru()
            if v is None:
                break
            evicted.append(v)
        return evicted


class _LayerTierState:
    def __init__(self, gpu_capacity: int, cpu_capacity: int):
        self.gpu = _LRUSet(gpu_capacity)
        self.cpu = _LRUSet(cpu_capacity)


class ExpertTierResidencyManager:
    """GPU/CPU/SSD expert residency manager for MoE expert ids.

    Tracks logical expert ids only; actual weight movement is managed by existing KT paths.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        *,
        cache_mode: str = "layerwise",
        gpu_capacity: int = 0,
        cpu_capacity: int = 0,
    ):
        if cache_mode not in {"global", "layerwise"}:
            raise ValueError("cache_mode must be 'global' or 'layerwise'")
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.cache_mode = cache_mode
        self.stats = ExpertTierCacheStats()
        if cache_mode == "global":
            self._global = _LayerTierState(gpu_capacity=gpu_capacity, cpu_capacity=cpu_capacity)
            self._layers = None
        else:
            self._global = None
            self._layers = [_LayerTierState(gpu_capacity=gpu_capacity, cpu_capacity=cpu_capacity) for _ in range(self.num_layers)]

    @staticmethod
    def validate_budget_or_raise(
        *,
        gpu_total_bytes: int,
        core_weights_bytes: int,
        kv_reservation_bytes: int,
    ) -> None:
        if core_weights_bytes + kv_reservation_bytes > gpu_total_bytes:
            raise ValueError(
                "Insufficient GPU memory budget: core weights + KV reservation exceed GPU capacity"
            )

    def _get_layer(self, layer_id: int) -> _LayerTierState:
        if self.cache_mode == "global":
            assert self._global is not None
            return self._global
        assert self._layers is not None
        return self._layers[layer_id]

    def _ensure_expert(self, layer: _LayerTierState, expert_id: int) -> None:
        if expert_id in layer.gpu:
            self.stats.gpu_hits += 1
            layer.gpu.touch(expert_id)
            return

        if expert_id in layer.cpu:
            self.stats.cpu_hits += 1
            layer.cpu.touch(expert_id)
            layer.gpu.touch(expert_id)
            self.stats.promotions_to_gpu += 1
        else:
            self.stats.ssd_loads += 1
            layer.cpu.touch(expert_id)
            layer.gpu.touch(expert_id)
            self.stats.promotions_to_cpu += 1
            self.stats.promotions_to_gpu += 1

        for ev in layer.gpu.ensure_capacity():
            self.stats.demotions_from_gpu += 1
            layer.cpu.touch(ev)

        for ev in layer.cpu.ensure_capacity():
            self.stats.demotions_from_cpu += 1

    def observe_experts(self, layer_id: int, expert_ids: Iterable[int]) -> None:
        layer = self._get_layer(layer_id)
        for eid in expert_ids:
            if eid < 0 or eid >= self.num_experts:
                continue
            self._ensure_expert(layer, int(eid))
