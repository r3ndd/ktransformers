from __future__ import annotations

from collections import Counter, defaultdict, deque


class BaseRoutingScheme:
    def get_pool(self, layer_id: int) -> set[int]:
        raise NotImplementedError

    def observe(self, layer_id: int, routed_experts: list[int]) -> None:
        raise NotImplementedError

    def end_token(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class SlidingWindowAdaptivePoolRouting(BaseRoutingScheme):
    """Per-layer sliding-window adaptive expert pool.

    The active pool for a layer is the top ``pool_size_per_layer`` experts by
    frequency over the previous ``window_size`` tokens.
    """

    def __init__(self, window_size: int, pool_size_per_layer: int, update_interval_tokens: int = 1):
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0; got {window_size}")
        if pool_size_per_layer < 0:
            raise ValueError(f"pool_size_per_layer must be >= 0; got {pool_size_per_layer}")
        if update_interval_tokens <= 0:
            raise ValueError(f"update_interval_tokens must be > 0; got {update_interval_tokens}")

        self.window_size = window_size
        self.pool_size_per_layer = pool_size_per_layer
        self.update_interval_tokens = update_interval_tokens

        self._histories: dict[int, deque[set[int]]] = defaultdict(lambda: deque(maxlen=self.window_size))
        self._pools: dict[int, set[int]] = defaultdict(set)
        self._token_counter = 0

    def get_pool(self, layer_id: int) -> set[int]:
        return set(self._pools.get(layer_id, set()))

    def observe(self, layer_id: int, routed_experts: list[int]) -> None:
        self._histories[layer_id].append(set(routed_experts))

    def _recompute_pools(self) -> None:
        if self.pool_size_per_layer == 0:
            self._pools = defaultdict(set)
            return

        out: dict[int, set[int]] = {}
        for layer_id, history in self._histories.items():
            freq: Counter[int] = Counter()
            for token_experts in history:
                for expert_id in token_experts:
                    freq[expert_id] += 1

            ordered = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
            out[layer_id] = {expert_id for expert_id, _ in ordered[: self.pool_size_per_layer]}

        self._pools = defaultdict(set, out)

    def end_token(self) -> None:
        self._token_counter += 1
        if self._token_counter % self.update_interval_tokens == 0:
            self._recompute_pools()

    def reset(self) -> None:
        self._histories.clear()
        self._pools.clear()
        self._token_counter = 0
