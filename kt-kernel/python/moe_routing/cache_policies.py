from __future__ import annotations

from collections import Counter, deque


class BasePolicy:
    def observe(self, experts: list[int]) -> None:
        raise NotImplementedError

    def cached(self) -> set[int]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class BaselinePolicy(BasePolicy):
    def observe(self, experts: list[int]) -> None:
        return

    def cached(self) -> set[int]:
        return set()


class SlidingWindowPolicy(BasePolicy):
    def __init__(self, capacity: int, window_size: int):
        self.capacity = capacity
        self.window = deque(maxlen=window_size)

    def observe(self, experts: list[int]) -> None:
        self.window.append(set(experts))

    def cached(self) -> set[int]:
        merged: set[int] = set()
        for s in self.window:
            merged.update(s)
        if len(merged) <= self.capacity:
            return merged
        return set(list(merged)[: self.capacity])


class FixedHotPolicy(BasePolicy):
    def __init__(self, hot_experts: list[int]):
        self._hot = set(hot_experts)

    def observe(self, experts: list[int]) -> None:
        return

    def cached(self) -> set[int]:
        return set(self._hot)


def build_hotset(freq: Counter[int], pool_size: int) -> list[int]:
    return [x for x, _ in freq.most_common(pool_size)]
