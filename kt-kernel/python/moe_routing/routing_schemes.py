from __future__ import annotations

from collections import defaultdict, deque


class BaseRoutingScheme:
    def smooth_scores(self, layer_id: int, current_scores: list[float]) -> list[float]:
        raise NotImplementedError

    def observe(self, layer_id: int, current_scores: list[float]) -> None:
        raise NotImplementedError

    def end_token(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class SlidingWindowScoreAveragingRouting(BaseRoutingScheme):
    """Per-layer sliding-window score smoothing.

    For each token/layer, smoothed scores are the mean of previous ``window_size-1``
    full score vectors and the current score vector. ``window_size=1`` is the
    baseline-equivalent (no smoothing) case.
    """

    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0; got {window_size}")
        self.window_size = window_size
        self._history: dict[int, deque[list[float]]] = defaultdict(lambda: deque(maxlen=self.window_size - 1))

    def smooth_scores(self, layer_id: int, current_scores: list[float]) -> list[float]:
        prev = self._history[layer_id]
        if not prev:
            return list(current_scores)

        n = len(current_scores)
        denom = float(len(prev) + 1)
        out = [0.0] * n

        for i in range(n):
            s = current_scores[i]
            for vec in prev:
                s += vec[i]
            out[i] = s / denom
        return out

    def observe(self, layer_id: int, current_scores: list[float]) -> None:
        self._history[layer_id].append(list(current_scores))

    def end_token(self) -> None:
        return

    def reset(self) -> None:
        self._history.clear()
