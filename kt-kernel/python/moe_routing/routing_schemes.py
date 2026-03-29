from __future__ import annotations

from collections import defaultdict, deque
import math


class BaseRoutingScheme:
    def smooth_scores(self, layer_id: int, current_scores: list[float]) -> list[float]:
        raise NotImplementedError

    def observe(self, layer_id: int, current_scores: list[float]) -> None:
        raise NotImplementedError

    def end_token(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class PrefillBlockMeanRouting(BaseRoutingScheme):
    """Prefill block mean routing with per-layer sliding history."""

    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0; got {window_size}")
        self.window_size = int(window_size)
        self._history: dict[int, deque[list[float]]] = defaultdict(lambda: deque(maxlen=max(self.window_size - 1, 1)))

    def smooth_scores(self, layer_id: int, current_scores: list[float]) -> list[float]:
        prev = self._history[layer_id]
        if not prev:
            return list(current_scores)
        n = len(current_scores)
        out = [0.0] * n
        denom = float(len(prev) + 1)
        for i in range(n):
            s = float(current_scores[i])
            for vec in prev:
                s += float(vec[i])
            out[i] = s / denom
        return out

    def observe(self, layer_id: int, current_scores: list[float]) -> None:
        self._history[layer_id].append(list(current_scores))

    def end_token(self) -> None:
        return

    def reset(self) -> None:
        self._history.clear()


class PrefillFullMeanRouting(BaseRoutingScheme):
    """Prefill full-span running mean routing per layer."""

    def __init__(self):
        self._sum: dict[int, list[float]] = {}
        self._count: dict[int, int] = {}

    def smooth_scores(self, layer_id: int, current_scores: list[float]) -> list[float]:
        if layer_id not in self._sum:
            return list(current_scores)
        s = self._sum[layer_id]
        c = max(self._count.get(layer_id, 1), 1)
        return [float(v) / float(c) for v in s]

    def observe(self, layer_id: int, current_scores: list[float]) -> None:
        if layer_id not in self._sum:
            self._sum[layer_id] = [float(v) for v in current_scores]
            self._count[layer_id] = 1
            return
        cur = self._sum[layer_id]
        for i, v in enumerate(current_scores):
            cur[i] += float(v)
        self._count[layer_id] = self._count.get(layer_id, 0) + 1

    def end_token(self) -> None:
        return

    def reset(self) -> None:
        self._sum.clear()
        self._count.clear()


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


class EMAScoreAveragingRouting(BaseRoutingScheme):
    """Per-layer EMA score smoothing.

    For each token/layer:
    s'_t = ema_beta * s_t + (1 - ema_beta) * s'_{t-1}
    """

    def __init__(self, ema_beta: float):
        if ema_beta <= 0.0 or ema_beta > 1.0:
            raise ValueError(f"ema_beta must be in (0, 1]; got {ema_beta}")
        self.ema_beta = float(ema_beta)
        self._ema: dict[int, list[float]] = {}

    def smooth_scores(self, layer_id: int, current_scores: list[float]) -> list[float]:
        prev = self._ema.get(layer_id)
        if prev is None:
            return list(current_scores)
        a = self.ema_beta
        b = 1.0 - a
        return [a * float(cur) + b * float(prev_i) for cur, prev_i in zip(current_scores, prev)]

    def observe(self, layer_id: int, current_scores: list[float]) -> None:
        prev = self._ema.get(layer_id)
        if prev is None:
            self._ema[layer_id] = list(current_scores)
            return
        a = self.ema_beta
        b = 1.0 - a
        self._ema[layer_id] = [a * float(cur) + b * float(prev_i) for cur, prev_i in zip(current_scores, prev)]

    def end_token(self) -> None:
        return

    def reset(self) -> None:
        self._ema.clear()


class TwoTimescaleEMARouting(BaseRoutingScheme):
    """Per-layer two-timescale EMA score smoothing.

    Uses fixed short/long EMA betas and mixes them with ``mix_lambda``:
    s'_t = mix_lambda * short_t + (1 - mix_lambda) * long_t
    where short_t is EMA(beta=0.5) and long_t is EMA(beta=0.05).
    """

    SHORT_BETA = 0.5
    LONG_BETA = 0.05

    def __init__(self, mix_lambda: float):
        if mix_lambda <= 0.0 or mix_lambda >= 1.0:
            raise ValueError(f"mix_lambda must be in (0, 1); got {mix_lambda}")
        self.mix_lambda = float(mix_lambda)
        self._short_ema: dict[int, list[float]] = {}
        self._long_ema: dict[int, list[float]] = {}

    def smooth_scores(self, layer_id: int, current_scores: list[float]) -> list[float]:
        short_prev = self._short_ema.get(layer_id)
        long_prev = self._long_ema.get(layer_id)

        if short_prev is None or long_prev is None:
            return list(current_scores)

        short_beta = self.SHORT_BETA
        long_beta = self.LONG_BETA
        short_t = [
            short_beta * float(cur) + (1.0 - short_beta) * float(prev_i)
            for cur, prev_i in zip(current_scores, short_prev)
        ]
        long_t = [
            long_beta * float(cur) + (1.0 - long_beta) * float(prev_i) for cur, prev_i in zip(current_scores, long_prev)
        ]
        lam = self.mix_lambda
        return [lam * s + (1.0 - lam) * l for s, l in zip(short_t, long_t)]

    def observe(self, layer_id: int, current_scores: list[float]) -> None:
        short_prev = self._short_ema.get(layer_id)
        long_prev = self._long_ema.get(layer_id)

        if short_prev is None or long_prev is None:
            init = list(current_scores)
            self._short_ema[layer_id] = list(init)
            self._long_ema[layer_id] = list(init)
            return

        short_beta = self.SHORT_BETA
        long_beta = self.LONG_BETA
        self._short_ema[layer_id] = [
            short_beta * float(cur) + (1.0 - short_beta) * float(prev_i)
            for cur, prev_i in zip(current_scores, short_prev)
        ]
        self._long_ema[layer_id] = [
            long_beta * float(cur) + (1.0 - long_beta) * float(prev_i) for cur, prev_i in zip(current_scores, long_prev)
        ]

    def end_token(self) -> None:
        return

    def reset(self) -> None:
        self._short_ema.clear()
        self._long_ema.clear()


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(float(s) - float(m)) for s in scores]
    denom = sum(exps)
    if denom == 0.0:
        return [0.0 for _ in scores]
    return [e / denom for e in exps]


class TwoTimescaleSoftmaxRouting(BaseRoutingScheme):
    """Scaled-softmax two-timescale EMA routing.

    Let x_t = softmax(rho * current_scores).
    Then:
    short_t = EMA_beta_0.5(x_t)
    long_t = EMA_beta_0.05(x_t)
    s'_t = mix_lambda * short_t + (1-mix_lambda) * long_t

    This is identical to scheme 3 structurally, but with scaled-softmax inputs.
    """

    SHORT_BETA = 0.5
    LONG_BETA = 0.05

    def __init__(self, mix_lambda: float = 0.2, rho: float = 1.0):
        if mix_lambda <= 0.0 or mix_lambda >= 1.0:
            raise ValueError(f"mix_lambda must be in (0, 1); got {mix_lambda}")
        self.mix_lambda = float(mix_lambda)
        self.rho = max(0.0, float(rho))
        self._short_ema: dict[int, list[float]] = {}
        self._long_ema: dict[int, list[float]] = {}

    def smooth_scores(self, layer_id: int, current_scores: list[float]) -> list[float]:
        x_t = _softmax([self.rho * float(s) for s in current_scores])
        short_prev = self._short_ema.get(layer_id)
        long_prev = self._long_ema.get(layer_id)

        if short_prev is None or long_prev is None:
            short_t = list(x_t)
            long_t = list(x_t)
        else:
            short_beta = self.SHORT_BETA
            long_beta = self.LONG_BETA
            short_t = [
                short_beta * float(cur) + (1.0 - short_beta) * float(prev_i) for cur, prev_i in zip(x_t, short_prev)
            ]
            long_t = [long_beta * float(cur) + (1.0 - long_beta) * float(prev_i) for cur, prev_i in zip(x_t, long_prev)]

        lam = self.mix_lambda
        return [lam * s + (1.0 - lam) * l for s, l in zip(short_t, long_t)]

    def observe(self, layer_id: int, current_scores: list[float]) -> None:
        x_t = _softmax([self.rho * float(s) for s in current_scores])
        short_prev = self._short_ema.get(layer_id)
        long_prev = self._long_ema.get(layer_id)

        if short_prev is None or long_prev is None:
            init = list(x_t)
            self._short_ema[layer_id] = list(init)
            self._long_ema[layer_id] = list(init)
            return

        short_beta = self.SHORT_BETA
        long_beta = self.LONG_BETA
        self._short_ema[layer_id] = [
            short_beta * float(cur) + (1.0 - short_beta) * float(prev_i) for cur, prev_i in zip(x_t, short_prev)
        ]
        self._long_ema[layer_id] = [
            long_beta * float(cur) + (1.0 - long_beta) * float(prev_i) for cur, prev_i in zip(x_t, long_prev)
        ]

    def end_token(self) -> None:
        return

    def reset(self) -> None:
        self._short_ema.clear()
        self._long_ema.clear()


# Backward-compatible alias
TwoTimescalePlusCurrentSoftmaxRouting = TwoTimescaleSoftmaxRouting
