from __future__ import annotations

from collections import Counter, OrderedDict, deque
from typing import TypeAlias, Union

# Type alias for layer-qualified expert: (layer_id, expert_id)
ExpertKey = tuple[int, int]

# Additional type aliases for backward compatibility and clarity
ExpertId: TypeAlias = int
QualifiedExpertId: TypeAlias = tuple[int, int]  # (layer_id, expert_id) - same as ExpertKey
ExpertSet: TypeAlias = Union[set[ExpertId], set[QualifiedExpertId]]
ExpertList: TypeAlias = Union[list[ExpertId], list[QualifiedExpertId]]


class BasePolicy:
    """Base class for cache policies.

    Supports both simple expert IDs (int) for backward compatibility
    and qualified expert IDs (layer_id, expert_id) for the simulation path.
    """

    def observe(self, experts: list[ExpertKey]) -> None:
        """Observe a set of experts accessed in a token processing step."""
        raise NotImplementedError

    def cached(self) -> set[ExpertKey]:
        """Return the current set of cached experts."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the policy state."""
        pass


class BaselinePolicy(BasePolicy):
    """Baseline policy that caches nothing."""

    def observe(self, experts: list[ExpertKey]) -> None:
        return

    def cached(self) -> set[ExpertKey]:
        return set()


class SlidingWindowPolicy(BasePolicy):
    """Sliding window policy that keeps experts from recent windows.

    For qualified identities (layer_id, expert_id), maintains a merged
    view of all experts seen in the window, respecting capacity limits.
    """

    def __init__(self, capacity: int, window_size: int):
        self.capacity = capacity
        self.window_size = window_size
        self.window: deque[set[ExpertKey]] = deque(maxlen=window_size)

    def observe(self, experts: list[ExpertKey]) -> None:
        self.window.append(set(experts))

    def cached(self) -> set[ExpertKey]:
        merged: set[ExpertKey] = set()
        for s in self.window:
            merged.update(s)
        if len(merged) <= self.capacity:
            return merged
        # Return first 'capacity' experts deterministically (sorted)
        return set(sorted(merged)[: self.capacity])

    def reset(self) -> None:
        self.window.clear()


class FixedHotPolicy(BasePolicy):
    """Fixed hot policy that caches a predetermined set of experts.

    Works with qualified (layer_id, expert_id) tuples.
    """

    def __init__(self, hot_experts: list[ExpertKey]):
        self._hot = set(hot_experts)

    def observe(self, experts: list[ExpertKey]) -> None:
        return

    def cached(self) -> set[ExpertKey]:
        return set(self._hot)

    def reset(self) -> None:
        pass


def build_hotset(freq: Counter[ExpertKey], pool_size: int) -> list[ExpertKey]:
    """Build a hotset of experts from frequency counts.

    Works with qualified (layer_id, expert_id) tuples.

    Args:
        freq: Counter mapping expert keys to frequencies
        pool_size: Maximum number of experts to include in hotset

    Returns:
        List of expert keys sorted by frequency (most common first)
    """
    return [x for x, _ in freq.most_common(pool_size)]


class LayerAwareSlidingWindowPolicy(BasePolicy):
    """Layer-aware sliding window that maintains separate windows per layer.

    This policy treats (layer_id, expert_id) as the unit of caching
    and maintains separate sliding windows for each layer, then combines
    results into qualified identities.
    """

    def __init__(self, capacity: int, window_size: int, num_layers: int):
        self.capacity = capacity
        self.window_size = window_size
        self.num_layers = num_layers
        self.layer_windows: dict[int, deque[set[int]]] = {
            layer_id: deque(maxlen=window_size) for layer_id in range(num_layers)
        }

    def observe(self, experts: list[ExpertKey]) -> None:
        """Observe qualified experts grouped by layer.

        For qualified identities [(layer_id, expert_id), ...],
        groups by layer and updates per-layer windows.
        """
        if not experts:
            return

        # Group by layer
        by_layer: dict[int, set[int]] = {i: set() for i in range(self.num_layers)}
        for layer_id, expert_id in experts:
            by_layer[layer_id].add(expert_id)

        # Update each layer's window
        for layer_id, expert_set in by_layer.items():
            self.layer_windows[layer_id].append(expert_set)

    def cached(self) -> set[ExpertKey]:
        """Return cached experts across all layers.

        Returns qualified identities (layer_id, expert_id) tuples.
        """
        qualified: set[ExpertKey] = set()

        for layer_id, window in self.layer_windows.items():
            layer_experts: set[int] = set()
            for s in window:
                layer_experts.update(s)

            # Add qualified identities for this layer
            for expert_id in layer_experts:
                qualified.add((layer_id, expert_id))

        # Apply capacity limit deterministically (sorted by layer, then expert)
        if len(qualified) <= self.capacity:
            return qualified
        return set(sorted(qualified)[: self.capacity])

    def cached_for_layer(self, layer_id: int) -> set[int]:
        """Get cached experts for a specific layer (expert IDs only)."""
        if layer_id not in self.layer_windows:
            return set()

        window = self.layer_windows[layer_id]
        layer_experts: set[int] = set()
        for s in window:
            layer_experts.update(s)
        return layer_experts

    def reset(self) -> None:
        for window in self.layer_windows.values():
            window.clear()


class PerLayerLRUPolicy(BasePolicy):
    """Per-layer LRU cache with fixed expert slots per layer.

    The cache tracks recency independently for each layer and returns
    layer-qualified keys ``(layer_id, expert_id)``.
    """

    def __init__(self, capacity_per_layer: int):
        if capacity_per_layer < 0:
            raise ValueError(f"capacity_per_layer must be >= 0; got {capacity_per_layer}")
        self.capacity_per_layer = capacity_per_layer
        self._layers: dict[int, OrderedDict[int, None]] = {}

    def observe(self, experts: list[ExpertKey]) -> None:
        if self.capacity_per_layer == 0:
            return

        for layer_id, expert_id in sorted(experts):
            lru = self._layers.setdefault(layer_id, OrderedDict())
            if expert_id in lru:
                lru.move_to_end(expert_id)
            else:
                lru[expert_id] = None
            while len(lru) > self.capacity_per_layer:
                lru.popitem(last=False)

    def cached(self) -> set[ExpertKey]:
        out: set[ExpertKey] = set()
        for layer_id, lru in self._layers.items():
            for expert_id in lru.keys():
                out.add((layer_id, expert_id))
        return out

    def cached_for_layer(self, layer_id: int) -> set[int]:
        lru = self._layers.get(layer_id)
        return set(lru.keys()) if lru is not None else set()

    def reset(self) -> None:
        self._layers.clear()
