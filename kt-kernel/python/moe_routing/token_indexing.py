from __future__ import annotations

import numpy as np
import pandas as pd


def add_absolute_token_position(traces: pd.DataFrame) -> pd.DataFrame:
    """Add per-context absolute token positions derived from trace ordering.

    The trace hook emits batch-local ``token_position`` values. This helper derives
    a monotonic, per-context ``absolute_token_position`` that is robust to changing
    microbatch sizes over time.

    Derivation strategy:
    - Sort rows by ``context_id`` and ``token_id`` when available (fallback to
      original row order otherwise).
    - Split each context stream into microbatches whenever ``layer_id`` decreases.
    - Within each microbatch, map distinct ``token_position`` values to dense
      offsets [0..B-1] in sorted order.
    - Accumulate offsets across microbatches to produce absolute positions.
    """

    required_columns = {"context_id", "layer_id", "token_position"}
    missing = sorted(required_columns - set(traces.columns))
    if missing:
        raise ValueError(
            "add_absolute_token_position requires columns " f"{sorted(required_columns)}; missing {missing}"
        )

    working = traces.copy()
    working["_orig_pos"] = np.arange(len(working), dtype=np.int64)

    sort_cols = ["context_id", "token_id"] if "token_id" in working.columns else ["context_id", "_orig_pos"]
    working = working.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    absolute_positions = np.empty(len(working), dtype=np.int64)

    for _, ctx in working.groupby("context_id", sort=False):
        layer_ids = ctx["layer_id"].to_numpy()
        token_positions = ctx["token_position"].to_numpy()
        orig_positions = ctx["_orig_pos"].to_numpy()

        if len(ctx) == 0:
            continue

        breaks = [0]
        for i in range(1, len(layer_ids)):
            if int(layer_ids[i]) < int(layer_ids[i - 1]):
                breaks.append(i)
        breaks.append(len(layer_ids))

        absolute_for_ctx = np.empty(len(ctx), dtype=np.int64)
        cursor = 0

        for start, end in zip(breaks[:-1], breaks[1:]):
            microbatch_positions = token_positions[start:end]
            distinct_positions = sorted({int(p) for p in microbatch_positions})
            pos_to_offset = {pos: offset for offset, pos in enumerate(distinct_positions)}

            for i in range(start, end):
                absolute_for_ctx[i] = cursor + pos_to_offset[int(token_positions[i])]

            cursor += len(distinct_positions)

        absolute_positions[orig_positions] = absolute_for_ctx

    out = traces.copy()
    out["absolute_token_position"] = absolute_positions
    return out
