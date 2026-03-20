import sys
import os

# Add python directory to path for development testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import pandas as pd

from moe_routing.metrics import temporal_reuse_curve, expert_entropy_by_layer


def test_temporal_reuse_curve_returns_probabilities():
    df = pd.DataFrame(
        {
            "layer_id": [0, 0, 0],
            "token_position": [0, 1, 2],
            "expert_ids": [[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 10, 11], [2, 7, 12, 13, 14, 15]],
        }
    )
    out = temporal_reuse_curve(df, max_distance=2)
    assert 1 in out and 2 in out
    assert 0.0 <= out[1] <= 1.0


def test_entropy_by_layer_non_negative():
    df = pd.DataFrame({"layer_id": [0, 0], "expert_ids": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]})
    ent = expert_entropy_by_layer(df)
    assert ent[0] >= 0.0
