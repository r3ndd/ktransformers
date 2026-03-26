import os
import sys

import pandas as pd

# Add python directory to path for local test execution/import resolution
_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "python")
sys.path.insert(0, _python_dir)

from moe_routing.token_indexing import add_absolute_token_position


def test_add_absolute_token_position_handles_singleton_then_batched_pattern():
    # Pattern mirrors real traces:
    # - Two singleton microbatches (bs=1)
    # - One batched microbatch (bs=3)
    rows = [
        # microbatch 0 (bs=1)
        {"token_id": 0, "context_id": "ctx", "layer_id": 0, "token_position": 0},
        {"token_id": 1, "context_id": "ctx", "layer_id": 1, "token_position": 0},
        # microbatch 1 (bs=1)
        {"token_id": 2, "context_id": "ctx", "layer_id": 0, "token_position": 0},
        {"token_id": 3, "context_id": "ctx", "layer_id": 1, "token_position": 0},
        # microbatch 2 (bs=3)
        {"token_id": 4, "context_id": "ctx", "layer_id": 0, "token_position": 0},
        {"token_id": 5, "context_id": "ctx", "layer_id": 0, "token_position": 1},
        {"token_id": 6, "context_id": "ctx", "layer_id": 0, "token_position": 2},
        {"token_id": 7, "context_id": "ctx", "layer_id": 1, "token_position": 0},
        {"token_id": 8, "context_id": "ctx", "layer_id": 1, "token_position": 1},
        {"token_id": 9, "context_id": "ctx", "layer_id": 1, "token_position": 2},
    ]
    df = pd.DataFrame(rows)

    out = add_absolute_token_position(df)

    # Expected absolute positions by row: [0,0,1,1,2,3,4,2,3,4]
    assert out["absolute_token_position"].tolist() == [0, 0, 1, 1, 2, 3, 4, 2, 3, 4]


def test_add_absolute_token_position_is_per_context():
    df = pd.DataFrame(
        {
            "token_id": [0, 1, 0, 1],
            "context_id": ["a", "a", "b", "b"],
            "layer_id": [0, 1, 0, 1],
            "token_position": [0, 0, 0, 0],
        }
    )

    out = add_absolute_token_position(df)
    ctx_a = out[out["context_id"] == "a"]["absolute_token_position"].unique().tolist()
    ctx_b = out[out["context_id"] == "b"]["absolute_token_position"].unique().tolist()

    assert ctx_a == [0]
    assert ctx_b == [0]
