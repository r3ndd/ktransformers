## prompt_id: mixed_001 | experiment: decode_two_timescale_softmax_f375d90f | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 16.0}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s a complete implementation of binary search tree (BST) in Python with `insert`, `search`, and `delete` operations, plus explanations of time complexity and walkthroughs using a small example.

## 1. Implementation Code

```python
class Node:
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        # Insert the value into root or recursively
        if self.root is None:
            self.root = Node(value)
        else:
            _insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.data:
            if node.left is None:
                node.left = Node(value)
            else:
                _insert_recursive(node.left, value)
        elif value > node.data:
            if node.right is None:
                node.right = Node(value)
            else:
                _insert_recursive(node.right, value)
        else:
            # duplicate values not inserted (optional)
            pass

    def search(self
