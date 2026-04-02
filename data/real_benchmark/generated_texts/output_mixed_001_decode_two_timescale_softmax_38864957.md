## prompt_id: mixed_001 | experiment: decode_two_timescale_softmax_38864957 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 64.0}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s a complete implementation of a Binary Search Tree (BST) in Python with `insert`, `search`, and `delete` operations, plus explanations of time complexity and walkthroughs using a small example.

## 1. Implementation Code

```python
class BSTNode:
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    # Insert operation
    def insert(self, value):
        if self.root is None:
            self.root = BSTNode(value)
            return
        self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if node.data is None:
            node.data = value
            return
        if value < node.data:
            if node.left is None:
                node.left = BSTNode(value)
            else:
                self._insert_recursive(node.left, value)
        elif value > node.data:
            if node.right is None:
                node.right = BSTNode(value)
            else:
                self._insert_recursive(node.right, value)
