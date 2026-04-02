## prompt_id: mixed_001 | experiment: decode_two_timescale_ema_601fe0ea | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_ema
decode_params: {"mix_lambda": 0.4}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s is a implementation of a binary search tree (BST) in Python with `insert`, `search`, and `delete` operations, followed explanations for each operation, including time complexity and step-by-step walkthrough using a small example.

```
code:
```python
class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    # Insert operation
    def insert(self, key):
        """Insert a key into the BST"""
        if self.root is None:
            self.root = BSTNode(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node, key):
        """Helper recursive insertion"""
        if key < node.key:
            if node.left is None:
                node.left = BSTNode(key)
        else:
            self._insert(node.left, key)
        elif key > node.key:
            if node.right is None:
                node.right = BSTNode(key)
        else:
            self._insert(node.right, key
