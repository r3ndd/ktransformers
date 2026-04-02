## prompt_id: mixed_001 | experiment: decode_ema_score_averaging_bf390907 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.9}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s a complete implementation of a Binary Search Tree (BST) in Python with `insert`, `search`, and `delete` operations, along with explanations of time complexity and step-by-step walkthroughs.

---

## BST Implementation in Python

```python
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    # Insert a key into the BST
    def insert(self, key):
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if node is None:
            return TreeNode(key)
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        # If key == node.key, we do nothing (no duplicates)
        return node

    # Search for a key in the BST
    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node,
