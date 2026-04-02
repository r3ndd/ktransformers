## prompt_id: mixed_001 | experiment: prefill_prefill_block_mean_81581d58 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s a complete implementation of a binary search tree (BST) in Python with `insert`, `search`, and `delete` operations, along with explanations of time complexity and step-by-step walkthroughs.

---

## BST Implementation in Python

```python
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BinarySearchTree:
    def __init__(self):
        self.root = None

    # ---------- INSERT ----------
    def insert(self, key):
        """Insert a key into the BST."""
        self.root = self._insert_recursive(self.root, key)

    def _insert_recursive(self, node, key):
        if node is None:
            return TreeNode(key)
        if key < node.key:
            node.left = self._insert_recursive(node.left, key)
        elif key > node.key:
            node.right = self._insert_recursive(node.right, key)
        # If key == node.key, we do nothing (no duplicates)
        return node

    # ---------- SEARCH ----------
    def search(self, key):
        """Return True if key exists in the BST,
