## prompt_id: mixed_001 | experiment: decode_ema_score_averaging_ce23db51 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.5}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s a complete implementation of a Binary Search Tree (BST) in Python with `insert`, `search`, and `delete` operations, along with explanations of time complexity and step-by-step walkthroughs using a small example.

---

## 🌲 Binary Search Tree Implementation in Python

```python
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    # Insert operation
    def insert(self, key):
        """Insert a key into the BST."""
        self.root = self._insert(self.root, key)

    def _insert(self, node, key):
        if node is None:
            return TreeNode(key)
        
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        # else: key == node.key → do nothing (no duplicates allowed)
        
        return node

    # Search operation
    def search(self, key):
 """Search a key in the
