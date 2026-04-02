## prompt_id: mixed_001 | experiment: decode_ema_score_averaging_b5aee75c | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.3}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s is a complete implementation of a Binary Search Tree (BST) in Python with `insert`, `search`, and `delete` operations, along with explanations of time complexity and step-by-step walkthroughs.



```python
class Node:
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
    if self.root is None:
    self.root = Node(key)
    return
    self._insert(self.root, key)

    def _insert(self, node, key):
    if key < node.key:
    if node.left is None:
    node.left = Node(key)
    else:
    self._insert(node.left, key)
    elif key > node.key:
    if node.right is None:
    node.right = Node(key)
    else:
    self._insert(node.right, key)
    # else: key == node.key → do nothing (no duplicates allowed)

    # Search
