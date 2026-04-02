## prompt_id: mixed_001 | experiment: decode_sliding_window_score_averaging_d4dee748 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: sliding_window_score_averaging
decode_params: {"window_size": 4}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s a complete implementation of a Binary Search Tree (BST) in Python with `insert`, `search`, and `delete` operations, along explanations for time complexity and step-by-step walkthroughs.

```python
class Node:
    def __init__(self, key):
    self.key = key
    self.left = None
    self.right = None

class BST:
    def __init__(self):
    self.root = None

    # Insert: O average O(log n), worst O(n)
    def insert(self, key):
    self._root = self._insert(self._root, key)

    def _insert(self, node, key):
    if node is None:
    return Node(key)
    if key < node.key:
    node.left = self._insert(node.left, key)
else:
    node.right = self._insert(node.right, key)
return node

    # Search: O average O(log n), worst O(n)
    def search(self, key):
    return self._search(self._root, key)

    def _search(self, node, key):
    if node is None or node.key != key:
    return
