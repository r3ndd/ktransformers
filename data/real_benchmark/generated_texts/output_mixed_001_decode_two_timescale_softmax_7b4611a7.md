## prompt_id: mixed_001 | experiment: decode_two_timescale_softmax_7b4611a7 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 1024.0}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s a implementation of a binary search tree (BST) in Python with `insert`, `search`, and `delete` operations, plus explanations of time complexity and step-by-step walkthroughs using a small example.

## BST Definition
A Binary Search Tree is ordered by:
- each node has a left child smaller than the parent, right child larger than the parent.
- no duplicate keys allowed (or we can handle duplicates if you want).

## Implementation
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
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert_recursive(self.root, key)

    def _insert_recursive(self, node, key):
        if key < node.key:
            if node.left is None:
                node.left = Node(key)
            else:
                self._insert_recursive(node.left, key)
        elif key > node.key:
            if node.right
