## prompt_id: mixed_001 | experiment: decode_two_timescale_softmax_cc6c43c8 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 4.0}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s a complete implementation of a Binary Search Tree (BST) in Python with `insert`, `search`, and `delete` operations, along with explanations of time complexity and step-by-step walkthroughs using a small example.

### 1. BST Definition & Properties
- A **Binary Search Tree** is a tree where each node has a left child and right child.
- For every node:
  - All keys in the left subtree are less than the key at node.
  - All keys in the right subtree are greater than the key at node.
- This property allows efficient search, insert, and delete operations.

### 2. Implementation Code

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    # Insert operation
    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
