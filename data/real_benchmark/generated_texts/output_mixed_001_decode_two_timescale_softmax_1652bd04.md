## prompt_id: mixed_001 | experiment: decode_two_timescale_softmax_1652bd04 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 1.0}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s is a implementation of a binary search tree (BST) in Python, with `insert`, `search`, and `delete` operations. I’ll also explain the time complexity for each operation and walk through a small example.

---

## 1. BST Implementation in Python

```python
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    # Insert: O(log n) (average), worst-case O(n)
    def insert(self, key):
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert_node(self.root, key)

    def _insert_node(self, node, key):
        if key < node.key:
            if node.left is None:
                node.left = Node(key)
            else:
                self._insert_node(node.left, key)
        elif key > node.key:
            if node.right is None:
                node.right = Node(key)
            else:
                self._insert_node(node.right, key)
