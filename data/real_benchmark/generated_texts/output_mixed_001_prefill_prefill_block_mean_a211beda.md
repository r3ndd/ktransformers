## prompt_id: mixed_001 | experiment: prefill_prefill_block_mean_a211beda | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 32}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

assistant
<think>

</think>

Here is a complete implementation of a Binary Search Tree (BST) in Python, including `insert`, `search`, and `delete` operations. Following the code, I will provide a detailed breakdown of the time complexity and a step-by-step walkthrough for each operation using a small example.

### Python Implementation

```python
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    # --- INSERT OPERATION ---
    def insert(self, key):
        """Inserts a new key into the BST."""
        if self.root is None:
            self.root = TreeNode(key)
        else:
            self._insert_recursive(self.root, key)

    def _insert_recursive(self, node, key):
        if key < node.key:
            if node.left is None:
                node.left = TreeNode(key)
            else:
                self._insert_recursive(node.left, key)
        elif key > node.key:
            if node.right is None:
                node.right = TreeNode(key)
