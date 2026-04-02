## prompt_id: mixed_001 | experiment: prefill_prefill_block_mean_64393c5a | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 128}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

assistant
<think>

</think>

Here is a complete implementation of a Binary Search Tree (BST) in Python, including `insert`, `search`, and `delete` operations. Following the code, I will provide a detailed walkthrough using a small example and analyze the time complexity for each operation.

### Python Implementation

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    # --- INSERT OPERATION ---
    def insert(self, val):
        """Inserts a value into the BST."""
        if self.root is None:
            self.root = TreeNode(val)
        else:
            self._insert_recursive(self.root, val)

    def _insert_recursive(self, node, val):
        if val < node.val:
            if node.left is None:
                node.left = TreeNode(val)
            else:
                self._insert_recursive(node.left, val)
        elif val > node.val:
            if node.right is None:
                node.right = TreeNode(val)
            else:
                self._
