## prompt_id: mixed_001 | experiment: prefill_prefill_full_mean_97497bc8 | seed: 42

### MoE Routing

prefill: prefill_full_mean
prefill_params: {}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

assistant
<think>

</think>

Here is a complete implementation of a Binary Search Tree (BST) in Python, including `insert`, `search`, and `delete` operations. Following the code, I will provide a detailed walkthrough with time complexity analysis for each operation using a small example.

### Python Implementation

```python
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None  # Optional: helps in deletion logic

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        """Inserts a new key into the BST."""
        if self.root is None:
            self.root = TreeNode(key)
            return

        current = self.root
        while True:
            if key < current.key:
                if current.left is None:
                    current.left = TreeNode(key)
                    break
                current = current.left
            elif key > current.key:
                if current.right is None:
                    current.right = TreeNode(key)
                    break
                current = current.right
            else:
                # Key already exists;
