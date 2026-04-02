## prompt_id: mixed_001 | experiment: decode_two_timescale_softmax_ec8a4bbc | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 256.0}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s a complete implementation of a binary search tree (BST) in Python with `insert`, `search`, and `delete` operations, plus explanations of time complexity and step-by-step walkthroughs using a small example.

## Implementation

```python
class BSTNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.value = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        # Insert the new node at the root if tree is empty
        if self.root is None:
            self.root = BSTNode(value)
            return
        # Find the correct position to insert the new node
        current_node = self.root
        while True:
            if value < current_node.value:
                if current_node.left is None:
                    current_node.left = BSTNode(value)
                    break
                else:
                    current_node = current_node.left
            elif value > current_node.value:
                if current_node.right is None:
                    current_node.right = BSTNode(value)
                    break
                else:
                    current_node = current_node.right
            else:
