## prompt_id: mixed_001 | experiment: decode_ema_score_averaging_d27b125a | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.1}

### Prompt

Implement a binary search tree in Python with insert, search, and delete operations. For each operation, explain the time complexity and walk through how the algorithm works using a small example.

### Generated Text

Here’s is a implementation of a binary search tree (BST) in Python, with `insert`, `search`, and `delete` operations.

Each operation:
  
Time complexity explanation + walk through using a small example.

  

The BST is a ordered tree where each node has a key and left/right children.

  

Insert:  
O‑log N (average case), worst O‑N (sorted case).

  

Search:  
O‑log N (average case), worst O‑N (sorted case).

  

Delete:  
O‑log N (average case), worst O‑N (sorted case).

  

Python code:  

  

```python
class Node:
    def __init__(self, key):
    self.key = key
    self.left = None
    self.right = None

class BinarySearchTree:
    def __init__(self):
    self.root = None

    def insert(self, key):
    if self root is None:
    self root = Node(key)
    else:
    self._insert(self root, key)

    def _insert(self, node, key):
    if key < node key:
    if node left is None:
    node
