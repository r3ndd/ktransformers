## prompt_id: factual_001 | experiment: decode_ema_score_averaging_7c3f44a7 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.7}

### Prompt

Explain how garbage collection works in Python. Include a discussion of reference counting, the generational garbage collector, and when memory might not be freed as expected.

### Generated Text

Python’s memory management is a hybrid system that combines **reference counting** with a **generational garbage collector**. This design allows Python to reclaim most memory quickly while also handling complex cases (like reference cycles) that reference counting alone cannot solve.

---

## 1. Reference Counting: The First Line of Defense

### How It Works
- Every object in Python has an internal counter (`ob_refcnt`) that tracks how many references point to it.
- When you create an object, its reference count starts at 1.
- Each time a new reference is created (e.g., assigning to a variable, adding to a list), the count increments.
- When a reference goes out of scope or is reassigned, the count decrements.
- **As soon as the reference count drops to zero**, the object’s `__del__` method (if defined) is called, and its memory is immediately freed.

### Example
```python
a = [1, 2, 3]      # ref count = 1
b = a              # ref count = 2
del b              # ref count = 1
del a              # ref count = 0 → object freed immediately
```

### Adv
