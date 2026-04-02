## prompt_id: coding_002 | experiment: decode_ema_score_averaging_7c3f44a7 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.7}

### Prompt

I have this Python code that's supposed to flatten a nested list, but it's not working correctly. Can you debug it?

def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# Test
print(flatten([1, [2, 3], [[4, 5], 6]]))  # Expected: [1, 2, 3, 4, 5, 6]

### Generated Text

Your code is actually correct for the given example and will produce the expected output:

```python
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# Test
print(flatten([1, [2, 3], [[4, 5], 6]]))  # Output: [1, 2, 3, 4, 5, 6]
```

This works because:
- It recursively processes each element.
- If an element is a list, it flattens that sublist and extends the result.
- Otherwise, it appends the element directly.

If you’re seeing unexpected behavior, possible causes could be:
- You’re running an older or modified version of the function.
- There’s extra code around this function that modifies `nested` or `result`.
- You’re testing with something other than a list (e.g., tuples, dicts) and expecting different behavior.

If you want to support other iterable types (like tuples) as well, you can adjust the check:

```python
from
