## prompt_id: coding_002 | experiment: decode_ema_score_averaging_bf390907 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.9}

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

If you’re seeing unexpected behavior, possible causes are:

- You’re running an older or modified version of the function.
- There’s another `flatten` function shadowing this one (e.g., from a library).
- The input isn’t what you think it is (e.g., tuples instead of lists, or non-list iterables).

To make it more robust (and handle tuples as well), you could do:

```python
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item
