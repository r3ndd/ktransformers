## prompt_id: coding_002 | experiment: decode_ema_score_averaging_ce23db51 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.5}

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

Your code is actually correct for the given example and typical use cases. It properly flattens nested lists of one level or more:

```python
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

print(flatten([1, [2, 3], [[4, 5], 6]]))
# Output: [1, 2, 3, 4, 5, 6]
```

If it’s “not working” in your environment, common issues are:

- You’re passing something that isn’t a list (e.g., `None` or a string) at the top level.
- There are other types (like tuples) you want to flatten but aren’t handled.
- The function name was overridden somewhere.



To make it more robust, you can also handle tuples and other iterables (while excluding strings):

```python
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item
