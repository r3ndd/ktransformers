## prompt_id: coding_002 | experiment: decode_ema_score_averaging_b5aee75c | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.3}

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

Your code is logic for a nested list, but it’s actually correct.

The function:

```python
def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
```

print `flatten([1, [2, 3], [[4, 5], 6]])` → `[1, 2, 3, 4, 5, 6]`.

If you’re seeing different output, the most common issues are:

1. **Python version**: using older Python where `isinstance` behaves differently? (unlikely).  
2. **Input data**: not exactly what you think (maybe some strings or other types).  
3. **Indent/typo**: copy-paste error.  

Let’s run it in a fresh environment:

```python
>>> def flatten(nested):
...     result = []
...     for item in nested:
...         if isinstance(item, list):
...             result.extend(flatten(item))
...         else:
...             result.append(item)
...     return result
...
