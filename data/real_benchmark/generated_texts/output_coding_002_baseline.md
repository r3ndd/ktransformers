## prompt_id: coding_002 | experiment: baseline | seed: 42

### MoE Routing

prefill: baseline (none)
decode: baseline (none)

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

If you’re seeing unexpected behavior, it’s likely due to one of these common issues:

1. **Non-list iterables**: If your input contains other iterable types (e.g., tuples, sets), they won’t be flattened unless you explicitly handle them.  
   Example fix to also flatten tuples:
   ```python
   def flatten(nested):
       result = []
       for item in nested:
           if isinstance(item, (list, tuple)):
               result.extend(flatten(item))
           else:
               result.append(item)
       return result
   ```

2. **Strings being treated as lists**: In Python, strings are iterable,
