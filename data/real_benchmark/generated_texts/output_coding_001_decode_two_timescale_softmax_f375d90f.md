## prompt_id: coding_001 | experiment: decode_two_timescale_softmax_f375d90f | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 16.0}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s a Python function that finds the longest palindromic substring using the “expand around center” approach:

```python
def longest_palindrome(s):
    """
    Returns the longest palindromic substring in string `s`.
    """
    if not s:
        return ""

    n = len(s)
    best_start, best_end = 0, 0

    for i in range(n):
        # Expand odd (center at i)
        left, right = i, i
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        length = right - left - 1
        if length > best_end - best_start + 1:
            best_start, best_end = left + 1, right - 1

        # Expand even (center between i and i+1)
        left, right = i, i + 1
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        length = right - left - 1
        if length > best
