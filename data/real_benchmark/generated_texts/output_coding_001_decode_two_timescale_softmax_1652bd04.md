## prompt_id: coding_001 | experiment: decode_two_timescale_softmax_1652bd04 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 1.0}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s is a Python function that finds the longest palindromic substring in a given string, using the “expand around center” approach:

```python
def longest_palindromic_substring(s: str) -> str:
    if not s:
        return ""

    n = len(s)
    start, end = 0, 0

    for i in range(n):
        # expand around center for odd-length palindromes (center at i)
        l, r = i, i
        while l > 0 and r < n - 1 and s[l-1] == s[r+1]:
            l -= 1
            r += 1
        if r - l + 1 > end - start + 1:
            start, end = l, r

        # expand around center for even-length palindromes (center between i and i+1)
        l, r = i, i + 1
        while l > 0 and r < n - 1 and s[l-1] == s[r+1]:
            l -= 1
            r += 1
        if r - l + 1
