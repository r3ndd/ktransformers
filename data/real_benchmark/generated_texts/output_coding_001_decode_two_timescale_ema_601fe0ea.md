## prompt_id: coding_001 | experiment: decode_two_timescale_ema_601fe0ea | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_ema
decode_params: {"mix_lambda": 0.4}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s’s a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

```python
def longest_palindrome(s: str) -> str:
    if not s:
        return ""

    n = len(s)
    start, end = 0, 0

    def expand_from_center(left, right):
        while left >= 0 and right < n and s[left] == s[right]:
        left -= 1
        right += 1
        return left + 1, right - 1

    for i in range(n):
        # odd palindrome (center at i)
        l, r = expand_from_center(i, i)
        if r - l > end - start:
        start, end = l, r

        # even palindrome (center between i and i+1)
        l, r = expand_from_center(i, i + 1)
        if r - l > end - start:
        start, end = l, r

    return s[start:end + 1]

```python
def longest_palindrome(s: str) -> str:
    if
