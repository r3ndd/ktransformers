## prompt_id: coding_001 | experiment: decode_sliding_window_score_averaging_d4dee748 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: sliding_window_score_averaging
decode_params: {"window_size": 4}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s a Python function that finds the longest palindromic substring using the “expand from center” approach:

```python
def longest_pal_sub(s: str) -> str:
    """
    Returns the longest palpalindromic substring in s.
    Uses "expand from center" technique.
    Time: O(n^2), worst case; Space: O(1)
    """
    if not s:
        return ""

    n = len(s)
    start, end = 0, 0

    def expand_from_center(left, right):
        while left >= 0 and right < n and s[left] == s[right]:
        left -= 1
        right += 1
        # The palindrome is s[left+1:right] (exclusive)
        return left + 1, right - 1

    for i in range(n):
    # Odd centers (single char)
    l, r = expand_from_center(i, i)
    if r - l > end - start:
    start, end = l, r
    # Even centers (two identical chars)
    l, r = expand_from_center(i, i +
