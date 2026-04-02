## prompt_id: coding_001 | experiment: decode_ema_score_averaging_ce23db51 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.5}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s a Python function that finds the longest palindromic substring using the “expand around center” approach:

```python
def longest_palindrome(s: str) -> str:
    if not s:
        return ""

    start, end = 0, 0  # indices of the best palindrome found so far

    def expand_from_center(left: int, right: int) -> tuple[int, int]:
        """Expand from left and right while characters match."""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        # After loop, [left+1, right-1] is the palindrome
        return left + 1, right - 1

    for i in range(len(s)):
        # Odd-length palindromes (center at i)
        l1, r1 = expand_from_center(i, i)
        # Even-length palindromes (center between i and i+1)
        l2, r2 = expand_from_center(i, i + 1)

        # Choose the longer one
        if r1 - l1 > end -
