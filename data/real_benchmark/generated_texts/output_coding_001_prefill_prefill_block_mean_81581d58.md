## prompt_id: coding_001 | experiment: prefill_prefill_block_mean_81581d58 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s a Python function that finds the longest palindromic substring using the “expand around center” approach:

```python
def longest_palindromic_substring(s: str) -> str:
    if not s:
        return ""

    start, end = 0, 0  # indices of the best palindrome found so far

    def expand_around_center(left: int, right: int) -> tuple[int, int]:
        """Expand from the given center and return (start, end) of the palindrome."""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        # When loop exits, s[left+1:right] is the palindrome
        return left + 1, right - 1

    for i in range(len(s)):
        # Odd-length palindromes (center at i)
        l1, r1 = expand_around_center(i, i)
        # Even-length palindromes (center between i and i+1)
        l2, r2 = expand_around_center(i, i + 1)

        # Update best if
