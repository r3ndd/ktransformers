## prompt_id: coding_001 | experiment: prefill_prefill_full_mean_97497bc8 | seed: 42

### MoE Routing

prefill: prefill_full_mean
prefill_params: {}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

assistant
<think>

</think>

Here’s a Python function that finds the longest palindromic substring using the “expand around center” approach, along with an explanation and time complexity.

```python
def longest_palindromic_substring(s: str) -> str:
    """
    Returns the longest palindromic substring in s.
    Uses the expand-around-center technique.
    """
    if not s:
        return ""

    start, end = 0, 0  # indices of the best palindrome found so far

    def expand_around_center(left: int, right: int) -> tuple[int, int]:
        """Expand from the given center (left, right) while it's a palindrome."""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        # When loop exits, s[left+1:right] is the palindrome
        return left + 1, right - 1

    for i in range(len(s)):
        # Odd-length palindromes (center at i)
        l1, r1 = expand_around_center(i, i)
