## prompt_id: coding_001 | experiment: decode_ema_score_averaging_b5aee75c | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.3}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s is a Python function that finds the longest palindromic substring using the “expand around center” approach:

```python
def longest_pal_substring(s: str) -> str:
    if not s:
        return ""

    start, end = 0, 0

    for i in range(len(s)):
    # expand around center for each character and between characters
        left, right = i, i
        while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
        # after loop, palindrome is s[left+1:right]
        if right - left - 1 > end - start + 1:
        start, end = left + 1, right - 1

        left, right = i, i + 1
        while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
        # after loop, palindrome is s[left+1:right]
        if right - left - 1 > end - start + 1:
        start, end = left + 1
