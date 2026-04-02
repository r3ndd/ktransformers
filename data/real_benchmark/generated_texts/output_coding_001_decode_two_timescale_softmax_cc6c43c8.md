## prompt_id: coding_001 | experiment: decode_two_timescale_softmax_cc6c43c8 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 4.0}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s a Python function that finds the longest palindromic substring using the **expand-around-center** approach (also known as “center expansion” or “expand around center”).

### Approach Explanation
- For each character in the string, treat it as the potential center of a palindrome.
- If the center is a single character (odd-length palindrome), expand outward while characters match.
- If the center is two adjacent characters (even-length palindrome), expand outward while characters match.
- Track the longest palindrome found during this process.
- This runs in `O(n)` time for number of centers and `O(n)` for expansion in worst case, giving overall `O(n²)` time complexity, but with average-case much faster than brute-force.

### Time Complexity
- **Time**: `O(n²)` where `n` is length of input string.
- **Space**: `O(1)` auxiliary space (just tracking indices).

### Implementation

```python
def longest_palindromic_substring(s: str) -> str:
    if not s:
        return ""

    n = len(s)
    start = 0
    max_len = 0

    def expand(left: int,
