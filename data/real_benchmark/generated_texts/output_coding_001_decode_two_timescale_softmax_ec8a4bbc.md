## prompt_id: coding_001 | experiment: decode_two_timescale_softmax_ec8a4bbc | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 256.0}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s a Python function to find the longest palindromic substring using the **expand-around-center** approach, which is efficient and easy to implement.

### Approach
The idea is:
- For every character in string `s`, treat it as a potential center of palindrome (odd or even length).
- Expand outward from that center while checking characters match each other until mismatch occurs.
- Keep track of the longest palindrome found.

This is called “center expansion” method.

### Time Complexity
- Each character is processed once for odd and even centers.
- In worst case, we can expand up to O(n) per character.
- Total time complexity is O(n²) where n = length of string.
- Space complexity is O(1) because we only store some variables.

### Implementation

```python
def longest_palindrome_substring(s):
    """
    Find the longest palindromic substring in s.
    Returns the longest palindromic substring.
    """
    if not s:
        return ""
    
    def helper_expand_center(start, end):
        """
        Expand around center start and end.
        Return the left and right indices.
