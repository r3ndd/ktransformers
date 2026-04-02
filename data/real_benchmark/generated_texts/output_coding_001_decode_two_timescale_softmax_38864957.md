## prompt_id: coding_001 | experiment: decode_two_timescale_softmax_38864957 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 64.0}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s a Python function that finds the longest palindromic substring in a given string, using the “expand around center” approach (also known as Man’s algorithm).

### Approach Explanation

- **Expand Around Center**:  
  For each character `i` in the string, we treat it as a potential center of a palindrome.
  - If `i` is an odd character, we expand outward from `i` and `i+1`.
  - If `i` is an even character, we expand outward from `i` alone.
  - We continue expanding until characters match on both sides, then stop when mismatch occurs or boundaries are reached.
  - This gives us the length of palindrome centered at `i`.
  - The maximum length among all centers is the answer.

 time complexity: O(n) because we expand outward for each center, which takes O(n) in worst case.

 space complexity: O(1) because we only keep track of current max length and indices.
