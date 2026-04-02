## prompt_id: coding_001 | experiment: decode_two_timescale_softmax_7c578a93 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: two_timescale_softmax
decode_params: {"mix_lambda": 0.2, "rho": 0.25}

### Prompt

Write a Python function to find the longest palindromic substring in a given string. Include a brief explanation of your approach and time complexity.

### Generated Text

Here’s is a Python function to find the longest palindic substring in a given string, using the **expanding approach** (centering expansion) and with a brief explanation of its approach and time complexity.

```python
def longest_palindromic_substring(s):
"""
Find the longest palindromic substring in a given string.
"""
if s:
return ""
n = len(s)
if n == 0:
return ""
start = 0
end = 0
for i in range(n):
# centering expansion for odd length
left = i - 1
right = i + 1
while left >= 0 and right < n and s[left] == s[right]:
left -= 1
right += 1
if right - left > end - start:
start = left + 1
end = right - 1
# centering expansion for even length
left = i - 1
right = i + 1
while left >= 0 and right < n and s[left] == s[right]:
left -= 1
right += 1
if right - left > end - start:
start = left + 1
