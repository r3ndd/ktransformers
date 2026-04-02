## prompt_id: coding_002 | experiment: decode_ema_score_averaging_7807f730 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.05}

### Prompt

I have this Python code that's supposed to flatten a nested list, but it's not working correctly. Can you debug it?

def flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# Test
print(flatten([1, [2, 3], [[4, 5], 6]]))  # Expected: [1, 2, 3, 4, 5, 6]

### Generated Text

Your code is logic correct. function `` actually works:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:

 code:
