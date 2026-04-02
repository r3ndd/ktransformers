## prompt_id: creative_002 | experiment: decode_ema_score_averaging_ce23db51 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.5}

### Prompt

Write a haiku about debugging code at 3 AM. Then write a brief explanation of what a haiku is and why you chose those particular images.

### Generated Text

Blue light on the face,
Coffee cold beside the chair,
One bug found at dawn.

A **haiku** is a traditional Japanese poetic form consisting of three lines with a syllac structure of 5-7-5. It often focuses on capturing a moment in nature or a specific feeling, rather than telling a long story.

I chose these images to reflect the universal experience of late-night coding:
*   **"Blue light on the face"** captures the harsh glow of monitors in a dark room, emphasizing the isolation and physical strain.
*   **"Coffee cold"** highlights the passage of time and the forgotten self-care that comes with deep focus.
*   **"One bug found at dawn"** contrasts the exhaustion of the night with the small victory of solving a problem, acknowledging that the rest of the code might still be broken, but for now, there is peace.
