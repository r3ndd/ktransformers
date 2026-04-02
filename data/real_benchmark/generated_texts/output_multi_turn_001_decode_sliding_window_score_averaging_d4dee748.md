## prompt_id: multi_turn_001 | experiment: decode_sliding_window_score_averaging_d4dee748 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: sliding_window_score_averaging
decode_params: {"window_size": 4}

### Prompt

Let's work through a coding problem together. I need to implement a LRU (Least Recently Used) cache in Python.

First, can you explain what an LRU cache is and when it's useful?

### Generated Text

An **LRU (Least Recently Used) cache** is a fixed-size data structure that stores key-value pairs, and when the it reaches capacity, it evates the least recently used item.

 It's useful in scenarios where: you need to limit memory usage but access patterns are "temp locality" (recently accessed items are likely to be accessed again soon).—common examples include web caching, database query results, session storage, or any other systems where you want fast O(1) access with bounded memory.



### Core How It Works
-
1. **Capacity**: You has a a maximum size `N`.
2. **Operations**:
   - `get(key`: str) -> value | None`: Returns the value if key exists, else `None` (or maybe raise error).
   - `put(key: str, value): None`: Adds/updates the entry; If full capacity, remove the *least recently used* entry before.
3. **Order**: The order of "access recency" tracks: most recent = front, least recent = back.

4. **Implementation**: Typically uses a **hash map + doubly linked list** for: O(1) get/put time.

5.
