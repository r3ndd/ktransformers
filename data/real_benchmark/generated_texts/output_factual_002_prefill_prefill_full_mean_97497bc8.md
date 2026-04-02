## prompt_id: factual_002 | experiment: prefill_prefill_full_mean_97497bc8 | seed: 42

### MoE Routing

prefill: prefill_full_mean
prefill_params: {}
decode: sliding_window_score_averaging
decode_params: {"window_size": 1}

### Prompt

Compare and contrast the transformer architecture with recurrent neural networks (RNNs). What are the key advantages of transformers that led to their dominance in NLP?

### Generated Text

assistant
<think>

</think>

The shift from **Recurrent Neural Networks (RNNs)** to **Transformers** represents one of the most significant paradigm shifts in the history of Natural Language Processing (NLP). While RNNs dominated for over a decade, Transformers have since become the foundational architecture for modern Large Language Models (LLMs) like BERT, GPT, and Llama.

Here is a detailed comparison, followed by an analysis of why Transformers achieved dominance.

---

### 1. Core Architectural Differences

#### Recurrent Neural Networks (RNNs)
*   **Mechanism:** RNNs process data sequentially, one token at a time ($t_1, t_2, \dots, t_n$). They maintain a hidden state vector that acts as "memory," carrying information from previous steps to the current step.
*   **Flow:** Information flows linearly through time. To understand the word "bank" in a sentence, the network must pass through every preceding word to accumulate context.
*   **Variants:** Standard RNNs suffer from vanishing gradients; Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) were developed to mitigate this, allowing
