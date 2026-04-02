## prompt_id: factual_002 | experiment: decode_ema_score_averaging_7c3f44a7 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.7}

### Prompt

Compare and contrast the transformer architecture with recurrent neural networks (RNNs). What are the key advantages of transformers that led to their dominance in NLP?

### Generated Text

The shift from Recurrent Neural Networks (RNNs) to Transformer architectures represents one of the most significant paradigm shifts in natural language processing (NLP). While RNNs dominated for over a decade, Transformers now underpin state-of-the-art models like BERT, GPT, and Llama.

Here is a detailed comparison of the two architectures, followed an analysis of why Transformers achieved dominance.

### 1. Core Architectural Differences

#### **Recurrent Neural Networks (RNNs)**
*   **Mechanism:** RNNs process data **sequentially**. They maintain a hidden state that acts as a "memory" of previous inputs. To predict the current word, the model processes the input token and combines it with the hidden state from the previous time step.
*   **Flow:** $h_t = f(x_t, h_{t-1})$. The information flows strictly from left to right (or right to left in bidirectional variants).
*   **Context Dependency:** In theory, an RNN can remember any part of the sequence, but in practice, it struggles to retain information from very distant steps due to the **vanishing gradient problem**.

#### **Transformers**
*   **Mechanism:** Transformers rely
