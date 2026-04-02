## prompt_id: factual_002 | experiment: decode_ema_score_averaging_bf390907 | seed: 42

### MoE Routing

prefill: prefill_block_mean
prefill_params: {"window_size": 1}
decode: ema_score_averaging
decode_params: {"ema_beta": 0.9}

### Prompt

Compare and contrast the transformer architecture with recurrent neural networks (RNNs). What are the key advantages of transformers that led to their dominance in NLP?

### Generated Text

The shift from Recurrent Neural Networks (RNNs) to Transformer architectures marks one of the most significant paradigm shifts in Natural Language Processing (NLP). While RNNs dominated for over a decade, Transformers have since become the standard backbone for modern large language models (LLMs).

Here is a detailed comparison and an analysis of why Transformers achieved dominance.

### 1. Core Architectural Differences

#### **Recurrent Neural Networks (RNNs)**
*   **Mechanism:** RNNs process data sequentially, one token at a time. At each time step $t$, the network takes the current input $x_t$ and the hidden state from the previous step $h_{t-1}$ to produce the new hidden state $h_t$.
*   **Memory:** Information is passed forward through the chain of hidden states. The model "remembers" context by compressing it into the vector $h_t$.
*   **Variants:** Standard RNNs suffer from vanishing/exploding gradients. Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) were developed to mitigate this, allowing them to retain information over longer sequences.

#### **Transformers**
*   **Mechanism
