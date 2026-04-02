## prompt_id: factual_002 | experiment: baseline | seed: 42

### MoE Routing

prefill: baseline (none)
decode: baseline (none)

### Prompt

Compare and contrast the transformer architecture with recurrent neural networks (RNNs). What are the key advantages of transformers that led to their dominance in NLP?

### Generated Text

The shift from Recurrent Neural Networks (RNNs) to Transformer architectures marks one of the most significant paradigm shifts in Natural Language Processing (NLP). While RNNs dominated for over a decade, Transformers have since become the standard foundation for modern large language models (LLMs).

Here is a detailed comparison and an analysis of why Transformers achieved dominance.

### 1. Core Architectural Differences

#### **Recurrent Neural Networks (RNNs)**
*   **Mechanism:** RNNs process data sequentially, one token at a time. They maintain a hidden state ($h_t$) that acts as a "memory" of previous inputs. To predict the next word, the model looks at the current input and the hidden state from the previous step.
*   **Flow:** $Input_t \rightarrow [Hidden State] \rightarrow Output_t$. The output of step $t$ feeds into step $t+1$.
*   **Variants:** Standard RNNs suffer from vanishing/exploding gradients. Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) were developed to mitigate this by adding gating mechanisms to control information flow.

#### **Transformers**
*   **Mechanism
