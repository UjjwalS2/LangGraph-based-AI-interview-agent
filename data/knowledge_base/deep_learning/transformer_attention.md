# Transformer Architecture and Multi-Head Attention

## Scaled Dot-Product Attention
Attention maps a query vector and a set of key-value vector pairs to an output:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
- $Q \in \mathbb{R}^{N \times d_k}$, $K \in \mathbb{R}^{M \times d_k}$, $V \in \mathbb{R}^{M \times d_v}$.
- **Scaling Factor $\frac{1}{\sqrt{d_k}}$**: For large projection dimensions $d_k$, dot products grow large in magnitude, pushing softmax into regions with extremely small gradients. Dividing by $\sqrt{d_k}$ stabilizes variance to 1.0.

## Multi-Head Attention (MHA)
Allows the model to jointly attend to information from different representation subspaces at different positions:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
with projection matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

## Positional Encoding
Because self-attention is permutation-equivariant (order-agnostic), token order must be injected:
1. **Sinusoidal Positional Encoding**: Fixed trigonometric frequencies:
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
2. **Rotary Position Embedding (RoPE)**: Applies complex 2D rotation matrices to $Q$ and $K$ representations, encoding relative token distance naturally into the inner product $q_m^T k_n$. Standard in modern LLMs (LLaMA, Mistral, Gemma).
3. **ALiBi (Attention with Linear Biases)**: Adds static linear distance penalties directly to attention logits: $\text{softmax}(q_i k_j^T - m \cdot |i - j|)$.

## Attention Variants for Inference Efficiency
- **Multi-Query Attention (MQA)**: Shares a single Key and Value head across all Query heads.
- **Grouped-Query Attention (GQA)**: Divides Query heads into $G$ groups, sharing one Key-Value head per group, dramatically reducing KV-cache VRAM usage during auto-regressive generation.
- **FlashAttention**: IO-aware exact attention tiling computation into GPU SRAM, avoiding $O(N^2)$ reads/writes to slow HBM memory.
