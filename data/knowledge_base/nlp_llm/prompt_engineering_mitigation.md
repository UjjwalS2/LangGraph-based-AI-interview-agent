# Prompt Engineering, Context Windows, and Hallucination Mitigation

## Prompt Engineering Paradigms
- **Zero-Shot Prompting**: Direct instruction without exemplar demonstrations.
- **Few-Shot In-Context Learning**: Provides input-output demonstration pairs in the prompt, steering output structure and tone without parameter fine-tuning.
- **Chain-of-Thought (CoT)**: Prompts model to generate intermediate reasoning steps before arriving at a final answer ("Let's think step by step"), significantly improving logical and mathematical reasoning.
- **ReAct (Reason + Act)**: Interleaves reasoning traces with tool/retrieval action calls, allowing dynamic interaction with external environments.

## Context Window Dynamics and The "Lost in the Middle" Effect
- LLM attention mechanisms exhibit position bias: information placed at the very beginning (primacy effect) or very end (recency effect) of long contexts is recalled with much higher accuracy than information buried in the middle $30-70\%$ of the context.
- **Mitigation in RAG**: Re-order retrieved context chunks so that the highest-scoring reranked passages are placed at the outer boundaries of the prompt.

## Hallucination Types and Mitigation
1. **Fact-Conflicting Hallucination**: Generated output contradicts source/retrieved facts.
2. **Input-Conflicting Hallucination**: Generated output contradicts explicit instructions or user query constraints.
3. **Context-Free Hallucination**: Output fabricates ungrounded entities or claims not supported by any known knowledge base.
- **Systematic Mitigations**:
  - Strict grounding system instructions ("Answer strictly using provided reference passages; if information is absent, state that explicitly").
  - Temperature reduction ($T \le 0.2$) and top-$p$ nucleus sampling.
  - Verification loops / Self-Reflection (generating initial response, verifying assertions against retrieved source, and revising discrepancies).
