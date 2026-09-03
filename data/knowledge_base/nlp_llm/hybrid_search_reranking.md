# Hybrid Retrieval, Reciprocal Rank Fusion, and Cross-Encoder Reranking

## Lexical vs. Dense Retrieval
- **Lexical Search (BM25 - Best Matching 25)**:
  - Probabilistic relevance framework based on exact term frequencies and inverse document frequency:
    $$\text{BM25}(D, Q) = \sum_{i=1}^N \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
  - Excels at exact keyword matches, domain acronyms, technical IDs, error codes, and specific variable names.
  - Fails when queries and documents use synonyms or paraphrased concepts (vocabulary mismatch).
- **Dense Semantic Retrieval (BGE-M3 + Qdrant)**:
  - Encodes entire passages into dense vector embeddings where semantic proximity is measured by cosine similarity.
  - Excels at conceptual understanding, paraphrasing, cross-lingual queries, and semantic themes.
  - Can struggle with exact keyword matching or rare identifiers not well-represented in pre-training.

## Hybrid Retrieval Fusion: Reciprocal Rank Fusion (RRF)
Because BM25 scores (unbounded positive floats) and Dense cosine similarities ($[-1, 1]$ or $[0, 1]$) follow completely different probability distributions and scale ranges, standard linear combination $(\alpha \cdot \text{score}_{\text{dense}} + (1-\alpha) \cdot \text{score}_{\text{bm25}})$ requires fragile manual tuning and heuristic normalization.

**Reciprocal Rank Fusion (RRF)** solves this by operating purely on the ranked order positions:
$$\text{RRF}(d \in D) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$
- $M$: The set of retrievers (e.g. $\{\text{BM25}, \text{Dense}\}$).
- $\text{rank}_m(d)$: The 1-indexed rank position of document $d$ in retriever $m$.
- $k$: Smoothing constant (typically $k=60$). Prevents top-ranked items in any single retriever from dominating the score excessively while penalizing items ranked lower down the lists.
- **Deduplication**: If a document appears in both BM25 top-20 and Dense top-20, its fused score combines both reciprocal rank terms $\left( \frac{1}{60 + r_1} + \frac{1}{60 + r_2} \right)$, naturally propelling it to the top.

## Cross-Encoder Reranking
Why not run cross-encoders directly over the entire database?
- **Bi-Encoder (Dense Retriever)**: Embeds query $q$ and document $d$ independently ($u = E(q), v = E(d)$) and computes dot product. Extremely fast ($O(1)$ vector indexing via HNSW), but lacks cross-attention between query tokens and document tokens.
- **Cross-Encoder (Reranker, e.g. `BAAI/bge-reranker-v2-m3`)**: Feeds the concatenated pair $[CLS] + \text{Query} + [SEP] + \text{Document} + [SEP]$ through all transformer layers simultaneously.
  - Allows full token-to-token bidirectional cross-attention, capturing intricate nuance, negation, and positional relationships.
  - Computational complexity is $O(N \cdot L^2)$, which is too slow to evaluate on thousands of documents in real-time.
- **Two-Stage Retrieval Architecture**:
  1. Retrieve candidate pool ($K=20-50$ items) via fast Hybrid Search + RRF.
  2. Rerank the top 15 fused candidates using the heavy Cross-Encoder to select the top 5 highest-quality reference passages.
