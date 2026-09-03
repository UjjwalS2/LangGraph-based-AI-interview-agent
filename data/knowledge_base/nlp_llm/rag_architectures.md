# RAG Paradigms: Naive, Advanced, and Modular Retrieval-Augmented Generation

## Evolution of RAG Paradigms
Retrieval-Augmented Generation (RAG) grounds Large Language Model responses on authoritative external knowledge bases, mitigating hallucinations, outdated knowledge cutoff, and lack of domain specificity.

### 1. Naive RAG
- Workflow: `Document Ingestion` $\to$ `Fixed Chunking` $\to$ `Embedding` $\to$ `Vector Search` $\to$ `Prompt Augmentation` $\to$ `LLM Generation`.
- Limitations: Poor precision, noisy chunk retrieval, context fragmentation, sensitivity to semantic drift, and high vulnerability to the "Lost in the Middle" phenomenon.

### 2. Advanced RAG
Introduces targeted pre-retrieval and post-retrieval optimizations:
- **Pre-Retrieval**:
  - Query Rewriting / Expansion (HyDE - Hypothetical Document Embeddings, Step-Back Prompting).
  - Multi-Query decomposition.
  - Hierarchical and parent-document chunking.
- **Post-Retrieval**:
  - Hybrid Search (combining Lexical BM25 and Dense embeddings).
  - Reciprocal Rank Fusion (RRF) for score-agnostic rank combination.
  - Cross-Encoder Reranking to filter top candidates with deep cross-attention.
  - Context compression and selective pruning.

### 3. Modular RAG
Decouples RAG into specialized independent services:
- Dynamic Routing (deciding whether to retrieve, query knowledge graphs, call external APIs, or respond directly).
- Self-RAG (reflection tokens determining retrieval necessity and output faithfulness).
- Semantic Caching layer reducing redundant model execution and retrieval overhead.

## Chunking Strategies and Tradeoffs
- **Fixed-Size Chunking with Overlap**: Splits text strictly by token or character count (e.g. 500 tokens with 100 token overlap). Simple, but risks cutting sentences and code blocks in half.
- **Recursive / Structure-Aware Chunking**: Splits iteratively on markdown headers (`#`, `##`), paragraphs (`\n\n`), sentences (`. `), and words (` `), keeping semantic units coherent.
- **Sentence Window Retrieval**: Embeds individual small sentences for precise retrieval, but expands context to surrounding $\pm 3$ sentences when sending to the LLM.
- **Hierarchical / Parent-Child Chunking**: Indexes small leaf chunks ($128$ tokens) in the vector database for high similarity recall, but retrieves the larger parent chunk ($512-1024$ tokens) for rich context.
