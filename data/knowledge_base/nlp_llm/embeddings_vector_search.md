# Embeddings, Vector Databases, and Approximate Nearest Neighbor Search

## Dense Embeddings: Static vs. Contextual
- **Static Embeddings (Word2Vec, GloVe)**: Map each token to a single static vector regardless of sentence context (cannot disambiguate "apple" the fruit vs "apple" the company).
- **Contextual Embeddings (BERT, BGE-M3, Text-Embedding-3)**: Generates high-dimensional vector representations ($d = 768 \dots 1024$) where token and sequence representations dynamically adapt based on surrounding tokens via self-attention.
- **BGE-M3 (BAAI Multi-Lingual, Multi-Functionality, Multi-Granularity)**:
  - Produces 1024-dimensional dense semantic vectors.
  - Trained with multi-stage contrastive learning on over 100 languages.
  - Supports dense retrieval, lexical multi-hot matching, and multi-vector ColBERT-style token retrieval.

## Similarity Metrics
Given vectors $u, v \in \mathbb{R}^d$:
1. **Cosine Similarity**: $\text{sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} \in [-1, 1]$. Invariant to vector scale. (When vectors are $L_2$-normalized, inner product equals cosine similarity).
2. **Dot Product (Inner Product)**: $\langle u, v \rangle = \sum u_i v_i$. Dependent on magnitude.
3. **Euclidean Distance ($L_2$)**: $\|u - v\|_2 = \sqrt{\sum (u_i - v_i)^2}$.

## Vector Indexing Algorithms (ANN)
Exact k-NN search requires $O(N \cdot d)$ linear scanning, which becomes prohibitive at scale ($N > 10^5$). Vector databases utilize Approximate Nearest Neighbor (ANN) indexes:
1. **HNSW (Hierarchical Navigable Small World)**:
   - Multi-layer graph index with skip-list properties.
   - Top layers have long-range links for fast coarse navigation; bottom layer contains dense local links for fine-grained exploration.
   - Provides logarithmic search time $O(\log N)$ with $>95\%$ recall. Default indexing in Qdrant, Milvus, and Weaviate.
2. **IVF (Inverted File Index)**:
   - Partitions vector space into Voronoi cells using K-Means centroids ($nlist$).
   - At query time, only vectors belonging to the $nprobe$ closest centroids are searched.
3. **Product Quantization (PQ)**:
   - Decomposes high-dimensional vectors into $m$ low-dimensional sub-vectors and quantizes each into centroid IDs, compressing memory footprint by $8\times \dots 16\times$.
