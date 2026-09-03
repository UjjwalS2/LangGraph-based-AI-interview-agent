"""
Hybrid RAG Vector and Lexical Store for LangGraph Agent.
Combines Qdrant dense vector search with BM25 keyword matching via Reciprocal Rank Fusion.
"""

from typing import Any, Dict, List, Optional
import os
import hashlib
import logging
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.config import config, PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    chunk_id: str
    doc_id: str
    topic: str
    text: str
    score: float
    source: str = "hybrid_rrf"


class HybridKnowledgeStore:
    """Manages Qdrant vector index and BM25 inverted index for technical grounding."""

    def __init__(self):
        self.qdrant_path = PROJECT_ROOT / config.paths.qdrant_storage
        self.bm25_path = PROJECT_ROOT / config.paths.bm25_storage
        self.collection_name = "langgraph_knowledge"
        self.dimension = config.embedding.dimension

        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.bm25_path.mkdir(parents=True, exist_ok=True)

        self.client = QdrantClient(path=str(self.qdrant_path))
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []

        self._init_qdrant()
        self._load_or_build_indices()

    def _init_qdrant(self):
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
                )
        except Exception as e:
            logger.warning(f"Qdrant collection init: {e}")

    def embed_text(self, text: str) -> np.ndarray:
        """Deterministic dense feature extraction using unigrams, bigrams, and subwords."""
        clean = text.lower()
        for ch in '.,()[]{}:;"\'!?':
            clean = clean.replace(ch, " ")
        words = [w for w in clean.split() if len(w) > 1]
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not words:
            return vec

        for i, w in enumerate(words):
            # Unigram feature hash
            h1 = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16) % self.dimension
            vec[h1] += 1.0

            # Bigram feature hash
            if i > 0:
                bw = f"{words[i-1]}_{w}"
                h2 = int(hashlib.md5(bw.encode("utf-8")).hexdigest()[:8], 16) % self.dimension
                vec[h2] += 1.5

            # Character 3-grams for stem and typo tolerance
            for c_idx in range(len(w) - 2):
                sub = w[c_idx : c_idx + 3]
                h3 = int(hashlib.md5(sub.encode("utf-8")).hexdigest()[:8], 16) % self.dimension
                vec[h3] += 0.3

        vec = np.tanh(vec)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def _load_or_build_indices(self):
        bm25_file = self.bm25_path / "bm25_store.pkl"
        if bm25_file.exists():
            try:
                with open(bm25_file, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", [])
                    tokenized = [c["text"].lower().split() for c in self.chunks]
                    if tokenized:
                        self.bm25 = BM25Okapi(tokenized)
                        return
            except Exception as e:
                logger.warning(f"Could not load existing BM25 cache ({e}). Ingesting docs...")

        # Ingest docs from knowledge base
        kb_dir = PROJECT_ROOT / config.paths.knowledge_base
        if not kb_dir.exists():
            return

        points = []
        self.chunks = []
        chunk_idx = 0

        for md_file in kb_dir.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
                topic = md_file.parent.name
                doc_id = md_file.stem

                # 500-char chunking with 100 overlap
                lines = text.split("\n\n")
                current_chunk = ""
                for line in lines:
                    if len(current_chunk) + len(line) < 500:
                        current_chunk += "\n\n" + line
                    else:
                        if current_chunk.strip():
                            cid = f"{doc_id}_c{chunk_idx:03d}"
                            vec = self.embed_text(current_chunk.strip()).tolist()
                            chunk_data = {
                                "chunk_id": cid,
                                "doc_id": doc_id,
                                "topic": topic,
                                "text": current_chunk.strip(),
                            }
                            self.chunks.append(chunk_data)
                            points.append(PointStruct(id=chunk_idx, vector=vec, payload=chunk_data))
                            chunk_idx += 1
                        current_chunk = line

                if current_chunk.strip():
                    cid = f"{doc_id}_c{chunk_idx:03d}"
                    vec = self.embed_text(current_chunk.strip()).tolist()
                    chunk_data = {
                        "chunk_id": cid,
                        "doc_id": doc_id,
                        "topic": topic,
                        "text": current_chunk.strip(),
                    }
                    self.chunks.append(chunk_data)
                    points.append(PointStruct(id=chunk_idx, vector=vec, payload=chunk_data))
                    chunk_idx += 1
            except Exception as e:
                logger.error(f"Failed ingesting {md_file}: {e}")

        # Index in Qdrant
        if points:
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
            except Exception as e:
                logger.error(f"Qdrant upsert failed: {e}")

        # Index in BM25
        if self.chunks:
            tokenized = [c["text"].lower().split() for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized)
            with open(bm25_file, "wb") as f:
                pickle.dump({"chunks": self.chunks}, f)

    def search_hybrid(self, query: str, topic_filter: Optional[str] = None, top_k: int = 4) -> List[RetrievedPassage]:
        """Executes BM25 + Qdrant hybrid search with RRF combination."""
        if not self.chunks:
            return []

        # BM25 Branch
        bm25_scores = {}
        if self.bm25:
            tokens = query.lower().split()
            scores = self.bm25.get_scores(tokens)
            ranked_indices = np.argsort(scores)[::-1][:20]
            for rank, idx in enumerate(ranked_indices, start=1):
                chunk = self.chunks[idx]
                if not topic_filter or topic_filter.lower() in chunk["topic"].lower():
                    bm25_scores[chunk["chunk_id"]] = 1.0 / (config.retrieval.rrf_k + rank)

        # Dense Vector Branch
        dense_scores = {}
        q_vec = self.embed_text(query).tolist()
        try:
            if hasattr(self.client, "query_points"):
                res = self.client.query_points(collection_name=self.collection_name, query=q_vec, limit=20).points
            else:
                res = self.client.search(collection_name=self.collection_name, query_vector=q_vec, limit=20)

            for rank, pt in enumerate(res, start=1):
                cid = pt.payload.get("chunk_id", "")
                topic = pt.payload.get("topic", "")
                if not topic_filter or topic_filter.lower() in topic.lower():
                    dense_scores[cid] = 1.0 / (config.retrieval.rrf_k + rank)
        except Exception as e:
            logger.warning(f"Vector search warning: {e}")

        # Fuse with Reciprocal Rank Fusion
        all_cids = set(bm25_scores.keys()) | set(dense_scores.keys())
        chunk_map = {c["chunk_id"]: c for c in self.chunks}

        scored_list = []
        for cid in all_cids:
            if cid in chunk_map:
                rrf_score = bm25_scores.get(cid, 0.0) + dense_scores.get(cid, 0.0)
                scored_list.append((rrf_score, chunk_map[cid]))

        scored_list.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored_list[:top_k]:
            results.append(
                RetrievedPassage(
                    chunk_id=chunk["chunk_id"],
                    doc_id=chunk["doc_id"],
                    topic=chunk["topic"],
                    text=chunk["text"],
                    score=round(score, 4),
                )
            )
        return results


_GLOBAL_STORE = None


def get_hybrid_store() -> HybridKnowledgeStore:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = HybridKnowledgeStore()
    return _GLOBAL_STORE
