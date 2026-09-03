"""
Configuration loader for LangGraph Agentic Interview Platform.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GraphConfig(BaseModel):
    max_rounds: int = 3
    checkpoint_type: str = "memory"
    escalation_enabled: bool = True
    quality_threshold: float = 0.70


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    dimension: int = 1024
    batch_size: int = 16
    device: str = "cpu"


class RetrievalConfig(BaseModel):
    top_k: int = 5
    rrf_k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5


class CacheConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.90
    max_entries: int = 5000


class PricingTier(BaseModel):
    input_per_1m_tokens: float
    output_per_1m_tokens: float


class PricingConfig(BaseModel):
    flash: PricingTier = Field(default_factory=lambda: PricingTier(input_per_1m_tokens=0.15, output_per_1m_tokens=0.60))
    pro: PricingTier = Field(default_factory=lambda: PricingTier(input_per_1m_tokens=1.25, output_per_1m_tokens=5.00))


class PathsConfig(BaseModel):
    knowledge_base: str = "data/knowledge_base"
    qdrant_storage: str = "storage/qdrant"
    bm25_storage: str = "storage/bm25"
    cache_storage: str = "storage/cache"
    simulation_output: str = "storage/simulation_results.json"


class AppConfig(BaseModel):
    backend: str = "offline"
    graph: GraphConfig = Field(default_factory=GraphConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    pricing: PricingConfig = Field(default_factory=PricingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}
    else:
        raw_dict = {}

    env_backend = os.getenv("APP_BACKEND_MODE")
    if env_backend:
        raw_dict["backend"] = env_backend.lower()

    return AppConfig(**raw_dict)


config = load_config()
