"""
RAG (Retrieval-Augmented Generation) modules.

This package provides:
- Embeddings: Generate vector embeddings using AWS Bedrock Titan
- VectorStore: Weaviate Cloud vector database for persistent storage
- InMemoryVectorStore: Simple in-memory store for testing
- Retriever: Semantic search over vector stores
"""

from .embeddings import get_embedding, get_embeddings_batch, cosine_similarity
from .vector_store import VectorStore, InMemoryVectorStore, Document, SearchResult
from .retriever import Retriever, MultiCollectionRetriever

__all__ = [
    # Embeddings
    "get_embedding",
    "get_embeddings_batch",
    "cosine_similarity",
    # Vector stores
    "VectorStore",
    "InMemoryVectorStore",
    "Document",
    "SearchResult",
    # Retrievers
    "Retriever",
    "MultiCollectionRetriever",
]
