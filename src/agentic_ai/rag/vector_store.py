"""
Weaviate Vector Store.

This module provides a vector store implementation using Weaviate Cloud
for storing and searching document embeddings with persistence.

Weaviate is a cloud-native vector database that provides:
- Cloud persistence (data survives program restarts)
- Efficient HNSW-based vector search
- Multi-collection support (e.g., "Regulations", "Procedures")
- Metadata filtering
- Built-in vector indexing

Example usage:
    from agentic_ai.rag.vector_store import VectorStore

    # Connect to Weaviate Cloud
    with VectorStore() as store:
        # Create a collection
        store.create_collection("Regulations", "Regulatory documents")

        # Add documents
        store.add_documents(
            collection_name="Regulations",
            documents=["Doc 1 text", "Doc 2 text"],
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            metadatas=[{"source": "reg1.pdf"}, {"source": "reg2.pdf"}],
            ids=["doc1", "doc2"]
        )

        # Search
        results = store.search("Regulations", query_embedding, k=5)

Environment variables required:
    WEAVIATE_URL: Your Weaviate Cloud instance URL
    WEAVIATE_API_KEY: Your Weaviate API key
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    A document with its embedding and metadata.

    This represents a chunk of text that has been embedded and can be
    searched. In a RAG system, documents are typically chunks of larger
    files (e.g., paragraphs from PDFs).

    Attributes:
        id: Unique identifier for the document.
        content: The original text content.
        embedding: Vector representation of the content.
        metadata: Optional dict with additional info (source file, page, etc.)

    Example:
        >>> doc = Document(
        ...     id="chunk_001",
        ...     content="The quick brown fox jumps over the lazy dog.",
        ...     embedding=[0.1, 0.2, 0.3, ...],
        ...     metadata={"source": "story.txt", "chunk_index": 0}
        ... )
    """
    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """
    A search result with similarity score.

    Attributes:
        document: The matched document.
        score: Similarity score (0 to 1, higher is better).
    """
    document: Document
    score: float


class VectorStore:
    """
    Weaviate Cloud vector store with multi-collection support.

    This class provides a wrapper around the Weaviate client for storing
    and searching document embeddings. Data is persisted in Weaviate Cloud,
    so it survives program restarts.

    Key features:
    - Multi-collection support (e.g., "Regulations", "Procedures")
    - Cloud persistence via Weaviate Cloud
    - Context manager support (with/as pattern)
    - Efficient HNSW-based vector search

    Attributes:
        client: Weaviate client instance
        url: Weaviate Cloud URL
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize connection to Weaviate Cloud.

        Args:
            url: Weaviate Cloud URL. If not provided, reads from WEAVIATE_URL env var.
            api_key: Weaviate API key. If not provided, reads from WEAVIATE_API_KEY env var.

        Raises:
            ImportError: If weaviate-client is not installed.
            ValueError: If URL or API key is not provided.

        Example:
            >>> store = VectorStore()
            >>> # Or with explicit credentials
            >>> store = VectorStore(
            ...     url="https://your-cluster.weaviate.network",
            ...     api_key="your-api-key"
            ... )
        """
        try:
            import weaviate
            from weaviate.classes.init import Auth
        except ImportError:
            raise ImportError(
                "weaviate-client is required. Install with: pip install weaviate-client>=4.4.0"
            )

        self.url = url or os.getenv("WEAVIATE_URL")
        self._api_key = api_key or os.getenv("WEAVIATE_API_KEY")

        if not self.url:
            raise ValueError(
                "Weaviate URL not provided. Set WEAVIATE_URL environment variable "
                "or pass url parameter."
            )

        if not self._api_key:
            raise ValueError(
                "Weaviate API key not provided. Set WEAVIATE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Connect to Weaviate Cloud
        logger.info(f"Connecting to Weaviate at {self.url}")

        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self._api_key),
        )

        logger.info("Successfully connected to Weaviate Cloud")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()
        return False

    def close(self) -> None:
        """
        Close the Weaviate connection.

        Always call this when done, or use the context manager.
        """
        if self.client:
            self.client.close()
            logger.info("Closed Weaviate connection")

    def create_collection(
        self,
        collection_name: str,
        description: str = "",
    ) -> bool:
        """
        Create a new collection in Weaviate.

        Collections are like tables - they store documents with similar schema.
        For example: "Regulations", "Procedures", "Policies".

        Args:
            collection_name: Name for the collection (will be capitalized).
            description: Optional description of what this collection contains.

        Returns:
            True if created, False if already exists.

        Example:
            >>> store.create_collection("Regulations", "Regulatory compliance documents")
        """
        from weaviate.classes.config import Configure, Property, DataType

        # Weaviate collection names must start with uppercase
        collection_name = collection_name.capitalize()

        # Check if collection already exists
        if self.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists")
            return False

        # Define collection schema
        # We store: content (text), metadata (as JSON string)
        # Vectors are stored separately by Weaviate
        self.client.collections.create(
            name=collection_name,
            description=description,
            vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The document text content",
                ),
                Property(
                    name="doc_id",
                    data_type=DataType.TEXT,
                    description="External document ID",
                ),
                Property(
                    name="metadata_json",
                    data_type=DataType.TEXT,
                    description="Document metadata as JSON",
                ),
            ],
        )

        logger.info(f"Created collection: {collection_name}")
        return True

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection.

        Returns:
            True if exists, False otherwise.
        """
        collection_name = collection_name.capitalize()
        return self.client.collections.exists(collection_name)

    def list_collections(self) -> list[str]:
        """
        List all collections in the Weaviate instance.

        Returns:
            List of collection names.
        """
        collections = self.client.collections.list_all()
        return list(collections.keys())

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection and all its documents.

        Warning: This is irreversible!

        Args:
            collection_name: Name of the collection to delete.

        Returns:
            True if deleted, False if didn't exist.
        """
        collection_name = collection_name.capitalize()

        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return False

        self.client.collections.delete(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return True

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Add documents with embeddings to a collection.

        Args:
            collection_name: Target collection name.
            documents: List of document text contents.
            embeddings: List of embedding vectors (must match documents length).
            metadatas: Optional list of metadata dicts for each document.
            ids: Optional list of document IDs. Auto-generated if not provided.

        Returns:
            List of document IDs that were added.

        Example:
            >>> store.add_documents(
            ...     collection_name="Regulations",
            ...     documents=["Text 1", "Text 2"],
            ...     embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            ...     metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}]
            ... )
        """
        import json
        import uuid

        collection_name = collection_name.capitalize()

        # Validate inputs
        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents ({len(documents)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        # Default metadata
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Get the collection
        collection = self.client.collections.get(collection_name)

        # Add documents in batch
        added_ids = []
        with collection.batch.dynamic() as batch:
            for doc_id, content, embedding, metadata in zip(
                ids, documents, embeddings, metadatas
            ):
                batch.add_object(
                    properties={
                        "content": content,
                        "doc_id": doc_id,
                        "metadata_json": json.dumps(metadata),
                    },
                    vector=embedding,
                )
                added_ids.append(doc_id)

        logger.info(f"Added {len(added_ids)} documents to '{collection_name}'")
        return added_ids

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents using vector similarity.

        Args:
            collection_name: Collection to search in.
            query_embedding: Query vector to find similar documents.
            k: Number of results to return (default: 5).

        Returns:
            List of result dicts, each containing:
                - id: Document ID
                - content: Document text
                - metadata: Document metadata
                - score: Similarity score (0-1, higher is better)

        Example:
            >>> results = store.search("Regulations", query_emb, k=3)
            >>> for r in results:
            ...     print(f"Score: {r['score']:.3f}, Content: {r['content'][:50]}")
        """
        import json
        from weaviate.classes.query import MetadataQuery

        collection_name = collection_name.capitalize()
        collection = self.client.collections.get(collection_name)

        # Perform vector search
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
        )

        # Format results
        results = []
        for obj in response.objects:
            # Weaviate returns distance, convert to similarity
            # For cosine distance: similarity = 1 - distance
            distance = obj.metadata.distance if obj.metadata.distance else 0
            similarity = 1 - distance

            # Parse metadata
            metadata = {}
            if obj.properties.get("metadata_json"):
                try:
                    metadata = json.loads(obj.properties["metadata_json"])
                except json.JSONDecodeError:
                    pass

            results.append({
                "id": obj.properties.get("doc_id", str(obj.uuid)),
                "content": obj.properties.get("content", ""),
                "metadata": metadata,
                "score": similarity,
            })

        logger.debug(
            f"Search in '{collection_name}' returned {len(results)} results "
            f"(top score: {results[0]['score']:.3f if results else 0})"
        )

        return results

    def get_collection_count(self, collection_name: str) -> int:
        """
        Get the number of documents in a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Number of documents in the collection.
        """
        collection_name = collection_name.capitalize()

        if not self.collection_exists(collection_name):
            return 0

        collection = self.client.collections.get(collection_name)
        response = collection.aggregate.over_all(total_count=True)
        return response.total_count or 0

    # =========================================================================
    # Legacy compatibility methods
    # =========================================================================

    def add_document(self, document: Document, collection_name: str = "Documents") -> None:
        """
        Add a single document (legacy compatibility).

        Args:
            document: Document object with content and embedding.
            collection_name: Target collection (default: "Documents").
        """
        # Create collection if it doesn't exist
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name, "Default document collection")

        self.add_documents(
            collection_name=collection_name,
            documents=[document.content],
            embeddings=[document.embedding],
            metadatas=[document.metadata],
            ids=[document.id],
        )

    def search_legacy(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection_name: str = "Documents",
    ) -> list[SearchResult]:
        """
        Search with legacy SearchResult format (backward compatibility).

        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            collection_name: Collection to search.

        Returns:
            List of SearchResult objects.
        """
        results = self.search(collection_name, query_embedding, k=top_k)

        return [
            SearchResult(
                document=Document(
                    id=r["id"],
                    content=r["content"],
                    embedding=[],  # Not returned by search
                    metadata=r["metadata"],
                ),
                score=r["score"],
            )
            for r in results
        ]


# =========================================================================
# In-Memory Vector Store (for testing/development without Weaviate)
# =========================================================================

class InMemoryVectorStore:
    """
    Simple in-memory vector store for testing and development.

    Use this when you don't have Weaviate configured or for quick testing.
    Data is lost when the program exits.

    Example:
        >>> store = InMemoryVectorStore()
        >>> store.add_document(Document(...))
        >>> results = store.search(query_embedding)
    """

    def __init__(self):
        """Initialize an empty in-memory store."""
        self.documents: dict[str, Document] = {}
        logger.info("Initialized InMemoryVectorStore (data will not persist)")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def close(self) -> None:
        """No-op for in-memory store."""
        pass

    def add_document(self, document: Document) -> None:
        """Add a document to the store."""
        self.documents[document.id] = document
        logger.debug(f"Added document: {document.id}")

    def add_documents_batch(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add multiple documents."""
        import uuid

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        if metadatas is None:
            metadatas = [{} for _ in documents]

        for doc_id, content, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            self.add_document(Document(
                id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata,
            ))

        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        from .embeddings import cosine_similarity

        if not self.documents:
            return []

        results = []
        for doc in self.documents.values():
            score = cosine_similarity(query_embedding, doc.embedding)
            results.append(SearchResult(document=doc, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self.documents)

    def clear(self) -> None:
        """Clear all documents."""
        self.documents.clear()
