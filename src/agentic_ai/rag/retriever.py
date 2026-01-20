"""
Retriever module for RAG systems.

The Retriever ties together embeddings and vector stores to provide
a simple interface for semantic search. It handles:
1. Embedding the query
2. Searching the vector store
3. Formatting results for use in prompts

This is the "R" in RAG - the retrieval step that finds relevant
context before generating a response.

Example usage:
    from agentic_ai.rag.retriever import Retriever
    from agentic_ai.rag.vector_store import VectorStore

    # Setup with Weaviate
    with VectorStore() as store:
        retriever = Retriever(store, collection_name="Regulations")

        # Simple retrieval
        results = retriever.retrieve("What is machine learning?", top_k=3)
        for result in results:
            print(f"Score: {result['score']:.3f}")
            print(f"Content: {result['content']}")

        # Get formatted context for prompt
        context = retriever.get_context_for_prompt("What is ML?")
        prompt = f"Context:\\n{context}\\n\\nQuestion: What is ML?"
"""

import logging
from typing import Any, Optional, Union

from .embeddings import get_embedding
from .vector_store import VectorStore, InMemoryVectorStore, SearchResult
from ..utils.config import Config

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retriever for semantic search over a vector store.

    The retriever provides a high-level interface for finding relevant
    documents based on semantic similarity. It:
    1. Converts queries to embeddings
    2. Searches the vector store
    3. Returns formatted results

    Works with both Weaviate VectorStore and InMemoryVectorStore.

    Attributes:
        vector_store: The underlying vector store.
        collection_name: Name of the collection to search (for Weaviate).
        config: Configuration for embeddings.
        default_top_k: Default number of results to return.
    """

    def __init__(
        self,
        vector_store: Union[VectorStore, InMemoryVectorStore],
        collection_name: str = "Documents",
        config: Optional[Config] = None,
        default_top_k: int = 5,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore or InMemoryVectorStore instance.
            collection_name: Collection to search (for Weaviate, default: "Documents").
            config: Optional Config for embedding API.
            default_top_k: Default number of results (default: 5).
        """
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.config = config
        self.default_top_k = default_top_k

        # Determine store type for logging
        store_type = type(vector_store).__name__
        logger.info(f"Initialized Retriever with {store_type}, collection: {collection_name}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        collection_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        This is the main retrieval method. It:
        1. Embeds the query using the same model as documents
        2. Searches the vector store for similar documents
        3. Returns results with content, metadata, and scores

        Args:
            query: The search query (natural language).
            top_k: Number of results to return (default: self.default_top_k).
            min_score: Minimum similarity score to include (default: 0.0).
            collection_name: Override the default collection (Weaviate only).

        Returns:
            List of dicts, each containing:
                - id: Document ID
                - content: Document text
                - metadata: Document metadata
                - score: Similarity score (0-1)

        Example:
            >>> retriever = Retriever(store, collection_name="Regulations")
            >>> results = retriever.retrieve("How do neural networks learn?")
            >>> for r in results:
            ...     print(f"[{r['score']:.2f}] {r['content'][:100]}...")
        """
        top_k = top_k or self.default_top_k
        collection = collection_name or self.collection_name

        logger.info(f"Retrieving top {top_k} results for: {query[:50]}...")

        # Step 1: Embed the query
        query_embedding = get_embedding(query, config=self.config)

        # Step 2: Search the vector store
        # Handle different store types
        if isinstance(self.vector_store, VectorStore):
            # Weaviate store - uses collection-based search
            search_results = self.vector_store.search(
                collection_name=collection,
                query_embedding=query_embedding,
                k=top_k,
            )
            # Results are already in dict format
            results = [r for r in search_results if r["score"] >= min_score]

        elif isinstance(self.vector_store, InMemoryVectorStore):
            # In-memory store - uses legacy SearchResult format
            search_results: list[SearchResult] = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )
            results = [
                {
                    "id": result.document.id,
                    "content": result.document.content,
                    "metadata": result.document.metadata,
                    "score": result.score,
                }
                for result in search_results
                if result.score >= min_score
            ]
        else:
            raise TypeError(f"Unsupported vector store type: {type(self.vector_store)}")

        logger.info(f"Retrieved {len(results)} results")
        return results

    def get_context_for_prompt(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        include_metadata: bool = True,
        separator: str = "\n\n---\n\n",
        collection_name: Optional[str] = None,
    ) -> str:
        """
        Get formatted context string for use in a prompt.

        This is a convenience method that retrieves documents and
        formats them into a single string suitable for including
        in an LLM prompt.

        Args:
            query: The search query.
            top_k: Number of documents to include.
            min_score: Minimum similarity score.
            include_metadata: If True, include source info (default: True).
            separator: String to separate documents (default: "---").
            collection_name: Override the default collection.

        Returns:
            Formatted string with all retrieved documents.

        Example:
            >>> context = retriever.get_context_for_prompt("What is AI?")
            >>> prompt = f'''Based on the following context:
            ... {context}
            ...
            ... Answer: What is AI?'''
        """
        results = self.retrieve(
            query,
            top_k=top_k,
            min_score=min_score,
            collection_name=collection_name,
        )

        if not results:
            return "No relevant documents found."

        # Format each result
        formatted_docs = []
        for i, result in enumerate(results, 1):
            doc_str = f"[Document {i}]"

            if include_metadata and result["metadata"]:
                # Add relevant metadata
                source = result["metadata"].get("source", "Unknown")
                page = result["metadata"].get("page")
                if page:
                    doc_str += f" (Source: {source}, Page: {page})"
                else:
                    doc_str += f" (Source: {source})"

            doc_str += f"\n{result['content']}"
            formatted_docs.append(doc_str)

        return separator.join(formatted_docs)

    def retrieve_with_scores_verbose(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Retrieve and print results with verbose output.

        Useful for debugging and understanding retrieval quality.

        Args:
            query: The search query.
            top_k: Number of results.
            collection_name: Override the default collection.
        """
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"Collection: {collection_name or self.collection_name}")
        print(f"{'='*60}")

        results = self.retrieve(query, top_k=top_k, collection_name=collection_name)

        if not results:
            print("No results found.")
            return

        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} (Score: {result['score']:.4f}) ---")
            if result["metadata"]:
                print(f"Metadata: {result['metadata']}")
            print(f"Content: {result['content'][:500]}...")

        print(f"\n{'='*60}\n")


class MultiCollectionRetriever:
    """
    Retriever that searches across multiple collections.

    Useful when you want to search both "Regulations" and "Procedures"
    collections and combine the results.

    Example:
        >>> retriever = MultiCollectionRetriever(
        ...     store,
        ...     collections=["Regulations", "Procedures"]
        ... )
        >>> results = retriever.retrieve("safety requirements")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        collections: list[str],
        config: Optional[Config] = None,
        default_top_k: int = 5,
    ):
        """
        Initialize multi-collection retriever.

        Args:
            vector_store: Weaviate VectorStore instance.
            collections: List of collection names to search.
            config: Optional Config for embeddings.
            default_top_k: Default results per collection.
        """
        self.vector_store = vector_store
        self.collections = collections
        self.config = config
        self.default_top_k = default_top_k

        logger.info(f"Initialized MultiCollectionRetriever for: {collections}")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Search across all collections and combine results.

        Results are merged and re-ranked by score.

        Args:
            query: The search query.
            top_k: Total number of results to return.
            min_score: Minimum similarity score.

        Returns:
            Combined results from all collections, sorted by score.
        """
        top_k = top_k or self.default_top_k

        logger.info(f"Searching {len(self.collections)} collections for: {query[:50]}...")

        # Embed query once
        query_embedding = get_embedding(query, config=self.config)

        # Search each collection
        all_results = []
        for collection in self.collections:
            try:
                results = self.vector_store.search(
                    collection_name=collection,
                    query_embedding=query_embedding,
                    k=top_k,
                )
                # Add collection name to metadata
                for r in results:
                    r["metadata"]["_collection"] = collection
                    if r["score"] >= min_score:
                        all_results.append(r)
            except Exception as e:
                logger.warning(f"Error searching collection '{collection}': {e}")

        # Sort by score and return top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"Found {len(all_results)} total results across collections")
        return all_results[:top_k]

    def get_context_for_prompt(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        include_metadata: bool = True,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Get formatted context from all collections.

        Args:
            query: The search query.
            top_k: Number of documents to include.
            min_score: Minimum similarity score.
            include_metadata: If True, include source info.
            separator: String to separate documents.

        Returns:
            Formatted string with retrieved documents.
        """
        results = self.retrieve(query, top_k=top_k, min_score=min_score)

        if not results:
            return "No relevant documents found."

        formatted_docs = []
        for i, result in enumerate(results, 1):
            collection = result["metadata"].get("_collection", "Unknown")
            doc_str = f"[Document {i} - {collection}]"

            if include_metadata:
                source = result["metadata"].get("source", "Unknown")
                page = result["metadata"].get("page")
                if page:
                    doc_str += f" (Source: {source}, Page: {page})"
                else:
                    doc_str += f" (Source: {source})"

            doc_str += f"\n{result['content']}"
            formatted_docs.append(doc_str)

        return separator.join(formatted_docs)
