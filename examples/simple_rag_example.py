"""
Simple RAG Example with Weaviate Cloud.

This example demonstrates the RAG (Retrieval-Augmented Generation) pipeline
using Weaviate Cloud for persistent vector storage:

1. Connect to Weaviate Cloud
2. Create a collection (if it doesn't exist)
3. Add documents with embeddings (on first run)
4. Query the vector store (data persists across runs!)
5. Use retrieved context to generate an answer

Run this example with:
    python examples/simple_rag_example.py

Make sure you have:
1. Set up your .env file with AWS and Weaviate credentials
2. Installed dependencies with: uv pip install -e .
3. A Weaviate Cloud account (https://console.weaviate.cloud/)

Key feature: Data persists in Weaviate Cloud! Run this script multiple times
and notice that documents are only added on the first run.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_ai.utils.config import get_config
from agentic_ai.llm.bedrock_client import BedrockClient, extract_text_from_response
from agentic_ai.rag.embeddings import get_embeddings_batch
from agentic_ai.rag.vector_store import VectorStore, InMemoryVectorStore, Document
from agentic_ai.rag.retriever import Retriever

# Collection name for this example
COLLECTION_NAME = "AiConcepts"


def get_sample_documents() -> list[dict]:
    """Return sample documents about AI concepts."""
    return [
        {
            "id": "ml_intro",
            "content": "Machine learning is a subset of artificial intelligence that enables "
                       "systems to learn and improve from experience without being explicitly "
                       "programmed. It focuses on developing algorithms that can access data "
                       "and use it to learn for themselves.",
            "metadata": {"source": "ml_intro.txt", "topic": "machine learning"}
        },
        {
            "id": "deep_learning",
            "content": "Deep learning is a type of machine learning that uses neural networks "
                       "with many layers (deep neural networks). It has been particularly "
                       "successful in image recognition, natural language processing, and "
                       "speech recognition tasks.",
            "metadata": {"source": "deep_learning.txt", "topic": "deep learning"}
        },
        {
            "id": "nlp_intro",
            "content": "Natural Language Processing (NLP) is a field of AI that focuses on "
                       "the interaction between computers and humans through natural language. "
                       "The ultimate goal of NLP is to enable computers to understand, interpret, "
                       "and generate human language in a valuable way.",
            "metadata": {"source": "nlp_intro.txt", "topic": "NLP"}
        },
        {
            "id": "rl_intro",
            "content": "Reinforcement learning is a type of machine learning where an agent "
                       "learns to make decisions by taking actions in an environment to maximize "
                       "cumulative reward. It's commonly used in robotics, game playing, and "
                       "autonomous systems.",
            "metadata": {"source": "rl_intro.txt", "topic": "reinforcement learning"}
        },
        {
            "id": "transfer_learning",
            "content": "Transfer learning is a machine learning technique where a model trained "
                       "on one task is re-purposed on a second related task. This is particularly "
                       "useful when you have limited data for the target task but abundant data "
                       "for a related source task.",
            "metadata": {"source": "transfer_learning.txt", "topic": "transfer learning"}
        },
    ]


def run_with_weaviate():
    """Run the RAG example with Weaviate Cloud (persistent storage)."""
    print("=" * 60)
    print("RAG Example with Weaviate Cloud")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration...")
    config = get_config()
    print(f"   AWS Region: {config.aws_region}")
    print(f"   LLM Model: {config.bedrock_model_id}")
    print(f"   Embedding Model: {config.embedding_model_id}")

    # Connect to Weaviate
    print("\n2. Connecting to Weaviate Cloud...")
    try:
        store = VectorStore()
        print("   Connected successfully!")
    except ValueError as e:
        print(f"   Error: {e}")
        print("   Make sure WEAVIATE_URL and WEAVIATE_API_KEY are set in .env")
        print("\n   Falling back to in-memory store for demonstration...")
        return run_with_inmemory()

    try:
        # Check if collection exists
        print(f"\n3. Checking collection '{COLLECTION_NAME}'...")
        collection_exists = store.collection_exists(COLLECTION_NAME)

        if collection_exists:
            doc_count = store.get_collection_count(COLLECTION_NAME)
            print(f"   Collection exists with {doc_count} documents")
            print("   (Data persisted from previous run!)")
        else:
            print(f"   Collection does not exist, creating it...")
            store.create_collection(COLLECTION_NAME, "AI and ML concepts for learning")
            print(f"   Created collection: {COLLECTION_NAME}")

            # Add documents
            print("\n4. Adding sample documents...")
            sample_docs = get_sample_documents()

            # Generate embeddings
            print("   Generating embeddings (calling AWS Bedrock Titan)...")
            texts = [doc["content"] for doc in sample_docs]
            embeddings = get_embeddings_batch(texts, config=config, show_progress=False)
            print(f"   Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")

            # Add to Weaviate
            print("   Adding documents to Weaviate...")
            store.add_documents(
                collection_name=COLLECTION_NAME,
                documents=texts,
                embeddings=embeddings,
                metadatas=[doc["metadata"] for doc in sample_docs],
                ids=[doc["id"] for doc in sample_docs],
            )
            print(f"   Added {len(sample_docs)} documents")

        # List all collections
        print("\n5. Listing all collections...")
        collections = store.list_collections()
        for coll in collections:
            count = store.get_collection_count(coll)
            print(f"   - {coll}: {count} documents")

        # Create retriever
        print("\n6. Creating retriever...")
        retriever = Retriever(store, collection_name=COLLECTION_NAME, config=config)

        # Test query
        query = "How does deep learning differ from regular machine learning?"
        print(f"\n7. Testing retrieval:")
        print(f"   Query: \"{query}\"")

        results = retriever.retrieve(query, top_k=3)

        print(f"\n   Found {len(results)} relevant documents:")
        for i, result in enumerate(results, 1):
            print(f"\n   --- Result {i} (Score: {result['score']:.4f}) ---")
            print(f"   Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"   Content: {result['content'][:150]}...")

        # Generate answer using retrieved context
        print("\n8. Generating answer with LLM (AWS Bedrock Claude)...")

        context = "\n\n".join([
            f"[{r['metadata'].get('source', 'Unknown')}]: {r['content']}"
            for r in results
        ])

        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        try:
            client = BedrockClient(config=config)
            response = client.invoke(
                prompt=prompt,
                system_prompt="You are a helpful AI assistant. Answer questions based on the provided context.",
                max_tokens=500,
                temperature=0.3,
            )

            answer = extract_text_from_response(response)
            print(f"\n   Answer:\n   {answer}")

        except Exception as e:
            print(f"   Error generating answer: {e}")

        print("\n" + "=" * 60)
        print("RAG Example Complete!")
        print("=" * 60)
        print("\nNote: Your data is now persisted in Weaviate Cloud.")
        print("Run this script again to see that documents are not re-added!")

    finally:
        # Always close the Weaviate connection
        print("\n9. Closing Weaviate connection...")
        store.close()


def run_with_inmemory():
    """Run the RAG example with in-memory store (no persistence)."""
    print("=" * 60)
    print("RAG Example with In-Memory Store")
    print("=" * 60)
    print("\nNote: Using in-memory store - data will not persist!")

    config = get_config()
    sample_docs = get_sample_documents()

    print("\n1. Generating embeddings...")
    try:
        texts = [doc["content"] for doc in sample_docs]
        embeddings = get_embeddings_batch(texts, config=config, show_progress=False)
        print(f"   Generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Using mock embeddings for demonstration...")
        import random
        embeddings = [[random.random() for _ in range(1536)] for _ in sample_docs]

    print("\n2. Creating in-memory vector store...")
    store = InMemoryVectorStore()

    for doc, embedding in zip(sample_docs, embeddings):
        store.add_document(Document(
            id=doc["id"],
            content=doc["content"],
            embedding=embedding,
            metadata=doc["metadata"],
        ))
    print(f"   Added {len(store)} documents")

    print("\n3. Testing retrieval...")
    query = "How does deep learning differ from regular machine learning?"
    print(f"   Query: \"{query}\"")

    try:
        from agentic_ai.rag.embeddings import get_embedding
        query_embedding = get_embedding(query, config=config)
    except Exception:
        import random
        query_embedding = [random.random() for _ in range(1536)]

    results = store.search(query_embedding, top_k=3)

    print(f"\n   Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. Score: {result.score:.4f} - {result.document.metadata.get('source', 'Unknown')}")

    print("\n" + "=" * 60)
    print("In-Memory RAG Example Complete!")
    print("=" * 60)


def main():
    """Run the RAG example."""
    print("\nThis example demonstrates RAG with Weaviate Cloud.")
    print("If Weaviate is not configured, it falls back to in-memory storage.\n")

    run_with_weaviate()


if __name__ == "__main__":
    main()
