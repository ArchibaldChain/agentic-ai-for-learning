"""
Embeddings module using AWS Bedrock Titan.

This module provides functions to generate text embeddings using
Amazon Titan Embeddings model via the Bedrock API.

Embeddings are vector representations of text that capture semantic meaning.
Similar texts will have similar embeddings (close in vector space).
This is the foundation of semantic search in RAG systems.

How embeddings work:
1. Text goes in (e.g., "The cat sat on the mat")
2. A vector comes out (e.g., [0.1, -0.3, 0.5, ...] with 1536 dimensions)
3. Similar texts have vectors that are close together
4. We can use cosine similarity to measure how close two vectors are

Example usage:
    from agentic_ai.rag.embeddings import get_embedding, get_embeddings_batch

    # Single embedding
    embedding = get_embedding("Hello, world!")
    print(f"Embedding dimension: {len(embedding)}")

    # Batch embeddings
    texts = ["First document", "Second document", "Third document"]
    embeddings = get_embeddings_batch(texts)
"""

import json
import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from ..utils.config import get_config, Config

logger = logging.getLogger(__name__)


def get_embedding(
    text: str,
    config: Optional[Config] = None,
    normalize: bool = True,
) -> list[float]:
    """
    Get the embedding vector for a single text.

    This function calls the Amazon Titan Embeddings model to convert
    text into a dense vector representation.

    Args:
        text: The text to embed. Should be non-empty.
        config: Optional Config object. If not provided, loads from environment.
        normalize: If True, normalize the embedding to unit length (default: True).
                   Normalized embeddings work better with cosine similarity.

    Returns:
        List of floats representing the embedding vector.
        For Titan Embeddings v1, this is 1536 dimensions.

    Raises:
        ValueError: If text is empty.
        ClientError: If the Bedrock API call fails.

    Example:
        >>> embedding = get_embedding("Hello, how are you?")
        >>> print(f"Dimension: {len(embedding)}")
        Dimension: 1536
        >>> print(f"First few values: {embedding[:5]}")
        First few values: [0.123, -0.456, 0.789, ...]
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    config = config or get_config()

    # Create Bedrock client
    session_kwargs = {"region_name": config.aws_region}
    if config.aws_access_key_id and config.aws_secret_access_key:
        session_kwargs["aws_access_key_id"] = config.aws_access_key_id
        session_kwargs["aws_secret_access_key"] = config.aws_secret_access_key

    client = boto3.client("bedrock-runtime", **session_kwargs)

    # Prepare the request body for Titan Embeddings
    # The format is specific to the Titan model
    request_body = {
        "inputText": text,
    }

    # Add normalization if supported (Titan v2 supports this natively)
    # For v1, we'll normalize manually
    if "v2" in config.embedding_model_id:
        request_body["normalize"] = normalize

    logger.debug(f"Getting embedding for text: {text[:50]}...")

    try:
        # Call the Bedrock API
        # Note: Titan uses invoke_model, not the converse API
        response = client.invoke_model(
            modelId=config.embedding_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body),
        )

        # Parse the response
        response_body = json.loads(response["body"].read())

        # Extract the embedding
        embedding = response_body["embedding"]

        logger.debug(f"Got embedding with dimension: {len(embedding)}")

        # Normalize if requested and not done by the model
        if normalize and "v2" not in config.embedding_model_id:
            embedding = _normalize_vector(embedding)

        return embedding

    except ClientError as e:
        logger.error(f"Error getting embedding: {e}")
        raise


def get_embeddings_batch(
    texts: list[str],
    config: Optional[Config] = None,
    normalize: bool = True,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Get embeddings for multiple texts.

    This function processes texts one at a time (Titan doesn't support batching).
    For large batches, consider using async or parallel processing.

    Args:
        texts: List of texts to embed.
        config: Optional Config object.
        normalize: If True, normalize embeddings to unit length.
        show_progress: If True, print progress updates.

    Returns:
        List of embedding vectors, one per input text.

    Example:
        >>> texts = ["First doc", "Second doc", "Third doc"]
        >>> embeddings = get_embeddings_batch(texts)
        >>> print(f"Got {len(embeddings)} embeddings")
        Got 3 embeddings
    """
    config = config or get_config()
    embeddings = []

    for i, text in enumerate(texts):
        if show_progress and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(texts)} embeddings...")

        embedding = get_embedding(text, config=config, normalize=normalize)
        embeddings.append(embedding)

    if show_progress:
        print(f"Completed {len(embeddings)} embeddings")

    return embeddings


def _normalize_vector(vector: list[float]) -> list[float]:
    """
    Normalize a vector to unit length.

    Unit vectors are important for cosine similarity because:
    - Cosine similarity of unit vectors = dot product
    - This makes computation faster and more stable

    Args:
        vector: The vector to normalize.

    Returns:
        Normalized vector with length 1.
    """
    import math

    # Calculate the magnitude (length) of the vector
    magnitude = math.sqrt(sum(x * x for x in vector))

    # Avoid division by zero
    if magnitude == 0:
        return vector

    # Divide each component by the magnitude
    return [x / magnitude for x in vector]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors:
    - 1.0 means identical direction (most similar)
    - 0.0 means perpendicular (unrelated)
    - -1.0 means opposite direction (most dissimilar)

    For normalized vectors, this is just the dot product.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity score between -1 and 1.

    Example:
        >>> vec1 = [1, 0, 0]
        >>> vec2 = [1, 0, 0]
        >>> cosine_similarity(vec1, vec2)
        1.0
        >>> vec3 = [0, 1, 0]
        >>> cosine_similarity(vec1, vec3)
        0.0
    """
    import math

    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate magnitudes
    magnitude1 = math.sqrt(sum(x * x for x in vec1))
    magnitude2 = math.sqrt(sum(x * x for x in vec2))

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Cosine similarity formula
    return dot_product / (magnitude1 * magnitude2)
