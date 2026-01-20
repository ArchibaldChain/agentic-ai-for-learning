"""
Document Parser using LangChain loaders and splitters.

This module provides document loading and chunking functionality.
It uses LangChain's document loaders (for convenience) but exposes
a simple interface that hides the LangChain complexity.

Supported formats:
- PDF files (.pdf)
- Word documents (.docx)
- Text files (.txt)
- Markdown files (.md)

The chunking strategy uses RecursiveCharacterTextSplitter which:
1. Tries to split on paragraph boundaries first
2. Then sentences
3. Then words
4. Ensures chunks don't exceed max size
5. Adds overlap between chunks for context continuity

Example usage:
    from agentic_ai.tools.document_parser import parse_and_chunk

    # Parse and chunk a PDF
    chunks = parse_and_chunk(
        "path/to/document.pdf",
        chunk_size=1000,
        chunk_overlap=200
    )

    for chunk in chunks:
        print(f"Chunk {chunk['metadata']['chunk_index']}:")
        print(chunk['content'][:100])
        print(f"Source: {chunk['metadata']['source']}")
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """
    A chunk of a document with its content and metadata.

    Attributes:
        content: The text content of the chunk.
        metadata: Dict with source info, page numbers, chunk index, etc.
    """
    content: str
    metadata: dict[str, Any]


class DocumentParser:
    """
    Parser for various document formats.

    This class provides a unified interface for loading documents
    of different types. It uses LangChain loaders internally but
    exposes a simple API.

    Supported formats:
    - .pdf: PDF documents
    - .docx: Microsoft Word documents
    - .txt: Plain text files
    - .md: Markdown files

    Example:
        >>> parser = DocumentParser()
        >>> pages = parser.load("report.pdf")
        >>> for page in pages:
        ...     print(f"Page {page.metadata.get('page', 0)}: {page.content[:50]}...")
    """

    def __init__(self):
        """Initialize the document parser."""
        logger.info("Initialized DocumentParser")

    def load(self, file_path: str) -> list[DocumentChunk]:
        """
        Load a document and return its pages/sections.

        Args:
            file_path: Path to the document file.

        Returns:
            List of DocumentChunk objects, one per page/section.

        Raises:
            ValueError: If file type is not supported.
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix == ".docx":
            return self._load_docx(file_path)
        elif suffix in [".txt", ".md"]:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _load_pdf(self, file_path: str) -> list[DocumentChunk]:
        """Load a PDF file using LangChain's PyPDFLoader."""
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            raise ImportError(
                "langchain-community is required for PDF loading. "
                "Install with: pip install langchain-community pypdf"
            )

        logger.info(f"Loading PDF: {file_path}")

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Convert LangChain documents to our format
        chunks = []
        for i, page in enumerate(pages):
            chunk = DocumentChunk(
                content=page.page_content,
                metadata={
                    "source": file_path,
                    "page": i + 1,  # 1-indexed for human readability
                    "total_pages": len(pages),
                    "file_type": "pdf",
                }
            )
            chunks.append(chunk)

        logger.info(f"Loaded {len(chunks)} pages from PDF")
        return chunks

    def _load_docx(self, file_path: str) -> list[DocumentChunk]:
        """Load a Word document using LangChain's Docx2txtLoader."""
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError:
            raise ImportError(
                "langchain-community is required for DOCX loading. "
                "Install with: pip install langchain-community python-docx"
            )

        logger.info(f"Loading DOCX: {file_path}")

        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        # DOCX loader returns single document
        chunks = []
        for doc in docs:
            chunk = DocumentChunk(
                content=doc.page_content,
                metadata={
                    "source": file_path,
                    "file_type": "docx",
                }
            )
            chunks.append(chunk)

        logger.info(f"Loaded DOCX document")
        return chunks

    def _load_text(self, file_path: str) -> list[DocumentChunk]:
        """Load a text or markdown file."""
        logger.info(f"Loading text file: {file_path}")

        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        chunk = DocumentChunk(
            content=content,
            metadata={
                "source": file_path,
                "file_type": path.suffix.lstrip("."),
            }
        )

        return [chunk]


class TextSplitter:
    """
    Split text into chunks with overlap.

    This implements a recursive splitting strategy that:
    1. Tries to split on larger boundaries first (paragraphs)
    2. Falls back to smaller boundaries (sentences, words)
    3. Ensures chunks don't exceed max size
    4. Adds overlap between chunks for context

    The overlap is important for RAG because:
    - Information might span chunk boundaries
    - Overlap ensures we don't lose context at boundaries

    Example:
        >>> splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        >>> chunks = splitter.split_text("Long document text...")
    """

    # Characters to try splitting on, in order of preference
    # We prefer to split on natural boundaries
    SEPARATORS = [
        "\n\n",  # Paragraph breaks (best)
        "\n",    # Line breaks
        ". ",    # Sentences
        ", ",    # Clauses
        " ",     # Words (last resort)
        "",      # Characters (emergency fallback)
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
    ):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Maximum size of each chunk in characters (default: 1000).
            chunk_overlap: Number of characters to overlap between chunks (default: 200).
            separators: Custom list of separators to try.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.SEPARATORS

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        logger.info(
            f"Initialized TextSplitter (size={chunk_size}, overlap={chunk_overlap})"
        )

    def split_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        chunks = []

        # Find the best separator that exists in the text
        separator = ""
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # Split into characters

        # Merge small splits into chunks
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split) + (len(separator) if current_chunk else 0)

            if current_length + split_length <= self.chunk_size:
                # Add to current chunk
                current_chunk.append(split)
                current_length += split_length
            else:
                # Current chunk is full
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    chunks.append(chunk_text)

                    # Start new chunk with overlap
                    # Keep the end of the previous chunk for context
                    overlap_text = self._get_overlap(chunk_text)
                    if overlap_text:
                        current_chunk = [overlap_text, split]
                        current_length = len(overlap_text) + len(separator) + len(split)
                    else:
                        current_chunk = [split]
                        current_length = len(split)
                else:
                    # Single split is too large, need to split further
                    if separators.index(separator) < len(separators) - 1:
                        # Try next separator
                        sub_chunks = self._split_text_recursive(
                            split, separators[separators.index(separator) + 1:]
                        )
                        chunks.extend(sub_chunks)
                    else:
                        # Can't split further, just add as is
                        chunks.append(split[:self.chunk_size])

                    current_chunk = []
                    current_length = 0

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def _get_overlap(self, text: str) -> str:
        """Get the overlap portion from the end of text."""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]


def parse_document(file_path: str) -> list[DocumentChunk]:
    """
    Parse a document file.

    Convenience function that creates a parser and loads the file.

    Args:
        file_path: Path to the document.

    Returns:
        List of DocumentChunk objects.

    Example:
        >>> chunks = parse_document("report.pdf")
    """
    parser = DocumentParser()
    return parser.load(file_path)


def parse_and_chunk(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """
    Parse a document and split it into chunks.

    This is the main function for preparing documents for RAG.
    It loads the document and splits the content into smaller
    overlapping chunks suitable for embedding.

    Args:
        file_path: Path to the document.
        chunk_size: Maximum chunk size in characters (default: 1000).
        chunk_overlap: Overlap between chunks (default: 200).

    Returns:
        List of dicts, each with 'content' and 'metadata' keys.

    Example:
        >>> chunks = parse_and_chunk("regulations.pdf", chunk_size=500)
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk['metadata']['chunk_index']}")
        ...     print(chunk['content'][:100])
    """
    # Load the document
    parser = DocumentParser()
    documents = parser.load(file_path)

    # Initialize splitter
    splitter = TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Split all documents
    all_chunks = []
    chunk_index = 0

    for doc in documents:
        text_chunks = splitter.split_text(doc.content)

        for text in text_chunks:
            # Skip empty chunks
            if not text.strip():
                continue

            chunk = {
                "content": text.strip(),
                "metadata": {
                    **doc.metadata,
                    "chunk_index": chunk_index,
                }
            }
            all_chunks.append(chunk)
            chunk_index += 1

    logger.info(
        f"Parsed {file_path}: {len(documents)} pages -> {len(all_chunks)} chunks"
    )

    return all_chunks


def load_directory(
    directory: str,
    pattern: str = "*",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict[str, Any]]:
    """
    Load and chunk all matching documents in a directory.

    Args:
        directory: Path to the directory.
        pattern: Glob pattern for files (default: "*" for all).
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of all chunks from all matching documents.

    Example:
        >>> # Load all PDFs in a directory
        >>> chunks = load_directory("data/regulations", pattern="*.pdf")
    """
    from pathlib import Path

    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    all_chunks = []
    supported_extensions = {".pdf", ".docx", ".txt", ".md"}

    for file_path in dir_path.glob(pattern):
        if file_path.suffix.lower() in supported_extensions:
            try:
                chunks = parse_and_chunk(
                    str(file_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                all_chunks.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

    logger.info(f"Loaded {len(all_chunks)} total chunks from {directory}")
    return all_chunks
