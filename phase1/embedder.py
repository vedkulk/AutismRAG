"""
phase1/embedder.py
──────────────────
Generates embeddings for TextChunks using Ollama's nomic-embed-text model.
Handles batching, retries, and progress tracking.

Before using:
    ollama pull nomic-embed-text

Usage:
    from phase1.embedder import OllamaEmbedder
    embedder = OllamaEmbedder()
    vectors = embedder.embed_texts(["text1", "text2"])
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """
    Wraps Ollama's embedding API via LangChain's OllamaEmbeddings.

    Adds:
      - Batch processing with configurable batch size
      - Retry logic on transient failures
      - Embedding dimension validation
      - Progress logging
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Args:
            model:       Ollama embedding model name
            base_url:    Ollama server URL
            batch_size:  Texts to embed per API call
            max_retries: Retry attempts on failure
            retry_delay: Seconds between retries
        """
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None
        self._embedding_dim: Optional[int] = None

    def _get_client(self):
        """Lazily initialise the LangChain OllamaEmbeddings client."""
        if self._client is None:
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError:
                from langchain_community.embeddings import OllamaEmbeddings

            self._client = OllamaEmbeddings(
                model=self.model,
                base_url=self.base_url,
            )
            logger.info(
                f"Initialized OllamaEmbedder: model={self.model}, "
                f"server={self.base_url}"
            )
        return self._client

    def check_connection(self) -> bool:
        """
        Verify Ollama is running and the embedding model is available.
        Call this before starting a long ingestion job.
        """
        import urllib.request
        import json

        try:
            url = f"{self.base_url}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read())

            available = [m["name"] for m in data.get("models", [])]
            model_base = self.model.split(":")[0]

            if not any(model_base in m for m in available):
                logger.error(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}\n"
                    f"Available models: {available}"
                )
                return False

            logger.info(f"Ollama connection OK. Model '{self.model}' available.")
            return True

        except Exception as e:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url}: {e}\n"
                f"Make sure Ollama is running: ollama serve"
            )
            return False

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.

        Processes in batches. Retries each batch on failure.

        Returns:
            List of embedding vectors (one per text)
        """
        client = self._get_client()
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_num in range(total_batches):
            start = batch_num * self.batch_size
            end = min(start + self.batch_size, len(texts))
            batch = texts[start:end]

            logger.debug(
                f"Embedding batch {batch_num + 1}/{total_batches} "
                f"({len(batch)} texts)"
            )

            for attempt in range(1, self.max_retries + 1):
                try:
                    batch_embeddings = client.embed_documents(batch)

                    # Validate dimensions on first batch
                    if self._embedding_dim is None and batch_embeddings:
                        self._embedding_dim = len(batch_embeddings[0])
                        logger.info(
                            f"Embedding dimension: {self._embedding_dim} "
                            f"(model: {self.model})"
                        )

                    all_embeddings.extend(batch_embeddings)
                    break  # Success

                except Exception as e:
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Embedding batch {batch_num + 1} failed "
                            f"(attempt {attempt}/{self.max_retries}): {e}. "
                            f"Retrying in {self.retry_delay}s..."
                        )
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(
                            f"Embedding batch {batch_num + 1} failed after "
                            f"{self.max_retries} attempts: {e}"
                        )
                        raise RuntimeError(
                            f"Embedding failed for batch {batch_num + 1}: {e}"
                        ) from e

        logger.info(f"Embedded {len(all_embeddings)}/{len(texts)} texts successfully")
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.
        Used at retrieval time (Phase 2+).
        """
        client = self._get_client()
        return client.embed_query(query)

    @property
    def embedding_dimension(self) -> Optional[int]:
        """Returns embedding dim after first embed_texts() call."""
        return self._embedding_dim

    def get_langchain_embeddings(self):
        """
        Return the raw LangChain embeddings object.
        Used when passing directly to ChromaDB / LangChain chains.
        """
        return self._get_client()

