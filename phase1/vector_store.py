"""
phase1/vector_store.py
──────────────────────
Manages the persistent ChromaDB vector store for the medical knowledge base.

Handles:
  - Creating / loading a persistent collection
  - Upserting chunks (idempotent — won't duplicate on re-runs)
  - Semantic similarity and MMR retrieval
  - Collection stats and inspection utilities

Usage:
    from phase1.vector_store import KnowledgeBaseStore
    store = KnowledgeBaseStore()
    store.add_chunks(chunks)
    results = store.similarity_search("child with social communication issues", k=5)
"""

import logging
from pathlib import Path
from typing import Optional

from phase1.chunker import TextChunk
from phase1.embedder import OllamaEmbedder

logger = logging.getLogger(__name__)


class KnowledgeBaseStore:
    def __init__(
        self,
        persist_dir: str = "./data/chroma_kb",
        collection_name: str = "autism_kb_medical_guidelines",
        embedder: Optional[OllamaEmbedder] = None,
        distance_metric: str = "cosine",
    ):
        """
        Args:
            persist_dir:       Where ChromaDB stores its files on disk
            collection_name:   Name of the ChromaDB collection
            embedder:          OllamaEmbedder instance (created if None)
            distance_metric:   "cosine" | "l2" | "ip"
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.embedder = embedder or OllamaEmbedder()
        self._chroma_client = None
        self._collection = None
        self._langchain_store = None

    def _get_chroma_client(self):
        if self._chroma_client is None:
            import chromadb

            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info(f"ChromaDB client initialized at: {self.persist_dir}")
        return self._chroma_client

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            client = self._get_chroma_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )
            logger.info(
                f"Collection '{self.collection_name}' loaded: "
                f"{self._collection.count()} existing chunks"
            )
        return self._collection

    def _get_langchain_store(self):

        return None

    def count(self) -> int:
        """Number of chunks currently in the collection."""
        return self._get_collection().count()

    def add_chunks(
        self,
        chunks: list[TextChunk],
        show_progress: bool = True,
        batch_size: int = 50,
    ) -> int:

        if not chunks:
            logger.warning("No chunks to add.")
            return 0

        collection = self._get_collection()
        total = len(chunks)
        upserted = 0
        num_batches = (total + batch_size - 1) // batch_size

        logger.info(
            f"Adding {total} chunks to '{self.collection_name}' "
            f"in {num_batches} batches..."
        )

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch = chunks[start:end]

            if show_progress:
                logger.info(
                    f"  Batch {batch_idx + 1}/{num_batches}: "
                    f"chunks {start}–{end - 1}"
                )

            # Extract texts for embedding
            texts = [c.text for c in batch]

            # Embed via Ollama
            embeddings = self.embedder.embed_texts(texts)

            # Prepare data for ChromaDB upsert (store plaintext documents)
            ids = [c.chunk_id for c in batch]
            metadatas = [c.metadata for c in batch]

            # Upsert (insert or update by ID)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            upserted += len(batch)

        logger.info(
            f"Upsert complete. Collection now has "
            f"{collection.count()} chunks total."
        )
        # Invalidate LangChain wrapper so it picks up new data
        self._langchain_store = None

        return upserted

    def similarity_search(
        self,
        query: str,
        k: int = 6,
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        
        query_embedding = self.embedder.embed_query(query)
        collection = self._get_collection()

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = collection.query(**query_kwargs)

        output = []
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            output.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": round(dist, 4),
                    "similarity": round(1 - dist, 4),  # cosine: similarity = 1 - distance
                    "rank": i + 1,
                }
            )

        return output

    def mmr_search(
        self,
        query: str,
        k: int = 6,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[dict]:

        sims = self.similarity_search(query, k=k)
        # match previous return shape: no similarity distance, just rank, text, metadata
        out: list[dict] = []
        for i, s in enumerate(sims, start=1):
            out.append(
                {
                    "text": s["text"],
                    "metadata": s["metadata"],
                    "rank": i,
                }
            )
        return out

    def get_retriever(self, k: int = 6, search_type: str = "mmr"):
        """
        Stub in this project; high-level retrieval is handled directly
        via similarity_search / mmr_search wrappers.
        """
        raise NotImplementedError("Use similarity_search or mmr_search directly.")

    def get_stats(self) -> dict:
        """
        Return metadata statistics about the collection.
        Useful for verifying ingestion quality.
        """
        collection = self._get_collection()
        count = collection.count()

        if count == 0:
            return {"total_chunks": 0}

        # Sample metadata to get source file distribution
        sample = collection.get(
            limit=min(count, 5000),
            include=["metadatas"],
        )

        from collections import Counter

        files = Counter()
        sections = Counter()

        for meta in sample["metadatas"]:
            files[meta.get("filename", "unknown")] += 1
            section = meta.get("section", "")
            if section:
                sections[section] += 1

        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "source_files": dict(files.most_common()),
            "top_sections": dict(sections.most_common(10)),
        }

    def delete_collection(self) -> None:
        """
        ⚠️ Delete the entire collection. Irreversible.
        Used for resetting during development.
        """
        client = self._get_chroma_client()
        client.delete_collection(self.collection_name)
        self._collection = None
        self._langchain_store = None
        logger.warning(f"Collection '{self.collection_name}' deleted.")

    def print_stats(self) -> None:
        """Print a human-readable summary of the vector store."""
        stats = self.get_stats()
        print("\n─── Knowledge Base Vector Store ────────────────────")
        print(f"  Collection     : {stats.get('collection_name', 'N/A')}")
        print(f"  Persist dir    : {stats.get('persist_dir', 'N/A')}")
        print(f"  Total chunks   : {stats.get('total_chunks', 0):,}")
        files = stats.get("source_files", {})
        if files:
            print(f"\n  Chunks per source file:")
            for fname, count in files.items():
                print(f"    {fname:<45} {count:>5}")
        sections = stats.get("top_sections", {})
        if sections:
            print(f"\n  Top sections in KB:")
            for section, count in list(sections.items())[:6]:
                print(f"    {section:<45} {count:>5}")
        print("────────────────────────────────────────────────────\n")

