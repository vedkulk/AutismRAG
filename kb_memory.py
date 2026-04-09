from __future__ import annotations

"""
In-memory medical knowledge base (KB) built from Phase 1 PDFs.

Supports hybrid retrieval:
  - Dense: cosine similarity via nomic-embed-text embeddings
  - Sparse: BM25 keyword matching for clinical terms (ADOS-2, DSM-5, etc.)
  - Fusion: Reciprocal Rank Fusion (RRF) to merge dense + sparse results

Caches parsed chunks + embeddings to disk to avoid re-embedding on every startup.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from phase1.pdf_parser import extract_documents_from_dir, ParsedDocument
from phase1.chunker import chunk_documents, TextChunk
from phase1.embedder import OllamaEmbedder

logger = logging.getLogger(__name__)

# RRF constant (standard value from the original RRF paper)
RRF_K = 60


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+(?:[-']\w+)*", text.lower())


def _compute_dir_hash(data_dir: str) -> str:
    """Hash filenames + sizes to detect when source docs change."""
    entries = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            fp = os.path.join(root, f)
            entries.append(f"{f}:{os.path.getsize(fp)}")
    return hashlib.md5("|".join(entries).encode()).hexdigest()


@dataclass
class KBEntry:
    text: str
    metadata: Dict[str, Any]
    embedding: np.ndarray


class InMemoryKB:
    def __init__(
        self,
        data_dir: str = "./data/kb_pdfs",
        embedder: Optional[OllamaEmbedder] = None,
        cache_dir: str = "./data/kb_cache",
    ) -> None:
        self.data_dir = data_dir
        self._embedder = embedder or OllamaEmbedder()
        self._cache_dir = cache_dir
        self._entries: List[KBEntry] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []
        self._loaded = False

    def _cache_path(self) -> str:
        return os.path.join(self._cache_dir, "kb_cache.npz")

    def _meta_cache_path(self) -> str:
        return os.path.join(self._cache_dir, "kb_meta.json")

    def _try_load_cache(self) -> bool:
        """Try to load cached embeddings from disk. Returns True if successful."""
        cache_path = self._cache_path()
        meta_path = self._meta_cache_path()

        if not os.path.exists(cache_path) or not os.path.exists(meta_path):
            return False

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Check if source docs have changed
            current_hash = _compute_dir_hash(self.data_dir)
            if meta.get("dir_hash") != current_hash:
                logger.info("KB source docs changed, cache invalidated")
                return False
            # Check if chunking params changed
            from phase1.chunker import chunk_document
            import inspect
            sig = inspect.signature(chunk_document)
            chunk_size = sig.parameters["chunk_size"].default
            chunk_overlap = sig.parameters["chunk_overlap"].default
            if (meta.get("chunk_size") != chunk_size or
                    meta.get("chunk_overlap") != chunk_overlap):
                logger.info("Chunking params changed, cache invalidated")
                return False

            data = np.load(cache_path, allow_pickle=True)
            texts = list(data["texts"])
            metadatas = list(data["metadatas"])
            embeddings = data["embeddings"]

            self._entries = [
                KBEntry(text=t, metadata=m, embedding=embeddings[i])
                for i, (t, m) in enumerate(zip(texts, metadatas))
            ]
            logger.info("Loaded %d KB entries from disk cache", len(self._entries))
            return True
        except Exception as e:
            logger.warning("Cache load failed, will re-embed: %s", e)
            return False

    def _save_cache(self, texts: List[str], metadatas: List[dict], embeddings: List[List[float]]) -> None:
        """Save embeddings to disk cache."""
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            np.savez(
                self._cache_path(),
                texts=np.array(texts, dtype=object),
                metadatas=np.array(metadatas, dtype=object),
                embeddings=np.array(embeddings, dtype=np.float32),
            )
            from phase1.chunker import chunk_document
            import inspect
            sig = inspect.signature(chunk_document)
            meta = {
                "dir_hash": _compute_dir_hash(self.data_dir),
                "num_entries": len(texts),
                "chunk_size": sig.parameters["chunk_size"].default,
                "chunk_overlap": sig.parameters["chunk_overlap"].default,
            }
            with open(self._meta_cache_path(), "w") as f:
                json.dump(meta, f)
            logger.info("Saved KB cache (%d entries) to disk", len(texts))
        except Exception as e:
            logger.warning("Failed to save KB cache: %s", e)

    def _build_bm25_index(self) -> None:
        """Build BM25 index from loaded entries."""
        self._tokenized_corpus = [_tokenize(e.text) for e in self._entries]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.info("Built BM25 index over %d documents", len(self._tokenized_corpus))
        else:
            self._bm25 = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        # Try disk cache first
        if self._try_load_cache():
            self._build_bm25_index()
            self._loaded = True
            return

        # Parse + chunk + embed from scratch
        docs: List[ParsedDocument] = extract_documents_from_dir(self.data_dir)
        chunks: List[TextChunk] = chunk_documents(docs)
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        self._entries = [
            KBEntry(
                text=t,
                metadata=m,
                embedding=np.asarray(e, dtype=np.float32),
            )
            for t, m, e in zip(texts, metadatas, embeddings)
        ]

        # Save cache for next time
        self._save_cache(texts, metadatas, embeddings)

        # Build BM25 index
        self._build_bm25_index()
        self._loaded = True

    def _dense_search(self, query_vec: np.ndarray, k: int) -> List[tuple[int, float]]:
        """Return top-k (index, similarity) by cosine similarity."""
        q_norm = np.linalg.norm(query_vec) + 1e-8
        sims = []
        for i, entry in enumerate(self._entries):
            v = entry.embedding
            sim = float(np.dot(query_vec, v) / (q_norm * (np.linalg.norm(v) + 1e-8)))
            sims.append((i, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def _bm25_search(self, query: str, k: int) -> List[tuple[int, float]]:
        """Return top-k (index, bm25_score) by BM25."""
        if self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idxs = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idxs if scores[i] > 0]

    def _rrf_fuse(
        self,
        dense_results: List[tuple[int, float]],
        bm25_results: List[tuple[int, float]],
        k: int,
    ) -> List[tuple[int, float]]:
        """Reciprocal Rank Fusion: score = sum(1/(RRF_K + rank_i))."""
        rrf_scores: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:k]

    def similarity_search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """
        Hybrid search: dense (cosine) + sparse (BM25) with RRF fusion.
        Returns list of {text, metadata, distance, similarity, rank}.
        """
        self._ensure_loaded()
        if not self._entries:
            return []

        q_vec = np.asarray(self._embedder.embed_query(query), dtype=np.float32)

        # Over-retrieve from both methods for better fusion
        dense_k = min(k * 3, len(self._entries))
        dense_results = self._dense_search(q_vec, dense_k)
        bm25_results = self._bm25_search(query, dense_k)

        # Fuse with RRF
        fused = self._rrf_fuse(dense_results, bm25_results, k)

        # Build dense similarity lookup for output
        dense_sim_map = {idx: sim for idx, sim in dense_results}

        results: List[Dict[str, Any]] = []
        for rank, (idx, rrf_score) in enumerate(fused, start=1):
            entry = self._entries[idx]
            sim = dense_sim_map.get(idx, 0.0)
            results.append(
                {
                    "text": entry.text,
                    "metadata": entry.metadata,
                    "distance": 1 - sim,
                    "similarity": round(sim, 4),
                    "rrf_score": round(rrf_score, 6),
                    "rank": rank,
                }
            )
        return results

    def similarity_search_by_vector(
        self, query_vector: list[float], k: int = 6
    ) -> List[Dict[str, Any]]:
        """Search KB using a pre-computed embedding vector (used by HyDE)."""
        self._ensure_loaded()
        if not self._entries:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-8

        sims: List[float] = []
        for entry in self._entries:
            v = entry.embedding
            sim = float(np.dot(q_vec, v) / (q_norm * (np.linalg.norm(v) + 1e-8)))
            sims.append(sim)

        idxs = np.argsort(sims)[::-1][:k]
        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(idxs, start=1):
            entry = self._entries[int(idx)]
            results.append(
                {
                    "text": entry.text,
                    "metadata": entry.metadata,
                    "distance": 1 - sims[int(idx)],
                    "similarity": round(sims[int(idx)], 4),
                    "rank": rank,
                }
            )
        return results
