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
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
        retrieval_type: str = "similarity",
        mmr_lambda: float = 0.5,
    ) -> None:
        self.data_dir = data_dir
        self._embedder = embedder or OllamaEmbedder()
        self._cache_dir = cache_dir
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._retrieval_type = retrieval_type.lower()
        self._mmr_lambda = float(mmr_lambda)
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
            if (meta.get("chunk_size") != self._chunk_size or
                    meta.get("chunk_overlap") != self._chunk_overlap or
                    meta.get("min_chunk_size") != self._min_chunk_size):
                logger.info(
                    "Chunking params changed (cached %s/%s/%s vs %s/%s/%s), cache invalidated",
                    meta.get("chunk_size"), meta.get("chunk_overlap"), meta.get("min_chunk_size"),
                    self._chunk_size, self._chunk_overlap, self._min_chunk_size,
                )
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
            meta = {
                "dir_hash": _compute_dir_hash(self.data_dir),
                "num_entries": len(texts),
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
                "min_chunk_size": self._min_chunk_size,
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
        chunks: List[TextChunk] = chunk_documents(
            docs,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_chunk_size=self._min_chunk_size,
        )
        logger.info(
            "Chunked %d docs → %d chunks (size=%d, overlap=%d, min=%d)",
            len(docs), len(chunks), self._chunk_size,
            self._chunk_overlap, self._min_chunk_size,
        )
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

    def _mmr_rerank(
        self,
        q_vec: np.ndarray,
        candidates: List[tuple[int, float]],
        k: int,
    ) -> List[tuple[int, float]]:
        """Maximal Marginal Relevance re-rank over candidate (idx, score) pairs.

        Score formula (standard MMR):
            score(c) = λ·sim(q, c) − (1 − λ)·max sim(c, c_selected)

        mmr_lambda convention matches config.ini comment:
            0 → max relevance, 1 → max diversity.
        So the effective λ for the relevance term is (1 − mmr_lambda).
        """
        if not candidates:
            return []

        relevance_weight = 1.0 - self._mmr_lambda
        diversity_weight = self._mmr_lambda

        q_norm = np.linalg.norm(q_vec) + 1e-8
        cand_idxs = [idx for idx, _ in candidates]
        cand_vecs = np.stack([self._entries[i].embedding for i in cand_idxs])
        cand_norms = np.linalg.norm(cand_vecs, axis=1) + 1e-8
        q_sims = (cand_vecs @ q_vec) / (cand_norms * q_norm)

        selected: List[int] = []  # positions within cand_idxs
        remaining = set(range(len(cand_idxs)))
        pairwise_max = np.full(len(cand_idxs), -np.inf, dtype=np.float32)

        k = min(k, len(cand_idxs))
        while len(selected) < k and remaining:
            best_pos = -1
            best_score = -np.inf
            for pos in remaining:
                div_penalty = 0.0 if not selected else float(pairwise_max[pos])
                mmr_score = relevance_weight * float(q_sims[pos]) - diversity_weight * div_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_pos = pos
            selected.append(best_pos)
            remaining.remove(best_pos)

            if remaining:
                new_vec = cand_vecs[best_pos]
                new_norm = cand_norms[best_pos]
                sims_to_new = (cand_vecs @ new_vec) / (cand_norms * new_norm)
                pairwise_max = np.maximum(pairwise_max, sims_to_new)

        return [(cand_idxs[p], float(q_sims[p])) for p in selected]

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

        # Fuse with RRF — for MMR mode, keep a larger candidate pool so the
        # diversity re-rank has something to work with.
        fuse_k = min(max(k * 3, 30), len(self._entries)) if self._retrieval_type == "mmr" else k
        fused = self._rrf_fuse(dense_results, bm25_results, fuse_k)

        dense_sim_map = {idx: sim for idx, sim in dense_results}

        if self._retrieval_type == "mmr":
            reranked = self._mmr_rerank(q_vec, fused, k)
            logger.info(
                "MMR rerank: %d candidates → %d (λ=%.2f)",
                len(fused), len(reranked), self._mmr_lambda,
            )
            fused_for_output = [(idx, dense_sim_map.get(idx, 0.0)) for idx, _ in reranked]
        else:
            fused_for_output = [(idx, score) for idx, score in fused[:k]]

        results: List[Dict[str, Any]] = []
        for rank, (idx, fused_score) in enumerate(fused_for_output, start=1):
            entry = self._entries[idx]
            sim = dense_sim_map.get(idx, 0.0)
            results.append(
                {
                    "text": entry.text,
                    "metadata": entry.metadata,
                    "distance": 1 - sim,
                    "similarity": round(sim, 4),
                    "rrf_score": round(fused_score, 6),
                    "rank": rank,
                }
            )
        return results

