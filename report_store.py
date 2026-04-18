from __future__ import annotations

"""
Ephemeral report-level vector store.

For each uploaded patient report PDF, we:
  - parse and clean it using phase1.pdf_parser
  - chunk it using phase1.chunker
  - embed + store chunks in an in-memory Chroma collection
  - build a BM25 index for sparse keyword matching

Retrieval is hybrid: dense (cosine via ChromaDB) + sparse (BM25)
fused with Reciprocal Rank Fusion (RRF).

The store is per-Streamlit session; when a new report is loaded,
the previous in-memory collection is replaced.
"""

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from phase1.pdf_parser import ParsedDocument, parse_pdf
from phase1.chunker import TextChunk, chunk_document
from phase1.embedder import OllamaEmbedder
from crypto_utils import encrypt_text, decrypt_text

logger = logging.getLogger(__name__)

# RRF constant (standard value from the original RRF paper)
RRF_K = 60


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+(?:[-']\w+)*", text.lower())


@dataclass
class ReportMetadata:
    report_id: str
    filename: str
    num_pages: int
    num_chunks: int


class ReportStore:
    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        distance_metric: str = "cosine",
    ) -> None:
        # Use an in-project persistent client for robustness with newer Chroma versions.
        # The collection itself is still per-session/per-report.
        self._client = chromadb.PersistentClient(
            path="./data/session_chroma",
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = None
        self._embedder = embedder or OllamaEmbedder()
        self._distance_metric = distance_metric
        self._meta: Optional[ReportMetadata] = None
        # Hybrid retrieval state
        self._plain_texts: List[str] = []
        self._metadatas: List[dict] = []
        self._bm25: Optional[BM25Okapi] = None

    @property
    def metadata(self) -> Optional[ReportMetadata]:
        return self._meta

    def _reset_collection(self) -> None:
        # Drop reference to current collection; a new one will be created on next use.
        self._collection = None
        self._meta = None
        self._plain_texts = []
        self._metadatas = []
        self._bm25 = None

    def _ensure_collection(self) -> chromadb.Collection:
        if self._collection is None:
            name = f"report_{uuid.uuid4().hex}"
            self._collection = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": self._distance_metric},
            )
        return self._collection

    def load_report_from_path(self, pdf_path: str) -> ReportMetadata:
        """
        Load a new patient report from a PDF path.
        Replaces any existing in-memory collection.
        """
        self._reset_collection()
        parsed: Optional[ParsedDocument] = parse_pdf(pdf_path)
        if parsed is None or not parsed.text.strip():
            raise ValueError(f"Failed to parse report PDF: {pdf_path}")

        chunks: List[TextChunk] = chunk_document(parsed)
        if not chunks:
            raise ValueError("Parsed report produced no chunks for embedding.")

        plain_texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        embeddings = self._embedder.embed_texts(plain_texts)
        encrypted_texts = [encrypt_text(t) for t in plain_texts]

        col = self._ensure_collection()
        col.add(
            ids=ids,
            embeddings=embeddings,
            documents=encrypted_texts,
            metadatas=metadatas,
        )

        # Store plain texts + build BM25 for hybrid retrieval
        self._plain_texts = plain_texts
        self._metadatas = metadatas
        tokenized = [_tokenize(t) for t in plain_texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        logger.info("Built BM25 index over %d report chunks", len(plain_texts))

        report_id = uuid.uuid4().hex
        self._meta = ReportMetadata(
            report_id=report_id,
            filename=parsed.filename,
            num_pages=parsed.num_pages,
            num_chunks=len(chunks),
        )
        return self._meta

    def _dense_search(self, query: str, k: int) -> List[tuple[int, str, dict, float]]:
        """Dense cosine search via ChromaDB.
        Returns list of (chunk_index, text, metadata, cosine_similarity)."""
        query_embedding = self._embedder.embed_query(query)
        n = min(k, self._collection.count())
        if n == 0:
            return []
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        out = []
        for enc_doc, meta, dist in zip(docs, metas, dists):
            text = decrypt_text(enc_doc)
            # Find the chunk index by matching text
            idx = next(
                (i for i, t in enumerate(self._plain_texts) if t == text), -1
            )
            out.append((idx, text, meta, 1.0 - float(dist)))
        return out

    def _bm25_search(self, query: str, k: int) -> List[tuple[int, float]]:
        """Sparse BM25 search. Returns list of (chunk_index, bm25_score)."""
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

    def similarity_search(self, query: str, k: int = 4) -> list[dict]:
        """
        Hybrid search: dense (cosine) + sparse (BM25) with RRF fusion.
        Returns list of {text, metadata, distance, similarity, rank, source_type}.
        """
        if self._collection is None:
            return []

        # Over-retrieve from both methods for better fusion
        dense_k = min(k * 3, len(self._plain_texts))
        dense_results = self._dense_search(query, dense_k)

        # Build dense similarity lookup by chunk index
        dense_sim_map: Dict[int, float] = {}
        dense_text_map: Dict[int, tuple[str, dict]] = {}
        for idx, text, meta, sim in dense_results:
            dense_sim_map[idx] = sim
            dense_text_map[idx] = (text, meta)

        bm25_results = self._bm25_search(query, dense_k)

        # Fuse with RRF
        fused = self._rrf_fuse(
            [(idx, sim) for idx, _, _, sim in dense_results],
            bm25_results,
            k,
        )

        out: list[dict] = []
        for rank, (idx, rrf_score) in enumerate(fused, start=1):
            if idx in dense_text_map:
                text, meta = dense_text_map[idx]
            else:
                text = self._plain_texts[idx]
                meta = self._metadatas[idx]
            sim = dense_sim_map.get(idx, 0.0)
            out.append(
                {
                    "text": text,
                    "metadata": meta,
                    "distance": 1.0 - sim,
                    "similarity": round(sim, 4),
                    "rrf_score": round(rrf_score, 6),
                    "rank": rank,
                    "source_type": "report",
                }
            )

        logger.info(
            "[REPORT] Hybrid search: %d dense + %d BM25 → %d fused",
            len(dense_results), len(bm25_results), len(out),
        )
        return out

