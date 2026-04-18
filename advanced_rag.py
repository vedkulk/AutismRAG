from __future__ import annotations

"""
Phase 4 — Advanced RAG techniques.

Provides four retrieval enhancements, each independently toggleable:
  1. HyDE          – hypothetical document embedding for KB search
  2. Query Rewrite – LLM-based clinical query reformulation
  3. Cross-Encoder – re-ranking retrieved chunks with a cross-encoder
  4. Compression   – extracting only query-relevant sentences from chunks

All techniques degrade gracefully: if one fails, the pipeline continues
with the unmodified input.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple


def _normalize_ce_score(score: float) -> float:
    """Map a cross-encoder logit to [0,1] via a scaled sigmoid.

    The previous linear formula (score + 10) / 20 clipped to 0 whenever
    the raw score was <= -10, which happens often for clinical text with
    a web-trained model. Sigmoid is smooth and never saturates to exactly 0.
    Scale factor 3 gives good dispersion in the typical [-12, +6] range.
    """
    return 1.0 / (1.0 + math.exp(-float(score) / 3.0))

if TYPE_CHECKING:
    from llm_engine import LLMEngine
    from phase1.embedder import OllamaEmbedder
    from dual_retriever import ContextChunk

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AdvancedRAGConfig:
    hyde_enabled: bool = False
    query_rewriting_enabled: bool = False
    cross_encoder_enabled: bool = False
    compression_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5
    compression_top_k: int = 5


# ══════════════════════════════════════════════════════════════════════════
#  HyDE — Hypothetical Document Embeddings
# ══════════════════════════════════════════════════════════════════════════

class HyDEGenerator:
    """Generate a hypothetical clinical passage and embed it for KB search."""

    _PROMPT = (
        "Given the following clinical question about a child's development or "
        "autism spectrum disorder, write a short clinical passage (3-5 sentences) "
        "that would be found in a medical textbook or clinical guideline answering "
        "this question. Write as if you are a pediatric developmental specialist.\n\n"
        "Question: {query}\n\n"
        "Clinical passage:"
    )

    def __init__(self, llm: LLMEngine, embedder: OllamaEmbedder) -> None:
        self._llm = llm
        self._embedder = embedder

    def generate(self, query: str) -> str:
        return self._llm.generate_raw(self._PROMPT.format(query=query))

    def generate_embedding(self, query: str) -> list[float]:
        hypothetical_doc = self.generate(query)
        logger.info("HyDE generated %d-char hypothetical document", len(hypothetical_doc))
        return self._embedder.embed_query(hypothetical_doc)


# ══════════════════════════════════════════════════════════════════════════
#  Query Rewriting
# ══════════════════════════════════════════════════════════════════════════

class QueryRewriter:
    """Rewrite a user question into multiple retrieval-optimized variants."""

    _PROMPT = (
        "Rewrite the following question into 3 different phrasings optimized for "
        "searching a medical knowledge base about Autism Spectrum Disorder. "
        "Each phrasing should use different clinical terminology or focus on a "
        "different aspect of the question. Use standard terms (DSM-5, ICD-10, "
        "ADOS-2, M-CHAT, etc.) where applicable.\n"
        "Return ONLY the 3 phrasings, one per line. No numbering, no extra text.\n\n"
        "Question: {query}\n\n"
        "Phrasings:"
    )

    def __init__(self, llm: LLMEngine) -> None:
        self._llm = llm

    def rewrite(self, query: str) -> str:
        """Backward-compatible: returns the first variant."""
        variants = self.rewrite_multi(query)
        return variants[0] if variants else query

    def rewrite_multi(self, query: str) -> List[str]:
        """Return multiple query variants for multi-query retrieval."""
        raw = self._llm.generate_raw(self._PROMPT.format(query=query)).strip()
        variants = [line.strip().lstrip("0123456789.-) ") for line in raw.split("\n") if line.strip()]
        # Deduplicate and limit to 3
        seen = set()
        unique = []
        for v in variants:
            if v and v.lower() not in seen:
                seen.add(v.lower())
                unique.append(v)
        unique = unique[:3]
        logger.info("Query rewritten into %d variants: %s", len(unique), [v[:60] for v in unique])
        return unique if unique else [query]


# ══════════════════════════════════════════════════════════════════════════
#  Cross-Encoder Re-ranking
# ══════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """Re-rank retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            logger.info(
                "Loading cross-encoder model '%s' (first run may download ~80 MB)…",
                self._model_name,
            )
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)

    def rerank(
        self, query: str, chunks: List[ContextChunk], top_k: int = 6
    ) -> List[ContextChunk]:
        if not chunks:
            return chunks
        self._ensure_model()

        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)

        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        # Keep top_k chunks. Always keep at least 2 report chunks.
        result: List[ContextChunk] = []
        report_kept = 0
        for chunk, score in scored[:top_k]:
            # Normalize cross-encoder logit to [0,1] for display
            chunk.similarity = _normalize_ce_score(score)
            result.append(chunk)

        # Ensure at least 2 report chunks survive reranking
        if report_kept < 2:
            report_in_result = {c.text for c in result if c.source_type == "report"}
            for chunk, score in scored:
                if chunk.source_type == "report" and chunk.text not in report_in_result:
                    chunk.similarity = _normalize_ce_score(score)
                    result.append(chunk)
                    report_in_result.add(chunk.text)
                    if len([c for c in result if c.source_type == "report"]) >= 2:
                        break

        logger.info(
            "Cross-encoder re-ranked %d → %d chunks (top=%.3f, min=%.3f)",
            len(chunks), len(result),
            result[0].similarity if result else 0,
            result[-1].similarity if result else 0,
        )
        return result


# ══════════════════════════════════════════════════════════════════════════
#  Contextual Compression (embedding-based — no LLM calls)
# ══════════════════════════════════════════════════════════════════════════

# Minimum cosine similarity between query embedding and chunk embedding
# to keep the chunk. Very permissive — the cross-encoder already filtered.
COMPRESSION_SIM_THRESHOLD = 0.30

import numpy as np


class ContextCompressor:
    """Filter chunks by embedding similarity to the query.

    Replaces the previous LLM-based compressor which:
      - made one LLM call per chunk (5 calls = 40-80s latency)
      - was too aggressive, dropping chunks the cross-encoder scored 3-8
    Now uses a single embedding comparison — fast and predictable.
    """

    def __init__(self, embedder: "OllamaEmbedder") -> None:
        self._embedder = embedder

    def compress(
        self, query: str, chunks: List["ContextChunk"], top_k: int = 5
    ) -> List["ContextChunk"]:
        if not chunks:
            return chunks

        q_vec = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec) + 1e-8

        chunk_texts = [c.text for c in chunks[:top_k]]
        chunk_embs = self._embedder.embed_texts(chunk_texts)

        compressed: List["ContextChunk"] = []
        for chunk, emb in zip(chunks[:top_k], chunk_embs):
            c_vec = np.asarray(emb, dtype=np.float32)
            cos_sim = float(np.dot(q_vec, c_vec) / (q_norm * (np.linalg.norm(c_vec) + 1e-8)))
            if cos_sim >= COMPRESSION_SIM_THRESHOLD:
                compressed.append(chunk)
            else:
                logger.debug(
                    "Compression dropped chunk (cos_sim=%.3f < %.2f)",
                    cos_sim, COMPRESSION_SIM_THRESHOLD,
                )

        # Keep any chunks beyond top_k unchanged
        compressed.extend(chunks[top_k:])

        logger.info(
            "Compression: %d → %d chunks (threshold=%.2f)",
            min(len(chunks), top_k), len(compressed) - len(chunks[top_k:]),
            COMPRESSION_SIM_THRESHOLD,
        )
        return compressed


# ══════════════════════════════════════════════════════════════════════════
#  Orchestrator
# ══════════════════════════════════════════════════════════════════════════

class AdvancedRAGOrchestrator:
    """Coordinates all advanced RAG techniques around the retrieval step."""

    def __init__(
        self,
        config: AdvancedRAGConfig,
        llm: LLMEngine,
        embedder: OllamaEmbedder,
    ) -> None:
        self.config = config
        self._hyde = HyDEGenerator(llm, embedder) if config.hyde_enabled else None
        self._rewriter = QueryRewriter(llm) if config.query_rewriting_enabled else None
        self._reranker = (
            CrossEncoderReranker(config.cross_encoder_model)
            if config.cross_encoder_enabled else None
        )
        self._compressor = ContextCompressor(embedder) if config.compression_enabled else None

    # ── Pre-retrieval: query transforms ──────────────────────────────────

    def pre_retrieval(
        self, query: str
    ) -> Tuple[List[str], Optional[list[float]]]:
        """
        Run HyDE and multi-query rewriting in parallel.

        Returns:
            (list_of_query_variants, hyde_embedding or None)
        """
        query_variants: List[str] = []
        hyde_emb: Optional[list[float]] = None

        futures = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            if self._rewriter:
                futures[pool.submit(self._rewriter.rewrite_multi, query)] = "rewrite"
            if self._hyde:
                futures[pool.submit(self._hyde.generate_embedding, query)] = "hyde"

            for future in as_completed(futures):
                label = futures[future]
                try:
                    if label == "rewrite":
                        query_variants = future.result()
                    elif label == "hyde":
                        hyde_emb = future.result()
                except Exception as e:
                    logger.warning("Advanced RAG %s failed, falling back: %s", label, e)

        return query_variants, hyde_emb

    # ── Post-retrieval: re-rank + compress ───────────────────────────────

    def post_retrieval(
        self, query: str, chunks: List[ContextChunk]
    ) -> List[ContextChunk]:
        """Apply cross-encoder re-ranking, then embedding-based compression."""
        chunks_before = len(chunks)
        top_score_before = max((c.similarity for c in chunks), default=0.0)

        # Re-rank
        if self._reranker:
            try:
                chunks = self._reranker.rerank(
                    query, chunks, top_k=self.config.rerank_top_k,
                )
            except Exception as e:
                logger.warning("Cross-encoder re-ranking failed, keeping original order: %s", e)

        chunks_after_rerank = len(chunks)
        top_rerank_score = max((c.similarity for c in chunks), default=0.0)

        # Compress
        if self._compressor:
            try:
                chunks = self._compressor.compress(
                    query, chunks, top_k=self.config.compression_top_k,
                )
            except Exception as e:
                logger.warning("Contextual compression failed, keeping original chunks: %s", e)

        chunks_after_compress = len(chunks)

        logger.info(
            "[DEBUG] chunks_before=%d, chunks_after_rerank=%d, "
            "chunks_after_compress=%d, top_rerank_score=%.3f",
            chunks_before, chunks_after_rerank,
            chunks_after_compress, top_rerank_score,
        )

        return chunks
