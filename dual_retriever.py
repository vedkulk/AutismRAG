from __future__ import annotations

"""
Dual-source retriever with query classification:
  - classifies queries as patient / medical / general
  - routes retrieval to the appropriate store(s)
  - fuses and deduplicates results into a single context object
  - (Phase 4) optionally applies HyDE, query rewriting, cross-encoder
    re-ranking, and contextual compression via AdvancedRAGOrchestrator
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from report_store import ReportStore
from kb_memory import InMemoryKB

if TYPE_CHECKING:
    from advanced_rag import AdvancedRAGOrchestrator

logger = logging.getLogger(__name__)


# ── Query classification ─────────────────────────────────────────────────

PATIENT_KEYWORDS = [
    "patient", "child's", "childs", "this child", "the child",
    "report", "observation", "observed", "documented",
    "tell me about", "summarize", "summary", "overview",
    "behaviour", "behavior", "behaviours", "behaviors",
    "what does the report", "from the report", "in the report",
    "their", "this kid",
]

MEDICAL_KEYWORDS = [
    "dsm", "icd", "ados", "m-chat", "mchat", "criteria",
    "diagnostic", "screening tool", "screening", "intervention", "treatment",
    "evidence-based", "guideline", "recommendation",
    "differential", "comorbid", "co-occurring", "co-occur",
    "prevalence", "prognosis", "etiology",
    "clinical significance", "appropriate for",
]


def classify_query(query: str) -> str:
    """Classify a query as 'patient', 'medical', or 'both'.

    - patient: about the uploaded report / this specific child
    - medical: about clinical guidelines, tools, criteria
    - both: references the patient in a medical context
    """
    q = query.lower()

    has_patient = any(kw in q for kw in PATIENT_KEYWORDS)
    has_medical = any(kw in q for kw in MEDICAL_KEYWORDS)

    if has_patient and has_medical:
        return "both"
    elif has_patient:
        return "patient"
    elif has_medical:
        return "medical"
    else:
        # Default: if a report is loaded, assume they're asking about the patient
        return "both"


# ── Query normalization ──────────────────────────────────────────────────
# Maps vague queries to document-aligned search terms so the retriever
# can match against actual chunk text. No LLM call — just pattern matching.

# Domain-specific terms that map to report section language
_DOMAIN_EXPANSIONS = {
    "social": "social interaction observations",
    "communication": "communication domain observations",
    "sensory": "sensory responses observations",
    "behavior": "behavioral patterns observations",
    "behaviour": "behavioral patterns observations",
    "speech": "communication domain speech language observations",
    "language": "communication domain speech language observations",
    "motor": "motor development observations",
    "play": "social interaction play observations",
    "eye contact": "eye contact social interaction observations",
    "repetitive": "repetitive behaviors behavioral patterns",
    "routine": "behavioral patterns routine change distress",
    "echolalia": "communication echolalia repeating words",
    "name": "respond to name social interaction",
}

# Vague query patterns → document-aligned rewrites
_VAGUE_PATTERNS = [
    # (regex pattern, rewrite template)
    (re.compile(r"^tell me about (?:the |this )?(?:patient|child|kid)\.?$", re.I),
     "summary of patient developmental report: communication, social interaction, behavioral patterns, sensory responses"),
    (re.compile(r"^(?:what is |what's |give me )?(?:a )?(?:summary|overview|recap).*$", re.I),
     "summary of patient developmental report: communication, social interaction, behavioral patterns, sensory responses"),
    (re.compile(r"^(?:tell me about|describe|explain|what about) (?:the |this )?(?:child'?s? )?(.+?)\.?$", re.I),
     None),  # handled dynamically below
    (re.compile(r"^how (?:is|are|does|do) (?:the |this )?(?:child'?s? )?(.+?)\.?\??$", re.I),
     None),  # handled dynamically below
]


def normalize_query(query: str, query_type: str) -> str:
    """Rewrite vague queries into document-aligned search terms.

    For patient queries: expand into report section terminology.
    For medical queries: pass through (advanced RAG handles these).
    For both: lightly expand with domain terms.

    Returns the original query if no normalization is needed.
    """
    if query_type == "medical":
        return query  # advanced RAG (HyDE + rewriting) handles medical queries

    q = query.strip()

    # Check exact vague patterns first
    for pattern, static_rewrite in _VAGUE_PATTERNS:
        match = pattern.match(q)
        if match:
            if static_rewrite:
                logger.info("[NORMALIZE] %r → %r (pattern match)", q, static_rewrite)
                return static_rewrite
            else:
                # Dynamic: extract the topic and expand it
                topic = match.group(1).strip().lower()
                expanded = _expand_topic(topic)
                result = f"{expanded} in patient developmental report"
                logger.info("[NORMALIZE] %r → %r (topic expansion)", q, result)
                return result

    # For patient queries that didn't match patterns, prepend report context
    if query_type == "patient":
        result = f"patient report: {q}"
        logger.info("[NORMALIZE] %r → %r (patient prefix)", q, result)
        return result

    # "both" type — lightly expand domain terms found in query
    q_lower = q.lower()
    expansions = []
    for term, expansion in _DOMAIN_EXPANSIONS.items():
        if term in q_lower:
            expansions.append(expansion)
    if expansions:
        result = f"{q} {' '.join(expansions)}"
        logger.info("[NORMALIZE] %r → added domain terms", q)
        return result

    return query  # no normalization needed


def _expand_topic(topic: str) -> str:
    """Expand a short topic into document-aligned search terms."""
    parts = []
    topic_lower = topic.lower()

    for term, expansion in _DOMAIN_EXPANSIONS.items():
        if term in topic_lower:
            parts.append(expansion)

    if parts:
        return " ".join(parts)

    # Fallback: use the topic as-is with report context
    return f"{topic} observations"


@dataclass
class ContextChunk:
    text: str
    metadata: Dict[str, Any]
    similarity: float
    source_type: str  # "report" or "kb"


@dataclass
class FusedContext:
    query: str
    chunks: List[ContextChunk]

    def as_prompt_block(self) -> str:
        """Human-readable block to inject into the LLM prompt."""
        lines = []
        for i, c in enumerate(self.chunks, start=1):
            src = "PATIENT REPORT" if c.source_type == "report" else "MEDICAL KB"
            lines.append(
                f"[{i}] ({src}, sim={c.similarity:.3f})\n{c.text}\n"
            )
        return "\n\n".join(lines)


# Minimum chunks to always pass to LLM (even if scores are low)
MIN_CHUNKS_FOR_LLM = 2


class DualRetriever:
    def __init__(
        self,
        kb_store: Optional[InMemoryKB] = None,
        report_store: Optional[ReportStore] = None,
        advanced_rag: Optional[AdvancedRAGOrchestrator] = None,
    ) -> None:
        self.kb_store = kb_store or InMemoryKB()
        self.report_store = report_store or ReportStore()
        self._advanced_rag = advanced_rag

    def _fuse(
        self,
        report_hits: List[Dict[str, Any]],
        kb_hits: List[Dict[str, Any]],
    ) -> List[ContextChunk]:
        """Fuse and deduplicate hits from both stores."""
        by_fingerprint: Dict[str, ContextChunk] = {}

        for hit in report_hits:
            key = hit["text"].strip()
            if not key:
                continue
            chunk = ContextChunk(
                text=hit["text"],
                metadata=hit["metadata"],
                similarity=float(hit.get("similarity", 0.0)),
                source_type="report",
            )
            if key not in by_fingerprint:
                by_fingerprint[key] = chunk

        for hit in kb_hits:
            text = hit["text"]
            key = text.strip()
            if not key or key in by_fingerprint:
                continue
            chunk = ContextChunk(
                text=text,
                metadata=hit["metadata"],
                similarity=float(hit.get("similarity", 0.8)),
                source_type="kb",
            )
            by_fingerprint[key] = chunk

        fused = list(by_fingerprint.values())
        fused.sort(
            key=lambda c: (0 if c.source_type == "report" else 1, -c.similarity)
        )
        return fused

    def retrieve(
        self,
        query: str,
        kb_k: int = 20,
        report_k: int = 4,
    ) -> FusedContext:
        """
        Retrieve relevant chunks from both the patient report and the KB.

        Query classification routes retrieval:
          - patient queries  → prioritize report, minimal KB
          - medical queries  → prioritize KB, skip report-specific transforms
          - both/general     → balanced retrieval from both sources
        """
        # ── Classify query ──────────────────────────────────────────────
        query_type = classify_query(query)
        logger.info("[ROUTING] Query type: %s — %r", query_type, query[:60])

        # ── Normalize vague queries into document-aligned terms ────────
        search_query = normalize_query(query, query_type)
        if search_query != query:
            logger.info("[ROUTING] Normalized query: %r", search_query[:80])

        # ── Adjust retrieval weights based on query type ────────────────
        if query_type == "patient":
            effective_report_k = max(report_k, 10)  # retrieve nearly all report chunks
            effective_kb_k = 2                        # minimal KB for clinical context
        elif query_type == "medical":
            effective_report_k = 2                    # minimal report
            effective_kb_k = kb_k                     # full KB
        else:  # "both"
            effective_report_k = max(report_k, 8)    # strong report presence
            effective_kb_k = min(kb_k, 10)            # moderate KB for cross-encoder

        # ── Pre-retrieval transforms ─────────────────────────────────────
        query_variants: list[str] = [search_query]
        hyde_embedding: Optional[list[float]] = None

        # Only run HyDE/rewriting for medical or both queries
        if self._advanced_rag and query_type != "patient":
            variants, hyde_embedding = self._advanced_rag.pre_retrieval(search_query)
            if variants:
                all_queries = [search_query] + [v for v in variants if v.lower() != search_query.lower()]
                query_variants = all_queries[:4]

        # ── Retrieval (multi-query) ──────────────────────────────────────
        report_hits = self.report_store.similarity_search(search_query, k=effective_report_k)

        all_kb_hits: list[dict] = []

        if hyde_embedding is not None:
            hyde_hits = self.kb_store.similarity_search_by_vector(
                hyde_embedding, k=effective_kb_k,
            )
            all_kb_hits.extend(hyde_hits)

        if effective_kb_k > 0:
            per_variant_k = max(effective_kb_k // len(query_variants), 4)
            for variant in query_variants:
                hits = self.kb_store.similarity_search(variant, k=per_variant_k)
                all_kb_hits.extend(hits)

        # Deduplicate KB hits by text content, keeping highest similarity
        seen: dict[str, dict] = {}
        for hit in all_kb_hits:
            key = hit["text"].strip()
            if key not in seen or hit.get("similarity", 0) > seen[key].get("similarity", 0):
                seen[key] = hit
        kb_hits = list(seen.values())

        logger.info(
            "[ROUTING] Retrieved %d report + %d KB chunks",
            len(report_hits), len(kb_hits),
        )

        # ── Fusion ───────────────────────────────────────────────────────
        fused = self._fuse(report_hits, kb_hits)

        # ── Post-retrieval transforms ────────────────────────────────────
        if self._advanced_rag:
            fused = self._advanced_rag.post_retrieval(query, fused)

        # ── Minimum chunk guarantee ──────────────────────────────────────
        if len(fused) < MIN_CHUNKS_FOR_LLM and (report_hits or kb_hits):
            raw_fused = self._fuse(report_hits, kb_hits)
            raw_fused.sort(key=lambda c: -c.similarity)
            fused = raw_fused[:MIN_CHUNKS_FOR_LLM]
            logger.warning(
                "[ROUTING] Post-retrieval dropped too many chunks, "
                "falling back to top-%d raw chunks", MIN_CHUNKS_FOR_LLM,
            )

        # ── Smart chunk selection: fixed 5 report + 3 KB ────────────────
        report_part = [c for c in fused if c.source_type == "report"]
        kb_part = [c for c in fused if c.source_type != "report"]

        report_part.sort(key=lambda c: -c.similarity)
        kb_part.sort(key=lambda c: -c.similarity)

        report_part = report_part[:5]
        kb_part = kb_part[:3]

        fused = report_part + kb_part

        logger.info(
            "[ROUTING] Final: %d chunks (%d report + %d KB) [type=%s]",
            len(fused), len(report_part), len(kb_part), query_type,
        )

        return FusedContext(query=query, chunks=fused)
