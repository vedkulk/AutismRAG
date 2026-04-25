from __future__ import annotations

"""
High-level RAG pipeline facade used by the Streamlit app.

This is the only interface the UI talks to.
It lazily initialises:
  - ReportStore (ephemeral per-session)
  - KnowledgeBaseStore (persistent KB from Phase 1)
  - DualRetriever
  - LLMEngine
"""

import configparser
import logging
from dataclasses import dataclass
from typing import Optional, Iterable

from phase1.embedder import OllamaEmbedder
from phase1.pdf_parser import parse_pdf_bytes
from report_store import ReportStore, ReportMetadata
from dual_retriever import DualRetriever, FusedContext
from llm_engine import LLMEngine
from kb_memory import InMemoryKB
from advanced_rag import AdvancedRAGConfig, AdvancedRAGOrchestrator
from dfs import EphemeralDFSStore

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    kb_data_dir: str = "./data/kb_pdfs"
    kb_persist_dir: str = "./data/chroma_kb"
    kb_collection_name: str = "autism_kb_medical_guidelines"
    kb_top_k: int = 20
    report_top_k: int = 8
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1"
    embed_model: str = "nomic-embed-text"
    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    # Retrieval strategy
    retrieval_type: str = "similarity"  # "similarity" | "mmr"
    mmr_lambda: float = 0.5
    # Phase 4 — Advanced RAG
    hyde_enabled: bool = False
    query_rewriting_enabled: bool = False
    cross_encoder_enabled: bool = False
    compression_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 15
    compression_top_k: int = 5
    # DFS — ephemeral encrypted+fragmented session store
    dfs_num_fragments: int = 10
    dfs_num_key_shares: int = 5
    dfs_key_threshold: int = 3

    @classmethod
    def from_ini(cls, path: str = "config.ini") -> "RAGConfig":
        """Build a RAGConfig from config.ini. Missing values fall back to defaults."""
        cfg = configparser.ConfigParser()
        read_ok = cfg.read(path)
        if not read_ok:
            logger.warning("config.ini not found at %s, using RAGConfig defaults", path)
            return cls()

        c = cls()
        c.ollama_base_url     = cfg.get("ollama",   "base_url",    fallback=c.ollama_base_url)
        c.llm_model           = cfg.get("ollama",   "llm_model",   fallback=c.llm_model)
        c.embed_model         = cfg.get("ollama",   "embed_model", fallback=c.embed_model)
        c.kb_data_dir         = cfg.get("paths",    "kb_data_dir", fallback=c.kb_data_dir)
        c.kb_persist_dir      = cfg.get("paths",    "chroma_db_dir", fallback=c.kb_persist_dir)
        c.chunk_size          = cfg.getint("chunking", "chunk_size",     fallback=c.chunk_size)
        c.chunk_overlap       = cfg.getint("chunking", "chunk_overlap",  fallback=c.chunk_overlap)
        c.min_chunk_size      = cfg.getint("chunking", "min_chunk_size", fallback=c.min_chunk_size)
        c.kb_top_k            = cfg.getint("retrieval", "kb_top_k",     fallback=c.kb_top_k)
        c.report_top_k        = cfg.getint("retrieval", "report_top_k", fallback=c.report_top_k)
        c.retrieval_type      = cfg.get("retrieval", "retrieval_type",  fallback=c.retrieval_type)
        c.mmr_lambda          = cfg.getfloat("retrieval", "mmr_lambda", fallback=c.mmr_lambda)
        c.hyde_enabled            = cfg.getboolean("advanced_rag", "hyde_enabled",            fallback=c.hyde_enabled)
        c.query_rewriting_enabled = cfg.getboolean("advanced_rag", "query_rewriting_enabled", fallback=c.query_rewriting_enabled)
        c.cross_encoder_enabled   = cfg.getboolean("advanced_rag", "cross_encoder_enabled",   fallback=c.cross_encoder_enabled)
        c.compression_enabled     = cfg.getboolean("advanced_rag", "compression_enabled",     fallback=c.compression_enabled)
        c.cross_encoder_model     = cfg.get("advanced_rag", "cross_encoder_model", fallback=c.cross_encoder_model)
        c.rerank_top_k            = cfg.getint("advanced_rag", "rerank_top_k",      fallback=c.rerank_top_k)
        c.compression_top_k       = cfg.getint("advanced_rag", "compression_top_k", fallback=c.compression_top_k)
        c.kb_collection_name  = cfg.get("chroma", "kb_collection_name", fallback=c.kb_collection_name)
        c.dfs_num_fragments   = cfg.getint("dfs", "num_fragments",  fallback=c.dfs_num_fragments)
        c.dfs_num_key_shares  = cfg.getint("dfs", "num_key_shares", fallback=c.dfs_num_key_shares)
        c.dfs_key_threshold   = cfg.getint("dfs", "key_threshold",  fallback=c.dfs_key_threshold)
        return c


class RAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or RAGConfig()
        self._embedder = OllamaEmbedder(
            model=self.config.embed_model,
            base_url=self.config.ollama_base_url,
        )
        self._kb_store = InMemoryKB(
            data_dir=self.config.kb_data_dir,
            embedder=self._embedder,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
            retrieval_type=self.config.retrieval_type,
            mmr_lambda=self.config.mmr_lambda,
        )
        self._report_store = ReportStore(embedder=self._embedder)
        self._llm = LLMEngine(
            model=self.config.llm_model,
            base_url=self.config.ollama_base_url,
        )
        self._dfs = EphemeralDFSStore(
            num_fragments=self.config.dfs_num_fragments,
            num_key_shares=self.config.dfs_num_key_shares,
            key_threshold=self.config.dfs_key_threshold,
        )

        # Phase 4 — Advanced RAG orchestrator
        adv_config = AdvancedRAGConfig(
            hyde_enabled=self.config.hyde_enabled,
            query_rewriting_enabled=self.config.query_rewriting_enabled,
            cross_encoder_enabled=self.config.cross_encoder_enabled,
            compression_enabled=self.config.compression_enabled,
            cross_encoder_model=self.config.cross_encoder_model,
            rerank_top_k=self.config.rerank_top_k,
            compression_top_k=self.config.compression_top_k,
        )
        any_enabled = (
            adv_config.hyde_enabled
            or adv_config.query_rewriting_enabled
            or adv_config.cross_encoder_enabled
            or adv_config.compression_enabled
        )
        advanced_rag = (
            AdvancedRAGOrchestrator(adv_config, self._llm, self._embedder)
            if any_enabled else None
        )

        self._retriever = DualRetriever(
            kb_store=self._kb_store,
            report_store=self._report_store,
            advanced_rag=advanced_rag,
        )

    @property
    def report_metadata(self) -> Optional[ReportMetadata]:
        return self._report_store.metadata

    def load_report_from_uploaded_file(self, uploaded_file) -> ReportMetadata:
        """Ingest an uploaded PDF without ever writing plaintext to disk.

        Flow:
            uploaded bytes
              → DFS.store (encrypt + fragment + scatter into tempdir nodes)
              → DFS.reconstruct (plaintext bytes back in memory only)
              → parse_pdf_bytes (in-memory parse)
              → ReportStore.load_report_from_parsed (Fernet-encrypted chunks
                in a tempdir-rooted Chroma)
            The plaintext bytes go out of scope at function exit.
            The encrypted DFS fragments persist for the session and are wiped
            on next upload, end_session(), or interpreter exit.
        """
        raw = bytes(uploaded_file.getbuffer())

        self._dfs.store(raw)
        plaintext = self._dfs.reconstruct()
        # Drop the original buffer ASAP
        del raw

        try:
            parsed = parse_pdf_bytes(plaintext, uploaded_file.name)
        finally:
            del plaintext

        if parsed is None:
            self._dfs.delete_active()
            raise ValueError(f"Failed to parse uploaded PDF: {uploaded_file.name}")

        return self._report_store.load_report_from_parsed(parsed)

    def end_session(self) -> None:
        """Wipe all session-scoped patient state. The pipeline is unusable
        after this — callers should construct a new RAGPipeline."""
        try:
            self._report_store.wipe()
        except Exception as e:
            logger.warning("end_session: report_store wipe failed: %s", e)
        try:
            self._dfs.wipe()
        except Exception as e:
            logger.warning("end_session: dfs wipe failed: %s", e)

    def _build_context(self, question: str) -> FusedContext:
        return self._retriever.retrieve(
            query=question,
            kb_k=self.config.kb_top_k,
            report_k=self.config.report_top_k,
        )

    def ask(self, question: str) -> tuple[str, FusedContext]:
        """
        Non-streaming answer. Returns (answer_text, fused_context).
        """
        ctx = self._build_context(question)
        answer = self._llm.ask(question, ctx)
        return answer, ctx

    def ask_stream(self, question: str) -> tuple[Iterable[str], FusedContext]:
        """
        Streaming answer. Returns (token_stream, fused_context).
        """
        ctx = self._build_context(question)
        stream = self._llm.ask_stream(question, ctx)
        return stream, ctx

