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

from dataclasses import dataclass
from typing import Optional, Iterable
from pathlib import Path

from phase1.embedder import OllamaEmbedder
from report_store import ReportStore, ReportMetadata
from dual_retriever import DualRetriever, FusedContext
from llm_engine import LLMEngine
from kb_memory import InMemoryKB
from advanced_rag import AdvancedRAGConfig, AdvancedRAGOrchestrator


@dataclass
class RAGConfig:
    kb_persist_dir: str = "./data/chroma_kb"
    kb_collection_name: str = "autism_kb_medical_guidelines"
    kb_top_k: int = 20
    report_top_k: int = 8
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1"
    embed_model: str = "nomic-embed-text"
    # Phase 4 — Advanced RAG
    hyde_enabled: bool = False
    query_rewriting_enabled: bool = False
    cross_encoder_enabled: bool = False
    compression_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 15
    compression_top_k: int = 5


class RAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or RAGConfig()
        self._embedder = OllamaEmbedder(
            model=self.config.embed_model,
            base_url=self.config.ollama_base_url,
        )
        self._kb_store = InMemoryKB(
            data_dir="./data/kb_pdfs",
            embedder=self._embedder,
        )
        self._report_store = ReportStore(embedder=self._embedder)
        self._llm = LLMEngine(
            model=self.config.llm_model,
            base_url=self.config.ollama_base_url,
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
        """
        Save the uploaded PDF to a temp path under ./data and ingest it.
        Streamlit passes an UploadedFile which provides a file-like object.
        """
        reports_dir = Path("./data/session_reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / uploaded_file.name
        with out_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())

        meta = self._report_store.load_report_from_path(str(out_path))
        return meta

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

