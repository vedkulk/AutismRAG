from __future__ import annotations

"""
Ephemeral report-level vector store.

For each uploaded patient report PDF, we:
  - parse and clean it using phase1.pdf_parser
  - chunk it using phase1.chunker
  - embed + store chunks in an in-memory Chroma collection

The store is per-Streamlit session; when a new report is loaded,
the previous in-memory collection is replaced.
"""

import uuid
from dataclasses import dataclass
from typing import Optional, List

import chromadb
from chromadb.config import Settings

from phase1.pdf_parser import ParsedDocument, parse_pdf
from phase1.chunker import TextChunk, chunk_document
from phase1.embedder import OllamaEmbedder
from crypto_utils import encrypt_text, decrypt_text


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

    @property
    def metadata(self) -> Optional[ReportMetadata]:
        return self._meta

    def _reset_collection(self) -> None:
        # Drop reference to current collection; a new one will be created on next use.
        self._collection = None
        self._meta = None

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

        report_id = uuid.uuid4().hex
        self._meta = ReportMetadata(
            report_id=report_id,
            filename=parsed.filename,
            num_pages=parsed.num_pages,
            num_chunks=len(chunks),
        )
        return self._meta

    def similarity_search(self, query: str, k: int = 4) -> list[dict]:
        """
        Simple cosine similarity search over the report chunks.
        Returns list of {text, metadata, distance, rank}.
        """
        if self._collection is None:
            return []

        query_embedding = self._embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        out: list[dict] = []
        for i, (enc_doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            doc = decrypt_text(enc_doc)
            out.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": float(dist),
                    "similarity": round(1 - float(dist), 4),
                    "rank": i + 1,
                    "source_type": "report",
                }
            )
        return out

