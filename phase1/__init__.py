"""
AutismRAG Phase 1 package.

Reuses the robust ingestion pipeline from the Capestone RAG project:
  - pdf_parser: clinical PDF parsing and cleaning
  - chunker: medically-aware semantic chunking
  - embedder: Ollama-based embedding wrapper
  - vector_store: ChromaDB knowledge base management
  - ingest: orchestration script / CLI
"""

