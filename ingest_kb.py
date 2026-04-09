"""
Legacy simple ingestion script.

Now acts as a thin wrapper around the richer Phase 1 pipeline in
`phase1.ingest`. You can still run:

    python ingest_kb.py

but internally this just calls `python -m phase1.ingest` with the
defaults from config.ini.
"""

import os
from phase1.ingest import run_ingestion, load_config, setup_logging


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(base_dir, "config.ini"))

    kb_dir = cfg.get("paths", "kb_data_dir", fallback="./data/kb_pdfs")
    chroma_dir = cfg.get("paths", "chroma_db_dir", fallback="./data/chroma_kb")
    embed_model = cfg.get("ollama", "embed_model", fallback="nomic-embed-text")
    ollama_url = cfg.get("ollama", "base_url", fallback="http://localhost:11434")
    collection_name = cfg.get(
        "chroma",
        "kb_collection_name",
        fallback="autism_kb_medical_guidelines",
    )
    chunk_size = cfg.getint("chunking", "chunk_size", fallback=800)
    chunk_overlap = cfg.getint("chunking", "chunk_overlap", fallback=150)
    min_chunk_size = cfg.getint("chunking", "min_chunk_size", fallback=100)
    log_dir = cfg.get("paths", "log_dir", fallback="./logs")

    setup_logging(log_dir=log_dir, level="INFO")

    os.makedirs(chroma_dir, exist_ok=True)

    run_ingestion(
        kb_dir=kb_dir,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embed_model=embed_model,
        ollama_url=ollama_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        dry_run=False,
        reset=False,
        verify_query=None,
    )


if __name__ == "__main__":
    main()


