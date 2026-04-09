
import argparse
import configparser
import logging
import sys
import time
from pathlib import Path


def setup_logging(log_dir: str = "./logs", level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "ingestion.log"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.ini") -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if Path(config_path).exists():
        cfg.read(config_path)
        logger.info(f"Config loaded from {config_path}")
    else:
        logger.warning(f"config.ini not found at {config_path}, using defaults.")
    return cfg


def run_ingestion(
    kb_dir: str,
    chroma_dir: str,
    collection_name: str,
    embed_model: str,
    ollama_url: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    dry_run: bool = False,
    reset: bool = False,
    verify_query: str = None,
) -> dict:
    from phase1.pdf_parser import extract_documents_from_dir
    from phase1.chunker import chunk_documents, print_chunk_stats
    from phase1.embedder import OllamaEmbedder
    from phase1.vector_store import KnowledgeBaseStore

    summary = {
        "documents_parsed": 0,
        "chunks_created": 0,
        "chunks_embedded": 0,
        "elapsed_seconds": 0,
        "errors": [],
    }

    start_time = time.time()

    if not dry_run:
        logger.info("=" * 55)
        logger.info("STEP 1: Checking Ollama connection...")
        logger.info("=" * 55)
        embedder = OllamaEmbedder(
            model=embed_model,
            base_url=ollama_url,
        )
        if not embedder.check_connection():
            logger.error(
                "\n  Cannot proceed without Ollama.\n"
                "    1. Start Ollama:           ollama serve\n"
                f"   2. Pull embedding model:  ollama pull {embed_model}\n"
            )
            sys.exit(1)
    else:
        embedder = None
        logger.info("DRY RUN: skipping Ollama connection check")

    logger.info("=" * 55)
    logger.info(f"STEP 2: Parsing documents from '{kb_dir}'...")
    logger.info("=" * 55)

    documents = extract_documents_from_dir(kb_dir)
    summary["documents_parsed"] = len(documents)

    if not documents:
        logger.error(
            f"  No documents found in '{kb_dir}'.\n"
            f"    Add PDF or .txt medical reference files to this folder."
        )
        sys.exit(1)

    logger.info(
        f"✓  Parsed {len(documents)} documents | "
        f"Total text: {sum(len(d.text) for d in documents):,} chars"
    )

    logger.info("=" * 55)
    logger.info("STEP 3: Chunking documents...")
    logger.info("=" * 55)

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )
    summary["chunks_created"] = len(chunks)

    print_chunk_stats(chunks)

    if dry_run:
        logger.info("DRY RUN complete — skipping embedding and storage.")
        summary["elapsed_seconds"] = round(time.time() - start_time, 1)
        return summary

    logger.info("=" * 55)
    logger.info("STEP 4: Initializing ChromaDB vector store...")
    logger.info("=" * 55)

    store = KnowledgeBaseStore(
        persist_dir=chroma_dir,
        collection_name=collection_name,
        embedder=embedder,
    )

    if reset:
        existing = store.count()
        if existing > 0:
            logger.warning(
                f"⚠️  --reset flag: deleting {existing} existing chunks "
                f"from '{collection_name}'..."
            )
            store.delete_collection()
            logger.info("Collection reset.")

    logger.info("=" * 55)
    logger.info(
        f"STEP 5: Embedding {len(chunks)} chunks with "
        f"'{embed_model}' and storing in ChromaDB..."
    )
    logger.info("=" * 55)
    logger.info(
        "⏳  This may take a few minutes depending on your hardware.\n"
        "    (nomic-embed-text is fast even on CPU)\n"
    )

    upserted = store.add_chunks(chunks, show_progress=True)
    summary["chunks_embedded"] = upserted

    logger.info("=" * 55)
    logger.info("STEP 6: Verification")
    logger.info("=" * 55)

    store.print_stats()

    if verify_query:
        logger.info(f"Running test query: '{verify_query}'")
        results = store.mmr_search(verify_query, k=3)
        print("\n─── Test Query Results ──────────────────────────────")
        print(f"  Query: \"{verify_query}\"\n")
        for r in results:
            src = r["metadata"].get("filename", "?")
            section = r["metadata"].get("section", "")
            print(f"  [{r['rank']}] Source: {src} | Section: {section}")
            print(f"      {r['text'][:200].replace(chr(10), ' ')}...")
            print()
        print("─────────────────────────────────────────────────────\n")

    elapsed = round(time.time() - start_time, 1)
    summary["elapsed_seconds"] = elapsed

    logger.info(
        f"\n  Phase 1 Ingestion Complete!\n"
        f"    Documents parsed : {summary['documents_parsed']}\n"
        f"    Chunks created   : {summary['chunks_created']:,}\n"
        f"    Chunks embedded  : {summary['chunks_embedded']:,}\n"
        f"    Time elapsed     : {elapsed}s\n"
        f"    KB stored at     : {chroma_dir}\n"
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutismRAG Phase 1 — Knowledge Base Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python -m phase1.ingest
        python -m phase1.ingest --kb-dir ./data/kb_pdfs
        python -m phase1.ingest --reset --verify "autism communication difficulties"
        python -m phase1.ingest --dry-run
                """,
    )
    parser.add_argument("--config", default="config.ini", help="Path to config.ini")
    parser.add_argument("--kb-dir", help="Override KB data directory")
    parser.add_argument("--chroma-dir", help="Override ChromaDB persist directory")
    parser.add_argument("--model", help="Override Ollama embedding model")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection and rebuild from scratch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only, skip embedding",
    )
    parser.add_argument(
        "--verify",
        metavar="QUERY",
        help="Run a test query after ingestion",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    kb_dir = args.kb_dir or cfg.get(
        "paths",
        "kb_data_dir",
        fallback="./data/kb_pdfs",
    )
    chroma_dir = args.chroma_dir or cfg.get(
        "paths",
        "chroma_db_dir",
        fallback="./data/chroma_kb",
    )
    embed_model = args.model or cfg.get(
        "ollama",
        "embed_model",
        fallback="nomic-embed-text",
    )
    ollama_url = cfg.get(
        "ollama",
        "base_url",
        fallback="http://localhost:11434",
    )
    collection_name = cfg.get(
        "chroma",
        "kb_collection_name",
        fallback="autism_kb_medical_guidelines",
    )
    chunk_size = cfg.getint("chunking", "chunk_size", fallback=800)
    chunk_overlap = cfg.getint("chunking", "chunk_overlap", fallback=150)
    min_chunk_size = cfg.getint("chunking", "min_chunk_size", fallback=100)
    log_dir = cfg.get("paths", "log_dir", fallback="./logs")

    setup_logging(log_dir=log_dir, level=args.log_level)

    logger.info("\n" + "=" * 55)
    logger.info("  AutismRAG — Phase 1: Knowledge Base Ingestion")
    logger.info("=" * 55)
    logger.info(f"  KB dir       : {kb_dir}")
    logger.info(f"  ChromaDB dir : {chroma_dir}")
    logger.info(f"  Embed model  : {embed_model}")
    logger.info(f"  Chunk size   : {chunk_size} chars (overlap: {chunk_overlap})")
    logger.info(f"  Dry run      : {args.dry_run}")
    logger.info(f"  Reset        : {args.reset}")
    logger.info("=" * 55 + "\n")

    run_ingestion(
        kb_dir=kb_dir,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embed_model=embed_model,
        ollama_url=ollama_url,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        dry_run=args.dry_run,
        reset=args.reset,
        verify_query=args.verify,
    )


if __name__ == "__main__":
    main()

