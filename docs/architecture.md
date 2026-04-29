# Architecture

This document covers the end-to-end data flow, the dual-store retrieval design, and the privacy model that constrains the architecture. Read this before making non-trivial changes.

## System overview

The app is a Streamlit chat UI in front of a RAG pipeline that grounds answers in two stores:

- **Knowledge Base (KB)** — persistent, on-disk, no PHI. Built once from `data/kb_pdfs/` (DSM-5, ADOS-2 manuals, AAP guidelines, etc.).
- **Report store** — ephemeral, encrypted, in-memory only. One uploaded patient PDF per session.

A retriever fuses hits from both stores into a single context block, an LLM (local Ollama) generates the answer, and the UI streams tokens back.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app.py)                                              │
│   - sidebar: upload, load, end-session                               │
│   - chat: streaming bubbles, sources expander, retrieval-quality bar │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │   RAGPipeline       │   rag_pipeline.py
                  │   (facade)          │   - the only thing app.py imports
                  └──────────┬──────────┘
                             │
        ┌────────────────────┼────────────────────────────┐
        │                    │                            │
┌───────▼──────┐  ┌──────────▼──────────┐  ┌──────────────▼──────┐
│ DualRetriev  │  │ AdvancedRAG         │  │ LLMEngine           │
│ - classify   │  │   Orchestrator      │  │   (Ollama via       │
│ - normalize  │  │ - QueryRewriter     │  │    LangChain)       │
│ - per-store  │  │ - CrossEncoder      │  │ - clinical system   │
│   hybrid     │  │   Reranker          │  │   prompt            │
│   retrieval  │  │ - ContextCompressor │  │ - streaming +       │
│ - fuse +     │  │                     │  │   non-streaming     │
│   dedupe     │  │                     │  │                     │
└──┬────────┬──┘  └─────────────────────┘  └─────────────────────┘
   │        │
   │        └──────────────────────┐
   │                               │
┌──▼─────────────┐         ┌───────▼──────────┐
│  InMemoryKB    │         │  ReportStore     │
│  (kb_memory.py)│         │ (report_store.py)│
│                │         │                  │
│ - reads cached │         │ - tempdir-rooted │
│   embeddings + │         │   Chroma         │
│   metadata     │         │ - Fernet         │
│ - dense (cos)  │         │   encrypted      │
│   + BM25       │         │   chunks         │
│ - RRF fusion   │         │ - dense + BM25   │
└────────────────┘         │ - RRF fusion     │
                           └─────────▲────────┘
                                     │
                           ┌─────────┴────────┐
                           │ EphemeralDFSStore│  dfs/ephemeral_store.py
                           │ - zstd compress  │
                           │ - AES-GCM encrypt│
                           │ - split + Shamir │
                           │   3-of-5 key     │
                           │ - tempdir nodes  │
                           └──────────────────┘
```

## End-to-end data flow

### 1. Knowledge base ingestion (one-time, offline)

`python ingest_kb.py` (a wrapper around `phase1.ingest`):

```
data/kb_pdfs/*.pdf
   └─► phase1.pdf_parser  (pdfplumber, falling back to PyPDF2)
        └─► phase1.chunker  (medical-section split → recursive char split)
             └─► phase1.embedder  (Ollama nomic-embed-text)
                  └─► phase1.vector_store.KnowledgeBaseStore
                       └─► data/chroma_kb/   (persistent ChromaDB)
```

At app boot, `InMemoryKB` reads the persistent KB plus a hash-fingerprinted cache (`data/kb_cache/kb_cache.npz` + `kb_meta.json`). If `data/kb_pdfs/` hasn't changed, the cache is used directly — no Chroma round-trip.

### 2. Report upload (per session)

When the user clicks **Load report**:

```
uploaded bytes
   ├─► EphemeralDFSStore.store(raw)
   │    └─► zstd compress
   │         └─► AES-256-GCM encrypt (fresh per-file key)
   │              └─► split into N fragments (default 10)
   │                   └─► scatter into rag_dfs_XXXX/node_*/  (system tempdir)
   │              └─► Shamir-split AES key (3-of-5 mnemonics) → key shares
   │
   ├─► EphemeralDFSStore.reconstruct(handle)
   │    └─► reassemble shares → AES key → AES-GCM decrypt → zstd decompress
   │         → plaintext bytes (in memory only)
   │
   ├─► phase1.pdf_parser.parse_pdf_bytes(plaintext, filename)
   │    └─► ParsedDocument
   │
   └─► ReportStore.load_report_from_parsed(parsed)
        └─► chunk → embed → write Fernet-encrypted chunk text into a
            tempdir-rooted Chroma collection + build BM25 index
```

The plaintext bytes go out of scope at function exit. The DFS fragments persist for the session and are wiped on next upload, **End session**, or interpreter exit (atexit + weakref finalizer).

### 3. Question handling (per turn)

```
user question  ──►  RAGPipeline.ask_stream(question)
                     │
                     ├─►  DualRetriever.retrieve(query, kb_k, report_k)
                     │     │
                     │     ├─ classify_query(query) → "patient" | "medical" | "both"
                     │     ├─ normalize_query(query, type)  (regex vague→specific)
                     │     ├─ adjust effective_kb_k / effective_report_k by type
                     │     │
                     │     ├─ AdvancedRAG.pre_retrieval(query)
                     │     │    └─ multi-query rewrite (only if non-patient)
                     │     │
                     │     ├─ ReportStore.similarity_search(...)   ◄─ hybrid: dense+BM25 (RRF)
                     │     ├─ for variant in query_variants:
                     │     │     InMemoryKB.similarity_search(...) ◄─ hybrid: dense+BM25 (RRF)
                     │     │
                     │     ├─ _fuse(report_hits, kb_hits)  → ContextChunk[]  (deduped)
                     │     │
                     │     ├─ AdvancedRAG.post_retrieval(query, fused)
                     │     │    ├─ CrossEncoderReranker (top rerank_top_k)
                     │     │    └─ ContextCompressor    (cosine ≥ 0.30 on top compression_top_k)
                     │     │
                     │     └─ smart final selection: 5 report + 3 KB → FusedContext
                     │
                     └─►  LLMEngine.ask_stream(question, ctx)
                           └─ system prompt + ctx.as_prompt_block() + history → Ollama
                              → token stream  ► UI placeholder
```

The UI then renders the answer plus a **Sources** expander listing each `ContextChunk` (badge for REPORT vs KB, similarity, filename, section), and a retrieval-quality strip below.

## Two layers of retrieval

The system runs hybrid retrieval **twice** at different granularities:

| Layer | Where | What it does |
|---|---|---|
| **Per-store hybrid** | inside `InMemoryKB.similarity_search` and `ReportStore.similarity_search` | Dense cosine search **and** BM25 keyword search, fused with Reciprocal Rank Fusion (RRF) per store |
| **Cross-store fusion** | `DualRetriever._fuse` | Merges KB hits and report hits into a single deduped list, sorted with report chunks first |

Cross-encoder reranking and compression then run on the cross-store result.

## Query classification and routing

`dual_retriever.classify_query` keyword-matches the question into:

| Type | Trigger | Effective `report_k` | Effective `kb_k` | Pre-retrieval |
|---|---|---|---|---|
| `patient` | "patient", "this child", "the report", … | `max(report_k, 10)` | `2` | skipped (no LLM rewrite) |
| `medical` | "DSM", "ADOS", "criteria", "treatment", … | `2` | full `kb_top_k` | rewrite enabled |
| `both` | both trigger sets fire, or default | `max(report_k, 8)` | `min(kb_top_k, 10)` | rewrite enabled |

Patient queries also bypass `_PROMPT`-driven advanced RAG entirely on the pre-retrieval side — the report-grounded chunks are what we want, and rewriting medical-vocabulary variants would dilute the signal.

`normalize_query` rewrites vague patient questions ("tell me about this kid") into document-aligned phrasing using a fixed regex pattern table plus a `_DOMAIN_EXPANSIONS` dict that maps "social" → "social interaction observations", etc. No LLM call.

## Privacy model

Patient data **never persists on disk in plaintext**. The architecture is built around this invariant:

- **No file logging.** `app.py:_enforce_in_memory_logging` strips every `FileHandler` from the root logger on boot and routes everything to stderr only.
- **DFS-encrypted upload.** The uploaded PDF is zstd-compressed, AES-256-GCM encrypted with a fresh per-file key, fragmented into ten pieces, and scattered into `node_*/` subdirs of a system tempdir (`/var/folders/.../rag_dfs_XXXX/`). The AES key is Shamir-split (3-of-5).
- **Plaintext only in memory.** `parse_pdf_bytes` operates on the reconstructed bytes; they go out of scope at the end of `RAGPipeline.load_report_from_uploaded_file`.
- **Encrypted chunks at rest.** `ReportStore` stores chunk *text* Fernet-encrypted (`crypto_utils.encrypt_text`) inside a tempdir-rooted Chroma collection. Embeddings are not encrypted (vectors don't reveal content directly).
- **Best-effort cleanup.** The DFS store registers `atexit` and `weakref.finalize` callbacks that recursively unlink the tempdir on interpreter exit. `sweep_orphans()` runs once per Streamlit boot to purge any `rag_dfs_*` left over from a prior crashed session.
- **Explicit wipe.** **End session** in the sidebar calls `pipeline.end_session()` → `report_store.wipe()` + `dfs.wipe()`.

What this is **not** designed to defend against: kernel-level memory introspection, swap, hibernation snapshots, malicious code in the same process. It's "best effort against casual disk inspection and crash leftovers."

## Where state lives

| Path | Lifetime | Encrypted? | Contains |
|---|---|---|---|
| `data/kb_pdfs/` | persistent | no | KB source documents (no PHI) |
| `data/chroma_kb/` | persistent | chunks encrypted (Fernet) | KB vector store |
| `data/kb_cache/{kb_cache.npz,kb_meta.json}` | persistent | no | KB embedding cache, hashed against `kb_pdfs/` |
| `data/eval_dataset.json`, `data/eval_dataset_report.json` | persistent | no | RAGAS eval datasets |
| `data/session_reports/*.pdf` | persistent (test only) | no | Test PDFs used by `run_evaluation.py`, `test_*.py` |
| `/var/folders/.../rag_dfs_XXXX/` | session | AES-GCM + Shamir | Encrypted PDF fragments |
| `/var/folders/.../rag_chroma_XXXX/` | session | chunks encrypted (Fernet) | Per-session report Chroma |
| `logs/eval_report_*.json` | persistent | no | RAGAS run outputs |
| `secret.key` | persistent | n/a (the key itself) | Fernet key for chunk encryption |
