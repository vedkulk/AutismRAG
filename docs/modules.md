# Module reference

One section per file, in roughly the order the runtime touches them. For each, the public surface and the key implementation choices.

---

## `app.py` — Streamlit UI

The presentation layer. Everything else can be driven without it via `RAGPipeline`.

**Responsibilities:**
- Render the chat (welcome screen, message bubbles, sources expander, retrieval-quality strip).
- Sidebar: PDF upload, **Load report**, **Clear chat**, **End session**.
- Session lifecycle: lazily build `RAGPipeline`, sweep orphan tempdirs once per app boot.
- Privacy enforcement: `_enforce_in_memory_logging()` strips every `FileHandler` from the root logger on import.

**Notable details:**
- `init_state()` runs `dfs.sweep_orphans()` on first run to clean up `rag_dfs_*` tempdirs from crashed sessions.
- `is_clearly_clinical(question)` is a defensive heuristic so obviously clinical questions are never rejected by upstream classifiers.
- All HTML is hand-rendered through `unsafe_allow_html=True` for layout precision.
- Streaming uses a single `st.empty()` placeholder that gets re-`markdown`-ed per token, with a typing indicator drawn before retrieval kicks in.

---

## `rag_pipeline.py` — pipeline facade

`RAGPipeline` is the single entry point the UI talks to.

**Public surface:**
- `RAGConfig` (dataclass) + `RAGConfig.from_ini(path)`
- `RAGPipeline(config)` — constructs embedder, KB store, report store, LLM engine, DFS store, advanced RAG orchestrator, dual retriever
- `load_report_from_uploaded_file(uploaded_file) -> ReportMetadata`
- `end_session() -> None` — wipes report store + DFS store
- `ask(question) -> (answer, FusedContext)`
- `ask_stream(question) -> (Iterable[str], FusedContext)`
- `report_metadata` property

**Notable details:**
- The advanced-RAG orchestrator is only constructed if at least one technique is enabled; otherwise `DualRetriever` runs without post-processing.
- `load_report_from_uploaded_file` deliberately drops references (`del raw`, `del plaintext`) ASAP to shorten the plaintext lifetime in memory.

---

## `dual_retriever.py` — query classification, routing, fusion

The retrieval brain. Decides where to look, normalizes vague language, fuses results.

**Public surface:**
- `ContextChunk` (dataclass: `text`, `metadata`, `similarity`, `source_type`)
- `FusedContext` (dataclass: `query`, `chunks`, `as_prompt_block()`)
- `classify_query(query) -> "patient" | "medical" | "both"`
- `normalize_query(query, query_type) -> str`
- `DualRetriever.retrieve(query, kb_k, report_k) -> FusedContext`

**Implementation notes:**
- `PATIENT_KEYWORDS` and `MEDICAL_KEYWORDS` drive classification. If both fire, the query is classified `"both"`. Default for an ambiguous query is also `"both"` (assumes a report is loaded).
- `_VAGUE_PATTERNS` table catches phrases like "tell me about this kid" and rewrites them into document-aligned terminology. `_DOMAIN_EXPANSIONS` adds clinical-section vocabulary to single-word topics.
- `_fuse()` keys on stripped chunk text so duplicate text from KB and report deduplicates correctly. Report chunks always sort before KB chunks.
- After post-retrieval, a smart selector caps the final context at **5 report + 3 KB** chunks. Tuned so the LLM gets enough patient signal without burying it in KB material.
- `MIN_CHUNKS_FOR_LLM = 2` guards against post-retrieval (rerank+compress) dropping everything; if so, the raw fused list is restored.

---

## `advanced_rag.py` — Phase 4 enhancements

Three independently toggleable retrieval enhancements, each fail-soft.

**Public surface:**
- `AdvancedRAGConfig` (dataclass)
- `AdvancedRAGOrchestrator.pre_retrieval(query) -> List[str]` — query variants
- `AdvancedRAGOrchestrator.post_retrieval(query, chunks) -> List[ContextChunk]`
- `QueryRewriter`, `CrossEncoderReranker`, `ContextCompressor` (used internally)

**Techniques:**

| Component | What it does | Tuning |
|---|---|---|
| `QueryRewriter` | LLM rewrites the question into up to 3 retrieval-optimized variants using DSM-5/ICD-10/ADOS-2 vocabulary. Dedupes case-insensitively. | hardcoded prompt; max 3 variants |
| `CrossEncoderReranker` | Scores `(query, chunk)` pairs with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Always preserves ≥2 report chunks. Sigmoid-normalizes logits to `[0,1]` for display. | `rerank_top_k` |
| `ContextCompressor` | Embedding-based: keeps chunks whose cosine similarity to the query is ≥ `COMPRESSION_SIM_THRESHOLD` (0.30). Replaces the previous LLM-per-chunk version (too slow, too aggressive). | `compression_top_k` |

**Implementation notes:**
- `_normalize_ce_score` uses a scaled sigmoid (`1/(1+exp(-x/3))`) instead of a clipped linear map. The old `(score+10)/20` saturated to 0 for clinical text.
- `pre_retrieval` only runs query rewriting; if it fails, returns `[]` and the retriever falls back to the single search query.
- `post_retrieval` runs rerank → compress in sequence; each step is wrapped in try/except so a runtime failure doesn't break the pipeline.

---

## `kb_memory.py` — persistent knowledge base store

`InMemoryKB` reads the persistent KB built by `phase1.ingest`, caches embeddings to disk, and serves hybrid retrieval.

**Public surface:**
- `KBEntry` (dataclass)
- `InMemoryKB(data_dir, embedder, chunk_size, chunk_overlap, min_chunk_size, retrieval_type, mmr_lambda, cache_dir="./data/kb_cache")`
- `similarity_search(query, k) -> List[Dict]` — hybrid (dense + BM25 via RRF)

**Implementation notes:**
- Caches `(embeddings, metadata)` to `data/kb_cache/kb_cache.npz` keyed by a SHA hash of `data/kb_pdfs/`. Re-runs reuse the cache; new PDFs trigger a re-embed.
- Dense search: cosine similarity over numpy. Sparse search: `rank_bm25.BM25Okapi` over a simple tokenizer.
- RRF fusion is a Reciprocal Rank Fusion merge of the two ranked lists.
- `retrieval_type` toggles `similarity` vs `mmr` (Maximal Marginal Relevance, parameterized by `mmr_lambda`) for diversity.

---

## `report_store.py` — per-session encrypted report store

`ReportStore` is the report-side counterpart of `InMemoryKB`: hybrid retrieval over a single uploaded PDF.

**Public surface:**
- `ReportMetadata` (dataclass: `filename`, `num_pages`, `num_chunks`)
- `ReportStore(embedder)`
- `load_report_from_parsed(parsed: ParsedDocument) -> ReportMetadata`
- `similarity_search(query, k) -> List[Dict]` — hybrid (dense + BM25 via RRF)
- `wipe() -> None`
- `metadata` property

**Implementation notes:**
- Persists Chroma data into a system tempdir (`tempfile.mkdtemp(prefix="rag_chroma_")`) so the on-disk vector files don't survive the session.
- Chunk *text* is encrypted with `crypto_utils.encrypt_text` (Fernet) before being added to Chroma. Embeddings are stored as-is.
- Loading a new report replaces the previous Chroma collection; the old tempdir is removed.
- An `atexit` callback wipes the tempdir on interpreter exit.

---

## `llm_engine.py` — Ollama wrapper

Thin LangChain shim around `ChatOllama`. Owns the clinical system prompt and conversation history.

**Public surface:**
- `LLMEngine(model, base_url, system_prompt=DEFAULT_SYSTEM_PROMPT)`
- `ask(question, ctx: FusedContext) -> str`
- `ask_stream(question, ctx) -> Iterable[str]`
- `generate_raw(prompt) -> str` — used by `QueryRewriter` for non-context-grounded prompts
- `clear_history() -> None`

**Implementation notes:**
- The default system prompt frames the LLM as a "clinical decision-support assistant for licensed healthcare providers" and explicitly distinguishes PATIENT REPORT vs MEDICAL KB context.
- History is bounded (recent N turns) to keep the prompt reasonable.
- Streaming uses LangChain's stream interface; non-streaming wraps it.

---

## `crypto_utils.py` — Fernet helpers

Application-level chunk-text encryption.

**Public surface:**
- `get_fernet() -> Fernet` (cached)
- `encrypt_text(plain) -> str`
- `decrypt_text(ciphertext) -> str`

**Key resolution order:**
1. `RAG_FERNET_KEY` environment variable
2. `./secret.key` file
3. Generate a new key, save it to `./secret.key`, and use that

This key is reused across the persistent KB Chroma store and per-session report Chroma store, so chunks encrypted under one are decryptable under the other.

---

## `dfs/` — ephemeral encrypted+fragmented store

The package's public API is exactly two names (see `dfs/__init__.py`):

```python
from dfs import EphemeralDFSStore, sweep_orphans
```

The remaining files in `dfs/` (`main.py`, `retrieve.py`, `transfer_encrypt.py`, `transfer_decrypt.py`, `drive_config.py`, `metadata_manager.py`, `storage_manager.py`) are a separate standalone CLI for cross-device transfer; the integrated app does not import them.

### `dfs/ephemeral_store.py`

`EphemeralDFSStore` is the single class the RAG pipeline interacts with.

**Public surface:**
- `StoredHandle` (dataclass: `paths`, `nonce`, `compressed_size`, `original_size`)
- `EphemeralDFSStore(num_fragments=10, num_key_shares=5, key_threshold=3)`
- `store(data: bytes) -> StoredHandle`
- `reconstruct(handle: StoredHandle = None) -> bytes`
- `delete_active() -> None`
- `wipe() -> None`
- `sweep_orphans() -> int` — module-level; cleans `rag_dfs_*` tempdirs from prior runs

**Lifecycle:**
1. Constructor creates a system tempdir at `/var/folders/.../rag_dfs_XXXX/`.
2. `store(bytes)` does: zstd compress → AES-256-GCM encrypt with a fresh key → split ciphertext into N fragments → write fragments into `node_*/` subdirs → Shamir-split the AES key (`threshold`-of-`num_shares`) → write key shares.
3. `reconstruct()` reverses the chain to produce plaintext bytes in memory.
4. `wipe()` recursively unlinks the tempdir. An `atexit` hook plus a `weakref.finalize` ensure cleanup on normal interpreter exit.

### Internal helpers (used only by `ephemeral_store`)

- `dfs/encryption.py` — `EncryptionManager`: AES-256-GCM via `cryptography.hazmat`.
- `dfs/key_manager.py` — `KeyManager`: Shamir secret-sharing via `shamir_mnemonic`.
- `dfs/erasure_manager.py` — `split_into_fragments` / `merge_fragments`: simple byte-range split, not erasure-coded despite the name.
- `dfs/compression_manager.py` — `CompressionManager`: zstd wrapper.

---

## `phase1/` — KB ingestion pipeline

Build-time only. Run via `python ingest_kb.py` (or directly `python -m phase1.ingest`).

| File | Purpose |
|---|---|
| `phase1/pdf_parser.py` | `parse_pdf`, `parse_pdf_bytes`, `extract_documents_from_dir`. Tries pdfplumber first, falls back to PyPDF2. `_clean_text` normalizes whitespace and headers. |
| `phase1/chunker.py` | `chunk_documents(documents, chunk_size, chunk_overlap)`. Splits on medical section headings ("Diagnosis", "Treatment", "History") first, then recursive character splitting within sections. Adds rich metadata. |
| `phase1/embedder.py` | `OllamaEmbedder(model, base_url)`. `embed_query`, `embed_texts` (batched, retries, progress). |
| `phase1/vector_store.py` | `KnowledgeBaseStore(persist_dir, collection_name, embedder)`. Idempotent `add_chunks` (won't duplicate), `similarity_search`, `mmr_search`. |
| `phase1/ingest.py` | CLI: `run_ingestion(...)` reads PDFs, chunks, embeds, upserts into Chroma. Reads `config.ini`. Logs to `logs/ingestion.log`. |

`parse_pdf_bytes` is also imported at runtime by `RAGPipeline.load_report_from_uploaded_file` for the in-memory parse of the uploaded PDF.

---

## `ragas_eval.py` + `run_evaluation.py` — evaluation

`ragas_eval.py` implements four RAGAS-style metrics, all using the local Ollama model as judge (no external API):

- `FaithfulnessScorer` — is the answer grounded in the retrieved context?
- `AnswerRelevancyScorer` — is the answer relevant to the question? (generates N hypothetical questions and scores cosine sim to the original)
- `ContextPrecisionScorer` — are retrieved contexts relevant and well-ordered?
- `ContextRecallScorer` — does the context cover the ground truth answer?

`RAGASEvaluator` runs all four over a list of `EvalSample`s and returns an `EvalReport`.

`run_evaluation.py` is the CLI that drives this against `data/eval_dataset.json` (KB-only) or `data/eval_dataset_report.json` (report-grounded, default). It loads the matching report PDF from `data/session_reports/` per question, runs the pipeline, and writes a JSON report to `logs/eval_report_<timestamp>.json`.

CLI flags:
- `--dataset PATH` — pick a different eval dataset
- `--report-pdf PATH` — override the per-question report (apply one PDF to all)
- `--advanced-rag` — force-enable all advanced RAG techniques
- `--cross-encoder` — only enable cross-encoder reranking
- `--limit N` — first N questions only

---

## `test_retrieval.py`, `test_validation.py`

Lightweight non-RAGAS tests. Both use `FakeUpload` from `run_evaluation.py` to feed local PDFs through `RAGPipeline.load_report_from_uploaded_file` without going through Streamlit.

- `test_retrieval.py` — prints what chunks come back per question, no LLM call. Useful for retrieval-tuning regressions.
- `test_validation.py` — runs a small fixed validation set against a specific dummy report.

---

## `ingest_kb.py`

A 50-line wrapper that loads `config.ini` and calls `phase1.ingest.run_ingestion`. Lives at the project root for ergonomics (`python ingest_kb.py`).
