# AutismRAG — Complete Technical Reference

> Single-document deep-dive. Complements `architecture.md`, `modules.md`, `configuration.md`, and `evaluation.md` with algorithm-level detail, sequence diagrams, and the full public API surface.

---

## Table of Contents

1. [System Overview & Invariants](#1-system-overview--invariants)
2. [Architecture (Layered)](#2-architecture-layered)
3. [End-to-End Flows](#3-end-to-end-flows)
   - 3.1 KB Ingestion (offline)
   - 3.2 Report Upload (per session)
   - 3.3 Question Handling (per turn)
4. [Cryptographic Protocol](#4-cryptographic-protocol)
5. [Storage Layer](#5-storage-layer)
6. [Retrieval Algorithms](#6-retrieval-algorithms)
7. [Advanced RAG Techniques](#7-advanced-rag-techniques)
8. [LLM Engine](#8-llm-engine)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Configuration Reference](#10-configuration-reference)
11. [Public API Surface](#11-public-api-surface)
12. [Data Structures](#12-data-structures)
13. [Failure Modes & Recovery](#13-failure-modes--recovery)
14. [Performance Characteristics](#14-performance-characteristics)
15. [Testing Strategy](#15-testing-strategy)
16. [Threat Model](#16-threat-model)
17. [Appendix — File Structure](#17-appendix--file-structure)

---

## 1. System Overview & Invariants

### What it is
AutismRAG is a privacy-conscious clinical decision-support system for pediatric Autism Spectrum Disorder (ASD) screening. A clinician uploads a patient developmental report (PDF), and the system answers natural-language clinical questions grounded in two evidence sources:

- **Persistent Knowledge Base (KB)** — DSM-5, ICD-11, ADOS-2, M-CHAT-R/F, NICE CG128, AAP guidelines, etc.
- **Per-session Report Store** — the uploaded patient PDF, encrypted at every storage layer.

### Five structural invariants
| # | Invariant | Where enforced |
|---|---|---|
| **I1** | **No plaintext patient data persists on disk** | `dfs/ephemeral_store.py` (AES-GCM + Shamir + tempdir), `report_store.py` (Fernet on Chroma chunks), `app.py:_enforce_in_memory_logging` |
| **I2** | **No conversation history retained** | `llm_engine.py:_build_messages` rebuilds messages from scratch every turn |
| **I3** | **No external API calls during inference** | All embeddings + LLM calls go to local Ollama; no OpenAI/Anthropic/Cohere |
| **I4** | **Fail-soft retrieval** | `advanced_rag.py` wraps every technique in try/except; `dual_retriever.py:MIN_CHUNKS_FOR_LLM=2` fallback |
| **I5** | **Idempotent KB ingestion** | `phase1/vector_store.py` deduplicates chunks by hash on upsert |

### Tech stack
| Layer | Component | Version |
|---|---|---|
| OS | macOS 13+ / Ubuntu 22.04 LTS | x86_64 or Apple Silicon |
| LLM runtime | Ollama | 0.1.30+ |
| LLM model | llama3.1 (8B Q4) | — |
| Embedding model | nomic-embed-text | 768-dim |
| Vector DB | ChromaDB | 0.4.x |
| LangChain | langchain-community + langchain-ollama | 0.1.x |
| BM25 | rank_bm25 | BM25Okapi |
| Cross-encoder | sentence-transformers | ms-marco-MiniLM-L-6-v2 (~80 MB) |
| Crypto | cryptography (AES-GCM, Fernet) | 41.0+ |
| Key sharing | shamir_mnemonic | 0.3+ (SLIP-39) |
| Compression | zstandard | 0.22+ (level 3) |
| PDF | pdfplumber → PyPDF2 fallback | 0.9+, 3.x |
| UI | Streamlit | 1.32+ |
| Python | CPython | 3.10+ |

---

## 2. Architecture (Layered)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 7: PRESENTATION                                                  │
│  app.py — Streamlit UI (chat bubbles, sources expander, sidebar)        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 6: PIPELINE FACADE                                               │
│  rag_pipeline.py — RAGPipeline (single entry point) + RAGConfig         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 5: RETRIEVAL ORCHESTRATION                                       │
│  dual_retriever.py — classify → normalize → route → fuse → final select │
│  advanced_rag.py — query rewrite (pre) + rerank + compress (post)       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 4: GENERATION                                                    │
│  llm_engine.py — ChatOllama wrapper (clinical prompt, streaming)        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3: VECTOR STORES                                                 │
│  kb_memory.py        report_store.py                                    │
│  - persistent KB     - per-session ephemeral                            │
│  - dense + BM25      - dense (Chroma) + BM25                            │
│  - RRF + MMR         - RRF + Fernet chunk encryption                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 2: STORAGE & PRIVACY                                             │
│  dfs/ephemeral_store.py — DFS storage orchestrator                      │
│  dfs/compression_manager.py — zstd L3                                   │
│  dfs/erasure_manager.py — byte-range fragmentation                      │
│  dfs/encryption.py — AES-256-GCM                                        │
│  dfs/key_manager.py — Shamir SLIP-39 (3-of-5)                           │
│  crypto_utils.py — Fernet symmetric (chunk-level)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 1: INGESTION (build-time only)                                   │
│  phase1/pdf_parser.py → chunker.py → embedder.py → vector_store.py      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module dependency direction
```
app.py
  └→ rag_pipeline.py
       ├→ dual_retriever.py
       │    ├→ kb_memory.py ←── phase1/* (build-time)
       │    ├→ report_store.py ←── crypto_utils.py
       │    └→ advanced_rag.py
       │         └→ llm_engine.py (for query rewriting)
       ├→ llm_engine.py
       └→ dfs/ephemeral_store.py
            ├→ dfs/compression_manager.py
            ├→ dfs/encryption.py
            ├→ dfs/erasure_manager.py
            └→ dfs/key_manager.py
```

**One-way crypto dependency:** `dfs/ephemeral_store.py` consumes `dfs/encryption.py` and `dfs/key_manager.py` as a black box. Crypto core is independently testable.

---

## 3. End-to-End Flows

### 3.1 KB Ingestion — Offline (`python ingest_kb.py`)

Build-time only. Run once after dropping new PDFs into `data/kb_pdfs/`.

```
┌──────────────────────┐
│ data/kb_pdfs/*.pdf   │
└──────────┬───────────┘
           │  for each PDF
           ▼
┌──────────────────────────────────────────┐
│ phase1.pdf_parser.parse_pdf(path)        │
│  1. Try pdfplumber (handles tables/      │
│     multi-column)                        │
│  2. On failure: PyPDF2 fallback          │
│  3. _clean_text:                         │
│     - normalize whitespace               │
│     - strip page-number headers/footers  │
│  → ParsedDocument(text, num_pages,       │
│     filename, source_path, metadata)     │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│ phase1.chunker.chunk_documents(...)      │
│  1. _split_into_sections — regex match   │
│     medical headings (DIAGNOSIS,         │
│     TREATMENT, HISTORY, …) split first   │
│  2. _recursive_split each section to     │
│     chunk_size=300 chars with            │
│     chunk_overlap=40 chars               │
│  3. Prepend section label to each chunk  │
│     so retrieval has context             │
│  4. Drop chunks shorter than             │
│     min_chunk_size=80                    │
│  → List[TextChunk] with metadata         │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│ phase1.embedder.OllamaEmbedder           │
│  - batches of 32                         │
│  - up to 3 retries with 2s backoff       │
│  - validates dimension on first batch    │
│    (must be 768)                         │
│  → List[List[float]]                     │
└──────────┬───────────────────────────────┘
           ▼
┌──────────────────────────────────────────┐
│ phase1.vector_store.KnowledgeBaseStore   │
│  - ChromaDB persistent client            │
│  - Idempotent upsert: chunk_id collision │
│    → skip (deduplicates on rerun)        │
│  - Fernet-encrypts chunk text BEFORE     │
│    adding to Chroma (crypto_utils)       │
│  → data/chroma_kb/                       │
└──────────────────────────────────────────┘
```

**Caching layer (runtime, not ingestion):**
After ingestion, the *next* app boot triggers `kb_memory.InMemoryKB._ensure_loaded()` which caches embeddings to `data/kb_cache/kb_cache.npz`. Cache key is MD5(filename+size for each PDF) plus chunk_size/overlap/min_chunk_size. Subsequent boots load in <1s.

### 3.2 Report Upload — Per Session

Triggered when the clinician clicks **Load Report** in the sidebar.

```
┌──────────────────────┐
│ uploaded_file        │  Streamlit UploadedFile
│ (in browser memory)  │
└──────────┬───────────┘
           │  raw = bytes(uploaded_file.getbuffer())
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ EphemeralDFSStore.store(raw)                                    │
│                                                                 │
│  1. CompressionManager.compress(raw)                            │
│     zstd level 3 → compressed bytes                             │
│                                                                 │
│  2. EncryptionManager.generate_key()                            │
│     AESGCM.generate_key(bit_length=256) → key                   │
│                                                                 │
│  3. EncryptionManager.encrypt(compressed, key)                  │
│     nonce = os.urandom(12) (96-bit)                             │
│     ciphertext = AESGCM(key).encrypt(nonce, compressed, None)   │
│     → (ciphertext, nonce)                                       │
│                                                                 │
│  4. split_into_fragments(ciphertext, num_fragments=10)          │
│     deterministic byte-range split → 10 fragments               │
│                                                                 │
│  5. KeyManager.split_key(key, num_shares=5, threshold=3)        │
│     shamir_mnemonic.generate_mnemonics(...) → 5 SLIP-39 strs   │
│                                                                 │
│  6. Write to /tmp/rag_dfs_XXXX/                                 │
│     - fragments → node_0/.../node_4/frag_NNN.bin (round-robin)  │
│     - shares    → node_0/.../node_4/share_NNN.txt (round-robin) │
│                                                                 │
│  7. del key  ← drop the AES key reference ASAP                  │
│                                                                 │
│  → StoredHandle(fragment_paths, key_share_paths, nonce,         │
│                 threshold, plaintext_size, metadata)            │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ EphemeralDFSStore.reconstruct(handle)                           │
│                                                                 │
│  1. ciphertext = merge_fragments([read(p) for p in frag_paths]) │
│  2. shares = [read(p) for p in key_share_paths[:threshold]]     │
│  3. key = KeyManager.reconstruct_key(shares)                    │
│     shamir_mnemonic.combine_mnemonics(shares) → 32 bytes        │
│  4. compressed = AESGCM(key).decrypt(ciphertext, nonce, None)   │
│  5. del key   ← drop reference inside try/finally               │
│  6. plaintext = CompressionManager.decompress(compressed)       │
│                                                                 │
│  → plaintext bytes (in MEMORY ONLY)                             │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ phase1.pdf_parser.parse_pdf_bytes(plaintext, filename)          │
│  - Same pdfplumber → PyPDF2 fallback as ingestion               │
│  - Operates on bytes, never writes to disk                      │
│  → ParsedDocument                                               │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ├── del plaintext  ← drop the decrypted bytes ASAP
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ReportStore.load_report_from_parsed(parsed)                     │
│                                                                 │
│  1. _reset_collection()  drop any prior collection              │
│  2. chunks = chunk_document(parsed, 300/40/50)                  │
│  3. embeddings = OllamaEmbedder.embed_texts(plain_texts)        │
│  4. encrypted_texts = [crypto_utils.encrypt_text(t)             │
│                        for t in plain_texts]                    │
│  5. ChromaDB.add(ids, embeddings, documents=encrypted_texts,    │
│                  metadatas)                                     │
│  6. Build BM25Okapi over plain_texts (in-process, in-memory)    │
│  7. Generate uuid for collection                                │
│                                                                 │
│  → ReportMetadata(report_id, filename, num_pages, num_chunks)   │
└─────────────────────────────────────────────────────────────────┘
```

**Critical privacy mechanics:**
- `raw` (uploaded bytes) is `del raw`'d after `dfs.store()`
- `plaintext` (decrypted bytes) is `del plaintext`'d after `parse_pdf_bytes()`
- Inside `dfs.reconstruct()`, the AES key is `del key`'d in a `try/finally`
- The `chroma_dir` is a system tempdir (`/tmp/rag_chroma_XXXX/`), wiped on session end
- BM25 index lives in process memory only — never serialized to disk

### 3.3 Question Handling — Per Turn

Triggered when the clinician types a question in the chat input.

```
┌────────────────────────────────────────┐
│ st.chat_input(...) returns the string  │
└──────────┬─────────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ RAGPipeline.ask_stream(question)                                │
│   ↓                                                             │
│  ctx = self._retriever.retrieve(query, kb_k=25, report_k=4)     │
│   ↓                                                             │
│  return self._llm.ask_stream(question, ctx), ctx                │
└──────────┬──────────────────────────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ DualRetriever.retrieve(query, kb_k, report_k)                   │
│                                                                 │
│  Step 1 — Query Classification                                  │
│    classify_query(query) keyword-matches                        │
│      PATIENT_KEYWORDS  ∩ query → "patient"                      │
│      MEDICAL_KEYWORDS  ∩ query → "medical"                      │
│      both fire OR neither   → "both"                            │
│                                                                 │
│  Step 2 — Query Normalization (no LLM call)                     │
│    For "patient" or "both":                                     │
│      regex match against _VAGUE_PATTERNS                        │
│      if match: rewrite or expand topic                          │
│      For "patient" no-match: prepend "patient report: "         │
│      For "both": append _DOMAIN_EXPANSIONS terms                │
│    For "medical": pass through (advanced RAG handles)           │
│                                                                 │
│  Step 3 — Route effective_kb_k / effective_report_k             │
│    "patient":  report_k = max(report_k, 10), kb_k = 2           │
│    "medical":  report_k = 2,                kb_k = full         │
│    "both":     report_k = max(report_k, 8), kb_k = min(kb_k,10) │
│                                                                 │
│  Step 4 — Pre-retrieval (advanced RAG, "medical" or "both" only)│
│    AdvancedRAGOrchestrator.pre_retrieval(search_query)          │
│      QueryRewriter.rewrite_multi(query)                         │
│        prompt LLM with DSM-5/ICD-10/ADOS-2 vocab                │
│        parse 3 variant lines, dedupe, → List[str]               │
│    query_variants = [search_query] + variants  (≤ 4 total)      │
│                                                                 │
│  Step 5 — Per-store hybrid retrieval                            │
│    A. ReportStore.similarity_search(search_query, k)            │
│       (a) dense:  ChromaDB cosine query → top-3k chunks         │
│       (b) BM25:   BM25Okapi.get_scores → top-3k chunks          │
│       (c) RRF:    fuse by rank, score = Σ 1/(60+rank)           │
│       (d) Decrypt chunk text via crypto_utils.decrypt_text      │
│                                                                 │
│    B. For each variant (multi-query if pre_retrieval ran):      │
│       InMemoryKB.similarity_search(variant, k)                  │
│         (a) dense: cosine over numpy embeddings                 │
│         (b) BM25:  BM25Okapi over corpus tokens                 │
│         (c) RRF: same formula                                   │
│         (d) if retrieval_type=mmr: MMR re-rank fused candidates │
│                                                                 │
│  Step 6 — Cross-store fusion (DualRetriever._fuse)              │
│    For each report_hit: by_fingerprint[strip(text)] = chunk     │
│    For each kb_hit:     skip if already in by_fingerprint       │
│    Sort: report-first, then by descending similarity            │
│                                                                 │
│  Step 7 — Post-retrieval (advanced RAG)                         │
│    AdvancedRAGOrchestrator.post_retrieval(query, fused)         │
│      CrossEncoderReranker.rerank(query, chunks, top_k=8)        │
│        scores = ms-marco-MiniLM(query, each chunk)              │
│        chunk.similarity = sigmoid(score / 3)                    │
│        ensure ≥ 2 report chunks survive                         │
│      ContextCompressor.compress(query, chunks, top_k=4)         │
│        cosine(embed(query), embed(chunk)) ≥ 0.30 → keep         │
│                                                                 │
│  Step 8 — MIN_CHUNKS_FOR_LLM=2 fallback                         │
│    if len(fused) < 2 and (report_hits or kb_hits):              │
│      restore raw fused, take top-2 by similarity                │
│                                                                 │
│  Step 9 — Smart selection: 5 report + 3 KB                      │
│    sort report_part by similarity, take top 5                   │
│    sort kb_part by similarity,     take top 3                   │
│    fused = report_part + kb_part   (8 chunks max)               │
│                                                                 │
│  → FusedContext(query, chunks)                                  │
└──────────┬──────────────────────────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ LLMEngine.ask_stream(question, ctx)                             │
│                                                                 │
│  Step A — Sufficiency check                                     │
│    if not ctx.chunks: yield INSUFFICIENT_CONTEXT_MSG; return    │
│                                                                 │
│  Step B — Build messages (stateless — no history)               │
│    SystemMessage(DEFAULT_SYSTEM_PROMPT)                         │
│    HumanMessage(f"Clinician question: ...                       │
│                  Below are excerpts: ...                        │
│                  {ctx.as_prompt_block()}                        │
│                  Summarize ...")                                │
│                                                                 │
│  Step C — Stream from ChatOllama                                │
│    for chunk in self._chat.stream(messages):                    │
│      if chunk.content: yield chunk.content                      │
└──────────┬──────────────────────────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Streamlit UI                                                    │
│  - st.empty() placeholder updated per token                     │
│  - typing indicator before first token                          │
│  - on stream end: render Sources expander + retrieval-quality   │
│    strip (high/medium/low based on chunk coverage)              │
└─────────────────────────────────────────────────────────────────┘
```

**Latency budget per turn (current config, full pipeline):**
| Stage | Time | Note |
|---|---|---|
| Query rewriting | ~3-4 s | 1 LLM call |
| Per-store retrieval | <0.5 s | hybrid + RRF |
| Cross-encoder rerank | ~1-2 s | first call loads model |
| Compression | <0.2 s | 1 batch embed call |
| LLM generation | 10-15 s | 250-word answer |
| **Total** | **~14-19 s** | |

---

## 4. Cryptographic Protocol

### 4.1 AES-256-GCM (per-session PDF encryption)

**Library:** `cryptography.hazmat.primitives.ciphers.aead.AESGCM`

**Key generation:**
```python
key = AESGCM.generate_key(bit_length=256)  # 32 bytes from OS CSPRNG
```

**Encryption:**
```python
nonce = os.urandom(12)  # 96-bit, fresh per call
aesgcm = AESGCM(key)
ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
# ciphertext is plaintext_len + 16 (16-byte auth tag appended)
```

**Decryption:**
```python
plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)
# raises cryptography.exceptions.InvalidTag if ciphertext was modified
```

**Why AES-256-GCM:**
- Confidentiality + integrity in one primitive
- 96-bit nonce is the standard for GCM (longer requires hashing internally)
- Authentication tag detects any modification — flipping a single ciphertext bit raises InvalidTag

**Nonce strategy:** Fresh per call from `os.urandom(12)`. Since the key itself is fresh per session and only used for one encryption, nonce reuse cannot occur in practice.

### 4.2 Shamir Secret Sharing (key distribution)

**Library:** `shamir_mnemonic` (Trezor implementation of SLIP-39)

**Why SLIP-39 mnemonics rather than raw bytes:**
- Human-auditable share contents (each share is a 33-word English mnemonic)
- Built-in checksums detect typos when shares are recombined
- Standard format with broad library support

**Splitting:**
```python
mnemonics = generate_mnemonics(
    group_threshold=1,        # one group of shares
    groups=[(3, 5)],          # 3-of-5 threshold within the group
    master_secret=key,        # the AES key (32 bytes)
)
shares = mnemonics[0]         # list of 5 SLIP-39 strings
```

**Reconstruction:**
```python
recovered_key = combine_mnemonics(shares[:3])  # any 3 of the 5
```

**Threshold properties:**
- Any 3 shares → full key reconstruction
- Any 2 shares → information-theoretic zero knowledge of the key
- Tolerates up to 2 simultaneous share losses

### 4.3 Fernet (chunk-level encryption in ChromaDB)

**Library:** `cryptography.fernet.Fernet`

**Format:** `Fernet` is a thin wrapper around AES-128-CBC + HMAC-SHA256. Tokens encode:
```
version (1 byte) | timestamp (8 bytes) | IV (16 bytes) | ciphertext (var) | HMAC (32 bytes)
```

**Key resolution order** (`crypto_utils.get_fernet`):
1. `RAG_FERNET_KEY` environment variable (production)
2. `./secret.key` file (development)
3. Auto-generate and persist to `./secret.key` on first use

**API:**
```python
encrypt_text(plain: str) -> str         # base64-encoded Fernet token
decrypt_text(ciphertext: str) -> str    # falls back to input if not Fernet
```

**Why same key across KB and report stores:** Lets retrieved chunks decrypt regardless of source. The KB collection has no PHI but is encrypted defensively in case KB content is borderline (clinical guideline excerpts).

### 4.4 Application-layer logging policy

**Enforcement:** `app.py:_enforce_in_memory_logging` runs at module import time:
```python
def _enforce_in_memory_logging():
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(logging.StreamHandler())
    root.setLevel(logging.INFO)
```

**Why at import time:** Closes the window in which a downstream library import (e.g., LangChain, ChromaDB) could attach a `FileHandler`. The check runs again every Streamlit script rerun.

**What still goes to files:** ingestion logs (`logs/ingestion.log`) and RAGAS eval reports (`logs/eval_report_*.json`) — these contain no PHI by design.

---

## 5. Storage Layer

### 5.1 EphemeralDFSStore (`dfs/ephemeral_store.py`)

**Public interface:**
```python
class EphemeralDFSStore:
    def __init__(num_fragments=10, num_key_shares=5, key_threshold=3, num_nodes=5)
    def store(self, data: bytes) -> StoredHandle
    def reconstruct(self, handle: StoredHandle = None) -> bytes
    def delete_active(self) -> None
    def wipe(self) -> None
    @property
    def root: Path
    @property
    def has_active: bool
```

**Module-level:**
```python
def sweep_orphans() -> int
```

**Lifecycle states:**
```
[fresh]  →  [tempdir created]  →  [active handle]  →  [active handle]  →  [wiped]
            (constructor)        (after store)      (after store again)  (wipe / atexit)
```

**Tempdir layout (after `store()`):**
```
/tmp/rag_dfs_a8f3.../
├── node_0/
│   ├── frag_000.bin    (~one tenth of ciphertext)
│   ├── frag_005.bin
│   ├── share_000.txt   (SLIP-39 mnemonic)
│   └── share_005.txt
├── node_1/
│   ├── frag_001.bin
│   ├── frag_006.bin
│   ├── share_001.txt
│   └── share_006.txt
├── node_2/ ... node_4/
```

**Cleanup architecture (4-layer defence in depth):**
| Layer | Trigger | Mechanism |
|---|---|---|
| 1 | `End Session` button | `pipeline.end_session()` → `dfs.wipe()` |
| 2 | Normal interpreter exit | `atexit.register(_atexit_cleanup)` walks `_active_stores` |
| 3 | Garbage collection | `weakref.finalize(self, _wipe_dir, root)` |
| 4 | App boot after crash | `sweep_orphans()` from `app.py:init_state` |

**Secure unlink:** `_secure_unlink(path)` overwrites file with `b'\x00' * size`, fsyncs, then unlinks. Symbolic on SSDs due to wear-leveling — the AES-GCM is doing the real protection.

### 5.2 ReportStore (`report_store.py`)

**Per-session ChromaDB collection in a tempdir.**

**Tempdir:** `/tmp/rag_chroma_XXXX/`

**Storage characteristics:**
- ChromaDB uses `PersistentClient` (not `EphemeralClient`) for consistent behaviour across versions, but rooted at a tempdir so it dies with the session
- Collection name is a fresh UUID per report (`report_{uuid.uuid4().hex}`)
- Distance metric: cosine

**What's stored encrypted vs plaintext:**
| Field | Encryption | Why |
|---|---|---|
| `documents` | Fernet-encrypted before `add()` | The actual chunk text (would reveal patient observations) |
| `embeddings` | NOT encrypted | 768-dim vectors don't reveal content directly; required for similarity search |
| `metadatas` | NOT encrypted | Filename, section, char offset — not PHI |
| `ids` | NOT encrypted | UUIDs |

**Hybrid retrieval state (in-memory only, never persisted):**
- `self._plain_texts: List[str]` — keeps the unencrypted text in process memory for BM25
- `self._bm25: BM25Okapi` — built from tokenized plain texts

This means the encrypted ChromaDB is the durable artifact; the BM25 index is rebuilt fresh on every `load_report_from_parsed` call.

### 5.3 InMemoryKB (`kb_memory.py`)

**Persistent KB with disk-cached embeddings.**

**Cache file structure:**
```
data/kb_cache/
├── kb_cache.npz   # numpy: {texts, metadatas, embeddings}
└── kb_meta.json   # {dir_hash, num_entries, chunk_size, chunk_overlap, min_chunk_size}
```

**Cache key (MD5 of):**
```python
hashlib.md5("|".join(f"{filename}:{filesize}" for f in sorted(data/kb_pdfs/))).hexdigest()
```

**Cache invalidation triggers:**
- Any PDF added/removed/renamed/resized in `data/kb_pdfs/`
- `chunk_size`, `chunk_overlap`, or `min_chunk_size` changed in `config.ini`
- Cache file deleted manually

**Cache hit cold-start:** ~0.5-1 s
**Cache miss cold-start (5 PDFs, ~2400 chunks):** ~30-45 s (Ollama embed time)

### 5.4 KB Persistent ChromaDB (`data/chroma_kb/`)

Built once by `python ingest_kb.py`. Contains the same chunks as the cache but in ChromaDB's SQLite + binary format. Stored Fernet-encrypted per chunk text.

The runtime path uses `InMemoryKB` (cache) for retrieval, NOT this persistent Chroma — Chroma is the source of truth for re-ingestion only.

---

## 6. Retrieval Algorithms

### 6.1 Query Classification (`dual_retriever.classify_query`)

```python
PATIENT_KEYWORDS = ["patient", "child's", "this child", "report", "observation",
                    "tell me about", "summarize", "behaviour", "from the report",
                    "their", "this kid", ...]

MEDICAL_KEYWORDS = ["dsm", "icd", "ados", "m-chat", "criteria", "screening",
                    "intervention", "treatment", "evidence-based", "guideline",
                    "differential", "comorbid", "prevalence", ...]

def classify_query(query: str) -> str:
    q = query.lower()
    has_patient = any(kw in q for kw in PATIENT_KEYWORDS)
    has_medical = any(kw in q for kw in MEDICAL_KEYWORDS)
    if has_patient and has_medical: return "both"
    elif has_patient: return "patient"
    elif has_medical: return "medical"
    else: return "both"  # safe default — assume report is loaded
```

**Routing table (effective k):**
| Type | Trigger | report_k | kb_k | Multi-query rewriting |
|---|---|---|---|---|
| `patient` | only PATIENT_KEYWORDS fire | `max(report_k, 10)` | 2 | **disabled** |
| `medical` | only MEDICAL_KEYWORDS fire | 2 | full | **enabled** |
| `both` | both fire OR ambiguous | `max(report_k, 8)` | `min(kb_k, 10)` | **enabled** |

Query rewriting is skipped for `patient` because rewriting the patient's specific question into clinical-vocabulary variants would dilute the report-grounded signal.

### 6.2 Query Normalization (`dual_retriever.normalize_query`)

**No LLM call** — pure regex/string substitution. Two stages:

**Stage 1: Vague pattern match** (e.g., "tell me about this kid")
```python
_VAGUE_PATTERNS = [
    (re.compile(r"^tell me about (?:the |this )?(?:patient|child|kid)\.?$", re.I),
     "summary of patient developmental report: communication, social interaction, "
     "behavioral patterns, sensory responses"),
    (re.compile(r"^(?:tell me about|describe|explain|what about) "
                r"(?:the |this )?(?:child'?s? )?(.+?)\.?$", re.I),
     None),  # dynamic: extract topic, expand via _DOMAIN_EXPANSIONS
    ...
]
```

**Stage 2: Domain expansion** for `both` queries (no static rewrite):
```python
_DOMAIN_EXPANSIONS = {
    "social":      "social interaction observations",
    "communication": "communication domain observations",
    "sensory":     "sensory responses observations",
    "speech":      "communication domain speech language observations",
    "eye contact": "eye contact social interaction observations",
    "echolalia":   "communication echolalia repeating words",
    "name":        "respond to name social interaction",
    ...
}
```

### 6.3 Dense (Cosine) Search

```python
def _dense_search(self, query_vec, k):
    q_norm = ||query_vec||
    sims = []
    for i, entry in enumerate(self._entries):
        v = entry.embedding
        sim = (query_vec · v) / (q_norm * ||v||)
        sims.append((i, sim))
    sims.sort(by sim descending)
    return sims[:k]
```

For ReportStore (ChromaDB-backed), this is delegated to `collection.query()` which uses HNSW under the hood.

### 6.4 BM25 Sparse Search

**Tokenizer:**
```python
def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+(?:[-']\w+)*", text.lower())
```

Captures hyphenated terms (`m-chat-r`), apostrophes (`child's`), and treats everything as lowercase.

**Scoring:** standard BM25Okapi with default `k1=1.5, b=0.75`.

**Why include BM25:** clinical acronyms (ADOS-2, M-CHAT-R/F, ICD-11) often don't embed compactly into 768-dim dense space because they're rare in the embedding model's training data. BM25 captures exact lexical matches.

### 6.5 Reciprocal Rank Fusion (RRF)

Standard formula from Cormack et al. (SIGIR 2009):
```
RRF(d) = Σ_over_each_ranking 1 / (k + rank_in_ranking(d))
```

Where `k = 60` (project default; hyperparameter from the original paper).

```python
RRF_K = 60

def _rrf_fuse(dense_results, bm25_results, k):
    rrf_scores: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (idx, _) in enumerate(bm25_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:k]
```

**Why RRF over weighted sum:** RRF is rank-based, so it doesn't require normalization between dense (cosine, in [-1,1]) and BM25 (unbounded positive). Dramatically less sensitive to score scale differences.

### 6.6 MMR Re-ranking (`InMemoryKB._mmr_rerank`)

Maximal Marginal Relevance, used when `retrieval_type=mmr`:

```
score(c) = (1 - λ) * sim(q, c) - λ * max_{c' ∈ selected} sim(c, c')
```

- `mmr_lambda = 0` → pure relevance (same as similarity)
- `mmr_lambda = 1` → pure diversity
- `mmr_lambda = 0.5` (default) → balanced

**Algorithm:**
```
selected = []
remaining = all_candidates
while |selected| < k and remaining:
    for each c in remaining:
        relevance = (1 - λ) * cosine(q, c)
        diversity_penalty = λ * max(cosine(c, s) for s in selected) if selected else 0
        mmr_score(c) = relevance - diversity_penalty
    pick c* with max mmr_score(c)
    selected.append(c*)
    remaining.remove(c*)
return selected
```

**Why use MMR:** without it, the KB retriever often returns multiple chunks from the same DSM-5 section, all with similar embeddings. MMR adds diversity so the LLM sees broader evidence.

### 6.7 Cross-store Fusion (`DualRetriever._fuse`)

```python
def _fuse(self, report_hits, kb_hits) -> List[ContextChunk]:
    by_fingerprint: Dict[str, ContextChunk] = {}
    
    # Report chunks first
    for hit in report_hits:
        key = hit["text"].strip()
        if key and key not in by_fingerprint:
            by_fingerprint[key] = ContextChunk(
                text=hit["text"], metadata=hit["metadata"],
                similarity=hit.get("similarity", 0.0),
                source_type="report",
            )
    
    # KB chunks, deduplicating against report
    for hit in kb_hits:
        key = hit["text"].strip()
        if key and key not in by_fingerprint:
            by_fingerprint[key] = ContextChunk(
                text=hit["text"], metadata=hit["metadata"],
                similarity=hit.get("similarity", 0.8),
                source_type="kb",
            )
    
    fused = list(by_fingerprint.values())
    fused.sort(key=lambda c: (
        0 if c.source_type == "report" else 1,  # report first
        -c.similarity,
    ))
    return fused
```

### 6.8 Smart 5+3 Selection (`DualRetriever.retrieve`, end of method)

```python
report_part = [c for c in fused if c.source_type == "report"]
kb_part     = [c for c in fused if c.source_type != "report"]
report_part.sort(key=lambda c: -c.similarity)
kb_part.sort(key=lambda c: -c.similarity)
report_part = report_part[:5]   # top 5 report
kb_part     = kb_part[:3]       # top 3 KB
fused = report_part + kb_part   # max 8 chunks to LLM
```

**Why 5+3:** empirically tuned. More report chunks = stronger patient grounding; 3 KB chunks is enough for clinical context (DSM criteria, screening tool reference) without burying the patient signal.

---

## 7. Advanced RAG Techniques

### 7.1 Query Rewriting (`advanced_rag.QueryRewriter`)

**Prompt:**
```
Rewrite the following question into 3 different phrasings optimized for searching
a medical knowledge base about Autism Spectrum Disorder. Each phrasing should use
different clinical terminology or focus on a different aspect of the question.
Use standard terms (DSM-5, ICD-10, ADOS-2, M-CHAT, etc.) where applicable.
Return ONLY the 3 phrasings, one per line. No numbering, no extra text.

Question: {query}

Phrasings:
```

**LLM head:** the classifier head with `temperature=0.0` (deterministic) — `LLMEngine.generate_raw`.

**Post-processing:**
1. Split on newline
2. Strip leading list markers (`0123456789.-) `)
3. Lowercase-deduplicate
4. Cap at 3 variants

The variants are added to the search-query list for KB retrieval (multi-query). Each variant gets `effective_kb_k / num_variants` chunks (minimum 4).

### 7.2 Cross-Encoder Reranking (`advanced_rag.CrossEncoderReranker`)

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers, ~80 MB, downloaded on first call).

**How it differs from bi-encoder retrieval:** a bi-encoder embeds query and document independently; a cross-encoder takes both as joint input and produces a relevance score. More accurate but quadratic — only feasible after first-stage retrieval has narrowed to ~25 candidates.

**Logit normalization:**
```python
def _normalize_ce_score(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(score) / 3.0))
```

Why scaled sigmoid (factor 3) and not simple `(score+10)/20`:
- ms-marco logits typically range [-12, +6] for clinical text
- Linear `(score+10)/20` clips to 0 at score ≤ -10 (very common)
- Scaled sigmoid is smooth, never saturates to exactly 0, gives good visual dispersion

**Report-chunk preservation:**
```python
# Always keep ≥ 2 report chunks even if their cross-encoder scores are low
report_in_result = {c.text for c in result if c.source_type == "report"}
for chunk, score in scored:
    if chunk.source_type == "report" and chunk.text not in report_in_result:
        chunk.similarity = _normalize_ce_score(score)
        result.append(chunk)
        if len([c for c in result if c.source_type == "report"]) >= 2:
            break
```

### 7.3 Embedding-Based Compression (`advanced_rag.ContextCompressor`)

**Why embedding-based not LLM-based:** the previous LLM-per-chunk version made one LLM call per chunk (5-8 calls = 40-80 s latency) and was too aggressive (dropped chunks the cross-encoder scored 3-8). Replaced with a single embedding similarity check.

```python
COMPRESSION_SIM_THRESHOLD = 0.30

def compress(self, query, chunks, top_k=5):
    q_vec = embed_query(query)
    chunk_embs = embed_texts([c.text for c in chunks[:top_k]])
    
    compressed = []
    for chunk, emb in zip(chunks[:top_k], chunk_embs):
        cos_sim = dot(q_vec, emb) / (norm(q_vec) * norm(emb))
        if cos_sim >= COMPRESSION_SIM_THRESHOLD:
            compressed.append(chunk)
    
    compressed.extend(chunks[top_k:])  # don't touch chunks beyond top_k
    return compressed
```

**Threshold rationale:** 0.30 is permissive — the cross-encoder already filtered. Dropping at >0.50 would over-prune; <0.20 would never drop anything. 0.30 catches the obvious off-topic chunks.

### 7.4 Orchestrator (`advanced_rag.AdvancedRAGOrchestrator`)

**Fail-soft per technique:**
```python
def post_retrieval(self, query, chunks):
    if self._reranker:
        try:
            chunks = self._reranker.rerank(query, chunks, top_k=self.config.rerank_top_k)
        except Exception as e:
            logger.warning("Cross-encoder re-ranking failed, keeping original: %s", e)
    
    if self._compressor:
        try:
            chunks = self._compressor.compress(query, chunks, top_k=self.config.compression_top_k)
        except Exception as e:
            logger.warning("Compression failed, keeping original: %s", e)
    
    return chunks
```

If a technique fails at runtime (e.g., cross-encoder model fails to download, embedding API timeout), the pipeline continues with the unmodified chunk list. The user gets a slightly worse-quality answer instead of a hard error.

---

## 8. LLM Engine

### 8.1 Two heads (`llm_engine.LLMEngine`)

```python
self._chat = ChatOllama(model="llama3.1", temperature=0.2)        # answer head
self._classifier = ChatOllama(model="llama3.1", temperature=0.0)  # deterministic head
```

| Head | Use cases | Temperature |
|---|---|---|
| `_chat` | `ask()`, `ask_stream()` (final answers) | 0.2 (slight creativity) |
| `_classifier` | `generate_raw()` (RAGAS scorers, query rewriting) | 0.0 (deterministic) |

### 8.2 Clinical system prompt

```
You are a clinical decision-support assistant for licensed healthcare providers
evaluating children for Autism Spectrum Disorder (ASD).

You receive context from two sources:
- PATIENT REPORT: observations from a child's developmental report
- MEDICAL KB: clinical guidelines and medical literature

INSTRUCTIONS:
1. Answer the clinician's SPECIFIC question directly and concisely.
   Do not add unrequested background or tangential information.
2. Use ONLY the provided context. Do not use prior knowledge.
3. When patient report context is provided, always cite specific observations from it.
4. Connect patient observations to relevant clinical criteria (DSM-5, ICD-10)
   when medical KB context supports it.
5. Use cautious clinical language ("may be consistent with", "warrants further evaluation").
6. Do NOT refuse to answer — this tool exists to surface evidence for clinicians.
7. Keep answers focused and under 250 words.

Structure (use only when both source types are present):
[From Patient Report]: specific observations relevant to the question
[From Medical KB]: relevant clinical criteria or guidelines
[Clinical Correlation]: how observations relate to criteria
```

**Design notes:**
- "Do NOT refuse to answer" — without this, Llama 3.1 frequently refuses clinical questions claiming it can't give medical advice. The system prompt frames this as decision *support*, not advice.
- Hedged language enforced via "may be consistent with" patterns
- 250-word cap keeps the answer scannable in the chat UI

### 8.3 Stateless message construction

```python
def _build_messages(self, question, ctx):
    # Each turn is independent — no conversation history is retained
    user_prompt = (
        f"A clinician has asked the following question while reviewing a patient case:\n"
        f"{question}\n\n"
        f"Below are excerpts from the patient's developmental report and "
        f"relevant medical reference documents:\n\n"
        f"{ctx.as_prompt_block()}\n\n"
        f"Summarize the relevant information from these documents to address "
        f"the clinician's question."
    )
    return [
        SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
```

**Why stateless:** prevents prior questions/answers about other patients from bleeding into a new query. Cross-patient privacy is enforced at the LLM input boundary.

### 8.4 Sufficiency guard

```python
def _context_is_sufficient(self, ctx) -> bool:
    if not ctx.chunks:
        return False
    return True

def ask_stream(self, question, ctx):
    if not self._context_is_sufficient(ctx):
        yield INSUFFICIENT_CONTEXT_MSG
        return
    for chunk in self._chat.stream(self._build_messages(question, ctx)):
        if chunk.content:
            yield chunk.content
```

The retrieval pipeline (BM25 + dense + cross-encoder + compression) has already filtered, so the only "insufficient" case is literally zero chunks. The fallback message is a fixed string explaining no relevant information was found.

---

## 9. Evaluation Framework

### 9.1 The four RAGAS metrics

#### Faithfulness (`FaithfulnessScorer`)

**Question:** *Is the answer grounded in the retrieved context?*

**Algorithm:**
1. LLM extracts atomic factual claims from the answer (`_EXTRACT_PROMPT`)
2. For each claim, LLM judges YES/NO whether the claim is supported by the combined context (`_VERIFY_PROMPT`)
3. Score = (# supported claims) / (# total claims)

**Range:** [0, 1]. 1.0 = every claim grounded, 0.0 = none.

#### Answer Relevancy (`AnswerRelevancyScorer`)

**Question:** *Is the answer relevant to the question?*

**Algorithm:**
1. LLM generates `n=3` hypothetical questions that the answer could be answering
2. Embed original question + each generated question via nomic-embed-text
3. Compute cosine similarity between original and each generated
4. Score = mean similarity

**Range:** [0, 1]. 1.0 = generated questions semantically identical to original, 0.0 = no overlap.

#### Context Precision (`ContextPrecisionScorer`)

**Question:** *Are relevant chunks ranked higher than irrelevant ones?*

**Algorithm:** Average Precision (AP) formula:
1. LLM judges YES/NO for each chunk's relevance to the question
2. For each rank position k where chunk is relevant: precision@k = (cumulative relevant up to k) / k
3. Score = (sum of precision@k for relevant positions) / total_relevant

**Range:** [0, 1]. 1.0 = all relevant chunks at the top.

#### Context Recall (`ContextRecallScorer`)

**Question:** *Does the context cover the ground truth answer?*

**Algorithm:**
1. LLM extracts atomic claims from the **ground truth** answer (`_EXTRACT_PROMPT`)
2. For each ground-truth claim, LLM judges YES/NO whether attributable to retrieved context
3. Score = (# attributable claims) / (# ground-truth claims)

**Range:** [0, 1]. 1.0 = all ground-truth claims covered.

### 9.2 RAGAS aggregate score

Harmonic mean of the four metrics:
```python
ragas_score = len(metrics) / sum(1.0 / v for v in metrics if v > 0)
```

Harmonic mean penalizes any single weak metric — a system that scores 1.0/1.0/1.0/0.0 gets RAGAS = 0, not 0.75.

### 9.3 CLI (`run_evaluation.py`)

```bash
python run_evaluation.py                      # report-grounded eval (default)
python run_evaluation.py --dataset ./data/eval_dataset.json  # KB-only
python run_evaluation.py --report-pdf x.pdf   # override report
python run_evaluation.py --advanced-rag       # force-enable Phase 4
python run_evaluation.py --cross-encoder      # only enable cross-encoder
python run_evaluation.py --limit 3            # first 3 questions
```

**Output:**
- Console: per-sample table + aggregate
- File: `logs/eval_report_<timestamp>.json` with full per-metric breakdown + config snapshot

### 9.4 Latest measured results

**Full pipeline (5+3, all advanced RAG)** on `dummy_autism_screening_report.pdf`:
| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Answer Relevancy | 0.75 |
| Context Precision | 0.73 |
| Context Recall | 0.72 |
| **RAGAS Score (harmonic)** | **0.77** |

**Ablation:**
| Config | Recall | Precision | Faith | Relev |
|---|---|---|---|---|
| Baseline (dense only) | 0.54 | 0.61 | 0.82 | 0.73 |
| Cross-encoder reranking only | 0.73 | 0.69 | 0.91 | 0.69 |
| Full pipeline | 0.72 | 0.73 | 0.91 | 0.75 |

---

## 10. Configuration Reference

All settings in `config.ini`, loaded by `RAGConfig.from_ini()`. Missing values fall back to dataclass defaults.

### `[ollama]`
| Key | Default | Effect |
|---|---|---|
| `base_url` | `http://localhost:11434` | Ollama endpoint |
| `llm_model` | `llama3.1` | Generation model |
| `embed_model` | `nomic-embed-text` | **Must match what KB was indexed with — switching invalidates `data/chroma_kb/` and `data/kb_cache/`** |

### `[paths]`
| Key | Default |
|---|---|
| `kb_data_dir` | `./data/kb_pdfs` |
| `chroma_db_dir` | `./data/chroma_kb` |
| `log_dir` | `./logs` |

### `[chunking]`
| Key | Default | Tuning |
|---|---|---|
| `chunk_size` | 300 | smaller = better precision, larger = better context |
| `chunk_overlap` | 40 | ~12.5% of `chunk_size` |
| `min_chunk_size` | 80 | drops too-small chunks |

**Cache invalidation:** any change here triggers re-ingest.

### `[retrieval]`
| Key | Default | Notes |
|---|---|---|
| `kb_top_k` | 25 | KB chunks per query (over-retrieve for reranking) |
| `report_top_k` | 4 | Report chunks per query (raised by routing for `patient` queries) |
| `retrieval_type` | `mmr` | `similarity` or `mmr` |
| `mmr_lambda` | 0.5 | 0=relevance, 1=diversity |

### `[advanced_rag]`
| Key | Default | Effect |
|---|---|---|
| `query_rewriting_enabled` | true | LLM rewrites into 3 variants |
| `cross_encoder_enabled` | true | rerank with ms-marco-MiniLM |
| `compression_enabled` | true | filter chunks below cosine 0.30 |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | auto-downloaded |
| `rerank_top_k` | 8 | top-K kept after rerank |
| `compression_top_k` | 4 | chunks compressor inspects (≤ rerank_top_k) |

Any technique fails soft at runtime.

### `[evaluation]`
| Key | Default |
|---|---|
| `eval_dataset` | `./data/eval_dataset.json` |
| `eval_output_dir` | `./logs` |
| `num_relevancy_questions` | 3 |

### `[chroma]`
| Key | Default |
|---|---|
| `kb_collection_name` | `autism_kb_medical_guidelines` |
| `distance_metric` | `cosine` |

### `[dfs]`
| Key | Default | Notes |
|---|---|---|
| `num_fragments` | 10 | encrypted PDF byte-range pieces |
| `num_key_shares` | 5 | Shamir total shares |
| `key_threshold` | 3 | Shamir reconstruction threshold |

### Environment variables
| Variable | Purpose |
|---|---|
| `RAG_FERNET_KEY` | Override `secret.key` for chunk encryption |

### What invalidates what (re-ingest required)
| Change | What dies |
|---|---|
| `embed_model` | `data/chroma_kb/` + `data/kb_cache/` |
| `chunk_size` / `chunk_overlap` / `min_chunk_size` | `data/chroma_kb/` re-ingest |
| `kb_collection_name` | new collection |
| `distance_metric` | new collection |
| `retrieval_type` / `mmr_lambda` | nothing on disk |
| `[advanced_rag]` toggles | nothing on disk |

---

## 11. Public API Surface

### RAGPipeline (`rag_pipeline.py`)
```python
class RAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None)
    
    @property
    def report_metadata(self) -> Optional[ReportMetadata]
    
    def load_report_from_uploaded_file(self, uploaded_file) -> ReportMetadata
    def end_session(self) -> None
    def ask(self, question: str) -> tuple[str, FusedContext]
    def ask_stream(self, question: str) -> tuple[Iterable[str], FusedContext]
```

### RAGConfig (`rag_pipeline.py`)
```python
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
    chunk_size: int = 800
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    retrieval_type: str = "similarity"
    mmr_lambda: float = 0.5
    query_rewriting_enabled: bool = False
    cross_encoder_enabled: bool = False
    compression_enabled: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 15
    compression_top_k: int = 5
    dfs_num_fragments: int = 10
    dfs_num_key_shares: int = 5
    dfs_key_threshold: int = 3
    
    @classmethod
    def from_ini(cls, path: str = "config.ini") -> "RAGConfig"
```

### DualRetriever (`dual_retriever.py`)
```python
class DualRetriever:
    def __init__(self, kb_store, report_store, advanced_rag=None)
    def retrieve(self, query: str, kb_k: int = 20, report_k: int = 4) -> FusedContext
```

### ReportStore (`report_store.py`)
```python
class ReportStore:
    def __init__(self, embedder=None, distance_metric="cosine")
    
    @property
    def metadata: Optional[ReportMetadata]
    
    def load_report_from_path(self, pdf_path: str) -> ReportMetadata
    def load_report_from_parsed(self, parsed: ParsedDocument) -> ReportMetadata
    def similarity_search(self, query: str, k: int = 4) -> List[Dict]
    def wipe(self) -> None
```

### InMemoryKB (`kb_memory.py`)
```python
class InMemoryKB:
    def __init__(self, data_dir, embedder, cache_dir, chunk_size,
                 chunk_overlap, min_chunk_size, retrieval_type, mmr_lambda)
    def similarity_search(self, query: str, k: int = 6) -> List[Dict]
```

### LLMEngine (`llm_engine.py`)
```python
class LLMEngine:
    def __init__(self, model="llama3.1", base_url="http://localhost:11434", temperature=0.2)
    def ask(self, question: str, ctx: FusedContext) -> str
    def ask_stream(self, question: str, ctx: FusedContext) -> Iterable[str]
    def generate_raw(self, prompt: str) -> str
    def is_asd_question(self, question: str) -> bool
```

### EphemeralDFSStore (`dfs/ephemeral_store.py`)
```python
class EphemeralDFSStore:
    def __init__(self, num_fragments=10, num_key_shares=5, key_threshold=3, num_nodes=5)
    
    @property
    def root: Path
    @property
    def has_active: bool
    
    def store(self, data: bytes) -> StoredHandle
    def reconstruct(self, handle: StoredHandle = None) -> bytes
    def delete_active(self) -> None
    def wipe(self) -> None

# Module-level
def sweep_orphans() -> int
```

### AdvancedRAGOrchestrator (`advanced_rag.py`)
```python
class AdvancedRAGOrchestrator:
    def __init__(self, config: AdvancedRAGConfig, llm: LLMEngine, embedder: OllamaEmbedder)
    def pre_retrieval(self, query: str) -> List[str]
    def post_retrieval(self, query: str, chunks: List[ContextChunk]) -> List[ContextChunk]
```

---

## 12. Data Structures

### ContextChunk (`dual_retriever.py`)
```python
@dataclass
class ContextChunk:
    text: str               # decrypted chunk text
    metadata: Dict[str, Any] # filename, section, char_start, etc.
    similarity: float        # cosine sim or normalized cross-encoder score
    source_type: str         # "report" or "kb"
```

### FusedContext (`dual_retriever.py`)
```python
@dataclass
class FusedContext:
    query: str
    chunks: List[ContextChunk]
    
    def as_prompt_block(self) -> str:
        """Render chunks for LLM prompt:
        [1] (PATIENT REPORT, sim=0.812)
        <chunk text>
        
        [2] (MEDICAL KB, sim=0.760)
        <chunk text>
        ..."""
```

### ReportMetadata (`report_store.py`)
```python
@dataclass
class ReportMetadata:
    report_id: str   # uuid4 hex
    filename: str
    num_pages: int
    num_chunks: int
```

### StoredHandle (`dfs/ephemeral_store.py`)
```python
@dataclass
class StoredHandle:
    fragment_paths: List[Path]
    key_share_paths: List[Path]
    nonce: bytes
    threshold: int
    plaintext_size: int = 0
    metadata: dict = field(default_factory=dict)
```

### KBEntry (`kb_memory.py`)
```python
@dataclass
class KBEntry:
    text: str
    metadata: Dict[str, Any]
    embedding: np.ndarray  # 768-dim float32
```

### TextChunk (`phase1/chunker.py`)
```python
@dataclass
class TextChunk:
    chunk_id: str           # filename_chunkNNNN
    text: str
    source_file: str
    source_path: str
    chunk_index: int
    total_chunks: int
    section_hint: str
    char_start: int
    metadata: dict
```

### EvalSample / EvalReport (`ragas_eval.py`)
```python
@dataclass
class EvalSample:
    question: str
    ground_truth: str
    report: str = ""
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    context_sources: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    latency_seconds: float = 0.0

@dataclass
class EvalReport:
    samples: List[EvalSample]
    aggregate_scores: Dict[str, float] = field(default_factory=dict)
    config_snapshot: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
```

---

## 13. Failure Modes & Recovery

### Fail-soft pipeline patterns

| Component | Failure | Behavior |
|---|---|---|
| Cross-encoder model download | Network error | Pipeline continues without rerank |
| Compression embed call | Timeout | Pipeline continues without compression |
| Query rewriting LLM call | LLM error | Falls back to single original query |
| Post-retrieval drops everything | < 2 chunks | Restores raw fused list, takes top-2 |
| Empty FusedContext | 0 chunks | LLM returns INSUFFICIENT_CONTEXT_MSG |
| PDF parse | Both pdfplumber + PyPDF2 fail | Raises ValueError, DFS handle wiped |
| Ollama unavailable | Connection error | Embedding call raises, app shows error |
| Cache file corrupted | Invalid NPZ | Re-embed from scratch, refresh cache |

### Cleanup architecture (DFS)

| Trigger | Path | Coverage |
|---|---|---|
| `End Session` button | `pipeline.end_session() → dfs.wipe()` | Normal user flow |
| Browser tab close | None directly; relies on layer 2 below | — |
| `streamlit` Ctrl+C | `atexit.register(_atexit_cleanup)` | Clean shutdown |
| Python interpreter exit | `weakref.finalize(self, _wipe_dir, root)` | GC-driven |
| Crash (SIGKILL) | `sweep_orphans()` on next app boot | Recovery |

**Coverage matrix:**
| Scenario | wipe() | atexit | weakref | sweep_orphans |
|---|---|---|---|---|
| Click End Session | ✓ | — | — | — |
| Close browser, app keeps running | — | — | — | — (no trigger) |
| Quit streamlit cleanly | — | ✓ | — | — |
| Python crash mid-session | — | — | — | ✓ (next boot) |
| OOM kill | — | — | — | ✓ (next boot) |
| GC of orphaned RAGPipeline | — | — | ✓ | — |

### Retry & timeout behavior

**Embedder (`phase1/embedder.py`):**
- Up to 3 retries per batch with 2s back-off
- Last retry raises RuntimeError

**LLM streaming:**
- LangChain handles transient retries
- No per-token timeout (relies on Ollama default)

---

## 14. Performance Characteristics

### Cold-start times
| Phase | Time | Notes |
|---|---|---|
| KB cache hit | <1 s | NPZ load + BM25 rebuild |
| KB cache miss | ~30-45 s | re-embed 5 PDFs / ~2400 chunks |
| Cross-encoder model first download | ~30 s | one-time, ~80 MB |
| Cross-encoder warm load | ~10 s | first query after boot |

### Per-turn latency (current config)
| Component | Time |
|---|---|
| Query classification + normalization | <10 ms |
| Query rewriting (LLM call) | ~3-4 s |
| Per-store hybrid retrieval | <500 ms |
| Cross-store fusion + dedup | <50 ms |
| Cross-encoder rerank | ~1-2 s |
| Compression (1 embed batch) | <200 ms |
| Smart 5+3 selection | <10 ms |
| LLM streaming generation | 10-15 s |
| **Total user-perceived** | **~14-19 s** |

### RAGAS evaluation latency
| Per sample | Time |
|---|---|
| Mean | 91-130 s |
| Min | 21 s |
| Max | 210 s |

Each sample makes ~5-10 LLM calls (claim extraction × 2, claim verification × N, relevance judgement × M, hypothetical question generation).

### Memory footprint
| Component | RAM |
|---|---|
| KB embeddings (~2400 × 768 × 4B) | ~7 MB |
| BM25 KB index | ~5 MB |
| Cross-encoder model | ~80 MB |
| ChatOllama (Llama 3.1 8B Q4) | ~5-6 GB (in Ollama, not Python) |
| Ephemeral DFS tempdir | size of compressed PDF |
| Per-session report Chroma | ~1-3 MB depending on report length |

---

## 15. Testing Strategy

### `test_retrieval.py`
Per-question retrieval inspection without LLM. Prints what chunks come back. Best signal for "is the right context being retrieved?"

```bash
python test_retrieval.py
```

### `test_validation.py`
Small fixed validation set against a specific dummy report. Runs full pipeline including LLM.

```bash
python test_validation.py
```

### Cryptographic primitives
- AES-GCM round-trip with byte-exact assertion
- AES-GCM tamper detection (flip 1 bit → assert InvalidTag)
- Shamir 3-of-5 with 2 shares → assert raises
- Shamir 3-of-5 with 3, 4, 5 shares → assert each yields same key
- Fernet round-trip across KB and report stores

### Privacy invariants (manual)
- After `End Session`: grep `data/chroma_kb/` and `/tmp/rag_*` for patient-identifying strings — should find none
- After app crash: confirm `sweep_orphans` removes leftover `/tmp/rag_dfs_*`
- Enable DEBUG logging across all modules: confirm no log file is created

### Eval ablation
```bash
# Baseline
# (set query_rewriting_enabled=false, cross_encoder_enabled=false, compression_enabled=false in config.ini)
python run_evaluation.py

# Cross-encoder only
python run_evaluation.py --cross-encoder

# Full pipeline
python run_evaluation.py --advanced-rag
```

---

## 16. Threat Model

### What the privacy layer DOES defend against

| Threat | Mitigation |
|---|---|
| Casual disk inspection mid-session | AES-GCM ciphertext + Shamir sharded key |
| Crashed-session leftover files | `sweep_orphans()` on next boot |
| ChromaDB on-disk read | Fernet chunk encryption |
| Log file leakage | `_enforce_in_memory_logging` strips FileHandlers at boot |
| Cross-patient context bleed | Stateless LLM, fresh ReportStore per upload |
| Tampering with stored ciphertext | AES-GCM auth tag detects modification |
| Loss of up to 2 Shamir shares | 3-of-5 threshold |
| Fragmenter exposing PDF magic bytes | Bytes split AFTER compression+encryption (no PDF header in fragments) |

### What it DOES NOT defend against

| Threat | Why not |
|---|---|
| Kernel-level memory introspection | Plaintext exists transiently in process memory |
| Swap-out of memory pages | OS paging is outside our control |
| Hibernation snapshots | Same as swap |
| Malicious code in same Python process | Crypto keys are reachable in the process address space |
| Compromised Ollama server | We trust the local Ollama process |
| Side-channel attacks on AES | Library responsibility (cryptography.hazmat is constant-time) |
| Coercion / lawful subpoena | Out of scope for technical mitigations |

### Trust boundaries

```
┌─────────────────────────────────────────────┐
│ TRUSTED                                     │
│ - Local Python process                      │
│ - Local Ollama server                       │
│ - Local filesystem (after our cleanup)      │
│ - cryptography library                      │
│ - shamir_mnemonic library                   │
└─────────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │ UNTRUSTED         │
        │ - Network         │
        │ - Other processes │
        │ - Disk after      │
        │   sleep/swap      │
        └───────────────────┘
```

---

## 17. Appendix — File Structure

```
autism_rag_assistant/
├── app.py                          # Streamlit UI + logging policy enforcement
├── rag_pipeline.py                 # RAGPipeline facade + RAGConfig
├── dual_retriever.py               # Query routing, normalization, fusion
├── advanced_rag.py                 # Query rewrite + cross-encoder + compression
├── kb_memory.py                    # Persistent KB with disk cache + hybrid retrieval
├── report_store.py                 # Per-session encrypted Chroma + BM25
├── llm_engine.py                   # ChatOllama wrapper, two heads, clinical prompt
├── crypto_utils.py                 # Fernet helpers + key resolution
├── ingest_kb.py                    # CLI wrapper for phase1.ingest
├── ragas_eval.py                   # 4 RAGAS scorers + RAGASEvaluator
├── run_evaluation.py               # CLI for evaluation runs
├── test_retrieval.py               # Retrieval-only test (no LLM)
├── test_validation.py              # Full pipeline validation
├── config.ini                      # All runtime configuration
├── requirements.txt                # Python deps
├── secret.key                      # Fernet key (auto-generated)
│
├── dfs/
│   ├── __init__.py                 # exports EphemeralDFSStore + sweep_orphans
│   ├── ephemeral_store.py          # Storage orchestrator (Layer A: Storage)
│   ├── compression_manager.py      # zstd
│   ├── erasure_manager.py          # byte-range fragment split/merge
│   ├── encryption.py               # AES-256-GCM (Layer B: Crypto core)
│   ├── key_manager.py              # Shamir SLIP-39
│   └── (transfer_*.py, drive_config.py, main.py — standalone CLI, NOT used by app)
│
├── phase1/
│   ├── pdf_parser.py               # pdfplumber → PyPDF2 fallback + parse_pdf_bytes
│   ├── chunker.py                  # medical-section split + recursive char split
│   ├── embedder.py                 # OllamaEmbedder (batched, retries)
│   ├── vector_store.py             # KnowledgeBaseStore (Chroma upsert with dedup)
│   └── ingest.py                   # CLI orchestrator
│
├── data/
│   ├── kb_pdfs/                    # source PDFs (DSM-5, ICD-11, M-CHAT, NICE)
│   ├── chroma_kb/                  # persistent KB Chroma collection
│   ├── kb_cache/                   # NPZ embedding cache + meta JSON
│   ├── eval_dataset.json           # KB-only eval samples
│   ├── eval_dataset_report.json    # report-grounded eval samples
│   └── session_reports/            # test PDFs for run_evaluation.py
│
├── logs/
│   ├── ingestion.log               # KB build logs (no PHI)
│   ├── eval_run.log                # latest eval run console
│   ├── eval_cross_encoder_run.log  # cross-encoder ablation log
│   └── eval_report_*.json          # full RAGAS reports
│
├── docs/
│   ├── architecture.md             # high-level overview
│   ├── modules.md                  # per-module reference
│   ├── configuration.md            # config.ini reference
│   ├── evaluation.md               # RAGAS metrics + CLI
│   └── TECHNICAL_REFERENCE.md      # this document
│
└── README.md                       # project entry point
```

### Runtime tempdirs (NOT in repo)
```
/tmp/rag_dfs_*/         # encrypted PDF fragments (AES-GCM + Shamir)
└── node_0/ … node_4/
    ├── frag_*.bin
    └── share_*.txt

/tmp/rag_chroma_*/      # per-session report ChromaDB
└── chroma.sqlite3 + binary index files
```

Both are created via `tempfile.mkdtemp` and wiped on session end / atexit / sweep_orphans.

---

## End of Reference

Last updated for the codebase as of the May 2026 capstone submission. For runtime questions not covered here, the code itself is the source of truth — every module has docstrings explaining the why, and the existing `docs/architecture.md`, `docs/modules.md`, `docs/configuration.md`, and `docs/evaluation.md` cover the same ground at higher level.
