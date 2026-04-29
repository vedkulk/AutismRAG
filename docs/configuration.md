# Configuration reference

All runtime tuning lives in `config.ini`. `RAGConfig.from_ini()` reads it; missing fields fall back to dataclass defaults.

## `[ollama]`

| Key | Default | Notes |
|---|---|---|
| `base_url` | `http://localhost:11434` | Ollama server endpoint. |
| `llm_model` | `llama3.1` | Generation model. Anything Ollama supports works. |
| `embed_model` | `nomic-embed-text` | Embedding model. **Must match what the KB was indexed with** — switching mid-flight invalidates `data/chroma_kb/` and `data/kb_cache/`. |

## `[paths]`

| Key | Default | Notes |
|---|---|---|
| `kb_data_dir` | `./data/kb_pdfs` | Source PDFs for KB ingestion. |
| `chroma_db_dir` | `./data/chroma_kb` | Persistent KB Chroma collection. |
| `log_dir` | `./logs` | Where ingestion + eval logs go. App runtime logs go only to stderr. |

## `[chunking]`

| Key | Default | Notes |
|---|---|---|
| `chunk_size` | `300` | Characters per chunk. ~256–512 tokens for `nomic-embed-text`. |
| `chunk_overlap` | `40` | Overlap between adjacent chunks (~12.5% of `chunk_size`). |
| `min_chunk_size` | `80` | Smaller chunks are dropped. |

Tuning notes: smaller chunks improve retrieval precision but increase Chroma size and BM25 index cost. 300 is the empirically tuned value.

## `[retrieval]`

| Key | Default | Notes |
|---|---|---|
| `kb_top_k` | `25` | KB chunks retrieved per query (over-retrieve for reranking). |
| `report_top_k` | `4` | Report chunks retrieved per query. |
| `retrieval_type` | `mmr` | `similarity` or `mmr` (Maximal Marginal Relevance, for diversity). |
| `mmr_lambda` | `0.5` | MMR diversity vs relevance balance: `0` = max relevance, `1` = max diversity. Only used when `retrieval_type=mmr`. |

`DualRetriever` overrides these per query type — see the routing table in `architecture.md`.

## `[advanced_rag]`

| Key | Default | Notes |
|---|---|---|
| `query_rewriting_enabled` | `true` | LLM rewrites the question into up to 3 retrieval-optimized variants. |
| `cross_encoder_enabled` | `true` | Re-rank retrieved chunks with a cross-encoder. |
| `compression_enabled` | `true` | Embedding-based filtering of low-relevance chunks (cosine ≥ 0.30). |
| `cross_encoder_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Auto-downloaded on first use (~80 MB). |
| `rerank_top_k` | `8` | Top-K kept after cross-encoder reranking. |
| `compression_top_k` | `4` | Number of chunks the compressor inspects. Should be ≤ `rerank_top_k`. |

Each technique fails soft: if it errors at runtime, the pipeline keeps going with the unmodified chunks. Disabling all three short-circuits the orchestrator to `None`.

## `[evaluation]`

| Key | Default | Notes |
|---|---|---|
| `eval_dataset` | `./data/eval_dataset.json` | Default dataset for `run_evaluation.py`. The CLI accepts `--dataset` to override. |
| `eval_output_dir` | `./logs` | Where `eval_report_<timestamp>.json` is written. |
| `num_relevancy_questions` | `3` | How many hypothetical questions `AnswerRelevancyScorer` generates per sample. |

## `[chroma]`

| Key | Default | Notes |
|---|---|---|
| `kb_collection_name` | `autism_kb_medical_guidelines` | Collection name in `data/chroma_kb/`. |
| `distance_metric` | `cosine` | `cosine`, `l2`, or `ip`. Matches what Chroma was created with — changing it requires a re-ingest. |

## `[dfs]`

Parameters for the ephemeral encrypted+fragmented per-session PDF store (`dfs/ephemeral_store.py`).

| Key | Default | Notes |
|---|---|---|
| `num_fragments` | `10` | How many pieces the encrypted PDF is split into. Stored under `node_*/` subdirs of a system tempdir. |
| `num_key_shares` | `5` | Total Shamir shares of the AES-256-GCM key. |
| `key_threshold` | `3` | Shares required to reconstruct the key. Must be ≤ `num_key_shares`. |

These are mostly defence-in-depth on a single host — AES-GCM is doing the real protection. Lowering `num_fragments` reduces filesystem chatter; lowering `key_threshold` makes reconstruction faster but the key easier to recover.

## Environment variables

| Variable | Purpose |
|---|---|
| `RAG_FERNET_KEY` | Overrides `secret.key`. Useful for production deployments where you don't want the key file in the repo. Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. |

## What changes when you flip these

| Change | What invalidates |
|---|---|
| `embed_model` | `data/chroma_kb/` (re-ingest) and `data/kb_cache/` |
| `chunk_size`, `chunk_overlap`, `min_chunk_size` | `data/chroma_kb/` (re-ingest) |
| `kb_collection_name` | `data/chroma_kb/` (effectively a new collection) |
| `distance_metric` | `data/chroma_kb/` (Chroma rejects mismatch) |
| `retrieval_type`, `mmr_lambda` | nothing on disk; takes effect next query |
| `[advanced_rag]` toggles | nothing on disk; takes effect next query |
| `[dfs]` | nothing on disk; takes effect next upload |
