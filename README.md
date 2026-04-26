# ASD Clinical Assistant

A privacy-conscious RAG (Retrieval-Augmented Generation) decision-support tool for pediatric Autism Spectrum Disorder. A clinician uploads a patient PDF report and asks clinical questions; the assistant grounds its answers in (a) the uploaded report and (b) a persistent knowledge base of ASD literature and guidelines (DSM-5, ADOS-2, M-CHAT, etc.).

> **For educational and decision-support use only. Not a substitute for formal diagnosis.**

---

## Privacy model

Patient data **never persists on disk in plaintext**. The architecture is built around this invariant:

- The uploaded PDF is encrypted, fragmented (Shamir-split key shares), and scattered across a system tempdir for the duration of a single session — see `dfs/`.
- Plaintext bytes only exist in memory during PDF parsing, then go out of scope.
- Report chunks in the per-session Chroma store are Fernet-encrypted at rest.
- File-based logging is stripped from the root logger on app boot — everything goes to stderr only.
- Clicking **End session**, loading a new report, or closing the tab wipes all patient state.

The persistent knowledge base (general medical guidelines, no PHI) is the only thing that survives between sessions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  app.py  (Streamlit UI)                                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   RAGPipeline       │   rag_pipeline.py
                │   (facade)          │
                └──────────┬──────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────┐   ┌────────▼────────┐   ┌──────▼──────┐
│ DualRetriev │   │ AdvancedRAG     │   │ LLMEngine   │
│ - classify  │   │ - HyDE          │   │ (Ollama)    │
│ - normalize │   │ - QueryRewrite  │   └─────────────┘
│ - fuse      │   │ - CrossEncoder  │
└──┬────────┬─┘   │ - Compression   │
   │        │    └─────────────────┘
   │        │
┌──▼──┐  ┌──▼──────────┐         ┌────────────┐
│ KB  │  │ ReportStore │◄────────│ Ephemeral  │
│(Chr │  │ (encrypted) │         │ DFS Store  │
│oma) │  │             │         │ (encrypt+  │
└─────┘  └─────────────┘         │  fragment) │
                                 └────────────┘
```

### Key modules

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI, CSS, message rendering, session lifecycle |
| `rag_pipeline.py` | `RAGPipeline` facade + `RAGConfig` (loads from `config.ini`) |
| `dual_retriever.py` | Query classification (patient/medical/both), query normalization, fusion of KB + report hits |
| `advanced_rag.py` | Toggleable Phase 4 enhancements: HyDE, multi-query rewriting, cross-encoder rerank, embedding-based compression |
| `kb_memory.py` | Persistent KB store wrapper (`InMemoryKB`) |
| `report_store.py` | Per-session Fernet-encrypted Chroma store |
| `dfs/` | Ephemeral encrypt-and-fragment store for the uploaded PDF |
| `llm_engine.py` | Ollama wrapper (streaming + non-streaming) |
| `phase1/` | KB ingestion: PDF parse, chunk, embed, vector store |
| `ingest_kb.py` | Thin wrapper around `phase1.ingest` |
| `ragas_eval.py`, `run_evaluation.py` | RAGAS-based evaluation |

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- ~2 GB free disk space (KB Chroma + cross-encoder model)

### Install

```bash
git clone <repo-url> autism_rag_assistant
cd autism_rag_assistant

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Pull Ollama models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### Build the knowledge base

Drop ASD-related PDFs (DSM-5 chapters, ADOS-2 manuals, AAP guidelines, etc.) into `data/kb_pdfs/`, then:

```bash
python ingest_kb.py
```

This parses, chunks, embeds, and persists the KB into `data/chroma_kb/`. Re-run to add new documents (existing ones are deduplicated by hash).

### Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically http://localhost:8501).

---

## Usage

1. **Upload** a patient PDF report in the sidebar
2. Click **Load report** — the file is encrypted, fragmented, parsed in memory, and indexed
3. **Ask** clinical questions in the chat box, e.g.:
   - *"Tell me about this child"* → patient-routed, summarizes the report
   - *"What are the DSM-5 criteria for ASD?"* → KB-routed
   - *"Does this patient meet ADOS-2 thresholds?"* → both stores
4. Answers stream in with a **Sources** expander showing which chunks grounded the response
5. Click **End session** when done — wipes all patient state

---

## Configuration

All tuning lives in `config.ini`. Notable sections:

- `[ollama]` — model names and base URL
- `[chunking]` — `chunk_size`, `chunk_overlap`, `min_chunk_size`
- `[retrieval]` — `kb_top_k`, `report_top_k`, `retrieval_type` (`similarity` | `mmr`), `mmr_lambda`
- `[advanced_rag]` — toggle HyDE / query rewriting / cross-encoder / compression independently
- `[dfs]` — `num_fragments`, `num_key_shares`, `key_threshold` (Shamir secret-sharing parameters)

Each advanced-RAG technique degrades gracefully: if it errors at runtime, the pipeline continues without it.

---

## Evaluation

```bash
python run_evaluation.py
```

Runs RAGAS metrics (faithfulness, answer relevancy, context precision/recall) against `data/eval_dataset.json` and writes a report into `logs/`.

---

## Tests

```bash
python test_retrieval.py
python test_validation.py
```

---

## Disclaimer

This tool is intended for clinical decision support and education. It does not provide medical advice, does not constitute a diagnosis, and must not be used as the sole basis for clinical decisions. Always verify outputs against primary sources and clinical judgment.
