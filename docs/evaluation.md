# Evaluation

The project uses RAGAS-style metrics computed by the local Ollama LLM (no external API). Implementation lives in `ragas_eval.py`; the CLI is `run_evaluation.py`.

## Metrics

All metrics return a float in `[0, 1]`. Higher is better for all four.

| Metric | Question it answers | How it's computed |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | LLM extracts atomic claims from the answer. For each claim, LLM judges whether the context supports it. Score = fraction of supported claims. |
| **Answer Relevancy** | Is the answer actually answering the question? | LLM generates `num_relevancy_questions` hypothetical questions that the answer could be answering. Score = mean cosine similarity (`nomic-embed-text` embeddings) between each generated question and the original. |
| **Context Precision** | Are relevant contexts ranked higher than irrelevant ones? | LLM judges relevance of each context to the question. Score = a precision-at-k variant that rewards relevant contexts appearing earlier in the list. |
| **Context Recall** | Does the context cover everything in the ground-truth answer? | LLM extracts atomic claims from the ground truth. Score = fraction of those claims that are supported by the retrieved context. |

`RAGASEvaluator.evaluate(samples) -> EvalReport` runs all four sequentially per sample, emits per-sample scores, and aggregates means.

## Running an evaluation

```bash
python run_evaluation.py                                      # report-grounded (default)
python run_evaluation.py --dataset ./data/eval_dataset.json   # KB-only
python run_evaluation.py --report-pdf sample.pdf              # override report for all questions
python run_evaluation.py --advanced-rag                       # force-enable Phase 4 techniques
python run_evaluation.py --cross-encoder                      # force-enable just cross-encoder
python run_evaluation.py --limit 3                            # first N questions only
```

Output:
- Console summary table (per-sample + aggregate)
- `logs/eval_report_<timestamp>.json` (full per-metric breakdown)

## Datasets

- **`data/eval_dataset.json`** — KB-only. Each sample has `question`, `ground_truth`, optional `contexts`. No `report` field.
- **`data/eval_dataset_report.json`** — report-grounded (default). Each sample has the same fields plus `report: <pdf_filename>`. The runner loads the matching PDF from `data/session_reports/` before that question group.

`run_evaluation.py:resolve_report_path` resolves the `report` field against `data/session_reports/`, so any PDF you reference must live there.

## Evaluation-specific config

In `[evaluation]` (see `configuration.md`):

- `eval_dataset` — default dataset path
- `eval_output_dir` — where the JSON report is written (also accepts `--output-dir`)
- `num_relevancy_questions` — how many hypothetical questions `AnswerRelevancyScorer` generates per sample (more = lower variance, more LLM calls)

## Quick non-RAGAS checks

For retrieval-tuning iterations, RAGAS is overkill — each sample makes ~5–10 LLM calls. Use these instead:

- `python test_retrieval.py` — dumps retrieved chunks per question, no LLM call. Best signal for "is the right context coming back?"
- `python test_validation.py` — small fixed validation set against a specific dummy report. Runs the full pipeline including LLM.

## Reading a report

The JSON has shape:

```json
{
  "config_snapshot": {...},
  "samples": [
    {
      "question": "...",
      "ground_truth": "...",
      "answer": "...",
      "contexts": [...],
      "scores": {
        "faithfulness": 0.83,
        "answer_relevancy": 0.91,
        "context_precision": 0.75,
        "context_recall": 0.66
      },
      "latency_seconds": 12.4
    }
  ],
  "aggregate": {
    "faithfulness": 0.81,
    "answer_relevancy": 0.88,
    ...
  }
}
```

When iterating on retrieval, watch **context recall** — it isolates whether the right chunks are being retrieved, independent of generation quality. **Context precision** catches over-retrieval (relevant chunks buried under irrelevant ones).

When iterating on prompts/generation, watch **faithfulness** and **answer relevancy**.
