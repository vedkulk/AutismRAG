#!/usr/bin/env python3
"""
Phase 5 -- Run RAGAS evaluation on the RAG pipeline.

Usage:
    python run_evaluation.py                                           # report-grounded (default)
    python run_evaluation.py --dataset ./data/eval_dataset.json        # KB-only (old dataset)
    python run_evaluation.py --report-pdf sample.pdf                   # override report for all Qs
    python run_evaluation.py --advanced-rag                            # enable Phase 4 techniques
    python run_evaluation.py --limit 3                                 # evaluate first N only

The report-grounded dataset (eval_dataset_report.json) includes a "report" field per
question. The runner automatically loads the correct report from ./data/session_reports/
before running each question group.

Output:
    - Console summary table
    - JSON report at ./logs/eval_report_<timestamp>.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from rag_pipeline import RAGPipeline, RAGConfig
from ragas_eval import EvalSample, RAGASEvaluator, EvalReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_evaluation")

REPORTS_DIR = "./data/session_reports"


# -- CLI -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAGAS evaluation for ASD RAG pipeline")
    p.add_argument(
        "--dataset",
        default="./data/eval_dataset_report.json",
        help="Path to evaluation dataset JSON (default: report-grounded dataset)",
    )
    p.add_argument(
        "--report-pdf",
        default=None,
        help="Path to a single report PDF to load for ALL questions (overrides per-Q reports)",
    )
    p.add_argument(
        "--advanced-rag",
        action="store_true",
        help="Enable all Phase 4 advanced RAG techniques",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions",
    )
    p.add_argument(
        "--output-dir",
        default="./logs",
        help="Directory for the JSON report",
    )
    return p.parse_args()


# -- Helpers ---------------------------------------------------------------

class FakeUpload:
    """Mimic Streamlit's UploadedFile for programmatic report loading."""
    def __init__(self, path: str):
        self.name = os.path.basename(path)
        self._data = open(path, "rb").read()
    def getbuffer(self):
        return self._data


def load_report_into_pipeline(pipeline: RAGPipeline, report_path: str) -> str:
    """Load a report PDF into the pipeline. Returns the filename."""
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report not found: {report_path}")
    meta = pipeline.load_report_from_uploaded_file(FakeUpload(report_path))
    logger.info(
        "Report loaded: %s (%d pages, %d chunks)",
        meta.filename, meta.num_pages, meta.num_chunks,
    )
    return meta.filename


def resolve_report_path(report_field: str) -> str:
    """Resolve a report filename from the dataset to an absolute path."""
    # If it's already an absolute/relative path that exists, use it
    if os.path.exists(report_field):
        return report_field
    # Otherwise look in REPORTS_DIR
    candidate = os.path.join(REPORTS_DIR, report_field)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"Report '{report_field}' not found in {REPORTS_DIR} or as a path"
    )


# -- Dataset loading -------------------------------------------------------

def load_dataset(path: str, limit: int | None = None) -> list[EvalSample]:
    with open(path) as f:
        raw = json.load(f)

    samples = [
        EvalSample(
            question=item["question"],
            ground_truth=item["ground_truth"],
            report=item.get("report", ""),
        )
        for item in raw
    ]
    if limit is not None:
        samples = samples[:limit]
    return samples


# -- Pipeline runner -------------------------------------------------------

def run_pipeline_on_samples(
    pipeline: RAGPipeline,
    samples: list[EvalSample],
    global_report: str | None = None,
) -> list[EvalSample]:
    """Run the RAG pipeline on each sample, populating answer + contexts.

    Handles per-question report loading: when a sample's 'report' field
    differs from the currently loaded report, the runner loads the new one.
    If global_report is set, it overrides all per-question reports.
    """
    total = len(samples)
    current_report: str | None = None

    for i, sample in enumerate(samples, 1):
        # Determine which report to load for this question
        target_report = global_report or sample.report or None

        if target_report and target_report != current_report:
            logger.info("Loading report for evaluation: %s", target_report)
            report_path = resolve_report_path(target_report)
            load_report_into_pipeline(pipeline, report_path)
            current_report = target_report

        logger.info("-- Question %d/%d --", i, total)
        logger.info("Q: %s", sample.question[:100])

        t0 = time.time()
        answer, ctx = pipeline.ask(sample.question)
        elapsed = time.time() - t0

        sample.answer = answer
        sample.contexts = [c.text for c in ctx.chunks]
        sample.context_sources = [c.source_type for c in ctx.chunks]
        sample.latency_seconds = round(elapsed, 2)

        logger.info("A: %s... (%d chunks, %.1fs)", answer[:80], len(ctx.chunks), elapsed)

    return samples


# -- Report formatting -----------------------------------------------------

def print_summary(report: EvalReport) -> None:
    """Print a formatted console summary."""
    agg = report.aggregate_scores
    n = len(report.samples)

    print("\n")
    print("=" * 68)
    print("  RAGAS EVALUATION REPORT")
    print("=" * 68)
    print(f"  Timestamp   : {report.timestamp}")
    print(f"  Samples     : {n}")
    if report.config_snapshot:
        for k, v in report.config_snapshot.items():
            print(f"  {k:<12}: {v}")
    print("-" * 68)
    print(f"  {'Metric':<24} {'Score':>8}")
    print("-" * 68)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        val = agg.get(metric, 0)
        bar = "#" * int(val * 20) + "." * (20 - int(val * 20))
        print(f"  {metric:<24} {val:>7.4f}  [{bar}]")
    print("-" * 68)
    ragas = agg.get("ragas_score", 0)
    bar = "#" * int(ragas * 20) + "." * (20 - int(ragas * 20))
    print(f"  {'RAGAS Score (harmonic)':<24} {ragas:>7.4f}  [{bar}]")
    print("=" * 68)

    # Per-question breakdown
    print(f"\n  {'#':<3} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7}  {'Latency':>7}  Question")
    print("-" * 95)
    for i, s in enumerate(report.samples, 1):
        sc = s.scores
        rpt = f" [{s.report}]" if s.report else ""
        print(
            f"  {i:<3} {sc.get('faithfulness',0):>7.3f} "
            f"{sc.get('answer_relevancy',0):>7.3f} "
            f"{sc.get('context_precision',0):>7.3f} "
            f"{sc.get('context_recall',0):>7.3f}  "
            f"{s.latency_seconds:>6.1f}s  "
            f"{s.question[:40]}{'...' if len(s.question) > 40 else ''}{rpt}"
        )
    print()


def save_report(report: EvalReport, output_dir: str) -> str:
    """Save JSON report and return the file path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"eval_report_{ts}.json")
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    return path


# -- Main ------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Build config
    config = RAGConfig()
    config_snapshot = {"llm_model": config.llm_model, "embed_model": config.embed_model}

    if args.advanced_rag:
        config.hyde_enabled = True
        config.query_rewriting_enabled = True
        config.cross_encoder_enabled = True
        config.compression_enabled = True
        config_snapshot["advanced_rag"] = "all enabled"
    else:
        config_snapshot["advanced_rag"] = "disabled"

    # Initialise pipeline
    logger.info("Initialising RAG pipeline...")
    pipeline = RAGPipeline(config=config)

    # If a global report is specified via CLI, note it
    global_report = None
    if args.report_pdf:
        global_report = args.report_pdf
        config_snapshot["report_override"] = os.path.basename(args.report_pdf)

    # Load evaluation dataset
    logger.info("Loading dataset: %s", args.dataset)
    samples = load_dataset(args.dataset, limit=args.limit)
    logger.info("Loaded %d evaluation samples", len(samples))

    # Count reports referenced
    reports_used = set(s.report for s in samples if s.report)
    if reports_used:
        config_snapshot["reports"] = ", ".join(sorted(reports_used))
        logger.info("Dataset references %d report(s): %s", len(reports_used), reports_used)

    # Phase 1: Run pipeline to get answers + contexts
    logger.info("Running pipeline on evaluation samples...")
    run_pipeline_on_samples(pipeline, samples, global_report=global_report)

    # Phase 2: Compute RAGAS metrics
    logger.info("Computing RAGAS metrics (this will make multiple LLM calls per sample)...")
    evaluator = RAGASEvaluator(
        llm=pipeline._llm,
        embedder=pipeline._embedder,
    )
    report = evaluator.evaluate_dataset(samples, config_snapshot=config_snapshot)

    # Output
    print_summary(report)
    report_path = save_report(report, args.output_dir)
    logger.info("Full report saved to: %s", report_path)


if __name__ == "__main__":
    main()
