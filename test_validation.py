#!/usr/bin/env python3
"""
Validation script -- runs 3 report-grounded questions through the updated
pipeline and prints retrieved chunks + answers.

These questions require BOTH the patient report AND the medical KB to answer
correctly, matching the real clinical workflow.

Usage:
    python test_validation.py
    python test_validation.py --advanced-rag
"""

import argparse
import os
import time
import logging

from rag_pipeline import RAGPipeline, RAGConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("test_validation")

REPORT_PATH = "./data/session_reports/dummy_autism_screening_report.pdf"

# Questions a doctor would ask about the ASD-TEST-001 report
VALIDATION_QUESTIONS = [
    {
        "id": "RQ1",
        "question": "Based on the communication observations in the report, does this child show signs consistent with ASD screening criteria?",
        "why": "Tests retrieval from BOTH report (echolalia, single words) and KB (DSM-5 criteria, M-CHAT)",
    },
    {
        "id": "RQ3",
        "question": "How do the repetitive behaviors described in this report align with the DSM-5 diagnostic criteria for ASD?",
        "why": "Tests report (hand flapping, lining up toys) + KB (DSM-5 Criterion B) synthesis",
    },
    {
        "id": "RQ6",
        "question": "Does the child's social interaction pattern suggest ASD or could it indicate social communication disorder instead?",
        "why": "Tests comparative reasoning: report (social observations) + KB (ASD vs SCD distinction)",
    },
]


class FakeUpload:
    def __init__(self, path: str):
        self.name = os.path.basename(path)
        self._data = open(path, "rb").read()
    def getbuffer(self):
        return self._data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--advanced-rag", action="store_true")
    args = parser.parse_args()

    config = RAGConfig()
    if args.advanced_rag:
        config.query_rewriting_enabled = True
        config.cross_encoder_enabled = True
        config.compression_enabled = True

    print("Initialising pipeline...")
    pipeline = RAGPipeline(config=config)

    # Load the patient report
    print(f"Loading report: {REPORT_PATH}")
    meta = pipeline.load_report_from_uploaded_file(FakeUpload(REPORT_PATH))
    print(f"Report loaded: {meta.filename} ({meta.num_pages} pages, {meta.num_chunks} chunks)")
    print(f"Config: kb_top_k={config.kb_top_k}, rerank_top_k={config.rerank_top_k}\n")

    for item in VALIDATION_QUESTIONS:
        qid = item["id"]
        question = item["question"]

        print("=" * 70)
        print(f"  {qid}: {question}")
        print(f"  Tests: {item['why']}")
        print("=" * 70)

        t0 = time.time()
        answer, ctx = pipeline.ask(question)
        elapsed = time.time() - t0

        from llm_engine import INSUFFICIENT_CONTEXT_MSG
        guard_triggered = answer.startswith(INSUFFICIENT_CONTEXT_MSG[:40])
        max_score = max((c.similarity for c in ctx.chunks), default=0.0)

        # Count sources
        report_chunks = sum(1 for c in ctx.chunks if c.source_type == "report")
        kb_chunks = sum(1 for c in ctx.chunks if c.source_type == "kb")

        print(f"\n  Retrieved {len(ctx.chunks)} chunks ({elapsed:.1f}s)")
        print(f"  Sources: {report_chunks} report + {kb_chunks} KB")
        print(f"  [DEBUG] chunks={len(ctx.chunks)}, max_score={max_score:.3f}, guard_triggered={guard_triggered}\n")

        for i, chunk in enumerate(ctx.chunks, 1):
            sim = chunk.similarity
            src = chunk.source_type.upper()
            preview = chunk.text[:200].replace("\n", " ")
            print(f"  [{i}] sim={sim:.3f} ({src}) {preview}...")
            print()

        print(f"  ANSWER ({len(answer)} chars):")
        print("-" * 70)
        print(answer[:600])
        if len(answer) > 600:
            print(f"  ... ({len(answer) - 600} more chars)")
        print("-" * 70)
        print()


if __name__ == "__main__":
    main()
