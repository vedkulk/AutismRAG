#!/usr/bin/env python3
"""
Quick retrieval test — no LLM, no RAGAS. Just check what chunks come back.
Tests whether the right report sections are retrieved for each question.
"""
import json
import os

from rag_pipeline import RAGPipeline, RAGConfig
from run_evaluation import FakeUpload

REPORTS_DIR = "./data/session_reports"


def load_report(pipeline, filename):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    pipeline.load_report_from_uploaded_file(FakeUpload(path))


def test_retrieval():
    # Basic config — no advanced RAG
    config = RAGConfig()
    pipeline = RAGPipeline(config=config)

    with open("./data/eval_dataset_report.json") as f:
        dataset = json.load(f)

    current_report = None

    for i, item in enumerate(dataset, 1):
        report = item.get("report", "")
        if report and report != current_report:
            load_report(pipeline, report)
            current_report = report

        question = item["question"]
        ctx = pipeline._build_context(question)

        report_chunks = [c for c in ctx.chunks if c.source_type == "report"]
        kb_chunks = [c for c in ctx.chunks if c.source_type == "kb"]

        print(f"\n{'='*70}")
        print(f"Q{i}: {question[:80]}...")
        print(f"  Report chunks: {len(report_chunks)}  |  KB chunks: {len(kb_chunks)}")
        print(f"  Total: {len(ctx.chunks)}")

        if report_chunks:
            print(f"  --- Report chunks ---")
            for j, c in enumerate(report_chunks):
                preview = c.text[:100].replace("\n", " ")
                print(f"    [{j}] sim={c.similarity:.3f}  {preview}...")
        else:
            print(f"  *** NO REPORT CHUNKS RETRIEVED ***")

        if kb_chunks:
            print(f"  --- Top 3 KB chunks ---")
            for j, c in enumerate(kb_chunks[:3]):
                preview = c.text[:100].replace("\n", " ")
                print(f"    [{j}] sim={c.similarity:.3f}  {preview}...")

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    test_retrieval()
