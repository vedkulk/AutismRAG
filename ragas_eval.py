from __future__ import annotations

"""
Phase 5 — RAGAS-style evaluation metrics for the RAG pipeline.

Implements four core metrics using the local Ollama LLM as judge:
  1. Faithfulness       — Is the answer grounded in the retrieved context?
  2. Answer Relevancy   — Is the answer relevant to the question?
  3. Context Precision   — Are retrieved contexts relevant and well-ordered?
  4. Context Recall      — Does the context cover the ground truth?

All metrics return a float in [0, 1].  No external API calls — everything
runs through the same Ollama instance used by the main pipeline.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

from llm_engine import LLMEngine
from phase1.embedder import OllamaEmbedder

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EvalSample:
    """A single evaluation sample."""
    question: str
    ground_truth: str
    report: str = ""  # filename of the patient report to load (empty = KB-only)
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    context_sources: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    latency_seconds: float = 0.0


@dataclass
class EvalReport:
    """Aggregate evaluation report across all samples."""
    samples: List[EvalSample]
    aggregate_scores: Dict[str, float] = field(default_factory=dict)
    config_snapshot: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "config": self.config_snapshot,
            "aggregate_scores": self.aggregate_scores,
            "num_samples": len(self.samples),
            "samples": [asdict(s) for s in self.samples],
        }


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def _parse_json_list(text: str) -> List[str]:
    """Best-effort parse a JSON list from LLM output."""
    text = text.strip()
    # Find the JSON array in the response
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    # Fallback: split by numbered lines
    lines = []
    for line in text.split("\n"):
        line = line.strip().lstrip("0123456789.-) ").strip('"').strip()
        if line:
            lines.append(line)
    return lines


def _parse_verdict(text: str) -> bool:
    """Parse a YES/NO verdict from LLM output."""
    t = text.strip().upper()
    return t.startswith("Y") or t.startswith("1") or "YES" in t


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


# ══════════════════════════════════════════════════════════════════════════
#  1. Faithfulness
# ══════════════════════════════════════════════════════════════════════════

class FaithfulnessScorer:
    """
    Measures whether the answer is grounded in the retrieved context.

    Algorithm:
      1. Extract atomic claims/statements from the answer.
      2. For each claim, ask the LLM if it can be inferred from the context.
      3. Score = (# supported claims) / (# total claims).
    """

    _EXTRACT_PROMPT = (
        "Given the following answer, extract a list of independent factual "
        "claims or statements. Return ONLY a JSON array of strings, nothing else.\n\n"
        "Answer:\n{answer}\n\n"
        "JSON array of claims:"
    )

    _VERIFY_PROMPT = (
        "Given the following context and claim, determine if the claim can be "
        "inferred or supported by the context. Answer with a single word: "
        "YES or NO.\n\n"
        "Context:\n{context}\n\n"
        "Claim: {claim}\n\n"
        "Verdict:"
    )

    def __init__(self, llm: LLMEngine) -> None:
        self._llm = llm

    def score(self, answer: str, contexts: List[str]) -> float:
        if not answer.strip() or not contexts:
            return 0.0

        # Step 1: Extract claims
        claims = _parse_json_list(
            self._llm.generate_raw(self._EXTRACT_PROMPT.format(answer=answer))
        )
        if not claims:
            logger.warning("Faithfulness: no claims extracted from answer")
            return 0.0

        # Step 2: Verify each claim
        combined_context = "\n\n".join(contexts)
        supported = 0
        for claim in claims:
            verdict = self._llm.generate_raw(
                self._VERIFY_PROMPT.format(context=combined_context, claim=claim)
            )
            if _parse_verdict(verdict):
                supported += 1

        score = supported / len(claims)
        logger.info(
            "Faithfulness: %d/%d claims supported (%.3f)",
            supported, len(claims), score,
        )
        return _clamp(score)


# ══════════════════════════════════════════════════════════════════════════
#  2. Answer Relevancy
# ══════════════════════════════════════════════════════════════════════════

class AnswerRelevancyScorer:
    """
    Measures whether the answer is relevant to the question.

    Algorithm:
      1. Generate N hypothetical questions that the answer could be answering.
      2. Compute embedding cosine similarity between each generated question
         and the original question.
      3. Score = mean similarity.
    """

    _GENERATE_PROMPT = (
        "Given the following answer, generate {n} different questions that "
        "this answer could be responding to. The questions should capture the "
        "main topics of the answer. Return ONLY a JSON array of strings.\n\n"
        "Answer:\n{answer}\n\n"
        "JSON array of {n} questions:"
    )

    def __init__(self, llm: LLMEngine, embedder: OllamaEmbedder, n: int = 3) -> None:
        self._llm = llm
        self._embedder = embedder
        self._n = n

    def score(self, question: str, answer: str) -> float:
        if not answer.strip():
            return 0.0

        # Step 1: Generate hypothetical questions
        generated = _parse_json_list(
            self._llm.generate_raw(
                self._GENERATE_PROMPT.format(answer=answer, n=self._n)
            )
        )
        if not generated:
            logger.warning("AnswerRelevancy: no questions generated")
            return 0.0

        # Step 2: Embed original + generated questions
        original_emb = np.asarray(
            self._embedder.embed_query(question), dtype=np.float32
        )
        generated_embs = self._embedder.embed_texts(generated)

        # Step 3: Cosine similarities
        sims = []
        orig_norm = np.linalg.norm(original_emb) + 1e-8
        for emb in generated_embs:
            g = np.asarray(emb, dtype=np.float32)
            sim = float(np.dot(original_emb, g) / (orig_norm * (np.linalg.norm(g) + 1e-8)))
            sims.append(sim)

        score = float(np.mean(sims))
        logger.info(
            "AnswerRelevancy: %d generated questions, mean sim=%.3f",
            len(sims), score,
        )
        return _clamp(score)


# ══════════════════════════════════════════════════════════════════════════
#  3. Context Precision
# ══════════════════════════════════════════════════════════════════════════

class ContextPrecisionScorer:
    """
    Measures whether retrieved contexts are relevant, weighted by rank.

    Algorithm:
      1. For each chunk, ask the LLM if it is relevant to the question
         and useful for answering it.
      2. Compute precision@k at each position.
      3. Score = mean of precision@k weighted by relevance indicator.
         (Average Precision formula)
    """

    _RELEVANCE_PROMPT = (
        "Given a question and a context passage, determine if the context "
        "contains information useful for answering the question. Answer "
        "with a single word: YES or NO.\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Verdict:"
    )

    def __init__(self, llm: LLMEngine) -> None:
        self._llm = llm

    def score(self, question: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0

        # Judge relevance of each chunk
        relevance = []
        for ctx in contexts:
            verdict = self._llm.generate_raw(
                self._RELEVANCE_PROMPT.format(question=question, context=ctx)
            )
            relevance.append(1.0 if _parse_verdict(verdict) else 0.0)

        # Average Precision: AP = (1/R) * sum_k( P@k * rel_k )
        # where R = total relevant, P@k = precision at position k
        total_relevant = sum(relevance)
        if total_relevant == 0:
            return 0.0

        cumulative_relevant = 0.0
        ap_sum = 0.0
        for k, rel in enumerate(relevance):
            cumulative_relevant += rel
            if rel > 0:
                precision_at_k = cumulative_relevant / (k + 1)
                ap_sum += precision_at_k

        score = ap_sum / total_relevant
        logger.info(
            "ContextPrecision: %d/%d relevant (AP=%.3f)",
            int(total_relevant), len(contexts), score,
        )
        return _clamp(score)


# ══════════════════════════════════════════════════════════════════════════
#  4. Context Recall
# ══════════════════════════════════════════════════════════════════════════

class ContextRecallScorer:
    """
    Measures whether the retrieved context covers the ground truth.

    Algorithm:
      1. Extract claims/sentences from the ground truth answer.
      2. For each ground truth claim, check if it can be attributed
         to the retrieved contexts.
      3. Score = (# attributable claims) / (# total ground truth claims).
    """

    _EXTRACT_PROMPT = (
        "Given the following reference answer, extract a list of independent "
        "factual claims. Return ONLY a JSON array of strings, nothing else.\n\n"
        "Reference answer:\n{ground_truth}\n\n"
        "JSON array of claims:"
    )

    _ATTRIBUTE_PROMPT = (
        "Given the following context and a reference claim, determine if the "
        "claim can be attributed to or supported by the context. Answer with "
        "a single word: YES or NO.\n\n"
        "Context:\n{context}\n\n"
        "Claim: {claim}\n\n"
        "Verdict:"
    )

    def __init__(self, llm: LLMEngine) -> None:
        self._llm = llm

    def score(self, ground_truth: str, contexts: List[str]) -> float:
        if not ground_truth.strip() or not contexts:
            return 0.0

        # Step 1: Extract ground truth claims
        claims = _parse_json_list(
            self._llm.generate_raw(self._EXTRACT_PROMPT.format(ground_truth=ground_truth))
        )
        if not claims:
            logger.warning("ContextRecall: no claims extracted from ground truth")
            return 0.0

        # Step 2: Check attribution
        combined_context = "\n\n".join(contexts)
        attributed = 0
        for claim in claims:
            verdict = self._llm.generate_raw(
                self._ATTRIBUTE_PROMPT.format(context=combined_context, claim=claim)
            )
            if _parse_verdict(verdict):
                attributed += 1

        score = attributed / len(claims)
        logger.info(
            "ContextRecall: %d/%d ground truth claims attributed (%.3f)",
            attributed, len(claims), score,
        )
        return _clamp(score)


# ══════════════════════════════════════════════════════════════════════════
#  RAGAS Evaluator — orchestrates everything
# ══════════════════════════════════════════════════════════════════════════

class RAGASEvaluator:
    """
    End-to-end evaluator: runs the RAG pipeline on a set of questions,
    computes all four RAGAS metrics, and produces an EvalReport.
    """

    def __init__(
        self,
        llm: LLMEngine,
        embedder: OllamaEmbedder,
        num_relevancy_questions: int = 3,
    ) -> None:
        self._faithfulness = FaithfulnessScorer(llm)
        self._answer_relevancy = AnswerRelevancyScorer(llm, embedder, n=num_relevancy_questions)
        self._context_precision = ContextPrecisionScorer(llm)
        self._context_recall = ContextRecallScorer(llm)

    def evaluate_sample(self, sample: EvalSample) -> EvalSample:
        """Compute all metrics for a single pre-populated sample."""
        logger.info("Evaluating: %s", sample.question[:80])

        sample.scores["faithfulness"] = self._faithfulness.score(
            sample.answer, sample.contexts,
        )
        sample.scores["answer_relevancy"] = self._answer_relevancy.score(
            sample.question, sample.answer,
        )
        sample.scores["context_precision"] = self._context_precision.score(
            sample.question, sample.contexts,
        )
        sample.scores["context_recall"] = self._context_recall.score(
            sample.ground_truth, sample.contexts,
        )

        return sample

    def evaluate_dataset(
        self,
        samples: List[EvalSample],
        config_snapshot: Optional[Dict[str, str]] = None,
    ) -> EvalReport:
        """Evaluate an entire dataset and produce an aggregate report."""
        for sample in samples:
            self.evaluate_sample(sample)

        # Aggregate: mean of each metric across samples
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        aggregate: Dict[str, float] = {}
        for name in metric_names:
            values = [s.scores.get(name, 0.0) for s in samples]
            aggregate[name] = round(float(np.mean(values)), 4) if values else 0.0

        # Harmonic mean of all four as an overall score
        metric_values = [aggregate[m] for m in metric_names if aggregate[m] > 0]
        if metric_values:
            aggregate["ragas_score"] = round(
                float(len(metric_values) / sum(1.0 / v for v in metric_values)), 4
            )
        else:
            aggregate["ragas_score"] = 0.0

        return EvalReport(
            samples=samples,
            aggregate_scores=aggregate,
            config_snapshot=config_snapshot or {},
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
