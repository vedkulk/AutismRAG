from __future__ import annotations

"""
LLM Engine wrapping ChatOllama (via LangChain).

Responsibilities:
  - maintain a short conversation history per session
  - apply a clinical system prompt
  - format FusedContext into the prompt
  - provide both non-streaming and streaming answer interfaces
"""

from typing import List, Dict, Any, Iterable, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOllama

from dual_retriever import FusedContext


DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for licensed healthcare providers "
    "evaluating children for Autism Spectrum Disorder (ASD).\n\n"
    "You receive context from two sources:\n"
    "- PATIENT REPORT: observations from a child's developmental report\n"
    "- MEDICAL KB: clinical guidelines and medical literature\n\n"
    "INSTRUCTIONS:\n"
    "1. Answer the clinician's SPECIFIC question directly and concisely. "
    "Do not add unrequested background or tangential information.\n"
    "2. Use ONLY the provided context. Do not use prior knowledge.\n"
    "3. When patient report context is provided, always cite specific observations from it.\n"
    "4. Connect patient observations to relevant clinical criteria (DSM-5, ICD-10) "
    "when medical KB context supports it.\n"
    "5. Use cautious clinical language "
    '("may be consistent with", "warrants further evaluation").\n'
    "6. Do NOT refuse to answer — this tool exists to surface evidence for clinicians.\n"
    "7. Keep answers focused and under 250 words.\n\n"
    "Structure (use only when both source types are present):\n"
    "[From Patient Report]: specific observations relevant to the question\n"
    "[From Medical KB]: relevant clinical criteria or guidelines\n"
    "[Clinical Correlation]: how observations relate to criteria\n"
)

INSUFFICIENT_CONTEXT_MSG = (
    "The provided documents do not contain sufficient information to answer this question. "
    "Please consult authoritative clinical guidelines or a specialist for this query."
)

# Guard thresholds — only skip LLM when context is truly empty/useless.
# Cross-encoder scores are raw logits (roughly -10 to +10), not [0,1] cosine.
# A cross-encoder score > 1.0 indicates meaningful relevance.
CROSS_ENCODER_MIN_SCORE = 1.0

import logging as _logging
_guard_logger = _logging.getLogger(__name__)


class LLMEngine:
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self._chat = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

        # Lightweight classifier chat head (low temperature, no history)
        self._classifier = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=0.0,
        )

    def _context_is_sufficient(self, ctx: FusedContext) -> bool:
        """Check if retrieved chunks are worth sending to the LLM.

        Only returns False when there are literally zero chunks.
        If chunks exist, always proceed — the retrieval pipeline
        (BM25 + vector + cross-encoder + compression) already filtered.
        """
        if not ctx.chunks:
            _guard_logger.info("[GUARD] No chunks — returning insufficient context")
            return False

        _guard_logger.info(
            "[GUARD] %d chunks available, proceeding (max_score=%.3f)",
            len(ctx.chunks),
            max(c.similarity for c in ctx.chunks),
        )
        return True

    def _build_messages(self, question: str, ctx: FusedContext) -> List[Any]:
        # Each turn is independent — no conversation history is retained,
        # so prior questions/answers about other patients can never bleed
        # into a new query.
        context_block = ctx.as_prompt_block()
        user_prompt = (
            f"A clinician has asked the following question while reviewing a patient case:\n"
            f"{question}\n\n"
            f"Below are excerpts from the patient's developmental report and "
            f"relevant medical reference documents:\n\n"
            f"{context_block}\n\n"
            f"Summarize the relevant information from these documents to address "
            f"the clinician's question."
        )
        return [
            SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

    def ask(self, question: str, ctx: FusedContext) -> str:
        if not self._context_is_sufficient(ctx):
            return INSUFFICIENT_CONTEXT_MSG
        resp = self._chat.invoke(self._build_messages(question, ctx))
        return resp.content

    def ask_stream(self, question: str, ctx: FusedContext) -> Iterable[str]:
        if not self._context_is_sufficient(ctx):
            yield INSUFFICIENT_CONTEXT_MSG
            return
        for chunk in self._chat.stream(self._build_messages(question, ctx)):
            if chunk.content:
                yield chunk.content

    def generate_raw(self, prompt: str) -> str:
        """Low-level LLM call with no history or context formatting.
        Uses the classifier head (temperature=0.0) for deterministic output."""
        resp = self._classifier.invoke([HumanMessage(content=prompt)])
        return resp.content or ""

    def is_asd_question(self, question: str) -> bool:
        """
        LLM-based intent classifier.

        Returns True if the question is about:
          - autism spectrum disorder
          - child/adolescent development or behaviour
          - interpreting symptoms, history, or questionnaire content
        Otherwise returns False.
        """
        prompt = (
            "You are a classifier.\n\n"
            "Decide if the user's question is about:\n"
            "- autism spectrum disorder (ASD), OR\n"
            "- developmental / behavioural concerns in a child or adolescent, OR\n"
            "- interpreting symptoms, history, or questionnaire content for ASD.\n\n"
            f"Question: \"{question}\"\n\n"
            "Answer with a single word only: YES or NO."
        )
        resp = self._classifier.invoke([HumanMessage(content=prompt)])
        ans = (resp.content or "").strip().upper()
        return ans.startswith("Y")

