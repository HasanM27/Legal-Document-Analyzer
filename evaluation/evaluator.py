"""
Step 7 — Evaluation & Safety Layer
=====================================
Two responsibilities:

1. EVALUATION — measures how good the RAG pipeline is:
   - Retrieval relevance: are the retrieved chunks actually relevant?
   - Answer groundedness: is the LLM answer based on retrieved chunks?
   - Answer completeness: did the LLM address the key facts from the doc?

2. SAFETY — protects users from bad outputs:
   - Confidence gating: if confidence is low, escalate to lawyer recommendation
   - Hallucination detection: flag answers that contradict retrieved law
   - Disclaimer injection: always present, always prominent
   - Escalation triggers: detect high-stakes situations needing real legal help
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from ingestion.retriever import RAGContext
from generation.generator import LegalAnalysis, ActionStep

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    retrieval_relevance: float
    answer_groundedness: float
    completeness:        float
    overall:             float
    flags:               list = field(default_factory=list)
    passed:              bool = True


@dataclass
class SafetyResult:
    is_safe:            bool
    escalate_to_lawyer: bool
    escalation_reason:  Optional[str]
    modified_analysis:  object


@dataclass
class SafeAnalysis:
    analysis:   object
    evaluation: EvaluationResult
    safety:     SafetyResult
    escalate:   bool


_STOP_WORDS = {
    "that","this","with","from","they","have","been","will","your","their",
    "which","when","were","also","each","more","shall","such","than","into",
    "upon","under","made","after","where","before","between","within",
    "without","against","there","these","those","being","would","could",
    "should","section","article","order","court","person","party",
}


class RAGEvaluator:

    MIN_RETRIEVAL_SCORE = 0.35
    MIN_COMPLETENESS    = 0.4

    def evaluate(self, context, analysis):
        retrieval    = self._score_retrieval(context)
        groundedness = self._score_groundedness(context, analysis)
        completeness = self._score_completeness(context, analysis)
        overall = round(retrieval*0.4 + groundedness*0.4 + completeness*0.2, 3)

        flags = []
        if retrieval    < self.MIN_RETRIEVAL_SCORE: flags.append("low_retrieval_relevance")
        if groundedness < 0.3:                      flags.append("low_groundedness")
        if completeness < self.MIN_COMPLETENESS:    flags.append("low_completeness")
        if not context.retrieved_chunks:            flags.append("no_chunks_retrieved")

        result = EvaluationResult(
            retrieval_relevance=retrieval,
            answer_groundedness=groundedness,
            completeness=completeness,
            overall=overall,
            flags=flags,
            passed=len(flags)==0,
        )
        log.info(f"Evaluation: retrieval={retrieval:.2f} groundedness={groundedness:.2f} "
                 f"completeness={completeness:.2f} overall={overall:.2f} flags={flags}")
        return result

    def _score_retrieval(self, context):
        if not context.retrieved_chunks:
            return 0.0
        scores = [c.score for c in context.retrieved_chunks]
        return round(sum(scores)/len(scores), 3)

    def _score_groundedness(self, context, analysis):
        if not context.retrieved_chunks:
            return 0.0
        reference = " ".join(c.text.lower() for c in context.retrieved_chunks)
        answer    = analysis.summary.lower() + " " + " ".join(analysis.rights).lower()
        ref_words = set(w for w in re.findall(r"\b[a-z]{4,}\b", reference) if w not in _STOP_WORDS)
        if not ref_words:
            return 0.5
        answer_words = set(re.findall(r"\b[a-z]{4,}\b", answer))
        overlap = ref_words & answer_words
        return min(round(len(overlap)/max(len(ref_words),1)*3, 3), 1.0)

    def _score_completeness(self, context, analysis):
        facts  = context.facts
        answer = (analysis.summary.lower() + " "
                  + " ".join(r.lower() for r in analysis.rights) + " "
                  + " ".join(s.instruction.lower() for s in analysis.action_steps))
        checks = []
        for amount in facts.monetary_amounts[:2]:
            numeric = re.sub(r"[^\d,]", "", amount)
            checks.append(numeric in answer.replace(",",""))
        if facts.deadlines:
            checks.append(any(d.split()[0] in answer for d in facts.deadlines[:2]))
        doc_keywords = {
            "eviction_notice":     ["evict","vacate","quit","rent"],
            "employment_contract": ["terminat","employ","dismiss","notice"],
            "debt_collection":     ["debt","pay","amount","due"],
            "court_summons":       ["court","summons","appear","statement"],
            "tenancy_agreement":   ["tenant","landlord","lease","rent"],
        }
        keywords = doc_keywords.get(context.doc_type, [])
        if keywords:
            checks.append(any(kw in answer for kw in keywords))
        return round(sum(checks)/len(checks), 3) if checks else 0.7


class SafetyChecker:

    HIGH_STAKES_PATTERNS = [
        r"criminal",r"arrest",r"imprisonment",r"fraud",r"forgery",
        r"assault",r"custody",r"divorce",r"inheritance",r"murder",r"drug",
    ]

    ESCALATION_STEP = ActionStep(
        step=99,
        instruction=(
            "This situation requires professional legal advice. "
            "Contact a lawyer or visit your nearest District Bar Association "
            "for free legal aid before taking any action."
        ),
        deadline="as soon as possible",
    )

    STRONG_DISCLAIMER = (
        "IMPORTANT: This analysis is AI-generated and is NOT a substitute "
        "for professional legal advice. The law may have changed and your "
        "specific circumstances matter greatly. Please consult a qualified "
        "lawyer before making any decisions based on this analysis."
    )

    def check(self, analysis, context, eval_result):
        escalate = False
        escalation_reason = None
        modified = analysis

        if analysis.confidence == "low" or not eval_result.passed:
            escalate = True
            escalation_reason = f"Low confidence (flags: {eval_result.flags})"
            modified = self._add_escalation(modified)
            modified.disclaimer = self.STRONG_DISCLAIMER

        if not context.retrieved_chunks:
            escalate = True
            escalation_reason = "No relevant law found in knowledge base"
            modified = self._add_escalation(modified)
            modified.disclaimer = self.STRONG_DISCLAIMER

        full_text = context.document_text.lower()
        for pattern in self.HIGH_STAKES_PATTERNS:
            if re.search(pattern, full_text):
                escalate = True
                escalation_reason = f"High-stakes content: '{pattern}'"
                modified = self._add_escalation(modified)
                break

        if not modified.disclaimer:
            modified.disclaimer = self.STRONG_DISCLAIMER

        if escalate:
            steps = [s for s in modified.action_steps if s.step != 99]
            steps.append(self.ESCALATION_STEP)
            modified.action_steps = steps

        log.info(f"Safety: escalate={escalate} reason={escalation_reason}")
        return SafetyResult(
            is_safe=True,
            escalate_to_lawyer=escalate,
            escalation_reason=escalation_reason,
            modified_analysis=modified,
        )

    def _add_escalation(self, analysis):
        if not any(s.step == 99 for s in analysis.action_steps):
            analysis.action_steps = analysis.action_steps + [self.ESCALATION_STEP]
        return analysis


class EvaluationAndSafetyPipeline:

    def __init__(self, evaluator=None, checker=None):
        self.evaluator = evaluator or RAGEvaluator()
        self.checker   = checker   or SafetyChecker()

    def run(self, analysis, context):
        eval_result   = self.evaluator.evaluate(context, analysis)
        safety_result = self.checker.check(analysis, context, eval_result)
        return SafeAnalysis(
            analysis=safety_result.modified_analysis,
            evaluation=eval_result,
            safety=safety_result,
            escalate=safety_result.escalate_to_lawyer,
        )