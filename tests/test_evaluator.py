"""
Tests for Step 7 — Evaluation & Safety Layer
Run with: pytest tests/test_evaluator.py -v
"""

import pytest
from unittest.mock import MagicMock
from ingestion.retriever import RAGContext, ExtractedFacts, RetrievedChunk
from generation.generator import LegalAnalysis, ActionStep
from evaluation.evaluator import (
    RAGEvaluator, SafetyChecker,
    EvaluationAndSafetyPipeline, EvaluationResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_context(doc_text="Eviction notice Karachi PKR 45,000 15 days.",
                 doc_type="eviction_notice", chunks=None):
    facts = ExtractedFacts(
        doc_type=doc_type, jurisdiction="sindh",
        monetary_amounts=["PKR 45,000"], deadlines=["15 days"],
        parties=["landlord","tenant"], key_phrases=["notice to quit"],
    )
    default_chunks = [
        RetrievedChunk(
            text="No tenant shall be evicted except in accordance with "
                 "the Sindh Rented Premises Ordinance. Landlord must give "
                 "fifteen days notice before filing for eviction.",
            score=0.82, source_id="sindh_rented_premises_1979",
            title="Sindh Rented Premises Ordinance 1979",
            jurisdiction="sindh", doc_category="tenancy",
        ),
        RetrievedChunk(
            text="A tenant who pays arrears of rent within the notice "
                 "period shall not be liable to eviction on that ground.",
            score=0.74, source_id="sindh_rented_premises_1979",
            title="Sindh Rented Premises Ordinance 1979",
            jurisdiction="sindh", doc_category="tenancy",
        ),
    ]
    return RAGContext(
        document_text=doc_text, doc_type=doc_type, jurisdiction="sindh",
        facts=facts, retrieved_chunks=chunks if chunks is not None else default_chunks,
        queries_used=["eviction notice tenant rights"],
    )


def make_analysis(confidence="high", urgency="critical", rights=None, steps=None):
    return LegalAnalysis(
        summary="You received an eviction notice demanding PKR 45,000 "
                "within 15 days or you must vacate the rented premises.",
        rights=rights or [
            "Right to pay rent within 15 days to stop eviction.",
            "Landlord cannot evict without Rent Controller order.",
        ],
        action_steps=steps or [
            ActionStep(step=1, instruction="Pay PKR 45,000 within 15 days.",
                      deadline="within 15 days"),
            ActionStep(step=2, instruction="Consult a lawyer.", deadline=None),
        ],
        urgency=urgency, confidence=confidence,
        disclaimer="This is not legal advice.",
        sources_cited=["Sindh Rented Premises Ordinance 1979"],
    )


# ── RAGEvaluator ──────────────────────────────────────────────────────────────

class TestRAGEvaluator:
    evaluator = RAGEvaluator()

    def test_returns_evaluation_result(self):
        result = self.evaluator.evaluate(make_context(), make_analysis())
        assert isinstance(result, EvaluationResult)

    def test_retrieval_score_is_avg_chunk_scores(self):
        ctx = make_context()
        result = self.evaluator.evaluate(ctx, make_analysis())
        expected = (0.82 + 0.74) / 2
        assert abs(result.retrieval_relevance - expected) < 0.01

    def test_no_chunks_gives_zero_retrieval(self):
        ctx = make_context(chunks=[])
        result = self.evaluator.evaluate(ctx, make_analysis())
        assert result.retrieval_relevance == 0.0

    def test_no_chunks_flags_correctly(self):
        ctx = make_context(chunks=[])
        result = self.evaluator.evaluate(ctx, make_analysis())
        assert "no_chunks_retrieved" in result.flags

    def test_good_analysis_passes(self):
        result = self.evaluator.evaluate(make_context(), make_analysis())
        assert result.passed is True
        assert result.flags == []

    def test_overall_is_weighted_average(self):
        result = self.evaluator.evaluate(make_context(), make_analysis())
        expected = (result.retrieval_relevance * 0.4
                   + result.answer_groundedness * 0.4
                   + result.completeness * 0.2)
        assert abs(result.overall - round(expected, 3)) < 0.001

    def test_completeness_checks_monetary_amount(self):
        ctx = make_context(doc_text="Pay PKR 45000 within 15 days.")
        analysis = make_analysis()
        result = self.evaluator.evaluate(ctx, analysis)
        assert result.completeness > 0

    def test_scores_are_between_0_and_1(self):
        result = self.evaluator.evaluate(make_context(), make_analysis())
        assert 0.0 <= result.retrieval_relevance <= 1.0
        assert 0.0 <= result.answer_groundedness <= 1.0
        assert 0.0 <= result.completeness       <= 1.0
        assert 0.0 <= result.overall            <= 1.0


# ── SafetyChecker ─────────────────────────────────────────────────────────────

class TestSafetyChecker:
    checker = SafetyChecker()

    def _eval(self, passed=True, flags=None):
        return EvaluationResult(
            retrieval_relevance=0.8 if passed else 0.2,
            answer_groundedness=0.7 if passed else 0.2,
            completeness=0.8 if passed else 0.2,
            overall=0.75 if passed else 0.2,
            flags=flags or ([] if passed else ["low_retrieval_relevance"]),
            passed=passed,
        )

    def test_good_analysis_not_escalated(self):
        result = self.checker.check(
            make_analysis(confidence="high"),
            make_context(),
            self._eval(passed=True),
        )
        assert result.escalate_to_lawyer is False

    def test_low_confidence_triggers_escalation(self):
        result = self.checker.check(
            make_analysis(confidence="low"),
            make_context(),
            self._eval(passed=True),
        )
        assert result.escalate_to_lawyer is True

    def test_failed_eval_triggers_escalation(self):
        result = self.checker.check(
            make_analysis(confidence="high"),
            make_context(),
            self._eval(passed=False),
        )
        assert result.escalate_to_lawyer is True

    def test_no_chunks_triggers_escalation(self):
        result = self.checker.check(
            make_analysis(),
            make_context(chunks=[]),
            self._eval(passed=False, flags=["no_chunks_retrieved"]),
        )
        assert result.escalate_to_lawyer is True

    def test_criminal_keyword_triggers_escalation(self):
        ctx = make_context(
            doc_text="You are accused of criminal fraud and may face arrest."
        )
        result = self.checker.check(
            make_analysis(), ctx, self._eval(passed=True)
        )
        assert result.escalate_to_lawyer is True

    def test_high_stakes_keywords(self):
        for keyword in ["criminal", "arrest", "fraud", "divorce", "imprisonment"]:
            ctx = make_context(doc_text=f"This involves {keyword} proceedings.")
            result = self.checker.check(
                make_analysis(), ctx, self._eval(passed=True)
            )
            assert result.escalate_to_lawyer is True, f"Should escalate for: {keyword}"

    def test_escalation_step_added_when_escalating(self):
        result = self.checker.check(
            make_analysis(confidence="low"),
            make_context(),
            self._eval(passed=True),
        )
        step_nums = [s.step for s in result.modified_analysis.action_steps]
        assert 99 in step_nums

    def test_escalation_step_not_duplicated(self):
        result = self.checker.check(
            make_analysis(confidence="low"),
            make_context(),
            self._eval(passed=False),
        )
        count = sum(1 for s in result.modified_analysis.action_steps if s.step == 99)
        assert count == 1

    def test_disclaimer_always_present(self):
        analysis = make_analysis()
        analysis.disclaimer = ""
        result = self.checker.check(analysis, make_context(), self._eval())
        assert len(result.modified_analysis.disclaimer) > 20

    def test_is_safe_always_true(self):
        result = self.checker.check(
            make_analysis(confidence="low"),
            make_context(chunks=[]),
            self._eval(passed=False),
        )
        assert result.is_safe is True


# ── EvaluationAndSafetyPipeline ───────────────────────────────────────────────

class TestEvaluationAndSafetyPipeline:
    pipeline = EvaluationAndSafetyPipeline()

    def test_returns_safe_analysis(self):
        from evaluation.evaluator import SafeAnalysis
        result = self.pipeline.run(make_analysis(), make_context())
        assert isinstance(result, SafeAnalysis)

    def test_safe_analysis_has_all_fields(self):
        result = self.pipeline.run(make_analysis(), make_context())
        assert result.analysis is not None
        assert result.evaluation is not None
        assert result.safety is not None
        assert isinstance(result.escalate, bool)

    def test_good_input_not_escalated(self):
        result = self.pipeline.run(
            make_analysis(confidence="high"),
            make_context(),
        )
        assert result.escalate is False

    def test_low_confidence_escalated(self):
        result = self.pipeline.run(
            make_analysis(confidence="low"),
            make_context(),
        )
        assert result.escalate is True

    def test_no_chunks_escalated(self):
        result = self.pipeline.run(
            make_analysis(),
            make_context(chunks=[]),
        )
        assert result.escalate is True

    def test_analysis_disclaimer_populated(self):
        result = self.pipeline.run(make_analysis(), make_context())
        assert len(result.analysis.disclaimer) > 20

    def test_evaluation_scores_in_range(self):
        result = self.pipeline.run(make_analysis(), make_context())
        ev = result.evaluation
        assert 0.0 <= ev.overall <= 1.0
        assert 0.0 <= ev.retrieval_relevance <= 1.0
