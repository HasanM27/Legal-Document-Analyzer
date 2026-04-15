"""
Tests for Step 5 — LLM generation layer
All unit tests use mocks — no Groq API key needed.
Live test requires GROQ_API_KEY set and KB built.

Run unit tests:  pytest tests/test_generator.py -v
Run live test:   pytest tests/test_generator.py -v -k live
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

from ingestion.parser import DocumentType
from ingestion.retriever import RAGContext, ExtractedFacts, RetrievedChunk
from generation.generator import (
    LegalAnalysis, ActionStep,
    PromptBuilder, OutputParser, LegalAnalysisGenerator,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_context(
    doc_type="eviction_notice",
    jurisdiction="sindh",
    doc_text="NOTICE TO QUIT: Pay PKR 45,000 or vacate within 14 days.",
    chunks=None,
):
    facts = ExtractedFacts(
        doc_type=doc_type,
        jurisdiction=jurisdiction,
        monetary_amounts=["PKR 45,000"],
        deadlines=["14 days"],
        parties=["landlord", "tenant"],
        key_phrases=["notice to quit"],
    )
    default_chunks = [
        RetrievedChunk(
            text="No tenant shall be evicted except in accordance with "
                 "the provisions of this Ordinance. The landlord shall "
                 "give fifteen days written notice before filing for eviction.",
            score=0.85,
            source_id="sindh_rented_premises_1979",
            title="Sindh Rented Premises Ordinance 1979",
            jurisdiction="sindh",
            doc_category="tenancy",
        ),
        RetrievedChunk(
            text="A tenant who pays the arrears of rent within the notice "
                 "period shall not be liable to eviction on that ground.",
            score=0.78,
            source_id="sindh_rented_premises_1979",
            title="Sindh Rented Premises Ordinance 1979",
            jurisdiction="sindh",
            doc_category="tenancy",
        ),
    ]
    return RAGContext(
        document_text=doc_text,
        doc_type=doc_type,
        jurisdiction=jurisdiction,
        facts=facts,
        retrieved_chunks=chunks if chunks is not None else default_chunks,
        queries_used=["eviction notice tenant rights"],
    )


VALID_JSON_RESPONSE = json.dumps({
    "summary": "Your landlord is demanding PKR 45,000 in unpaid rent and "
               "giving you 14 days to pay or leave. This is a formal eviction "
               "notice under Sindh tenancy law.",
    "rights": [
        "You have the right to pay the outstanding rent within the notice "
        "period to stop the eviction.",
        "Your landlord cannot evict you without an order from the Rent "
        "Controller — they cannot physically remove you themselves.",
        "You have the right to appeal any Rent Controller order to the "
        "District Judge within 30 days.",
    ],
    "action_steps": [
        {
            "step": 1,
            "instruction": "Check the exact date on the notice — your "
                           "14 days starts from that date.",
            "deadline": "immediately",
        },
        {
            "step": 2,
            "instruction": "If you can pay, do so by bank transfer or "
                           "pay order and keep the receipt as proof.",
            "deadline": "within 14 days",
        },
        {
            "step": 3,
            "instruction": "If you cannot pay, contact the Karachi Rent "
                           "Controller office or a legal aid centre today.",
            "deadline": "within 7 days",
        },
        {
            "step": 4,
            "instruction": "Consult a lawyer who specialises in tenancy "
                           "disputes for advice on your specific situation.",
            "deadline": None,
        },
    ],
    "urgency": "critical",
    "confidence": "high",
    "sources_cited": ["Sindh Rented Premises Ordinance 1979"],
})


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class TestPromptBuilder:
    builder = PromptBuilder()

    def test_returns_two_strings(self):
        ctx = make_context()
        system, user = self.builder.build(ctx)
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_contains_json_instruction(self):
        ctx = make_context()
        system, _ = self.builder.build(ctx)
        assert "JSON" in system

    def test_system_prompt_contains_urgency_levels(self):
        ctx = make_context()
        system, _ = self.builder.build(ctx)
        assert "critical" in system
        assert "high" in system
        assert "medium" in system
        assert "low" in system

    def test_user_prompt_contains_document_text(self):
        ctx = make_context(doc_text="Unique document text for testing.")
        _, user = self.builder.build(ctx)
        assert "Unique document text for testing." in user

    def test_user_prompt_contains_doc_type(self):
        ctx = make_context(doc_type="eviction_notice")
        _, user = self.builder.build(ctx)
        assert "Eviction" in user

    def test_user_prompt_contains_jurisdiction(self):
        ctx = make_context(jurisdiction="sindh")
        _, user = self.builder.build(ctx)
        assert "Sindh" in user

    def test_user_prompt_contains_retrieved_chunks(self):
        ctx = make_context()
        _, user = self.builder.build(ctx)
        assert "Sindh Rented Premises Ordinance" in user

    def test_user_prompt_contains_monetary_amounts(self):
        ctx = make_context()
        _, user = self.builder.build(ctx)
        assert "PKR 45,000" in user

    def test_user_prompt_contains_deadlines(self):
        ctx = make_context()
        _, user = self.builder.build(ctx)
        assert "14 days" in user

    def test_long_document_truncated(self):
        ctx = make_context(doc_text="X" * 5000)
        _, user = self.builder.build(ctx)
        assert "truncated" in user

    def test_no_chunks_handled_gracefully(self):
        ctx = make_context(chunks=[])
        _, user = self.builder.build(ctx)
        assert "No specific legal provisions" in user

    def test_chunk_scores_shown(self):
        ctx = make_context()
        _, user = self.builder.build(ctx)
        assert "relevance:" in user


# ── OutputParser ──────────────────────────────────────────────────────────────

class TestOutputParser:
    parser = OutputParser()

    def test_parses_valid_json(self):
        ctx = make_context()
        result = self.parser.parse(VALID_JSON_RESPONSE, ctx)
        assert isinstance(result, LegalAnalysis)

    def test_summary_extracted(self):
        ctx = make_context()
        result = self.parser.parse(VALID_JSON_RESPONSE, ctx)
        assert "PKR 45,000" in result.summary

    def test_rights_extracted(self):
        ctx = make_context()
        result = self.parser.parse(VALID_JSON_RESPONSE, ctx)
        assert len(result.rights) == 3
        assert all(isinstance(r, str) for r in result.rights)

    def test_action_steps_extracted(self):
        ctx = make_context()
        result = self.parser.parse(VALID_JSON_RESPONSE, ctx)
        assert len(result.action_steps) == 4
        assert all(isinstance(s, ActionStep) for s in result.action_steps)

    def test_step_deadlines_parsed(self):
        ctx = make_context()
        result = self.parser.parse(VALID_JSON_RESPONSE, ctx)
        assert result.action_steps[1].deadline == "within 14 days"
        assert result.action_steps[3].deadline is None

    def test_urgency_extracted(self):
        ctx = make_context()
        result = self.parser.parse(VALID_JSON_RESPONSE, ctx)
        assert result.urgency == "critical"

    def test_confidence_extracted(self):
        ctx = make_context()
        result = self.parser.parse(VALID_JSON_RESPONSE, ctx)
        assert result.confidence == "high"

    def test_sources_cited_extracted(self):
        ctx = make_context()
        result = self.parser.parse(VALID_JSON_RESPONSE, ctx)
        assert "Sindh Rented Premises Ordinance 1979" in result.sources_cited

    def test_disclaimer_always_present(self):
        ctx = make_context()
        result = self.parser.parse(VALID_JSON_RESPONSE, ctx)
        assert len(result.disclaimer) > 50
        assert "legal advice" in result.disclaimer.lower()

    def test_strips_markdown_fences(self):
        ctx = make_context()
        fenced = f"```json\n{VALID_JSON_RESPONSE}\n```"
        result = self.parser.parse(fenced, ctx)
        assert result.urgency == "critical"

    def test_extracts_json_from_surrounding_text(self):
        ctx = make_context()
        wrapped = f"Here is my analysis:\n{VALID_JSON_RESPONSE}\nHope this helps."
        result = self.parser.parse(wrapped, ctx)
        assert result.urgency == "critical"

    def test_invalid_json_returns_fallback(self):
        ctx = make_context()
        result = self.parser.parse("This is not JSON at all.", ctx)
        assert isinstance(result, LegalAnalysis)
        assert result.confidence == "low"
        assert len(result.action_steps) >= 1

    def test_invalid_urgency_defaults_to_medium(self):
        ctx = make_context()
        data = json.loads(VALID_JSON_RESPONSE)
        data["urgency"] = "EXTREMELY URGENT"
        result = self.parser.parse(json.dumps(data), ctx)
        assert result.urgency == "medium"

    def test_invalid_confidence_defaults_to_low(self):
        ctx = make_context()
        data = json.loads(VALID_JSON_RESPONSE)
        data["confidence"] = "very_sure"
        result = self.parser.parse(json.dumps(data), ctx)
        assert result.confidence == "low"

    def test_missing_rights_returns_empty_list(self):
        ctx = make_context()
        data = json.loads(VALID_JSON_RESPONSE)
        del data["rights"]
        result = self.parser.parse(json.dumps(data), ctx)
        assert result.rights == []


# ── LegalAnalysisGenerator (mocked) ──────────────────────────────────────────

class TestLegalAnalysisGenerator:

    def _make_generator(self, llm_response=None):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = llm_response or VALID_JSON_RESPONSE
        return LegalAnalysisGenerator(llm=mock_llm), mock_llm

    def test_returns_legal_analysis(self):
        gen, _ = self._make_generator()
        result = gen.generate(make_context())
        assert isinstance(result, LegalAnalysis)

    def test_llm_called_once(self):
        gen, mock_llm = self._make_generator()
        gen.generate(make_context())
        assert mock_llm.complete.call_count == 1

    def test_llm_receives_system_and_user_prompt(self):
        gen, mock_llm = self._make_generator()
        gen.generate(make_context())
        call_args = mock_llm.complete.call_args
        system = call_args[0][0]
        user   = call_args[0][1]
        assert "JSON" in system
        assert "DOCUMENT CONTENT" in user

    def test_analysis_has_all_required_fields(self):
        gen, _ = self._make_generator()
        result = gen.generate(make_context())
        assert result.summary
        assert isinstance(result.rights, list)
        assert isinstance(result.action_steps, list)
        assert result.urgency in ("critical", "high", "medium", "low")
        assert result.confidence in ("high", "medium", "low")
        assert result.disclaimer

    def test_graceful_on_bad_llm_response(self):
        gen, _ = self._make_generator(llm_response="I cannot help with that.")
        result = gen.generate(make_context())
        assert isinstance(result, LegalAnalysis)
        assert result.confidence == "low"

    def test_employment_contract_context(self):
        gen, _ = self._make_generator()
        ctx = make_context(
            doc_type="employment_contract",
            jurisdiction="federal",
            doc_text="Your employment is terminated effective immediately.",
        )
        result = gen.generate(ctx)
        assert isinstance(result, LegalAnalysis)

    def test_debt_collection_context(self):
        gen, _ = self._make_generator()
        ctx = make_context(
            doc_type="debt_collection",
            jurisdiction="federal",
            doc_text="You owe Rs. 250,000. Pay within 30 days.",
        )
        result = gen.generate(ctx)
        assert isinstance(result, LegalAnalysis)


# ── Live integration test ─────────────────────────────────────────────────────

class TestGeneratorLive:
    """
    Requires: GROQ_API_KEY env var set + KB built.
    Run with: pytest tests/test_generator.py -v -k live
    """

    @pytest.mark.live
    def test_live_eviction_analysis(self):
        import os
        if not os.environ.get("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")

        gen = LegalAnalysisGenerator()
        ctx = make_context(
            doc_text="""
            NOTICE TO PAY RENT OR QUIT

            To: Muhammad Ali
            Address: Flat 5, Block C, Gulshan-e-Iqbal, Karachi, Sindh

            You are hereby notified that you are in arrears of rent amounting
            to PKR 55,000 for the months of January and February 2025.

            You are required to pay the said amount within 14 days of this
            notice, failing which legal proceedings will be initiated against
            you under the Sindh Rented Premises Ordinance 1979.

            Issued by: Hassan Properties, Karachi
            Date: 1st March 2025
            """,
        )

        result = gen.generate(ctx)

        assert isinstance(result, LegalAnalysis)
        assert len(result.summary) > 50
        assert len(result.rights) >= 1
        assert len(result.action_steps) >= 2
        assert result.urgency in ("critical", "high", "medium", "low")
        assert result.disclaimer

        print(f"\n  LIVE ANALYSIS RESULT")
        print(f"  Urgency    : {result.urgency}")
        print(f"  Confidence : {result.confidence}")
        print(f"  Summary    : {result.summary[:200]}")
        print(f"  Rights     : {len(result.rights)} items")
        print(f"  Steps      : {len(result.action_steps)} items")
        print(f"  Sources    : {result.sources_cited}")
        print(f"\n  Full rights:")
        for r in result.rights:
            print(f"    - {r}")
        print(f"\n  Action steps:")
        for s in result.action_steps:
            deadline = f" [{s.deadline}]" if s.deadline else ""
            print(f"    {s.step}. {s.instruction}{deadline}")
