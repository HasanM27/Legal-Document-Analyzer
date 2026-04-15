"""
Tests for Step 4 — RAG retrieval pipeline
All unit tests use mocks — no ChromaDB or embedding model needed.
The live integration test at the bottom requires the KB to be built.

Run unit tests:  pytest tests/test_retriever.py -v
Run live test:   pytest tests/test_retriever.py -v -k live
"""

import pytest
from unittest.mock import MagicMock, patch
from ingestion.parser import ParsedDocument, DocumentType
from ingestion.retriever import (
    FactExtractor,
    QueryBuilder,
    ContextAssembler,
    RAGPipeline,
    ExtractedFacts,
    RetrievedChunk,
    RAGContext,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_parsed_doc(text: str, doc_type: DocumentType = DocumentType.UNKNOWN):
    return ParsedDocument(
        raw_text=text,
        clean_text=text,
        doc_type=doc_type,
        page_count=1,
        is_scanned=False,
        confidence=1.0,
        metadata={"source": "test"},
    )


def make_chunk(text: str, score: float = 0.8,
               source_id: str = "test_act",
               title: str = "Test Act",
               jurisdiction: str = "federal",
               doc_category: str = "tenancy") -> dict:
    return {
        "text": text, "score": score,
        "source_id": source_id, "title": title,
        "jurisdiction": jurisdiction, "doc_category": doc_category,
    }


# ── FactExtractor ─────────────────────────────────────────────────────────────

class TestFactExtractor:
    extractor = FactExtractor()

    def test_detects_sindh_jurisdiction(self):
        doc = make_parsed_doc("This is a notice issued in Karachi, Sindh.")
        facts = self.extractor.extract(doc)
        assert facts.jurisdiction == "sindh"

    def test_detects_punjab_jurisdiction(self):
        doc = make_parsed_doc("Property located in Lahore, Punjab.")
        facts = self.extractor.extract(doc)
        assert facts.jurisdiction == "punjab"

    def test_defaults_to_federal_when_no_signal(self):
        doc = make_parsed_doc("You are required to pay the outstanding amount.")
        facts = self.extractor.extract(doc)
        assert facts.jurisdiction == "federal"

    def test_extracts_pkr_amounts(self):
        doc = make_parsed_doc("You owe PKR 45,000 in unpaid rent.")
        facts = self.extractor.extract(doc)
        assert any("45,000" in a for a in facts.monetary_amounts)

    def test_extracts_rs_amounts(self):
        doc = make_parsed_doc("Outstanding balance of Rs. 12,500 is due.")
        facts = self.extractor.extract(doc)
        assert any("12,500" in a for a in facts.monetary_amounts)

    def test_extracts_deadlines(self):
        doc = make_parsed_doc("You have 14 days to respond to this notice.")
        facts = self.extractor.extract(doc)
        assert any("14" in d for d in facts.deadlines)

    def test_extracts_multiple_deadlines(self):
        doc = make_parsed_doc("Pay within 15 days or vacate within 30 days.")
        facts = self.extractor.extract(doc)
        assert len(facts.deadlines) >= 2

    def test_extracts_parties_landlord_tenant(self):
        doc = make_parsed_doc(
            "The landlord hereby notifies the tenant to vacate."
        )
        facts = self.extractor.extract(doc)
        assert "landlord" in facts.parties
        assert "tenant" in facts.parties

    def test_extracts_parties_employer_employee(self):
        doc = make_parsed_doc(
            "The employer hereby terminates the employee's contract."
        )
        facts = self.extractor.extract(doc)
        assert "employer" in facts.parties
        assert "employee" in facts.parties

    def test_extracts_key_phrases_eviction(self):
        doc = make_parsed_doc(
            "This is a notice to quit and vacate the rented premises."
        )
        facts = self.extractor.extract(doc)
        assert any("notice to quit" in p or "vacate" in p
                   for p in facts.key_phrases)

    def test_doc_type_preserved(self):
        doc = make_parsed_doc("test", DocumentType.EVICTION_NOTICE)
        facts = self.extractor.extract(doc)
        assert facts.doc_type == "eviction_notice"

    def test_empty_document(self):
        doc = make_parsed_doc("")
        facts = self.extractor.extract(doc)
        assert facts.doc_type == "unknown"
        assert facts.monetary_amounts == []
        assert facts.deadlines == []


# ── QueryBuilder ──────────────────────────────────────────────────────────────

class TestQueryBuilder:
    builder = QueryBuilder()

    def _facts(self, doc_type="eviction_notice", jurisdiction="sindh",
               deadlines=None, key_phrases=None,
               monetary_amounts=None, parties=None):
        return ExtractedFacts(
            doc_type=doc_type,
            jurisdiction=jurisdiction,
            monetary_amounts=monetary_amounts or [],
            deadlines=deadlines or [],
            parties=parties or [],
            key_phrases=key_phrases or [],
        )

    def test_returns_list_of_strings(self):
        queries = self.builder.build(self._facts())
        assert isinstance(queries, list)
        assert all(isinstance(q, str) for q in queries)

    def test_eviction_queries_contain_relevant_terms(self):
        queries = self.builder.build(self._facts(doc_type="eviction_notice"))
        combined = " ".join(queries).lower()
        assert "eviction" in combined or "tenant" in combined

    def test_employment_queries_contain_relevant_terms(self):
        queries = self.builder.build(
            self._facts(doc_type="employment_contract")
        )
        combined = " ".join(queries).lower()
        assert "employment" in combined or "termination" in combined

    def test_deadline_enriches_queries(self):
        queries = self.builder.build(
            self._facts(deadlines=["14 days", "30 days"])
        )
        combined = " ".join(queries)
        assert "14 days" in combined or "30 days" in combined

    def test_key_phrases_enrich_queries(self):
        queries = self.builder.build(
            self._facts(key_phrases=["eviction notice"])
        )
        combined = " ".join(queries)
        assert "eviction notice" in combined

    def test_unknown_doc_type_has_fallback_queries(self):
        queries = self.builder.build(self._facts(doc_type="unknown"))
        assert len(queries) >= 1

    def test_all_doc_types_produce_queries(self):
        for doc_type in ["eviction_notice", "employment_contract",
                         "debt_collection", "court_summons",
                         "tenancy_agreement", "government_letter", "unknown"]:
            queries = self.builder.build(self._facts(doc_type=doc_type))
            assert len(queries) >= 1, f"No queries for doc_type={doc_type}"


# ── ContextAssembler ──────────────────────────────────────────────────────────

class TestContextAssembler:
    assembler = ContextAssembler(max_chunks=5, min_score=0.3)

    def test_deduplicates_same_text(self):
        results = [
            make_chunk("Tenant shall receive 15 days notice.", score=0.8),
            make_chunk("Tenant shall receive 15 days notice.", score=0.7),
        ]
        chunks = self.assembler.assemble(results)
        assert len(chunks) == 1

    def test_filters_low_score(self):
        results = [
            make_chunk("High relevance text.", score=0.8),
            make_chunk("Low relevance text.", score=0.1),
        ]
        chunks = self.assembler.assemble(results)
        assert len(chunks) == 1
        assert chunks[0].score == 0.8

    def test_sorts_by_score_descending(self):
        results = [
            make_chunk("Third chunk.", score=0.5),
            make_chunk("First chunk.", score=0.9),
            make_chunk("Second chunk.", score=0.7),
        ]
        chunks = self.assembler.assemble(results)
        scores = [c.score for c in chunks]
        assert scores == sorted(scores, reverse=True)

    def test_caps_at_max_chunks(self):
        results = [
            make_chunk(f"Chunk number {i}.", score=0.9 - i * 0.05)
            for i in range(20)
        ]
        chunks = self.assembler.assemble(results)
        assert len(chunks) <= 5

    def test_returns_retrieved_chunk_objects(self):
        results = [make_chunk("Some legal text.", score=0.8)]
        chunks = self.assembler.assemble(results)
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    def test_empty_input_returns_empty(self):
        assert self.assembler.assemble([]) == []

    def test_all_below_min_score_returns_empty(self):
        results = [make_chunk("Text.", score=0.1)]
        assert self.assembler.assemble(results) == []


# ── RAGPipeline (mocked) ──────────────────────────────────────────────────────

class TestRAGPipeline:

    def _make_pipeline(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            make_chunk(
                "No tenant shall be evicted except in accordance with "
                "the provisions of this Ordinance.",
                score=0.85,
                source_id="sindh_rented_premises_1979",
                title="Sindh Rented Premises Ordinance 1979",
                jurisdiction="sindh",
                doc_category="tenancy",
            ),
            make_chunk(
                "The landlord shall give fifteen days notice before "
                "filing an application for eviction.",
                score=0.78,
                source_id="sindh_rented_premises_1979",
                title="Sindh Rented Premises Ordinance 1979",
                jurisdiction="sindh",
                doc_category="tenancy",
            ),
        ]
        return RAGPipeline(retriever=mock_retriever), mock_retriever

    def test_returns_rag_context(self):
        pipeline, _ = self._make_pipeline()
        doc = make_parsed_doc(
            "NOTICE TO QUIT: You are required to vacate the premises "
            "in Karachi. Pay PKR 45,000 or leave within 14 days.",
            DocumentType.EVICTION_NOTICE,
        )
        context = pipeline.run(doc)
        assert isinstance(context, RAGContext)

    def test_context_contains_document_text(self):
        pipeline, _ = self._make_pipeline()
        doc = make_parsed_doc("Eviction notice text here.", DocumentType.EVICTION_NOTICE)
        context = pipeline.run(doc)
        assert context.document_text == doc.clean_text

    def test_context_has_retrieved_chunks(self):
        pipeline, _ = self._make_pipeline()
        doc = make_parsed_doc("Tenant eviction notice Karachi.",
                              DocumentType.EVICTION_NOTICE)
        context = pipeline.run(doc)
        assert len(context.retrieved_chunks) > 0

    def test_context_has_correct_doc_type(self):
        pipeline, _ = self._make_pipeline()
        doc = make_parsed_doc("Employment contract termination.",
                              DocumentType.EMPLOYMENT_CONTRACT)
        context = pipeline.run(doc)
        assert context.doc_type == "employment_contract"

    def test_retriever_called_multiple_times(self):
        pipeline, mock_retriever = self._make_pipeline()
        doc = make_parsed_doc("Eviction notice Sindh tenant.",
                              DocumentType.EVICTION_NOTICE)
        pipeline.run(doc)
        # Should call retriever once per query
        assert mock_retriever.retrieve.call_count >= 3

    def test_queries_used_are_stored(self):
        pipeline, _ = self._make_pipeline()
        doc = make_parsed_doc("Debt collection notice.",
                              DocumentType.DEBT_COLLECTION)
        context = pipeline.run(doc)
        assert len(context.queries_used) >= 1
        assert all(isinstance(q, str) for q in context.queries_used)

    def test_facts_jurisdiction_detected(self):
        pipeline, _ = self._make_pipeline()
        doc = make_parsed_doc(
            "Property located in Karachi. Landlord demands rent.",
            DocumentType.EVICTION_NOTICE,
        )
        context = pipeline.run(doc)
        assert context.jurisdiction == "sindh"

    def test_chunks_are_deduplicated(self):
        mock_retriever = MagicMock()
        # Return same chunk from every query
        same_chunk = make_chunk("Same text repeated.", score=0.8)
        mock_retriever.retrieve.return_value = [same_chunk]

        pipeline = RAGPipeline(retriever=mock_retriever)
        doc = make_parsed_doc("Eviction notice.", DocumentType.EVICTION_NOTICE)
        context = pipeline.run(doc)

        # Despite multiple queries returning the same chunk, only 1 in context
        assert len(context.retrieved_chunks) == 1


# ── Live integration test (requires built KB) ─────────────────────────────────

class TestRAGPipelineLive:
    """
    Requires: python tests/test_kb_live.py to have been run first.
    Run with: pytest tests/test_retriever.py -v -k live
    """

    @pytest.mark.live
    def test_live_eviction_pipeline(self):
        pipeline = RAGPipeline()
        doc = make_parsed_doc(
            """
            NOTICE TO PAY RENT OR QUIT
            To: Muhammad Ali, 45 Gulshan-e-Iqbal, Karachi, Sindh.
            You are required to pay PKR 55,000 in arrears of rent for
            the months of January and February 2025, or vacate the
            rented premises within 15 days of this notice.
            Failure to comply will result in legal proceedings under
            the Sindh Rented Premises Ordinance.
            Issued by: Ahmed Properties, Karachi.
            """,
            DocumentType.EVICTION_NOTICE,
        )
        context = pipeline.run(doc)

        assert context.doc_type == "eviction_notice"
        assert context.jurisdiction == "sindh"
        assert len(context.retrieved_chunks) > 0

        top = context.retrieved_chunks[0]
        assert top.score > 0.3
        # Should retrieve from a tenancy-related source
        assert any(
            "rented" in c.title.lower() or
            "property" in c.title.lower() or
            "transfer" in c.title.lower()
            for c in context.retrieved_chunks
        )
        print(f"\n  Live test — top chunk:")
        print(f"  Source : {top.title}")
        print(f"  Score  : {top.score}")
        print(f"  Preview: {top.text[:150]}...")
