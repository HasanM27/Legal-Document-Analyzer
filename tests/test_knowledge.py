"""
Tests for Step 3 — legal knowledge base
Tests cover scraping logic, chunking of legal sources, metadata tagging,
and the retriever filter builder — all without needing ChromaDB or a live
embedding model.

Run with: pytest tests/test_knowledge.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from ingestion.knowledge import (
    LegalSource,
    StatuteScraper,
    KnowledgeBaseBuilder,
    KnowledgeRetriever,
    STATIC_SOURCES,
)
from ingestion.chunker import LegalTextChunker


# ── LegalSource ───────────────────────────────────────────────────────────────

class TestLegalSource:

    def test_source_fields(self):
        src = LegalSource(
            source_id="test_act",
            title="Test Act 2024",
            jurisdiction="sindh",
            doc_category="tenancy",
            static_text="Some legal text.",
        )
        assert src.source_id == "test_act"
        assert src.jurisdiction == "sindh"
        assert src.doc_category == "tenancy"
        assert src.url is None

    def test_static_sources_are_loaded(self):
        assert len(STATIC_SOURCES) >= 5

    def test_all_static_sources_have_required_fields(self):
        for src in STATIC_SOURCES:
            assert src.source_id, f"Missing source_id in {src}"
            assert src.title,     f"Missing title in {src}"
            assert src.jurisdiction in ("federal", "sindh", "punjab", "kpk", "balochistan")
            assert src.doc_category
            # Every source must have either a URL or static text
            assert src.url or src.static_text, f"No text for {src.source_id}"

    def test_all_static_sources_have_text(self):
        for src in STATIC_SOURCES:
            assert src.static_text and len(src.static_text.strip()) > 100, \
                f"Static text too short for {src.source_id}"

    def test_static_sources_cover_key_categories(self):
        categories = {src.doc_category for src in STATIC_SOURCES}
        assert "tenancy"       in categories
        assert "employment"    in categories
        assert "consumer"      in categories
        assert "court_summons" in categories
        # debt_collection is covered by "general" sources:
        # Limitation Act, Contract Act, Civil Procedure Code
        assert "general"       in categories


# ── StatuteScraper ────────────────────────────────────────────────────────────

class TestStatuteScraper:

    def test_returns_static_text_when_no_url(self):
        scraper = StatuteScraper()
        src = LegalSource(
            source_id="no_url",
            title="Test",
            jurisdiction="federal",
            doc_category="tenancy",
            static_text="This is the static fallback text for the test.",
        )
        text = scraper.fetch(src)
        assert text == "This is the static fallback text for the test."

    def test_falls_back_to_static_on_url_failure(self):
        scraper = StatuteScraper()
        src = LegalSource(
            source_id="bad_url",
            title="Test",
            jurisdiction="federal",
            doc_category="tenancy",
            url="http://thisurldoesnotexist.invalid/law.pdf",
            static_text="Fallback text when URL fails.",
        )
        # Should not raise — should return static text
        text = scraper.fetch(src)
        assert "Fallback text" in text

    def test_raises_when_no_text_available(self):
        scraper = StatuteScraper()
        src = LegalSource(
            source_id="empty",
            title="Empty Source",
            jurisdiction="federal",
            doc_category="tenancy",
            url=None,
            static_text=None,
        )
        with pytest.raises(ValueError, match="No text available"):
            scraper.fetch(src)


# ── Chunking of legal sources ─────────────────────────────────────────────────

class TestLegalSourceChunking:

    def test_sindh_tenancy_law_chunks(self):
        chunker = LegalTextChunker(chunk_size=600, chunk_overlap=80)
        src = next(s for s in STATIC_SOURCES if s.source_id == "sindh_rented_premises_1979")
        chunks = chunker.chunk(src.static_text)
        assert len(chunks) >= 3

    def test_key_provisions_preserved_in_chunks(self):
        chunker = LegalTextChunker(chunk_size=600, chunk_overlap=80)
        src = next(s for s in STATIC_SOURCES if s.source_id == "sindh_rented_premises_1979")
        chunks = chunker.chunk(src.static_text)
        combined = " ".join(c.text for c in chunks)
        assert "fifteen days" in combined.lower() or "15 days" in combined.lower()
        assert "Rent Controller" in combined

    def test_employment_law_chunks(self):
        chunker = LegalTextChunker(chunk_size=600, chunk_overlap=80)
        src = next(s for s in STATIC_SOURCES if s.source_id == "industrial_relations_act_2012")
        chunks = chunker.chunk(src.static_text)
        combined = " ".join(c.text for c in chunks)
        assert "thirty days" in combined.lower()
        assert "termination" in combined.lower()

    def test_metadata_tags_are_applied(self):
        chunker = LegalTextChunker()
        chunks = chunker.chunk(
            "Legal text about tenancy.",
            metadata={"source_id": "test", "jurisdiction": "sindh", "doc_category": "tenancy"},
        )
        assert chunks[0].metadata["jurisdiction"] == "sindh"
        assert chunks[0].metadata["doc_category"] == "tenancy"
        assert chunks[0].metadata["source_id"] == "test"


# ── KnowledgeBaseBuilder (mocked) ─────────────────────────────────────────────

class TestKnowledgeBaseBuilder:

    def _make_builder(self):
        """Builder with mocked embedder and vector store."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 384]   # fake embeddings
        mock_store = MagicMock()
        mock_store.count.return_value = 0
        mock_store._collection.return_value.get.return_value = {"ids": []}

        return KnowledgeBaseBuilder(
            embedder=mock_embedder,
            vector_store=mock_store,
        )

    def test_build_processes_all_sources(self):
        builder = self._make_builder()
        # Override embedder to return correct number of embeddings
        def fake_embed(texts):
            return [[0.1] * 384 for _ in texts]
        builder.embedder.embed.side_effect = fake_embed

        summary = builder.build(sources=STATIC_SOURCES)
        assert len(summary) == len(STATIC_SOURCES)

    def test_build_returns_ok_status_for_valid_sources(self):
        builder = self._make_builder()
        def fake_embed(texts):
            return [[0.1] * 384 for _ in texts]
        builder.embedder.embed.side_effect = fake_embed

        summary = builder.build(sources=STATIC_SOURCES)
        for sid, info in summary.items():
            assert info["status"] == "ok", f"Expected ok for {sid}, got {info}"

    def test_build_skips_duplicate_content(self):
        builder = self._make_builder()
        # Simulate already stored
        builder.vector_store._collection.return_value.get.return_value = {"ids": ["existing"]}

        def fake_embed(texts):
            return [[0.1] * 384 for _ in texts]
        builder.embedder.embed.side_effect = fake_embed

        summary = builder.build(sources=STATIC_SOURCES[:1])
        # Should be 0 chunks since content is "already stored"
        assert summary[STATIC_SOURCES[0].source_id]["chunks"] == 0

    def test_build_handles_source_error_gracefully(self):
        builder = self._make_builder()
        bad_source = LegalSource(
            source_id="bad",
            title="Bad Source",
            jurisdiction="federal",
            doc_category="tenancy",
            static_text=None,
            url=None,
        )
        summary = builder.build(sources=[bad_source])
        assert summary["bad"]["status"] == "error"


# ── KnowledgeRetriever filter builder ─────────────────────────────────────────

class TestKnowledgeRetriever:

    def _make_retriever(self):
        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = [0.1] * 384
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "text": "A tenant must receive 15 days notice.",
                "score": 0.91,
                "metadata": {
                    "source_id": "sindh_rented_premises_1979",
                    "title": "Sindh Rented Premises Ordinance 1979",
                    "jurisdiction": "sindh",
                    "doc_category": "tenancy",
                },
            }
        ]
        return KnowledgeRetriever(embedder=mock_embedder, vector_store=mock_store)

    def test_retrieve_returns_results(self):
        retriever = self._make_retriever()
        results = retriever.retrieve("eviction notice tenant rights")
        assert len(results) == 1
        assert results[0]["score"] == 0.91
        assert "15 days" in results[0]["text"]

    def test_retrieve_flattens_metadata(self):
        retriever = self._make_retriever()
        results = retriever.retrieve("tenant eviction")
        r = results[0]
        assert "source_id"    in r
        assert "title"        in r
        assert "jurisdiction" in r
        assert "doc_category" in r

    def test_filter_jurisdiction_only(self):
        retriever = self._make_retriever()
        retriever.retrieve("eviction", jurisdiction="sindh")
        call_kwargs = retriever.vector_store.query.call_args[1]
        where = call_kwargs.get("where")
        assert where == {"jurisdiction": {"$eq": "sindh"}}

    def test_filter_doc_category_only(self):
        retriever = self._make_retriever()
        retriever.retrieve("termination", doc_category="employment")
        call_kwargs = retriever.vector_store.query.call_args[1]
        where = call_kwargs.get("where")
        assert where == {"doc_category": {"$eq": "employment"}}

    def test_filter_both(self):
        retriever = self._make_retriever()
        retriever.retrieve("rent", jurisdiction="federal", doc_category="tenancy")
        call_kwargs = retriever.vector_store.query.call_args[1]
        where = call_kwargs.get("where")
        assert where == {"$and": [
            {"jurisdiction": {"$eq": "federal"}},
            {"doc_category": {"$eq": "tenancy"}},
        ]}

    def test_filter_neither(self):
        retriever = self._make_retriever()
        retriever.retrieve("general query")
        call_kwargs = retriever.vector_store.query.call_args[1]
        where = call_kwargs.get("where")
        assert where is None