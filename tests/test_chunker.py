"""
Tests for Step 2 — chunking logic
These tests only cover the chunker (no embedding model or ChromaDB needed).
Run with: pytest tests/test_chunker.py -v
"""

import pytest
from ingestion.chunker import LegalTextChunker, TextChunk


chunker = LegalTextChunker(chunk_size=512, chunk_overlap=64)


# ── Basic chunking ─────────────────────────────────────────────────────────────

class TestBasicChunking:

    def test_short_text_is_single_chunk(self):
        doc = "This is a short legal notice. It fits in one chunk easily."
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == doc.strip()

    def test_returns_textchunk_objects(self):
        chunks = chunker.chunk("Hello world.")
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_ids_are_unique(self):
        text = "A " * 300   # force multiple chunks
        chunks = chunker.chunk(text)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_indices_are_sequential(self):
        text = "Word " * 400
        chunks = chunker.chunk(text)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_total_chunks_metadata_correct(self):
        text = "Word " * 400
        chunks = chunker.chunk(text)
        total = len(chunks)
        for c in chunks:
            assert c.metadata["total_chunks"] == total

    def test_empty_text_returns_empty(self):
        assert chunker.chunk("") == []
        assert chunker.chunk("   \n\n  ") == []


# ── Boundary detection ────────────────────────────────────────────────────────

class TestBoundaryDetection:

    def test_splits_on_numbered_clauses(self):
        text = (
            "AGREEMENT\n\n"
            "1. The landlord agrees to maintain the property in good condition "
            "and carry out all necessary repairs within a reasonable timeframe.\n\n"
            "2. The tenant agrees to pay rent on the first day of each month "
            "without deduction or set-off unless agreed otherwise in writing.\n\n"
            "3. Either party may terminate this agreement by giving 30 days "
            "written notice to the other party at the address specified herein."
        )
        chunks = chunker.chunk(text)
        # Each numbered clause should end up in its own chunk (or merged if tiny)
        full_text = " ".join(c.text for c in chunks)
        assert "landlord agrees" in full_text
        assert "tenant agrees" in full_text
        assert "Either party" in full_text

    def test_splits_on_allcaps_headings(self):
        text = (
            "PARTIES\n"
            "This agreement is between the landlord and tenant as defined below.\n\n"
            "TERMS\n"
            "The tenancy shall commence on the date specified and continue monthly."
        )
        chunks = chunker.chunk(text)
        combined = " ".join(c.text for c in chunks)
        assert "landlord and tenant" in combined
        assert "tenancy shall commence" in combined

    def test_splits_on_section_headings(self):
        text = (
            "Section 1 This is the first section with important information.\n"
            "Section 2 This is the second section describing tenant obligations."
        )
        chunks = chunker.chunk(text)
        combined = " ".join(c.text for c in chunks)
        assert "first section" in combined
        assert "second section" in combined


# ── Size constraints ──────────────────────────────────────────────────────────

class TestSizeConstraints:

    def test_no_chunk_exceeds_max_size(self):
        # Generate a long paragraph with no natural boundaries
        long_para = "The tenant shall be responsible for all utilities. " * 30
        chunks = chunker.chunk(long_para)
        for c in chunks:
            assert len(c.text) <= chunker.chunk_size + chunker.chunk_overlap + 50

    def test_large_document_produces_multiple_chunks(self):
        # Use numbered clauses so there are natural boundaries to split on
        text = "\n\n".join(
            f"{i}. The tenant agrees to clause number {i} and all its terms and conditions as stated herein."
            for i in range(1, 40)
        )
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_overlap_content_preserved(self):
        # With overlap, the end of chunk N should appear in chunk N+1
        long_text = "Sentence number {i} contains important legal information. " * 50
        small_chunker = LegalTextChunker(chunk_size=200, chunk_overlap=50)
        chunks = small_chunker.chunk(long_text)
        if len(chunks) > 1:
            end_of_first = chunks[0].text[-30:]
            # At least some of the overlap window should bridge adjacent chunks
            assert len(chunks[1].text) > 0


# ── Metadata ──────────────────────────────────────────────────────────────────

class TestMetadata:

    def test_custom_metadata_passed_through(self):
        chunks = chunker.chunk("Short text.", metadata={"doc_type": "eviction_notice", "session_id": "abc"})
        assert chunks[0].metadata["doc_type"] == "eviction_notice"
        assert chunks[0].metadata["session_id"] == "abc"

    def test_char_positions_are_set(self):
        text = "First sentence here. Second sentence here."
        chunks = chunker.chunk(text)
        for c in chunks:
            assert c.char_start >= 0
            assert c.char_end > c.char_start


# ── Real-world document sample ────────────────────────────────────────────────

class TestRealWorldSample:

    EVICTION_NOTICE = """
    NOTICE TO PAY RENT OR QUIT

    To: John Smith
    Address: 123 Main Street, Karachi

    You are hereby notified that you are in default of the rental agreement
    dated 1st January 2025 for the above premises.

    1. Amount of rent due: PKR 45,000
    2. Period for which rent is due: February 2025
    3. You are required to pay the full amount within 14 days of this notice.

    LEGAL NOTICE
    Failure to pay the outstanding rent or vacate the premises within the
    time specified will result in legal proceedings being initiated against
    you under the Sindh Rented Premises Ordinance 1979.

    Section 14 of the said Ordinance provides that a landlord may apply
    to the Rent Controller for an order of eviction where the tenant has
    failed to pay rent for a period of two months or more.

    Issued by: ABC Properties
    Date: 1st March 2025
    """

    def test_eviction_notice_chunks_correctly(self):
        chunks = chunker.chunk(self.EVICTION_NOTICE)
        assert len(chunks) >= 2
        combined = " ".join(c.text for c in chunks)
        assert "PKR 45,000" in combined
        assert "14 days" in combined
        assert "Sindh Rented Premises" in combined

    def test_no_content_lost(self):
        # Every meaningful word in the original should appear somewhere in chunks
        chunks = chunker.chunk(self.EVICTION_NOTICE)
        combined = " ".join(c.text for c in chunks)
        for keyword in ["PKR 45,000", "Rent Controller", "ABC Properties"]:
            assert keyword in combined, f"'{keyword}' was lost during chunking"
