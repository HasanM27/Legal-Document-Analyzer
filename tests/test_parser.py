"""
Tests for Step 1 — document ingestion & parsing
Run with:  pytest tests/test_parser.py -v
"""

import pytest
from ingestion.parser import DocumentParser, DocumentType, ParsedDocument


parser = DocumentParser()


# ── Text parsing ───────────────────────────────────────────────────────────────

class TestTextParsing:

    def test_basic_text(self):
        doc = parser.parse_text("Hello world. This is a test document.")
        assert isinstance(doc, ParsedDocument)
        assert doc.clean_text == "Hello world. This is a test document."
        assert doc.is_scanned is False
        assert doc.confidence == 1.0
        assert doc.page_count == 1

    def test_empty_text_still_parses(self):
        doc = parser.parse_text("   \n\n  ")
        assert doc.clean_text == ""
        assert doc.doc_type == DocumentType.UNKNOWN

    def test_whitespace_collapse(self):
        doc = parser.parse_text("Hello   world\n\n\n\nNew paragraph")
        assert "   " not in doc.clean_text
        assert doc.clean_text.count("\n\n\n") == 0


# ── Document classification ───────────────────────────────────────────────────

class TestClassification:

    def test_eviction_notice(self):
        text = """
        NOTICE TO PAY RENT OR QUIT
        You are hereby required to pay the sum of $1,450 in unpaid rent
        or vacate and deliver up possession of the premises within 14 days.
        Failure to comply will result in eviction proceedings.
        """
        doc = parser.parse_text(text)
        assert doc.doc_type == DocumentType.EVICTION_NOTICE

    def test_tenancy_agreement(self):
        text = """
        TENANCY AGREEMENT
        This lease agreement is entered between the landlord and tenant.
        Monthly rent of PKR 45,000 is due on the 1st of each month.
        A security deposit of PKR 90,000 is required upon signing.
        The rental period commences on 1st January 2025.
        """
        doc = parser.parse_text(text)
        assert doc.doc_type == DocumentType.TENANCY_AGREEMENT

    def test_employment_contract(self):
        text = """
        EMPLOYMENT AGREEMENT
        This offer of employment confirms your job title as Senior Engineer.
        Your salary will be PKR 150,000 per month.
        A probationary period of 3 months applies.
        A termination clause requires 30 days written notice.
        """
        doc = parser.parse_text(text)
        assert doc.doc_type == DocumentType.EMPLOYMENT_CONTRACT

    def test_debt_collection(self):
        text = """
        DEBT COLLECTION NOTICE
        Your account has an outstanding balance of $3,200.
        This is a notice from a debt collector.
        The amount due must be paid within 30 days.
        Payment overdue notices have been sent previously.
        """
        doc = parser.parse_text(text)
        assert doc.doc_type == DocumentType.DEBT_COLLECTION

    def test_unknown_document(self):
        text = "The quick brown fox jumps over the lazy dog."
        doc = parser.parse_text(text)
        assert doc.doc_type == DocumentType.UNKNOWN

    def test_single_keyword_stays_unknown(self):
        # Only 1 keyword hit — should not classify
        doc = parser.parse_text("This document mentions eviction once.")
        assert doc.doc_type == DocumentType.UNKNOWN


# ── Text cleaning ─────────────────────────────────────────────────────────────

class TestTextCleaning:

    def test_smart_quotes_normalised(self):
        doc = parser.parse_text("\u201Chello\u201D and \u2018world\u2019")
        assert '"hello"' in doc.clean_text
        assert "'world'" in doc.clean_text

    def test_page_numbers_removed(self):
        text = "Some content\n\nPage 1 of 5\n\nMore content\n\n3\n\nEnd"
        doc = parser.parse_text(text)
        assert "Page 1 of 5" not in doc.clean_text
        # Standalone "3" (page number) removed
        lines = [l.strip() for l in doc.clean_text.splitlines()]
        assert "3" not in lines

    def test_excessive_blank_lines_collapsed(self):
        text = "Para one\n\n\n\n\n\nPara two"
        doc = parser.parse_text(text)
        assert "\n\n\n" not in doc.clean_text

    def test_multiple_spaces_collapsed(self):
        doc = parser.parse_text("Hello     world   here")
        assert "  " not in doc.clean_text


# ── Bytes routing ─────────────────────────────────────────────────────────────

class TestBytesRouting:

    def test_txt_bytes(self):
        text = "This is a tenancy agreement with landlord and tenant and monthly rent and security deposit and rental period."
        doc = parser.parse_bytes(text.encode("utf-8"), "doc.txt")
        assert doc.doc_type == DocumentType.TENANCY_AGREEMENT
        assert doc.is_scanned is False

    def test_unknown_extension_falls_back(self):
        text = b"Plain text content with no legal keywords."
        doc = parser.parse_bytes(text, "document.xyz")
        assert doc.doc_type == DocumentType.UNKNOWN


# ── Metadata ──────────────────────────────────────────────────────────────────

class TestMetadata:

    def test_pasted_text_source(self):
        doc = parser.parse_text("Some text")
        assert doc.metadata["source"] == "pasted_text"

    def test_txt_bytes_source(self):
        doc = parser.parse_bytes(b"Some text content here.", "file.txt")
        assert doc.metadata["source"] == "pasted_text"
