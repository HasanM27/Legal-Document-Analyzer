"""
Step 1 — Document ingestion & parsing
======================================
Handles: digital PDFs, scanned PDFs (OCR), plain images, raw text
Output : ParsedDocument dataclass with clean text + metadata
"""

import re
import io
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


# ── Types ──────────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    EVICTION_NOTICE      = "eviction_notice"
    EMPLOYMENT_CONTRACT  = "employment_contract"
    DEBT_COLLECTION      = "debt_collection"
    TENANCY_AGREEMENT    = "tenancy_agreement"
    COURT_SUMMONS        = "court_summons"
    GOVERNMENT_LETTER    = "government_letter"
    UNKNOWN              = "unknown"


@dataclass
class ParsedDocument:
    raw_text:      str
    clean_text:    str
    doc_type:      DocumentType
    page_count:    int
    is_scanned:    bool
    confidence:    float          # OCR confidence 0.0–1.0 (1.0 if not OCR)
    metadata:      dict = field(default_factory=dict)


# ── Classifier keywords ────────────────────────────────────────────────────────

DOC_TYPE_KEYWORDS: dict[DocumentType, list[str]] = {
    DocumentType.EVICTION_NOTICE: [
        "vacate", "quit", "eviction", "unlawful detainer",
        "pay or quit", "notice to leave", "possession",
    ],
    DocumentType.EMPLOYMENT_CONTRACT: [
        "employment agreement", "offer of employment", "salary",
        "probationary period", "termination clause", "job title",
    ],
    DocumentType.DEBT_COLLECTION: [
        "amount due", "outstanding balance", "debt collector",
        "creditor", "collection agency", "payment overdue",
    ],
    DocumentType.TENANCY_AGREEMENT: [
        "tenancy agreement", "lease agreement", "landlord", "tenant",
        "monthly rent", "security deposit", "rental period",
    ],
    DocumentType.COURT_SUMMONS: [
        "summons", "plaintiff", "defendant", "court order",
        "you are hereby ordered", "hearing date", "magistrate",
    ],
    DocumentType.GOVERNMENT_LETTER: [
        "ministry", "department", "government of", "official notice",
        "reference number", "nadra", "federal board",
    ],
}


# ── Core parser ────────────────────────────────────────────────────────────────

class DocumentParser:
    """
    Main entry point. Call parse_file() with a file path, or
    parse_bytes() with raw bytes + a filename for the extension hint.
    """

    # -- Public API ------------------------------------------------------------

    def parse_file(self, path: str | Path) -> ParsedDocument:
        path = Path(path)
        suffix = path.suffix.lower()
        raw_bytes = path.read_bytes()
        return self._route(raw_bytes, suffix)

    def parse_bytes(self, data: bytes, filename: str) -> ParsedDocument:
        suffix = Path(filename).suffix.lower()
        return self._route(data, suffix)

    def parse_text(self, text: str) -> ParsedDocument:
        """Accept raw pasted text directly — no file needed."""
        clean = self._clean(text)
        return ParsedDocument(
            raw_text=text,
            clean_text=clean,
            doc_type=self._classify(clean),
            page_count=1,
            is_scanned=False,
            confidence=1.0,
            metadata={"source": "pasted_text"},
        )

    # -- Routing ---------------------------------------------------------------

    def _route(self, data: bytes, suffix: str) -> ParsedDocument:
        if suffix == ".pdf":
            return self._parse_pdf(data)
        elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            return self._parse_image(data, suffix)
        elif suffix in (".txt", ".md"):
            return self.parse_text(data.decode("utf-8", errors="replace"))
        else:
            # Try PDF first, fall back to raw text decode
            try:
                return self._parse_pdf(data)
            except Exception:
                return self.parse_text(data.decode("utf-8", errors="replace"))

    # -- PDF parsing -----------------------------------------------------------

    def _parse_pdf(self, data: bytes) -> ParsedDocument:
        """
        Strategy:
          1. Try PyMuPDF (fitz) for digital PDFs — fast, accurate.
          2. If extracted text is too short (scanned), fall back to OCR
             via pdfplumber + Tesseract on each rendered page image.
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=data, filetype="pdf")
            pages_text = [page.get_text() for page in doc]
            raw = "\n\n".join(pages_text)

            # Heuristic: scanned PDFs have very little extractable text
            avg_chars = len(raw) / max(len(pages_text), 1)
            if avg_chars < 80:
                return self._ocr_pdf(data, len(pages_text))

            clean = self._clean(raw)
            return ParsedDocument(
                raw_text=raw,
                clean_text=clean,
                doc_type=self._classify(clean),
                page_count=len(pages_text),
                is_scanned=False,
                confidence=1.0,
                metadata={"source": "pymupdf"},
            )

        except ImportError:
            # PyMuPDF not installed — try pdfplumber
            return self._parse_pdf_pdfplumber(data)

    def _parse_pdf_pdfplumber(self, data: bytes) -> ParsedDocument:
        """Fallback PDF parser using pdfplumber."""
        import pdfplumber

        pages_text = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages_text.append(text)

        raw = "\n\n".join(pages_text)
        avg_chars = len(raw) / max(page_count, 1)

        if avg_chars < 80:
            return self._ocr_pdf(data, page_count)

        clean = self._clean(raw)
        return ParsedDocument(
            raw_text=raw,
            clean_text=clean,
            doc_type=self._classify(clean),
            page_count=page_count,
            is_scanned=False,
            confidence=1.0,
            metadata={"source": "pdfplumber"},
        )

    def _ocr_pdf(self, data: bytes, page_count: int) -> ParsedDocument:
        """
        Scanned PDF — render each page to an image, then run Tesseract.
        Requires: pdf2image + poppler-utils system package + pytesseract + Tesseract binary.
        """
        try:
            from pdf2image import convert_from_bytes
            import pytesseract

            images = convert_from_bytes(data, dpi=300)
            pages_text = []
            confidences = []

            for img in images:
                data_dict = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT
                )
                text = pytesseract.image_to_string(img)
                pages_text.append(text)

                # Average word-level confidence (ignore -1 entries)
                confs = [c for c in data_dict["conf"] if c != -1]
                confidences.append(sum(confs) / len(confs) / 100 if confs else 0.0)

            raw = "\n\n".join(pages_text)
            clean = self._clean(raw)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return ParsedDocument(
                raw_text=raw,
                clean_text=clean,
                doc_type=self._classify(clean),
                page_count=page_count,
                is_scanned=True,
                confidence=round(avg_conf, 3),
                metadata={"source": "ocr_tesseract", "dpi": 300},
            )

        except ImportError as e:
            raise RuntimeError(
                f"OCR dependencies missing: {e}. "
                "Install: pip install pdf2image pytesseract  "
                "and system package: poppler-utils"
            )

    # -- Image parsing ---------------------------------------------------------

    def _parse_image(self, data: bytes, suffix: str) -> ParsedDocument:
        """Direct image OCR — for photos of documents."""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(io.BytesIO(data))

            # Pre-process: convert to greyscale for better OCR accuracy
            img = img.convert("L")

            data_dict = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT
            )
            raw = pytesseract.image_to_string(img)
            confs = [c for c in data_dict["conf"] if c != -1]
            conf = sum(confs) / len(confs) / 100 if confs else 0.0

            clean = self._clean(raw)
            return ParsedDocument(
                raw_text=raw,
                clean_text=clean,
                doc_type=self._classify(clean),
                page_count=1,
                is_scanned=True,
                confidence=round(conf, 3),
                metadata={"source": f"ocr_image_{suffix.lstrip('.')}"},
            )

        except ImportError as e:
            raise RuntimeError(
                f"Image OCR dependencies missing: {e}. "
                "Install: pip install pytesseract Pillow"
            )

    # -- Text cleaning ---------------------------------------------------------

    def _clean(self, text: str) -> str:
        """
        Normalise raw extracted text:
          - Collapse excessive whitespace / blank lines
          - Remove garbled OCR artifacts (non-printable chars)
          - Normalise smart quotes and dashes
          - Strip page headers/footers (simple heuristic)
        """
        # Remove non-printable characters except newlines and tabs
        text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", " ", text)

        # Normalise smart quotes and dashes
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201C", '"').replace("\u201D", '"')
        text = text.replace("\u2013", "-").replace("\u2014", "-")

        # Remove common PDF header/footer noise (page numbers, running titles)
        text = re.sub(r"(?m)^\s*Page\s+\d+\s+of\s+\d+\s*$", "", text)
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)  # standalone page numbers

        # Collapse 3+ blank lines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    # -- Document type classifier ----------------------------------------------

    def _classify(self, text: str) -> DocumentType:
        """
        Keyword-based classifier. Returns the type with the most keyword hits.
        Falls back to UNKNOWN if nothing matches confidently.
        """
        lower = text.lower()
        scores: dict[DocumentType, int] = {}

        for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[doc_type] = score

        if not scores:
            return DocumentType.UNKNOWN

        best = max(scores, key=lambda t: scores[t])
        # Require at least 2 keyword hits to classify confidently
        return best if scores[best] >= 2 else DocumentType.UNKNOWN
