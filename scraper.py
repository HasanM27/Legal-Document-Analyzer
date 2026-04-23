"""
scraper.py — Pakistan Code Legal Document Scraper
===================================================
Fetches PDFs from https://pakistancode.gov.pk/pdffiles/, opens each one,
reads the first-page heading to determine the law title, detects language
(English / Urdu), and filters to only laws relevant to this app.

This module is ONLY responsible for fetching and parsing.
It does NOT touch ChromaDB — that is knowledge.py's job.

Usage (standalone test):
    python scraper.py               # scrape all relevant PDFs, print summary
    python scraper.py --limit 10    # test with first 10 PDFs only
"""

import re
import time
import logging
import argparse
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

PAKISTANCODE_INDEX = "https://pakistancode.gov.pk/pdffiles/"
REQUEST_HEADERS    = {"User-Agent": "LegalAssistant/1.0 (educational project)"}
REQUEST_DELAY      = 1.0   # seconds between requests — be polite to the server
REQUEST_TIMEOUT    = 30


# ── Relevant category whitelist ───────────────────────────────────────────────
#
# Only laws matching these keyword groups will be scraped.
# Anything else (Hajj regulations, fisheries, railway acts, etc.) is skipped.
#
# Each entry: (doc_category, [keywords that signal this category])

CATEGORY_WHITELIST: list[tuple[str, list[str]]] = [
    ("tenancy", [
        "rent", "tenancy", "tenant", "landlord", "lease",
        "premises", "eviction", "rented",
    ]),
    ("employment", [
        "employ", "labour", "labor", "worker", "wage",
        "industrial relation", "termination", "workmen",
        "minimum wage", "factories", "shops and establishment",
    ]),
    ("consumer", [
        "consumer", "defective", "trade practice",
        "goods", "product liability", "sale of goods",
    ]),
    ("court_summons", [
        "civil procedure", "code of civil", "court", "summon",
        "limitation", "evidence act", "oaths act",
    ]),
    ("debt_collection", [
        "debt", "recovery", "loan", "financial institution",
        "banking", "money lender", "negotiable instrument",
        "promissory", "creditor",
    ]),
    ("criminal", [
        "penal code", "criminal procedure", "offence",
        "punishment", "anti-terrorism", "corruption",
    ]),
    ("general", [
        "contract act", "specific relief", "transfer of property",
        "constitution", "limitation act", "stamp act",
        "registration act", "power of attorney",
    ]),
    ("family", [
        "family law", "muslim family", "marriage", "divorce",
        "guardian", "custody", "inheritance", "succession",
        "dowry", "dower",
    ]),
]


# ── Scraped law dataclass ─────────────────────────────────────────────────────

@dataclass
class ScrapedLaw:
    source_id:    str       # stable ID derived from filename
    title:        str       # law title extracted from first page
    url:          str       # original PDF URL
    full_text:    str       # complete extracted text (all pages)
    language:     str       # "english" | "urdu" | "mixed" | "unknown"
    doc_category: str       # matched category from whitelist
    page_count:   int


# ── Language detection ────────────────────────────────────────────────────────

_URDU_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def detect_language(text: str) -> str:
    if not text.strip():
        return "unknown"
    urdu_chars  = len(_URDU_RE.findall(text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    total       = urdu_chars + latin_chars
    if total == 0:
        return "unknown"
    ratio = urdu_chars / total
    if ratio > 0.7:
        return "urdu"
    elif ratio > 0.2:
        return "mixed"
    return "english"


# ── Category matching ─────────────────────────────────────────────────────────

def match_category(title: str) -> Optional[str]:
    """
    Check if a law title matches any whitelisted category.
    Returns the category string if matched, None if the law is irrelevant.
    """
    title_lower = title.lower()
    for category, keywords in CATEGORY_WHITELIST:
        for kw in keywords:
            if kw in title_lower:
                return category
    return None  # not relevant to this app


# ── Title extraction ──────────────────────────────────────────────────────────

_TITLE_KW = re.compile(
    r"\b(ACT|LAW|ORDINANCE|CODE|RULES?|REGULATIONS?|DECREE|STATUTE|ORDER)\b",
    re.IGNORECASE,
)


def extract_heading(text: str, language: str) -> str:
    """Extract the law title from first-page text."""
    if not text:
        return "Unknown Law"

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if language == "urdu":
        candidates = [l for l in lines if len(l) > 5][:2]
        return " ".join(candidates) if candidates else "Unknown Law (Urdu)"

    # English/mixed: find all-caps lines or lines with title keywords
    title_lines = []
    for line in lines[:30]:
        if re.match(r"^[\d\s\-/]+$", line) or len(line) < 4:
            continue
        if (line.isupper() and len(line) > 10) or _TITLE_KW.search(line):
            title_lines.append(line)
            if len(title_lines) >= 3:
                break

    if title_lines:
        return " ".join(title_lines)

    # Fallback: first substantive line
    for line in lines:
        if len(line) > 10:
            return line
    return "Unknown Law"


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, str, int]:
    """
    Extract full text from PDF bytes using PyMuPDF.
    Falls back to OCR (pytesseract) for scanned pages with no digital text.
    Returns (full_text, language, page_count).
    """
    try:
        doc        = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)
        pages      = []

        for page in doc:
            text = page.get_text("text").strip()
            if len(text) < 30:
                text = _ocr_page(page)
            if text:
                pages.append(text)

        doc.close()

        full_text = "\n\n".join(pages)
        language  = detect_language(full_text[:2000])
        return full_text, language, page_count

    except Exception as e:
        log.warning(f"PDF parse error: {e}")
        return "", "unknown", 0


def _ocr_page(page: fitz.Page) -> str:
    """OCR fallback for scanned PDF pages."""
    try:
        import pytesseract
        from PIL import Image

        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(img, lang="urd+eng")
        if not text.strip():
            text = pytesseract.image_to_string(img, lang="eng")
        return text.strip()
    except Exception as e:
        log.debug(f"OCR skipped: {e}")
        return ""


# ── Index fetcher ─────────────────────────────────────────────────────────────

def get_pdf_links(session: requests.Session) -> list[str]:
    """Fetch the pakistancode index page and return all unique PDF URLs."""
    log.info(f"Fetching index: {PAKISTANCODE_INDEX}")
    resp = session.get(PAKISTANCODE_INDEX, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    soup        = BeautifulSoup(resp.text, "html.parser")
    seen, links = set(), []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.lower().endswith(".pdf"):
            continue
        if href.startswith("http"):
            url = href
        elif href.startswith("/"):
            url = "https://pakistancode.gov.pk" + href
        else:
            url = PAKISTANCODE_INDEX.rstrip("/") + "/" + href

        if url not in seen:
            seen.add(url)
            links.append(url)

    log.info(f"Found {len(links)} PDF links")
    return links


# ── Main scrape function ──────────────────────────────────────────────────────

def scrape(
    limit:     Optional[int] = None,
    skip_urls: set            = None,
) -> list[ScrapedLaw]:
    """
    Scrape pakistancode.gov.pk, filter to relevant laws only, return results.

    Args:
        limit:     max PDFs to download (None = all). Useful for testing.
        skip_urls: URLs already covered by static sources — skip these.

    Returns:
        List of ScrapedLaw objects, one per relevant PDF found.
    """
    skip_urls = skip_urls or set()
    session   = requests.Session()
    results   = []

    try:
        all_urls = get_pdf_links(session)
    except Exception as e:
        log.error(f"Failed to fetch index: {e}")
        return []

    urls = [u for u in all_urls if u not in skip_urls]

    if limit:
        urls = urls[:limit]
        log.info(f"Limited to {limit} PDFs for this run")

    total    = len(urls)
    accepted = 0
    skipped  = 0

    for i, url in enumerate(urls, 1):
        filename = url.split("/")[-1]
        log.info(f"[{i}/{total}] {filename}")

        try:
            resp = session.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            full_text, language, page_count = extract_pdf_text(resp.content)

            if not full_text.strip():
                log.info(f"  → skipped (no text extracted)")
                skipped += 1
                time.sleep(REQUEST_DELAY)
                continue

            # Determine title from first page only
            title    = extract_heading(full_text[:3000], language)
            category = match_category(title)

            if category is None:
                log.info(f"  → skipped (not relevant): {title[:60]}")
                skipped += 1
                time.sleep(REQUEST_DELAY)
                continue

            source_id = re.sub(r"[^a-z0-9]", "_", filename.lower().replace(".pdf", ""))

            results.append(ScrapedLaw(
                source_id=source_id,
                title=title,
                url=url,
                full_text=full_text,
                language=language,
                doc_category=category,
                page_count=page_count,
            ))
            accepted += 1
            log.info(f"  → accepted [{category}] [{language}]: {title[:70]}")

        except requests.RequestException as e:
            log.warning(f"  → HTTP error: {e}")
        except Exception as e:
            log.warning(f"  → Error: {e}")

        time.sleep(REQUEST_DELAY)

    log.info(
        f"\nScrape complete: {accepted} relevant, "
        f"{skipped} skipped, out of {total} total PDFs"
    )
    return results


# ── CLI (standalone testing) ──────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Scrape relevant Pakistani law PDFs")
    parser.add_argument("--limit", type=int, default=None, help="Max PDFs to scrape")
    args = parser.parse_args()

    laws = scrape(limit=args.limit)
    print(f"\n{'='*60}")
    print(f"Found {len(laws)} relevant laws:\n")
    for law in laws:
        print(f"  [{law.doc_category:15}] [{law.language:7}] {law.title[:65]}")