"""
Step 1 — FastAPI upload endpoint
==================================
POST /ingest  →  accepts a file upload, returns ParsedDocument as JSON
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingestion.parser import DocumentParser, DocumentType


app = FastAPI(
    title="Legal Assistant API",
    description="Step 1: Document ingestion & parsing",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = DocumentParser()

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".txt", ".md"}
MAX_FILE_SIZE_MB = 20


# ── Response schema ────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_type:   str
    page_count: int
    is_scanned: bool
    confidence: float
    word_count: int
    preview:    str       # First 500 chars of clean text
    metadata:   dict


class TextIngestRequest(BaseModel):
    text: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload a PDF, image, or text file for parsing."""

    # Validate extension
    from pathlib import Path
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Read and size-check
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )

    # Parse
    try:
        doc = parser.parse_bytes(data, file.filename or "upload" + suffix)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return _build_response(doc)


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(body: TextIngestRequest):
    """Accept raw pasted text — no file upload needed."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text body is empty.")

    doc = parser.parse_text(body.text)
    return _build_response(doc)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Helper ─────────────────────────────────────────────────────────────────────

def _build_response(doc) -> IngestResponse:
    word_count = len(doc.clean_text.split())
    preview = doc.clean_text[:500] + ("..." if len(doc.clean_text) > 500 else "")
    return IngestResponse(
        doc_type=doc.doc_type.value,
        page_count=doc.page_count,
        is_scanned=doc.is_scanned,
        confidence=doc.confidence,
        word_count=word_count,
        preview=preview,
        metadata=doc.metadata,
    )
