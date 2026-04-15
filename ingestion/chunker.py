"""
Step 2 — Chunking & Embedding
===============================
Takes the clean_text from ParsedDocument, splits it into
clause-aware chunks, embeds each chunk, and stores them in ChromaDB.

Two collections:
  - "documents"  : chunks from user-uploaded documents (per session)
  - "knowledge"  : legal knowledge base chunks (built once, reused always)
"""

import uuid
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Chunk dataclass ────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    chunk_id:   str
    text:       str
    index:      int           # position in the original document
    char_start: int
    char_end:   int
    metadata:   dict = field(default_factory=dict)


# ── Chunker ────────────────────────────────────────────────────────────────────

class LegalTextChunker:
    """
    Splits legal text into clause-aware chunks.

    Why not just split every N characters?
    Legal documents have numbered clauses, sections, and paragraphs.
    Splitting mid-clause destroys the meaning. We split on those
    boundaries first, then enforce a max size to keep chunks
    manageable for embedding models.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """
        chunk_size    : max characters per chunk (512 ≈ 100–120 words, fits most embedding models)
        chunk_overlap : characters of overlap between adjacent chunks
                        so a clause split across a boundary isn't lost
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Patterns that signal a natural legal boundary (split HERE)
        self._boundary_patterns = [
            r"\n\s*\d+\.\s+",          # numbered clauses:  "1. ", "12. "
            r"\n\s*[A-Z]{2,}[\s:]+",   # ALL-CAPS headings: "WHEREAS ", "PARTIES:"
            r"\n\s*Section\s+\d+",     # "Section 3"
            r"\n\s*Article\s+\d+",     # "Article 5"
            r"\n\s*Clause\s+\d+",      # "Clause 2"
            r"\n\s*\([a-z]\)\s+",      # lettered sub-clauses: "(a) ", "(b) "
            r"\n{2,}",                 # blank lines (paragraph breaks)
        ]

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[TextChunk]:
        """Split text into overlapping clause-aware chunks."""
        metadata = metadata or {}

        # Step 1: find all natural boundary positions
        boundaries = self._find_boundaries(text)

        # Step 2: build initial segments at boundaries
        segments = self._split_at_boundaries(text, boundaries)

        # Step 3: merge tiny segments and split oversized ones
        chunks = self._normalise(segments)

        # Step 4: wrap in TextChunk dataclasses
        result = []
        cursor = 0
        for i, chunk_text in enumerate(chunks):
            start = text.find(chunk_text, cursor)
            if start == -1:
                start = cursor
            end = start + len(chunk_text)
            cursor = max(0, end - self.chunk_overlap)

            result.append(TextChunk(
                chunk_id=str(uuid.uuid4()),
                text=chunk_text.strip(),
                index=i,
                char_start=start,
                char_end=end,
                metadata={**metadata, "chunk_index": i, "total_chunks": 0},
            ))

        # Back-fill total_chunks now we know the count
        for chunk in result:
            chunk.metadata["total_chunks"] = len(result)

        return result

    def _find_boundaries(self, text: str) -> list[int]:
        positions = set()
        for pattern in self._boundary_patterns:
            for match in re.finditer(pattern, text):
                positions.add(match.start())
        return sorted(positions)

    def _split_at_boundaries(self, text: str, boundaries: list[int]) -> list[str]:
        if not boundaries:
            return [text]

        segments = []
        prev = 0
        for pos in boundaries:
            if pos > prev:
                segments.append(text[prev:pos])
            prev = pos
        segments.append(text[prev:])
        return [s for s in segments if s.strip()]

    def _normalise(self, segments: list[str]) -> list[str]:
        """
        - Merge segments that are too short (< 80 chars) with the next one
        - Split segments that are too long (> chunk_size) by sentence
        """
        # Merge short segments
        merged = []
        buffer = ""
        for seg in segments:
            if len(buffer) + len(seg) < self.chunk_size:
                buffer += " " + seg if buffer else seg
            else:
                if buffer:
                    merged.append(buffer)
                buffer = seg
        if buffer:
            merged.append(buffer)

        # Split oversized segments
        final = []
        for seg in merged:
            if len(seg) <= self.chunk_size:
                final.append(seg)
            else:
                final.extend(self._split_by_sentence(seg))

        return final

    def _split_by_sentence(self, text: str) -> list[str]:
        """Last-resort split on sentence boundaries with overlap."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        buffer = ""

        for sent in sentences:
            if len(buffer) + len(sent) <= self.chunk_size:
                buffer += " " + sent if buffer else sent
            else:
                if buffer:
                    chunks.append(buffer)
                # Start next chunk with overlap from end of previous
                overlap_text = buffer[-self.chunk_overlap:] if buffer else ""
                buffer = overlap_text + " " + sent if overlap_text else sent

        if buffer:
            chunks.append(buffer)

        return chunks


# ── Embedder ───────────────────────────────────────────────────────────────────

class Embedder:
    """
    Wraps a sentence-transformers model.
    Default model: all-MiniLM-L6-v2
      - Small (80MB), fast, good quality for English legal text
      - Produces 384-dimensional vectors
      - Free, runs locally, no API key needed

    Swap to "all-mpnet-base-v2" for higher quality (420MB, 768-dim).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None   # lazy load — only import when first used

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model '{self.model_name}'...")
                self._model = SentenceTransformer(self.model_name)
                print("Model loaded.")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings. Returns list of float vectors."""
        self._load()
        vectors = self._model.encode(texts, show_progress_bar=False)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ── Vector store ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around ChromaDB.

    Two named collections:
      DOCUMENTS  — user document chunks (cleared per session if needed)
      KNOWLEDGE  — legal knowledge base (persistent, never cleared by user uploads)
    """

    DOCUMENTS_COLLECTION = "documents"
    KNOWLEDGE_COLLECTION  = "knowledge"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._persist_dir = persist_dir
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self._persist_dir)
            except ImportError:
                raise RuntimeError(
                    "chromadb not installed. Run: pip install chromadb"
                )
        return self._client

    def _collection(self, name: str):
        return self._get_client().get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity for text
        )

    # -- Write -----------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        collection_name: str = DOCUMENTS_COLLECTION,
    ):
        """Store chunks + their embeddings in the named collection."""
        col = self._collection(collection_name)
        col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def clear_documents(self):
        """Wipe the user-document collection (call between sessions if needed)."""
        client = self._get_client()
        try:
            client.delete_collection(self.DOCUMENTS_COLLECTION)
        except Exception:
            pass

    # -- Read ------------------------------------------------------------------

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection_name: str = KNOWLEDGE_COLLECTION,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return top_k most similar chunks.
        Each result: { text, score, metadata }
        """
        col = self._collection(collection_name)
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, col.count() or 1),
            include=["documents", "distances", "metadatas"],
        )
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)

        output = []
        for text, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            output.append({
                "text": text,
                "score": round(1 - dist, 4),   # cosine distance → similarity
                "metadata": meta,
            })

        return output

    def count(self, collection_name: str) -> int:
        return self._collection(collection_name).count()


# ── Pipeline — puts it all together ───────────────────────────────────────────

class ChunkAndEmbedPipeline:
    """
    Single entry point for Step 2.
    Usage:
        pipeline = ChunkAndEmbedPipeline()
        chunks = pipeline.process_document(parsed_doc, session_id="abc123")
    """

    def __init__(
        self,
        chunker:      Optional[LegalTextChunker] = None,
        embedder:     Optional[Embedder]          = None,
        vector_store: Optional[VectorStore]       = None,
    ):
        self.chunker      = chunker      or LegalTextChunker()
        self.embedder     = embedder     or Embedder()
        self.vector_store = vector_store or VectorStore()

    def process_document(
        self,
        parsed_doc,                      # ParsedDocument from Step 1
        session_id: Optional[str] = None,
        store: bool = True,
    ) -> list[TextChunk]:
        """
        Full pipeline:
          1. Chunk the clean text
          2. Embed all chunks in one batch (efficient)
          3. Store in ChromaDB documents collection
          Returns the list of TextChunk objects.
        """
        session_id = session_id or str(uuid.uuid4())

        # 1. Chunk
        chunks = self.chunker.chunk(
            parsed_doc.clean_text,
            metadata={
                "session_id": session_id,
                "doc_type":   parsed_doc.doc_type.value,
                "is_scanned": parsed_doc.is_scanned,
            },
        )

        if not chunks:
            return []

        # 2. Embed (batch — much faster than one-by-one)
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        # 3. Store
        if store:
            self.vector_store.add_chunks(
                chunks, embeddings,
                collection_name=VectorStore.DOCUMENTS_COLLECTION,
            )

        return chunks
