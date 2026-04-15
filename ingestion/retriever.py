"""
Step 4 — RAG Retrieval Pipeline
=================================
Takes a ParsedDocument from Step 1, extracts key facts from it,
queries the knowledge base (Step 3), and returns a RAGContext object
ready to be handed to the LLM in Step 5.

Flow:
  ParsedDocument
      → FactExtractor        extract doc_type, jurisdiction, key claims
      → QueryBuilder         turn facts into targeted KB queries
      → KnowledgeRetriever   fetch top-k relevant legal chunks
      → ContextAssembler     rank, deduplicate, format into RAGContext
      → RAGContext            ready for LLM prompt assembly
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from ingestion.parser import ParsedDocument, DocumentType
from ingestion.knowledge import KnowledgeRetriever
from ingestion.chunker import Embedder, VectorStore

log = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtractedFacts:
    """Key facts pulled from the user's document."""
    doc_type:        str
    jurisdiction:    str                    # "sindh", "federal", "unknown"
    monetary_amounts: list[str]             # ["PKR 45,000", "Rs. 5,000"]
    deadlines:       list[str]              # ["14 days", "30 days"]
    parties:         list[str]              # ["landlord", "tenant"]
    key_phrases:     list[str]              # notable legal phrases found
    raw_queries:     list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """One retrieved legal knowledge chunk with its relevance score."""
    text:         str
    score:        float
    source_id:    str
    title:        str
    jurisdiction: str
    doc_category: str


@dataclass
class RAGContext:
    """
    Final output of the RAG pipeline — everything the LLM needs.
    Passed directly into the prompt builder in Step 5.
    """
    document_text:    str                   # cleaned user document
    doc_type:         str
    jurisdiction:     str
    facts:            ExtractedFacts
    retrieved_chunks: list[RetrievedChunk]  # ranked, deduplicated
    queries_used:     list[str]             # for debugging / logging


# ── Fact extractor ────────────────────────────────────────────────────────────

class FactExtractor:
    """
    Extracts structured facts from the parsed document.
    Uses regex patterns — no LLM needed here, keeping it fast and cheap.
    The LLM gets the raw document anyway so it can find anything we miss.
    """

    # Jurisdiction signals
    _JURISDICTION_PATTERNS = {
        "sindh":    [r"\bsindh\b", r"\bkarachi\b", r"\bhyderabad\b",
                     r"\bsukkur\b", r"\blarkana\b"],
        "punjab":   [r"\bpunjab\b", r"\blahore\b", r"\bfaisalabad\b",
                     r"\bmultan\b", r"\brawalpindi\b"],
        "kpk":      [r"\bkpk\b", r"\bkhyber\b", r"\bpeshawar\b"],
        "balochistan": [r"\bbalochistan\b", r"\bquetta\b"],
        "federal":  [r"\bislamabad\b", r"\bfederal\b"],
    }

    # Money patterns — PKR, Rs., rupees
    _MONEY_RE = re.compile(
        r"(?:PKR|Rs\.?|Rupees?)\s*[\d,]+(?:\.\d{1,2})?|"
        r"[\d,]+\s*(?:PKR|Rs\.?|rupees?)",
        re.IGNORECASE,
    )

    # Deadline patterns — "X days", "X months", "X years"
    _DEADLINE_RE = re.compile(
        r"\b(\d+)\s*(days?|months?|years?|hours?|weeks?)\b",
        re.IGNORECASE,
    )

    # Legal parties
    _PARTY_PATTERNS = [
        "landlord", "tenant", "lessee", "lessor",
        "employer", "employee", "worker",
        "plaintiff", "defendant", "creditor", "debtor",
        "buyer", "seller", "vendor", "purchaser",
        "consumer", "manufacturer", "contractor",
    ]

    # High-value legal phrases worth extracting
    _KEY_PHRASE_PATTERNS = [
        r"notice to (?:quit|vacate|pay)",
        r"eviction\s+(?:notice|proceedings?|order)",
        r"termination\s+(?:notice|letter|order)",
        r"breach\s+of\s+contract",
        r"wrongful\s+dismissal",
        r"unpaid\s+(?:rent|wages?|salary)",
        r"outstanding\s+(?:balance|amount|dues?)",
        r"legal\s+(?:notice|proceedings?|action)",
        r"court\s+(?:summons?|order|decree)",
        r"security\s+deposit",
        r"arrears?\s+of\s+rent",
        r"in\s+lieu\s+of\s+notice",
        r"retrenchment\s+compensation",
        r"defective\s+(?:goods?|product)",
        r"refund\s+(?:of\s+)?(?:price|amount|payment)",
    ]

    def extract(self, doc: ParsedDocument) -> ExtractedFacts:
        text = doc.clean_text
        lower = text.lower()

        return ExtractedFacts(
            doc_type=doc.doc_type.value,
            jurisdiction=self._detect_jurisdiction(lower),
            monetary_amounts=self._extract_money(text),
            deadlines=self._extract_deadlines(text),
            parties=self._extract_parties(lower),
            key_phrases=self._extract_key_phrases(lower),
        )

    def _detect_jurisdiction(self, lower: str) -> str:
        scores = {}
        for jurisdiction, patterns in self._JURISDICTION_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, lower))
            if score > 0:
                scores[jurisdiction] = score
        if not scores:
            return "federal"   # default to federal if no signal
        return max(scores, key=lambda k: scores[k])

    def _extract_money(self, text: str) -> list[str]:
        matches = self._MONEY_RE.findall(text)
        # Deduplicate while preserving order
        seen = set()
        result = []
        for m in matches:
            cleaned = m.strip()
            if cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result[:5]   # cap at 5

    def _extract_deadlines(self, text: str) -> list[str]:
        matches = self._DEADLINE_RE.findall(text)
        seen = set()
        result = []
        for num, unit in matches:
            deadline = f"{num} {unit}"
            if deadline not in seen:
                seen.add(deadline)
                result.append(deadline)
        return result[:5]

    def _extract_parties(self, lower: str) -> list[str]:
        return [p for p in self._PARTY_PATTERNS if p in lower]

    def _extract_key_phrases(self, lower: str) -> list[str]:
        found = []
        for pattern in self._KEY_PHRASE_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                found.append(match.group(0))
        return found


# ── Query builder ─────────────────────────────────────────────────────────────

class QueryBuilder:
    """
    Turns extracted facts into targeted queries for the knowledge base.

    Why multiple queries?
    A single query like "eviction notice" misses related concepts like
    "tenant rights", "rent arrears", "Rent Controller". Multiple focused
    queries cast a wider net and surface more relevant chunks.
    """

    # Maps document type → base queries that always apply
    _BASE_QUERIES: dict[str, list[str]] = {
        "eviction_notice": [
            "Sindh Rented Premises Ordinance eviction tenant notice",
            "landlord eviction grounds Rent Controller fifteen days",
            "tenant arrears of rent notice to quit rights",
            "eviction order Rent Controller application grounds",
        ],
        "tenancy_agreement": [
            "tenancy agreement rights obligations landlord tenant",
            "lease determination forfeiture security deposit",
            "rent increase fixation fair rent",
        ],
        "employment_contract": [
            "employment contract termination notice rights",
            "wrongful dismissal retrenchment compensation",
            "employer obligations wages unfair labour practices",
        ],
        "debt_collection": [
            "debt collection notice rights debtor creditor",
            "limitation period debt recovery civil suit",
            "court decree attachment salary property",
        ],
        "court_summons": [
            "court summons written statement response deadline",
            "civil procedure defendant rights appearance",
            "ex-parte decree set aside appeal",
        ],
        "government_letter": [
            "government notice official letter rights obligations",
            "fundamental rights due process constitutional",
        ],
        "unknown": [
            "legal rights obligations Pakistan civil law",
            "contract breach compensation remedy",
        ],
    }

    def build(self, facts: ExtractedFacts) -> list[str]:
        """Generate a list of queries to run against the knowledge base."""
        queries = []

        # 1. Base queries for this document type
        base = self._BASE_QUERIES.get(
            facts.doc_type,
            self._BASE_QUERIES["unknown"]
        )
        queries.extend(base)

        # 2. Enrich with extracted deadlines if found
        if facts.deadlines:
            deadline_str = " ".join(facts.deadlines[:2])
            queries.append(f"{facts.doc_type} {deadline_str} notice period rights")

        # 3. Enrich with key phrases
        for phrase in facts.key_phrases[:2]:
            queries.append(f"{phrase} rights Pakistan law")

        # 4. Monetary context query
        if facts.monetary_amounts:
            queries.append(
                f"{facts.doc_type} payment compensation remedy Pakistan"
            )

        # 5. Party-specific query
        if facts.parties:
            parties_str = " ".join(facts.parties[:2])
            queries.append(f"{parties_str} rights obligations Pakistan")

        return queries


# ── Context assembler ─────────────────────────────────────────────────────────

class ContextAssembler:
    """
    Takes raw results from multiple retrieval queries,
    deduplicates them, re-ranks by score, and caps at max_chunks.
    """

    def __init__(self, max_chunks: int = 8, min_score: float = 0.3):
        """
        max_chunks: maximum chunks to include in the final context.
                    8 is enough for a strong answer without overwhelming
                    the LLM context window.
        min_score:  discard chunks below this similarity threshold.
                    0.3 filters obvious noise while keeping marginal hits.
        """
        self.max_chunks = max_chunks
        self.min_score  = min_score

    def assemble(self, raw_results: list[dict]) -> list[RetrievedChunk]:
        """
        Deduplicate by source_id+text, filter by score, sort, cap.
        """
        seen_texts = set()
        unique = []

        for r in raw_results:
            # Deduplicate — same text from multiple queries
            text_key = r["text"][:100]   # first 100 chars as fingerprint
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            if r["score"] < self.min_score:
                continue

            unique.append(RetrievedChunk(
                text=r["text"],
                score=r["score"],
                source_id=r["source_id"],
                title=r["title"],
                jurisdiction=r["jurisdiction"],
                doc_category=r["doc_category"],
            ))

        # Sort by score descending, cap at max_chunks
        unique.sort(key=lambda c: c.score, reverse=True)
        return unique[:self.max_chunks]


# ── RAG Pipeline — main entry point ──────────────────────────────────────────

class RAGPipeline:
    """
    The single entry point for Step 4.

    Usage:
        pipeline = RAGPipeline()
        context  = pipeline.run(parsed_doc)
        # context.retrieved_chunks → hand to LLM in Step 5
    """

    def __init__(
        self,
        retriever:  Optional[KnowledgeRetriever] = None,
        extractor:  Optional[FactExtractor]      = None,
        qbuilder:   Optional[QueryBuilder]       = None,
        assembler:  Optional[ContextAssembler]   = None,
        top_k:      int = 5,     # chunks per query
    ):
        self.retriever = retriever or KnowledgeRetriever()
        self.extractor = extractor or FactExtractor()
        self.qbuilder  = qbuilder  or QueryBuilder()
        self.assembler = assembler or ContextAssembler()
        self.top_k     = top_k

    def run(self, doc: ParsedDocument) -> RAGContext:
        """
        Full pipeline:
          1. Extract facts from document
          2. Build targeted queries
          3. Retrieve relevant legal chunks for each query
          4. Assemble deduplicated, ranked context
        """
        log.info(f"RAG pipeline: doc_type={doc.doc_type.value}")

        # 1. Extract facts
        facts = self.extractor.extract(doc)
        log.info(f"  Facts: jurisdiction={facts.jurisdiction}, "
                 f"parties={facts.parties}, "
                 f"deadlines={facts.deadlines}")

        # 2. Build queries
        queries = self.qbuilder.build(facts)
        facts.raw_queries = queries
        log.info(f"  Running {len(queries)} queries against knowledge base")

        # 3. Retrieve — run all queries, collect all results
        all_results = []
        for query in queries:
            results = self.retriever.retrieve(
                query=query,
                top_k=self.top_k,
                jurisdiction=self._jurisdiction_filter(facts.jurisdiction),
                doc_category=self._category_filter(facts.doc_type),
            )
            all_results.extend(results)

        # 4. Assemble final context
        chunks = self.assembler.assemble(all_results)
        log.info(f"  Retrieved {len(chunks)} unique chunks "
                 f"(from {len(all_results)} raw results)")

        return RAGContext(
            document_text=doc.clean_text,
            doc_type=doc.doc_type.value,
            jurisdiction=facts.jurisdiction,
            facts=facts,
            retrieved_chunks=chunks,
            queries_used=queries,
        )

    def _jurisdiction_filter(self, jurisdiction: str) -> Optional[str]:
        """
        Only filter by jurisdiction when we're confident.
        'federal' is broad enough that filtering would miss relevant
        provincial laws, so we skip the filter and let ranking decide.
        """
        if jurisdiction in ("sindh", "punjab", "kpk", "balochistan"):
            return None   # include both provincial and federal results
        return None       # for federal/unknown — no filter, cast wide net

    def _category_filter(self, doc_type: str) -> Optional[str]:
        """
        Map document type to knowledge base category for pre-filtering.
        Only filter when the mapping is unambiguous.
        """
        mapping = {
            "eviction_notice":     None,   # tenancy + general both relevant
            "tenancy_agreement":   None,
            "employment_contract": None,
            "debt_collection":     None,
            "court_summons":       None,
            "government_letter":   None,
            "unknown":             None,
        }
        return mapping.get(doc_type)