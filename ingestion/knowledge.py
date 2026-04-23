"""
Step 3 — Legal Knowledge Base
===============================
Builds and manages the ChromaDB 'knowledge' collection.

This file is ONLY responsible for:
  - Maintaining the hand-picked STATIC_SOURCES
  - Calling scraper.py to get relevant scraped laws
  - Chunking, embedding, and storing everything in ChromaDB

It does NOT contain any scraping or PDF parsing logic — that lives in scraper.py.

Pipeline:
    STATIC_SOURCES  ──┐
                       ├──► chunk ──► embed ──► ChromaDB
    scraper.scrape() ──┘

Run once to build the KB, then periodically to refresh:
    python -m ingestion.knowledge                    # static + scraped
    python -m ingestion.knowledge --no-scrape        # static sources only
    python -m ingestion.knowledge --scrape-only      # scraped sources only
    python -m ingestion.knowledge --scrape-limit 20  # test with 20 PDFs
    python -m ingestion.knowledge --clear            # wipe and rebuild
    python -m ingestion.knowledge --stats            # print stats and exit
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from ingestion.chunker import LegalTextChunker, VectorStore, Embedder
from scraper import scrape, ScrapedLaw

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Source definition ─────────────────────────────────────────────────────────

@dataclass
class LegalSource:
    """Describes one legal document to load into the knowledge base."""
    source_id:    str
    title:        str
    jurisdiction: str           # "federal", "sindh", "punjab", etc.
    doc_category: str           # "tenancy", "employment", "consumer", etc.
    url:          Optional[str] = None
    static_text:  Optional[str] = None
    language:     str = "english"


# ── Hand-picked static sources ────────────────────────────────────────────────
#
# These are curated laws we always want in the KB regardless of scraping.
# static_text is the fallback if the URL is unreachable.

STATIC_SOURCES: list[LegalSource] = [

    LegalSource(
        source_id="constitution_of_pakistan_1973",
        title="Constitution of the Islamic Republic of Pakistan 1973",
        jurisdiction="federal",
        doc_category="general",
        url="https://sindhlaws.gov.pk/Constitution.pdf",
        static_text="""
CONSTITUTION OF THE ISLAMIC REPUBLIC OF PAKISTAN, 1973

ARTICLE 9 — SECURITY OF PERSON
No person shall be deprived of life or liberty save in accordance with law.

ARTICLE 10 — SAFEGUARDS AS TO ARREST AND DETENTION
A person who is arrested shall be informed of the grounds for arrest and
shall not be denied the right to consult and be defended by a legal
practitioner of his choice. Every person who is arrested and detained
shall be produced before a magistrate within a period of twenty-four hours.

ARTICLE 10-A — RIGHT TO FAIR TRIAL
For the determination of civil rights and obligations, every person shall
be entitled to a fair trial and due process.

ARTICLE 14 — INVIOLABILITY OF DIGNITY OF MAN
The dignity of man and, subject to law, the privacy of home shall be
inviolable. No person shall be subjected to torture for the purpose of
extracting evidence.

ARTICLE 23 — PROVISION AS TO PROPERTY
Every citizen shall have the right to acquire, hold and dispose of
property in any part of Pakistan, subject to the Constitution and any
reasonable restrictions imposed by law in the public interest.

ARTICLE 24 — PROTECTION OF PROPERTY RIGHTS
No person shall be deprived of his property save in accordance with law.

ARTICLE 25 — EQUALITY OF CITIZENS
All citizens are equal before law and are entitled to equal protection of law.

ARTICLE 37 — PROMOTION OF SOCIAL JUSTICE
The State shall ensure inexpensive and expeditious justice and just and
humane conditions of work.
        """,
    ),

    LegalSource(
        source_id="sindh_rented_premises_1979",
        title="Sindh Rented Premises Ordinance 1979",
        jurisdiction="sindh",
        doc_category="tenancy",
        url="https://sja.gos.pk/assets/Updated_Laws/Sindh%20Rented%20Premises%20Ordinance,1979.pdf",
        static_text="""
SINDH RENTED PREMISES ORDINANCE, 1979

SECTION 10 — EVICTION OF TENANT
No eviction order shall be made except on these grounds:
(a) The tenant has not paid rent due within fifteen days.
(b) The tenant has committed acts likely to impair the value of the premises.
(c) The tenant has sublet without the written consent of the landlord.
(d) The premises are required bona fide by the landlord for personal occupation.

SECTION 11 — NOTICE BEFORE EVICTION APPLICATION
Before filing for eviction on ground (a), the landlord shall give the tenant
fifteen days written notice to pay arrears. If the tenant pays within the
notice period, no eviction application shall lie on that ground.

SECTION 14 — APPEAL
Any person aggrieved by a Rent Controller order may, within thirty days,
prefer an appeal to the District Judge.

TENANT RIGHTS SUMMARY:
- Must receive 15 days written notice before landlord can apply for eviction.
- Paying arrears within the 15-day window stops the eviction on non-payment grounds.
- Landlord cannot evict without a Rent Controller order.
- Can appeal any order to the District Judge within 30 days.
        """,
    ),

    LegalSource(
        source_id="transfer_of_property_act_1882",
        title="Transfer of Property Act 1882",
        jurisdiction="federal",
        doc_category="tenancy",
        url="https://pakistancode.gov.pk/pdffiles/administrator77923ce792b475e339e1f46ba0442da3.pdf",
        static_text="""
TRANSFER OF PROPERTY ACT, 1882

SECTION 105 — LEASE DEFINED
A lease of immovable property is a transfer of a right to enjoy such property
for a certain time in consideration of rent.

SECTION 106 — DURATION OF CERTAIN LEASES
In the absence of a contract, a lease for non-agricultural purpose shall be
deemed a month-to-month lease, terminable by fifteen days notice.

SECTION 108 — RIGHTS AND LIABILITIES OF LESSOR AND LESSEE
The lessor is bound to disclose any material defect in the property.
The lessee may use the property as agreed and shall keep it in good repair.

SECTION 111 — DETERMINATION OF LEASE
A lease determines by efflux of time, surrender, merger, or forfeiture.

SECTION 114 — RELIEF AGAINST FORFEITURE
Where a lease has determined by forfeiture for non-payment of rent,
the court may give relief if the lessee pays all arrears and costs.
        """,
    ),

    LegalSource(
        source_id="industrial_relations_act_2012",
        title="Industrial Relations Act 2012",
        jurisdiction="federal",
        doc_category="employment",
        url="https://pakistancode.gov.pk/pdffiles/administrator964ce81cc171ed5dcd0960630e922422.pdf",
        static_text="""
INDUSTRIAL RELATIONS ACT, 2012

SECTION 25 — UNFAIR LABOUR PRACTICES BY EMPLOYER
No employer shall dismiss or retrench a worker for union activity or
discriminate in employment on account of union membership.

SECTION 33 — NOTICE OF TERMINATION
An employer shall not terminate the services of a permanent worker unless:
(a) A notice of thirty days has been given in writing, or
(b) The worker has been paid one month's wages in lieu of notice.
This does not apply to termination for misconduct proven through inquiry.

SECTION 46 — RETRENCHMENT COMPENSATION
When an employer retrenches any worker, compensation shall be paid equal
to thirty days average pay for every completed year of continuous service.

EMPLOYEE RIGHTS SUMMARY:
- Permanent employees must receive 30 days written notice before termination.
- Employer can pay one month salary instead of serving the notice period.
- Retrenchment: 30 days pay per completed year of service.
- Termination for misconduct requires a formal inquiry first.
        """,
    ),

    LegalSource(
        source_id="payment_of_wages_act_1936",
        title="Payment of Wages Act 1936",
        jurisdiction="federal",
        doc_category="employment",
        url="https://pakistancode.gov.pk/pdffiles/administrator8820e88efaf7eedabf5c1d8c73b3dee5.pdf",
        static_text="""
PAYMENT OF WAGES ACT, 1936

SECTION 3 — RESPONSIBILITY FOR PAYMENT
Wages must be paid before the seventh day after the last day of the wage
period for establishments with fewer than 1000 workers, or the tenth day
for larger establishments.

SECTION 7 — PERMITTED DEDUCTIONS
Deductions from wages may only be made for: fines, absence from duty,
damage attributable to the worker, house accommodation, income tax,
and provident fund contributions.

SECTION 8 — FINES
Total fines shall not exceed three percent of wages in any wage period.

SECTION 15 — CLAIMS FOR UNLAWFUL DEDUCTIONS
Any worker may apply for refund of any unlawful deduction within twelve
months of the deduction.

WORKER RIGHTS SUMMARY:
- Wages must be paid by the 7th of the following month.
- Salary cannot be withheld without a lawful reason.
- Fines cannot exceed 3% of wages per pay period.
- Unlawful deduction refunds can be claimed within 12 months.
        """,
    ),

    LegalSource(
        source_id="limitation_act_1908",
        title="Limitation Act 1908",
        jurisdiction="federal",
        doc_category="court_summons",
        url="https://pakistancode.gov.pk/pdffiles/administrator3294e35255f255ea96b3356091fb4844.pdf",
        static_text="""
LIMITATION ACT, 1908

The Limitation Act prescribes time periods within which civil suits must be filed.

KEY LIMITATION PERIODS:
- Suit for money due under a contract: 3 years from when money became due.
- Suit for possession of immovable property: 12 years.
- Suit for rent: 3 years from when the rent became due.
- Suit for wages: 3 years from when wages became due.
- Suit for wrongful dismissal compensation: 3 years from dismissal.
- Application to execute a court decree: 12 years.
- Application to set aside an ex-parte decree: 30 days from knowledge.
        """,
    ),

    LegalSource(
        source_id="civil_procedure_code_1908",
        title="Code of Civil Procedure 1908",
        jurisdiction="federal",
        doc_category="court_summons",
        url="https://pakistancode.gov.pk/pdffiles/administrator6598dabbad120033d4d42d717dcf9755.pdf",
        static_text="""
CODE OF CIVIL PROCEDURE, 1908

ORDER V — ISSUE AND SERVICE OF SUMMONS
A summons shall be served by delivering a copy to the defendant personally.
Where the defendant cannot be found, the summons may be served by leaving
a copy at his last known residence.

ORDER VIII — WRITTEN STATEMENT
The defendant shall present a written statement within thirty days from
service of summons. The court may extend this period up to ninety days.

ORDER IX — APPEARANCE OF PARTIES
Where the plaintiff appears but the defendant does not, the court may
proceed ex-parte. An ex-parte decree may be set aside on application
within 30 days of learning of the decree.

ORDER XXI — EXECUTION OF DECREES
Attachment of salary is limited to one-third of net monthly salary.

DEFENDANT RIGHTS SUMMARY:
- Must file written statement within 30 days (max 90 days with extension).
- Ex-parte decree can be set aside within 30 days of learning of it.
- Salary attachment limited to one-third of net monthly pay.
        """,
    ),

    LegalSource(
        source_id="consumer_protection_act_2014",
        title="Sindh Consumer Protection Act 2014",
        jurisdiction="sindh",
        doc_category="consumer",
        url="https://sja.gos.pk/assets/Updated_Laws/Sindh%20Consumer%20Protection%20Act%2C%202014.pdf",
        static_text="""
SINDH CONSUMER PROTECTION ACT, 2014

SECTION 13 AND 14 — RIGHT TO COMPENSATION
A consumer who suffers damage due to defective goods or deficient services
is entitled to: removal of defects, replacement of goods, refund of price,
or compensation for actual loss suffered.

SECTION 29 — COMPLAINT PROCEDURE
A consumer must send a 15-day legal notice before filing a complaint.
A complaint can be filed with the Consumer Court within 30 days of the
cause of action. No court fee is required.

SINDH CONSUMER RIGHTS SUMMARY:
- Right to replacement, refund, or compensation for defective goods.
- Right to a receipt for every purchase.
- Must send 15-day notice before filing Consumer Court complaint.
- Consumer Court shall aim to decide within 90 days.
        """,
    ),

    LegalSource(
        source_id="contract_act_1872",
        title="Contract Act 1872",
        jurisdiction="federal",
        doc_category="general",
        url="https://pakistancode.gov.pk/pdffiles/administrator8332a6df32386960ac7d81a5cf7aade2.pdf",
        static_text="""
CONTRACT ACT, 1872

SECTION 10 — WHAT AGREEMENTS ARE CONTRACTS
All agreements are contracts if made by free consent of parties competent
to contract, for a lawful consideration and lawful object.

SECTION 14 — FREE CONSENT
Consent is said to be free when it is not caused by coercion, undue
influence, fraud, misrepresentation, or mistake.

SECTION 19 — VOIDABILITY OF AGREEMENTS
When consent is caused by coercion, fraud, or misrepresentation, the
contract is voidable at the option of the aggrieved party.

SECTION 73 — COMPENSATION FOR LOSS FROM BREACH
When a contract is broken, the party who suffers the breach is entitled
to receive compensation for any loss that naturally arose from the breach.

CONTRACT RIGHTS SUMMARY:
- Contracts obtained by fraud or coercion are voidable.
- Breaching party must compensate for losses that naturally arise.
- Oral contracts are generally enforceable but harder to prove.
        """,
    ),

    LegalSource(
        source_id="specific_relief_act_1877",
        title="Specific Relief Act 1877",
        jurisdiction="federal",
        doc_category="general",
        url="https://pakistancode.gov.pk/pdffiles/administratorf257754bbb3c6863d879492bc8cd8f6e.pdf",
        static_text="""
SPECIFIC RELIEF ACT, 1877

SECTION 12 — WHEN SPECIFIC PERFORMANCE OF CONTRACT ENFORCED
Specific performance may be enforced where compensation in money would not
afford adequate relief. Commonly granted for contracts involving land.

SECTION 39 — CANCELLATION OF INSTRUMENTS
Any person against whom a written instrument is void or voidable may sue
to have it cancelled. Applies to fraudulent sale deeds and forged agreements.

SECTION 42 — DECLARATORY DECREES
Any person entitled to a right to any property may sue for a declaration
that he is so entitled, and the court may make a binding declaration.
        """,
    ),
]


# ── Text fetcher for static sources ──────────────────────────────────────────

class StaticSourceFetcher:
    """
    Fetches text for STATIC_SOURCES.
    If static_text is provided and the URL fails, uses static_text as fallback.
    """

    DELAY   = 1.5
    TIMEOUT = 30
    HEADERS = {"User-Agent": "LegalAssistant/1.0 (educational project)"}

    def fetch(self, source: LegalSource) -> str:
        if source.url:
            text = self._try_url(source.url)
            if text and len(text.strip()) > 200:
                log.info(f"  Fetched live: {source.title} ({len(text)} chars)")
                return text
            log.info(f"  URL failed, using static text: {source.title}")

        if source.static_text:
            return source.static_text.strip()

        raise ValueError(f"No text available for: {source.source_id}")

    def _try_url(self, url: str) -> Optional[str]:
        try:
            import requests
            from scraper import extract_pdf_text
            from bs4 import BeautifulSoup
            import time

            resp = requests.get(url, headers=self.HEADERS, timeout=self.TIMEOUT)
            resp.raise_for_status()
            time.sleep(self.DELAY)

            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                text, _, _ = extract_pdf_text(resp.content)
                return text

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n")

        except Exception as e:
            log.warning(f"  Could not fetch {url}: {e}")
            return None


# ── Knowledge base builder ────────────────────────────────────────────────────

class KnowledgeBaseBuilder:
    """
    Orchestrates the full pipeline:
      fetch/scrape → chunk → embed → store in ChromaDB 'knowledge' collection

    scraper.py handles: downloading PDFs, extracting text, filtering by category
    knowledge.py handles: chunking, embedding, storing
    """

    # Minimum cosine similarity score to accept a retrieval result.
    # ChromaDB returns distances (lower = more similar), so we filter
    # out weak matches to avoid noisy context reaching the LLM.
    RELEVANCE_THRESHOLD = 0.35

    def __init__(
        self,
        chunker:      Optional[LegalTextChunker] = None,
        embedder:     Optional[Embedder]          = None,
        vector_store: Optional[VectorStore]       = None,
        fetcher:      Optional[StaticSourceFetcher] = None,
    ):
        self.chunker      = chunker      or LegalTextChunker(chunk_size=600, chunk_overlap=80)
        self.embedder     = embedder     or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.fetcher      = fetcher      or StaticSourceFetcher()

    def build(
        self,
        clear_existing:  bool = False,
        include_scrape:  bool = True,
        scrape_limit:    Optional[int] = None,
        static_only:     bool = False,
        scrape_only:     bool = False,
    ) -> dict:
        """
        Build the knowledge base.

        Args:
            clear_existing: wipe ChromaDB collection before building
            include_scrape: call scraper.py to add scraped laws (default True)
            scrape_limit:   max PDFs to scrape (None = all)
            static_only:    only process STATIC_SOURCES, skip scraper
            scrape_only:    only process scraped laws, skip STATIC_SOURCES
        """
        if clear_existing:
            log.info("Clearing existing knowledge collection...")
            self._clear_knowledge()

        summary = {}

        # ── Step 1: Static sources ──
        if not scrape_only:
            log.info(f"\nIngesting {len(STATIC_SOURCES)} static sources...")
            for source in STATIC_SOURCES:
                log.info(f"  Processing: {source.title}")
                try:
                    text  = self.fetcher.fetch(source)
                    count = self._ingest(
                        source_id=source.source_id,
                        title=source.title,
                        jurisdiction=source.jurisdiction,
                        doc_category=source.doc_category,
                        language=source.language,
                        url=source.url or "",
                        text=text,
                    )
                    summary[source.source_id] = {"status": "ok", "chunks": count}
                    log.info(f"  Stored {count} chunks")
                except Exception as e:
                    log.error(f"  Failed: {e}")
                    summary[source.source_id] = {"status": "error", "error": str(e)}

        # ── Step 2: Scraped sources ──
        if include_scrape and not static_only:
            static_urls = {s.url for s in STATIC_SOURCES if s.url}
            log.info("\nScraping pakistancode.gov.pk...")
            scraped_laws = scrape(limit=scrape_limit, skip_urls=static_urls)

            log.info(f"\nIngesting {len(scraped_laws)} scraped laws...")
            for law in scraped_laws:
                log.info(f"  Processing: {law.title}")
                try:
                    count = self._ingest(
                        source_id=law.source_id,
                        title=law.title,
                        jurisdiction="federal",
                        doc_category=law.doc_category,
                        language=law.language,
                        url=law.url,
                        text=law.full_text,
                    )
                    summary[law.source_id] = {"status": "ok", "chunks": count}
                    log.info(f"  Stored {count} chunks")
                except Exception as e:
                    log.error(f"  Failed: {e}")
                    summary[law.source_id] = {"status": "error", "error": str(e)}

        total = sum(v.get("chunks", 0) for v in summary.values())
        log.info(f"\nKnowledge base built: {total} chunks across {len(summary)} sources")
        return summary

    def _ingest(
        self,
        source_id:    str,
        title:        str,
        jurisdiction: str,
        doc_category: str,
        language:     str,
        url:          str,
        text:         str,
    ) -> int:
        """Chunk, embed, and store one law. Returns number of chunks stored."""
        content_hash = hashlib.md5(text.encode()).hexdigest()

        if self._already_stored(source_id, content_hash):
            log.info(f"  Skipping (unchanged): {title}")
            return 0

        chunks = self.chunker.chunk(text, metadata={
            "source_id":    source_id,
            "title":        title,
            "jurisdiction": jurisdiction,
            "doc_category": doc_category,
            "language":     language,
            "content_hash": content_hash,
            "url":          url,
        })

        if not chunks:
            return 0

        embeddings = self.embedder.embed([c.text for c in chunks])
        self.vector_store.add_chunks(
            chunks, embeddings,
            collection_name=VectorStore.KNOWLEDGE_COLLECTION,
        )
        return len(chunks)

    def _already_stored(self, source_id: str, content_hash: str) -> bool:
        try:
            col = self.vector_store._collection(VectorStore.KNOWLEDGE_COLLECTION)
            results = col.get(
                where={"$and": [
                    {"source_id":    {"$eq": source_id}},
                    {"content_hash": {"$eq": content_hash}},
                ]},
                limit=1,
            )
            return len(results["ids"]) > 0
        except Exception:
            return False

    def _clear_knowledge(self):
        client = self.vector_store._get_client()
        try:
            client.delete_collection(VectorStore.KNOWLEDGE_COLLECTION)
            log.info("Knowledge collection cleared.")
        except Exception:
            pass

    def stats(self) -> dict:
        count = self.vector_store.count(VectorStore.KNOWLEDGE_COLLECTION)
        return {"total_chunks": count}


# ── Retriever — used by Step 4 ────────────────────────────────────────────────

class KnowledgeRetriever:
    """
    Query the knowledge base with relevance filtering.
    Only returns chunks that score above RELEVANCE_THRESHOLD so the LLM
    never receives weakly-matched context (e.g. Hajj laws for a rent dispute).
    """

    RELEVANCE_THRESHOLD = 0.35   # discard chunks with distance > this value

    def __init__(
        self,
        embedder:     Optional[Embedder]    = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.embedder     = embedder     or Embedder()
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query:        str,
        top_k:        int = 5,
        jurisdiction: Optional[str] = None,
        doc_category: Optional[str] = None,
        language:     Optional[str] = None,
    ) -> list[dict]:
        query_embedding = self.embedder.embed_one(query)
        where           = self._build_filter(jurisdiction, doc_category, language)

        raw = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            collection_name=VectorStore.KNOWLEDGE_COLLECTION,
            where=where,
        )

        # Filter out weak matches — these are irrelevant laws that happen to
        # share a few words with the query. Keeping them would pollute the
        # LLM's context and produce generic, inaccurate answers.
        results = []
        for r in raw:
            if r["score"] >= self.RELEVANCE_THRESHOLD:
                results.append({
                    "text":         r["text"],
                    "score":        r["score"],
                    "source_id":    r["metadata"].get("source_id", ""),
                    "title":        r["metadata"].get("title", ""),
                    "jurisdiction": r["metadata"].get("jurisdiction", ""),
                    "doc_category": r["metadata"].get("doc_category", ""),
                    "language":     r["metadata"].get("language", "english"),
                })

        if not results:
            log.info("No relevant law chunks found above threshold for this query.")

        return results

    def _build_filter(
        self,
        jurisdiction: Optional[str],
        doc_category: Optional[str],
        language:     Optional[str],
    ) -> Optional[dict]:
        filters = []
        if jurisdiction:
            filters.append({"jurisdiction": {"$eq": jurisdiction}})
        if doc_category:
            filters.append({"doc_category": {"$eq": doc_category}})
        if language:
            filters.append({"language": {"$eq": language}})

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the legal knowledge base")
    parser.add_argument("--clear",        action="store_true",  help="Clear existing KB before building")
    parser.add_argument("--stats",        action="store_true",  help="Print KB stats and exit")
    parser.add_argument("--no-scrape",    action="store_true",  help="Static sources only, skip scraper")
    parser.add_argument("--scrape-only",  action="store_true",  help="Scraped sources only, skip static")
    parser.add_argument("--scrape-limit", type=int, default=None, help="Max PDFs to scrape (for testing)")
    args = parser.parse_args()

    builder = KnowledgeBaseBuilder()

    if args.stats:
        print(f"\nKnowledge base stats: {builder.stats()}")
    else:
        summary = builder.build(
            clear_existing=args.clear,
            include_scrape=not args.no_scrape,
            scrape_limit=args.scrape_limit,
            static_only=args.no_scrape,
            scrape_only=args.scrape_only,
        )
        print("\nBuild summary:")
        for sid, info in summary.items():
            status = info["status"]
            chunks = info.get("chunks", 0)
            print(f"  {sid}: {status} ({chunks} chunks)")