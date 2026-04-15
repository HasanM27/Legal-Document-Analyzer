"""
Step 3 — Legal Knowledge Base
===============================
Fetches targeted Pakistani law PDFs from official government sources,
chunks them, embeds them, and loads them into the ChromaDB 'knowledge'
collection with rich metadata for jurisdiction-aware retrieval.

Run once to build the KB, then periodically to refresh it:
    python -m ingestion.knowledge

Primary sources (PDF URLs):
  - sindhlaws.gov.pk / sja.gos.pk   (Sindh provincial laws)
  - pakistancode.gov.pk              (Federal acts)

Each source has a static_text fallback so the app works offline
or when a government website is down.
"""

import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from ingestion.chunker import LegalTextChunker, VectorStore, Embedder

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Source definition ─────────────────────────────────────────────────────────

@dataclass
class LegalSource:
    """Describes one legal document to scrape or load."""
    source_id:    str
    title:        str
    jurisdiction: str          # "federal", "sindh", "punjab", etc.
    doc_category: str          # "tenancy", "employment", "consumer", etc.
    url:          Optional[str] = None
    static_text:  Optional[str] = None   # fallback if URL unreachable


# ── Legal sources ─────────────────────────────────────────────────────────────
#
# Priority order for each source:
#   1. Fetch full PDF from official government URL  (best — authoritative text)
#   2. Fall back to static_text below               (works offline / site down)
#
# To refresh from live URLs:  python -m ingestion.knowledge --clear

STATIC_SOURCES: list[LegalSource] = [

    # ── 1. Constitution of Pakistan 1973 ──────────────────────────────────────
    LegalSource(
        source_id="constitution_of_pakistan_1973",
        title="Constitution of the Islamic Republic of Pakistan 1973",
        jurisdiction="federal",
        doc_category="general",
        url="https://sindhlaws.gov.pk/Constitution.pdf",
        static_text="""
CONSTITUTION OF THE ISLAMIC REPUBLIC OF PAKISTAN, 1973

PART II — FUNDAMENTAL RIGHTS AND PRINCIPLES OF POLICY

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
No property shall be compulsorily acquired except for a public purpose,
and except by the authority of law which provides for compensation.

ARTICLE 25 — EQUALITY OF CITIZENS
All citizens are equal before law and are entitled to equal protection
of law. There shall be no discrimination on the basis of sex.

ARTICLE 37 — PROMOTION OF SOCIAL JUSTICE
The State shall ensure inexpensive and expeditious justice, make
provision for securing just and humane conditions of work, ensure that
children and women are not employed in vocations unsuited to their age
or sex, and make provision for securing the right to work.
        """,
    ),

    # ── 2. Sindh Rented Premises Ordinance 1979 ───────────────────────────────
    LegalSource(
        source_id="sindh_rented_premises_1979",
        title="Sindh Rented Premises Ordinance 1979",
        jurisdiction="sindh",
        doc_category="tenancy",
        url="https://sja.gos.pk/assets/Updated_Laws/Sindh%20Rented%20Premises%20Ordinance,1979.pdf",
        static_text="""
SINDH RENTED PREMISES ORDINANCE, 1979

SECTION 1 — SHORT TITLE AND COMMENCEMENT
This Ordinance may be called the Sindh Rented Premises Ordinance, 1979.
It extends to the whole of the Province of Sindh.

SECTION 4 — RENT CONTROLLER
The Government shall appoint a Rent Controller for each area. The Rent
Controller shall be a judicial officer not below the rank of Civil Judge.

SECTION 6 — FIXATION OF FAIR RENT
A tenant or landlord may apply to the Rent Controller for fixation of
fair rent. The Rent Controller shall fix fair rent after giving notice
to both parties and conducting such inquiry as it deems fit.

SECTION 10 — EVICTION OF TENANT
A landlord who seeks eviction shall apply to the Rent Controller.
No eviction order shall be made except on these grounds:
(a) The tenant has not paid rent due within fifteen days of it becoming payable.
(b) The tenant has committed acts likely to impair the value of the premises.
(c) The tenant has sublet without the written consent of the landlord.
(d) The premises are required bona fide by the landlord for personal occupation.
(e) The premises require repairs that cannot be done without vacation.

SECTION 11 — NOTICE BEFORE EVICTION APPLICATION
Before filing for eviction on ground (a) of Section 10, the landlord
shall give the tenant fifteen days written notice to pay arrears.
If the tenant pays within the notice period, no eviction application
shall lie on that ground.

SECTION 14 — APPEAL
Any person aggrieved by a Rent Controller order may, within thirty days,
prefer an appeal to the District Judge, who may confirm, modify or reverse.

TENANT RIGHTS SUMMARY:
- Must receive 15 days written notice before landlord can apply for eviction.
- Paying arrears within the 15-day window stops the eviction on non-payment grounds.
- Landlord cannot evict without a Rent Controller order.
- Can appeal any order to the District Judge within 30 days.
- Rent can only be increased through the Rent Controller process.
        """,
    ),

    # ── 3. Transfer of Property Act 1882 ──────────────────────────────────────
    LegalSource(
        source_id="transfer_of_property_act_1882",
        title="Transfer of Property Act 1882",
        jurisdiction="federal",
        doc_category="tenancy",
        url="https://pakistancode.gov.pk/pdffiles/administrator77923ce792b475e339e1f46ba0442da3.pdf",
        static_text="""
TRANSFER OF PROPERTY ACT, 1882

SECTION 105 — LEASE DEFINED
A lease of immovable property is a transfer of a right to enjoy such
property for a certain time, in consideration of rent. The transferor
is called the lessor, the transferee is called the lessee.

SECTION 106 — DURATION OF CERTAIN LEASES
In the absence of a contract or local law, a lease for any purpose other
than agriculture shall be deemed a lease from month to month, terminable
by fifteen days notice by either party.

SECTION 108 — RIGHTS AND LIABILITIES OF LESSOR AND LESSEE
The lessor is bound to disclose any material defect in the property.
The lessee may use the property as agreed and shall keep it in good repair.
On determination of the lease, the lessee shall restore possession to the lessor.

SECTION 111 — DETERMINATION OF LEASE
A lease determines by efflux of time, surrender, merger, or forfeiture
where the lessee breaks an express condition providing for re-entry.

SECTION 114 — RELIEF AGAINST FORFEITURE
Where a lease has determined by forfeiture for non-payment of rent,
the court may give relief against forfeiture if the lessee pays all
arrears and costs within such time as the court thinks fit.
        """,
    ),

    # ── 4. Industrial Relations Act 2012 ──────────────────────────────────────
    LegalSource(
        source_id="industrial_relations_act_2012",
        title="Industrial Relations Act 2012",
        jurisdiction="federal",
        doc_category="employment",
        url="https://pakistancode.gov.pk/pdffiles/administrator964ce81cc171ed5dcd0960630e922422.pdf",
        static_text="""
INDUSTRIAL RELATIONS ACT, 2012

SECTION 25 — UNFAIR LABOUR PRACTICES BY EMPLOYER
No employer shall restrain, coerce, or compel any worker in the exercise
of rights. No employer shall dismiss or retrench a worker for union
activity or discriminate in employment on account of union membership.

SECTION 33 — NOTICE OF TERMINATION
An employer shall not terminate the services of a permanent worker unless:
(a) A notice of thirty days has been given in writing, or
(b) The worker has been paid one month's wages in lieu of notice.
This does not apply to termination for misconduct proven through inquiry.

SECTION 46 — RETRENCHMENT COMPENSATION
When an employer retrenches any worker, compensation shall be paid equal
to thirty days average pay for every completed year of continuous service.

SECTION 58 — SETTLEMENT OF DISPUTES
Any industrial dispute may be reported to the Conciliator by either party.
The Conciliator shall try to bring about a settlement within thirty days.
If no settlement, the dispute may be referred to the NIRC.

EMPLOYEE RIGHTS SUMMARY:
- Permanent employees must receive 30 days written notice before termination.
- Employer can pay one month salary instead of serving the notice period.
- Retrenchment: 30 days pay per completed year of service.
- Cannot be dismissed for union membership or activity.
- Termination for misconduct requires a formal inquiry first.
        """,
    ),

    # ── 5. Payment of Wages Act 1936 ──────────────────────────────────────────
    LegalSource(
        source_id="payment_of_wages_act_1936",
        title="Payment of Wages Act 1936",
        jurisdiction="federal",
        doc_category="employment",
        url="https://pakistancode.gov.pk/pdffiles/administrator8820e88efaf7eedabf5c1d8c73b3dee5.pdf",
        static_text="""
PAYMENT OF WAGES ACT, 1936

SECTION 3 — RESPONSIBILITY FOR PAYMENT
Every employer is responsible for payment of all wages. Wages must be
paid before the seventh day after the last day of the wage period
for establishments with fewer than 1000 workers, or the tenth day
for larger establishments.

SECTION 7 — PERMITTED DEDUCTIONS
Deductions from wages may only be made for:
(a) Fines imposed under Section 8.
(b) Absence from duty.
(c) Damage or loss of goods directly attributable to the worker.
(d) House accommodation supplied by the employer.
(e) Income tax payable by the worker.
(f) Provident fund contributions.

SECTION 8 — FINES
No fine shall be imposed unless the employer has prior approval.
Total fines shall not exceed three percent of wages in any wage period.

SECTION 15 — CLAIMS FOR UNLAWFUL DEDUCTIONS
Any worker may apply for refund of any deduction made in contravention
of this Act within twelve months of the deduction.

WORKER RIGHTS SUMMARY:
- Wages must be paid by the 7th of the following month.
- Deductions strictly limited to categories listed in Section 7.
- Fines cannot exceed 3% of wages per pay period.
- Unlawful deduction refunds can be claimed within 12 months.
        """,
    ),

    # ── 6. Limitation Act 1908 ────────────────────────────────────────────────
    # NOTE: Use the correct PDF URL for the Limitation Act specifically.
    # The URL below is a placeholder — find the right one on pakistancode.gov.pk/pdffiles/
    LegalSource(
        source_id="limitation_act_1908",
        title="Limitation Act 1908",
        jurisdiction="federal",
        doc_category="general",
        url="https://pakistancode.gov.pk/pdffiles/administrator3294e35255f255ea96b3356091fb4844.pdf",
        static_text="""
LIMITATION ACT, 1908

PURPOSE
The Limitation Act prescribes time periods within which civil suits must
be filed. After expiry, a suit is time-barred and unenforceable in court.

KEY LIMITATION PERIODS:

MONEY AND DEBT SUITS:
- Suit for money due under a contract: 3 years from when money became due.
- Suit on a promissory note: 3 years from date of the note.
- Suit by a creditor to recover a debt: 3 years from when debt was payable.

PROPERTY AND TENANCY SUITS:
- Suit for possession of immovable property: 12 years.
- Suit for rent: 3 years from when the rent became due.
- Landlord recovering possession after lease determination: 12 years.

EMPLOYMENT SUITS:
- Suit for wages: 3 years from when the wages became due.
- Suit for wrongful dismissal compensation: 3 years from dismissal.

COURT DECREES:
- Application to execute a court decree: 12 years from enforceability.
- Application to set aside an ex-parte decree: 30 days from knowledge.

NOTE: The limitation period may be extended where the plaintiff was
under disability (minor or insane) or where the defendant fraudulently
concealed the facts giving rise to the cause of action.
        """,
    ),

    # ── 7. Civil Procedure Code 1908 ──────────────────────────────────────────
    # NOTE: This needs a DIFFERENT URL from the Limitation Act above.
    # Find the CPC PDF separately on pakistancode.gov.pk/pdffiles/
    LegalSource(
        source_id="civil_procedure_code_1908",
        title="Code of Civil Procedure 1908",
        jurisdiction="federal",
        doc_category="court_summons",
        url="https://pakistancode.gov.pk/pdffiles/administrator6598dabbad120033d4d42d717dcf9755.pdf",   # ← add the correct CPC PDF URL here once you find it
        static_text="""
CODE OF CIVIL PROCEDURE, 1908

ORDER V — ISSUE AND SERVICE OF SUMMONS
A summons shall be served by delivering a copy to the defendant personally
or to his agent. Where the defendant cannot be found, the summons may be
served by leaving a copy at his last known residence.

ORDER VIII — WRITTEN STATEMENT
The defendant shall present a written statement within thirty days from
service of summons. The court may extend this period up to ninety days
total for sufficient cause. The written statement must specifically admit
or deny each allegation. Any defence not raised shall be deemed waived.

ORDER IX — APPEARANCE OF PARTIES
Where the plaintiff appears but the defendant does not, the court may
proceed ex-parte. An ex-parte decree may be set aside on application
within 30 days of learning of the decree, on showing sufficient cause.

ORDER XXI — EXECUTION OF DECREES
A decree for payment of money may be executed by attachment and sale
of property. Attachment of salary is limited to one-third of net monthly
salary. Exempt from attachment: necessary wearing apparel, cooking
vessels, tools of trade, books of account.

DEFENDANT RIGHTS SUMMARY:
- Must file written statement within 30 days (max 90 days with extension).
- Ex-parte decree can be set aside within 30 days of learning of it.
- Right to legal representation at all hearings.
- Salary attachment limited to one-third of net monthly pay.
- Necessary household items and tools of trade exempt from attachment.
        """,
    ),

    # ── 8. Sindh Consumer Protection Act 2014 ────────────────────────────────
    LegalSource(
        source_id="consumer_protection_act_2014",
        title="Sindh Consumer Protection Act 2014",
        jurisdiction="sindh",
        doc_category="consumer",
        url="https://sja.gos.pk/assets/Updated_Laws/Sindh%20Consumer%20Protection%20Act%2C%202014.pdf",
        static_text="""
SINDH CONSUMER PROTECTION ACT, 2014

SECTION 2 — DEFINITIONS
Consumer: Any person who buys goods or hires services for personal,
domestic, or household purposes, excluding those who obtain goods for
resale or commercial purposes.
Defective Good: A product that does not meet the manufacturer's
specifications, is unsafe due to design, lacks adequate warnings,
or does not conform to an express warranty.
Deficient Service: Any fault, imperfection, or inadequacy in the quality
or nature of performance of a service which does not meet reasonable
expected standards.

SECTION 11, 16, 18, 19 — PROHIBITION OF UNFAIR TRADE PRACTICES
No person shall engage in unfair trade practices including:
(a) False representations about standard, quality, or grade of goods.
(b) False claims regarding price reductions or free items.
(c) Bait advertising — advertising goods not intended to be sold.
(d) Failing to provide receipts on demand.
(e) Failure to label products with ingredients, quality, or expiry dates.

SECTION 13 AND 14 — RIGHT TO COMPENSATION
A consumer who suffers damage due to defective goods or deficient
services is entitled to:
(a) Removal of defects or deficiencies.
(b) Replacement of the defective goods.
(c) Refund of the price paid.
(d) Compensation for any actual loss or damage suffered.

SECTION 29 — COMPLAINT PROCEDURE
A consumer must send a 15-day legal notice to the manufacturer or
service provider before filing a complaint. A complaint can be filed
with the Consumer Court within 30 days of the cause of action. No court
fee is required. The Consumer Court shall aim to decide within 90 days.

SINDH CONSUMER RIGHTS SUMMARY:
- Right to safety: protection against hazardous products (Section 4).
- Right to information: labeling, pricing, and ingredient disclosure.
- Right to redress: replacement, refund, or compensation for damages.
- Prohibition of misleading ads: false claims are illegal (Section 16).
- Mandatory receipt: right to a receipt for every purchase.
        """,
    ),

    # ── 9. Contract Act 1872 ──────────────────────────────────────────────────
    LegalSource(
        source_id="contract_act_1872",
        title="Contract Act 1872",
        jurisdiction="federal",
        doc_category="general",
        url="https://pakistancode.gov.pk/pdffiles/administrator8332a6df32386960ac7d81a5cf7aade2.pdf",
        static_text="""
CONTRACT ACT, 1872

SECTION 2 — DEFINITIONS
A proposal (offer) when accepted becomes a promise. A contract is an
agreement enforceable by law. An agreement is a promise or set of
promises forming consideration for each other.

SECTION 10 — WHAT AGREEMENTS ARE CONTRACTS
All agreements are contracts if they are made by free consent of parties
competent to contract, for a lawful consideration and lawful object,
and are not expressly declared void.

SECTION 14 — FREE CONSENT
Consent is said to be free when it is not caused by coercion, undue
influence, fraud, misrepresentation, or mistake.

SECTION 19 — VOIDABILITY OF AGREEMENTS
When consent is caused by coercion, fraud, or misrepresentation, the
contract is voidable at the option of the party whose consent was so
caused. The party may either affirm or rescind the contract.

SECTION 73 — COMPENSATION FOR LOSS FROM BREACH
When a contract is broken, the party who suffers the breach is entitled
to receive compensation for any loss or damage caused to him thereby
which naturally arose in the usual course of things, or which the
parties knew when they made the contract to be likely to result from
the breach.

SECTION 74 — COMPENSATION FOR BREACH WITH PENALTY CLAUSE
When a contract is broken and a sum is named as the amount to be paid
in case of breach, the party complaining of breach is entitled to
receive reasonable compensation not exceeding the amount so named.

SECTION 126 — CONTRACT OF GUARANTEE
A contract of guarantee is a contract to perform the promise or discharge
the liability of a third person in case of his default.

CONTRACT RIGHTS SUMMARY:
- A contract must have offer, acceptance, consideration, and free consent.
- Contracts obtained by fraud, coercion or misrepresentation are voidable.
- Breaching party must compensate for losses that naturally arise from breach.
- Penalty clauses are enforceable up to the amount named in the contract.
- Oral contracts are generally enforceable but harder to prove in court.
        """,
    ),

    # ── 10. Specific Relief Act 1877 ──────────────────────────────────────────
    LegalSource(
        source_id="specific_relief_act_1877",
        title="Specific Relief Act 1877",
        jurisdiction="federal",
        doc_category="general",
        url="https://pakistancode.gov.pk/pdffiles/administratorf257754bbb3c6863d879492bc8cd8f6e.pdf",
        static_text="""
SPECIFIC RELIEF ACT, 1877

SECTION 5 — RECOVERY OF SPECIFIC IMMOVABLE PROPERTY
A person entitled to the possession of specific immovable property may
recover it in the manner provided by the Code of Civil Procedure.

SECTION 8 — RECOVERY OF SPECIFIC MOVABLE PROPERTY
A person entitled to the possession of specific movable property may
recover it in a suit. The court may order delivery of the property or,
where delivery is not possible, award compensation.

SECTION 12 — WHEN SPECIFIC PERFORMANCE OF CONTRACT ENFORCED
Specific performance of a contract may be enforced where:
(a) There is no standard for ascertaining actual damages, or
(b) Compensation in money would not afford adequate relief.
Specific performance is commonly granted for contracts involving
immovable property (land, buildings) and unique goods.

SECTION 21 — POWER TO AWARD COMPENSATION
In a suit for specific performance, the court may award compensation
in addition to or instead of specific performance, if the contract
has become incapable of specific performance.

SECTION 39 — CANCELLATION OF INSTRUMENTS
Any person against whom a written instrument is void or voidable may
sue to have it adjudged void and to have it delivered up and cancelled.
This applies to fraudulent sale deeds, forged agreements, and
instruments obtained by misrepresentation.

SECTION 42 — DECLARATORY DECREES
Any person entitled to a legal character or a right to any property
may sue for a declaration that he is so entitled, and the court may
make a binding declaration of that right.

PRACTICAL USE CASES:
- Landlord refuses to hand over property after sale — sue for specific performance.
- Employer promises a role in writing then backtracks — specific performance or damages.
- A sale deed was obtained by fraud — apply for cancellation under Section 39.
- Dispute over ownership — apply for declaration under Section 42.
        """,
    ),
]


# ── Scraper ───────────────────────────────────────────────────────────────────

class StatuteScraper:
    """
    Attempts to fetch the full text of a legal source from its URL.
    Falls back gracefully to static_text if the URL is unreachable.
    """

    REQUEST_DELAY = 1.5   # seconds between requests — be polite to servers

    def fetch(self, source: LegalSource) -> str:
        """Return the best available text for this source."""
        if source.url:
            text = self._try_fetch(source.url)
            if text and len(text.strip()) > 200:
                log.info(f"  Fetched live: {source.title} ({len(text)} chars)")
                return text
            else:
                log.info(f"  URL unreachable, using static text: {source.title}")

        if source.static_text:
            return source.static_text.strip()

        raise ValueError(f"No text available for source: {source.source_id}")

    def _try_fetch(self, url: str) -> Optional[str]:
        try:
            import requests
            from bs4 import BeautifulSoup

            headers = {"User-Agent": "LegalAssistant/1.0 (educational project)"}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            time.sleep(self.REQUEST_DELAY)

            content_type = resp.headers.get("content-type", "")

            # PDF — extract text with pdfplumber
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                return self._extract_pdf(resp.content)

            # HTML — strip tags, keep text
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return soup.get_text(separator="\n")

        except Exception as e:
            log.warning(f"  Could not fetch {url}: {e}")
            return None

    def _extract_pdf(self, data: bytes) -> Optional[str]:
        try:
            import pdfplumber, io
            pages = []
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n\n".join(pages)
        except Exception as e:
            log.warning(f"  PDF extraction failed: {e}")
            return None


# ── Knowledge base builder ────────────────────────────────────────────────────

class KnowledgeBaseBuilder:
    """
    Orchestrates the full pipeline:
      fetch → chunk → embed → store in ChromaDB 'knowledge' collection
    """

    def __init__(
        self,
        chunker:      Optional[LegalTextChunker] = None,
        embedder:     Optional[Embedder]          = None,
        vector_store: Optional[VectorStore]       = None,
        scraper:      Optional[StatuteScraper]    = None,
    ):
        self.chunker      = chunker      or LegalTextChunker(chunk_size=600, chunk_overlap=80)
        self.embedder     = embedder     or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.scraper      = scraper      or StatuteScraper()

    def build(
        self,
        sources: list[LegalSource] = None,
        clear_existing: bool = False,
    ) -> dict:
        sources = sources or STATIC_SOURCES

        if clear_existing:
            log.info("Clearing existing knowledge collection...")
            self._clear_knowledge()

        summary = {}

        for source in sources:
            log.info(f"Processing: {source.title}")
            try:
                chunk_count = self._process_source(source)
                summary[source.source_id] = {"status": "ok", "chunks": chunk_count}
                log.info(f"  Stored {chunk_count} chunks")
            except Exception as e:
                log.error(f"  Failed: {e}")
                summary[source.source_id] = {"status": "error", "error": str(e)}

        total = sum(v.get("chunks", 0) for v in summary.values())
        log.info(f"\nKnowledge base built: {total} total chunks across {len(sources)} sources")
        return summary

    def _process_source(self, source: LegalSource) -> int:
        text = self.scraper.fetch(source)

        content_hash = hashlib.md5(text.encode()).hexdigest()
        if self._already_stored(source.source_id, content_hash):
            log.info(f"  Skipping (unchanged): {source.title}")
            return 0

        chunks = self.chunker.chunk(text, metadata={
            "source_id":    source.source_id,
            "title":        source.title,
            "jurisdiction": source.jurisdiction,
            "doc_category": source.doc_category,
            "content_hash": content_hash,
            "url":          source.url or "",
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
                    {"source_id": {"$eq": source_id}},
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
    Query the knowledge base.
    This is the interface Step 4 (RAG pipeline) will call.
    """

    def __init__(
        self,
        embedder:     Optional[Embedder]    = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.embedder     = embedder     or Embedder()
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        jurisdiction: Optional[str] = None,
        doc_category: Optional[str] = None,
    ) -> list[dict]:
        query_embedding = self.embedder.embed_one(query)
        where = self._build_filter(jurisdiction, doc_category)

        raw = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            collection_name=VectorStore.KNOWLEDGE_COLLECTION,
            where=where,
        )

        return [
            {
                "text":         r["text"],
                "score":        r["score"],
                "source_id":    r["metadata"].get("source_id", ""),
                "title":        r["metadata"].get("title", ""),
                "jurisdiction": r["metadata"].get("jurisdiction", ""),
                "doc_category": r["metadata"].get("doc_category", ""),
            }
            for r in raw
        ]

    def _build_filter(
        self,
        jurisdiction: Optional[str],
        doc_category: Optional[str],
    ) -> Optional[dict]:
        filters = []
        if jurisdiction:
            filters.append({"jurisdiction": {"$eq": jurisdiction}})
        if doc_category:
            filters.append({"doc_category": {"$eq": doc_category}})

        if len(filters) == 0:
            return None
        if len(filters) == 1:
            return filters[0]
        return {"$and": filters}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the legal knowledge base")
    parser.add_argument("--clear", action="store_true", help="Clear existing KB before building")
    parser.add_argument("--stats", action="store_true", help="Print KB stats and exit")
    args = parser.parse_args()

    builder = KnowledgeBaseBuilder()

    if args.stats:
        stats = builder.stats()
        print(f"\nKnowledge base stats: {stats}")
    else:
        summary = builder.build(clear_existing=args.clear)
        print("\nBuild summary:")
        for sid, info in summary.items():
            status = info["status"]
            chunks = info.get("chunks", 0)
            print(f"  {sid}: {status} ({chunks} chunks)")