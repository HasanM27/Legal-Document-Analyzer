"""
Quick knowledge base test
Run with: python tests/test_kb_live.py

Tests:
  1. Builder loads all 10 sources without crashing
  2. Chunks are actually stored in ChromaDB
  3. Retriever returns relevant results for real queries
  4. Metadata (jurisdiction, category) is stored correctly
  5. Deduplication works — re-running doesn't double the chunks
"""

import sys
sys.path.insert(0, '.')

from ingestion.knowledge import (
    KnowledgeBaseBuilder,
    KnowledgeRetriever,
    STATIC_SOURCES,
)
from ingestion.chunker import VectorStore

passed = 0
failed = 0

def check(name, condition, detail=''):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}", f"  ({detail})" if detail else '')
        failed += 1

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── Step 1: Build the knowledge base ─────────────────────────────────────────
section("1. Building knowledge base (this may take a minute...)")

builder = KnowledgeBaseBuilder()
summary = builder.build(sources=STATIC_SOURCES, clear_existing=True)

check("All 10 sources processed",
      len(summary) == len(STATIC_SOURCES),
      f"got {len(summary)}")

ok_count = sum(1 for v in summary.values() if v['status'] == 'ok')
check("All sources returned ok status",
      ok_count == len(STATIC_SOURCES),
      f"{ok_count}/{len(STATIC_SOURCES)} ok")

total_chunks = sum(v.get('chunks', 0) for v in summary.values())
check(f"Reasonable chunk count (got {total_chunks})",
      total_chunks >= 30,
      "expected at least 30 chunks across 10 sources")

print(f"\n  Chunks per source:")
for sid, info in summary.items():
    status = info['status']
    chunks = info.get('chunks', 0)
    bar = '█' * min(chunks, 40)
    print(f"    {sid[:45]:45}  {chunks:3} chunks  {bar}")


# ── Step 2: Verify ChromaDB storage ──────────────────────────────────────────
section("2. Verifying ChromaDB storage")

store = VectorStore()
stored_count = store.count(VectorStore.KNOWLEDGE_COLLECTION)

check(f"Chunks in ChromaDB match built count ({stored_count})",
      stored_count == total_chunks,
      f"stored={stored_count}, built={total_chunks}")

check("Collection is non-empty",
      stored_count > 0)


# ── Step 3: Retrieval quality ─────────────────────────────────────────────────
section("3. Testing retrieval quality")

retriever = KnowledgeRetriever()

# Query 1 — eviction / tenancy
results = retriever.retrieve("eviction notice tenant 15 days pay rent", top_k=3)
check("Eviction query returns results", len(results) > 0)
if results:
    top = results[0]
    check("Top eviction result has good score (>0.4)",
          top['score'] > 0.4,
          f"score={top['score']}")
    check("Top eviction result is from tenancy source",
          top['doc_category'] in ('tenancy', 'general'),
          f"got category='{top['doc_category']}'")
    print(f"\n  Top result for 'eviction notice':")
    print(f"    Source : {top['title']}")
    print(f"    Score  : {top['score']}")
    print(f"    Preview: {top['text'][:120].strip()}...")

# Query 2 — employment termination
results = retriever.retrieve("termination notice 30 days employment contract", top_k=3)
check("Employment query returns results", len(results) > 0)
if results:
    top = results[0]
    check("Top employment result has good score (>0.4)",
          top['score'] > 0.4,
          f"score={top['score']}")
    print(f"\n  Top result for 'termination notice':")
    print(f"    Source : {top['title']}")
    print(f"    Score  : {top['score']}")
    print(f"    Preview: {top['text'][:120].strip()}...")

# Query 3 — consumer complaint
results = retriever.retrieve("defective product refund consumer rights complaint", top_k=3)
check("Consumer query returns results", len(results) > 0)
if results:
    top = results[0]
    print(f"\n  Top result for 'defective product refund':")
    print(f"    Source : {top['title']}")
    print(f"    Score  : {top['score']}")
    print(f"    Preview: {top['text'][:120].strip()}...")

# Query 4 — court summons
results = retriever.retrieve("court summons written statement 30 days respond", top_k=3)
check("Court summons query returns results", len(results) > 0)
if results:
    top = results[0]
    print(f"\n  Top result for 'court summons':")
    print(f"    Source : {top['title']}")
    print(f"    Score  : {top['score']}")
    print(f"    Preview: {top['text'][:120].strip()}...")


# ── Step 4: Metadata filtering ────────────────────────────────────────────────
section("4. Testing metadata filtering")

# Filter by jurisdiction
sindh_results = retriever.retrieve(
    "tenant rights eviction", top_k=5, jurisdiction="sindh"
)
check("Jurisdiction filter returns results", len(sindh_results) > 0)
check("All results are sindh jurisdiction",
      all(r['jurisdiction'] == 'sindh' for r in sindh_results),
      f"got: {[r['jurisdiction'] for r in sindh_results]}")

# Filter by category
emp_results = retriever.retrieve(
    "wages salary payment", top_k=5, doc_category="employment"
)
check("Category filter returns results", len(emp_results) > 0)
check("All results are employment category",
      all(r['doc_category'] == 'employment' for r in emp_results),
      f"got: {[r['doc_category'] for r in emp_results]}")


# ── Step 5: Deduplication ─────────────────────────────────────────────────────
section("5. Testing deduplication (re-running builder)")

summary2 = builder.build(sources=STATIC_SOURCES, clear_existing=False)
new_chunks = sum(v.get('chunks', 0) for v in summary2.values())
count_after = store.count(VectorStore.KNOWLEDGE_COLLECTION)

check("Re-run adds 0 new chunks (all skipped as unchanged)",
      new_chunks == 0,
      f"added {new_chunks} chunks unexpectedly")

check("Total count unchanged after re-run",
      count_after == stored_count,
      f"before={stored_count}, after={count_after}")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═'*55}")

if failed == 0:
    print("  Knowledge base is working correctly.")
    print("  Ready to move to Step 4 — RAG retrieval pipeline.")
else:
    print("  Some tests failed — check the output above.")

sys.exit(0 if failed == 0 else 1)
