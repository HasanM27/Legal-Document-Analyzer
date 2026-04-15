"""
Step 6 — Streamlit Frontend
=============================
Run with: streamlit run app.py
"""

import os
import sys
import logging
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Haqdar — Pakistani Legal Document Assistant",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0f0e0c;
    color: #e8e4dc;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1100px; }

/* ── Hero header ── */
.hero {
    border-bottom: 1px solid #2a2820;
    padding-bottom: 2rem;
    margin-bottom: 2.5rem;
}
.hero-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #e8e4dc;
    line-height: 1;
    margin: 0;
}
.hero-wordmark span {
    color: #c8a96e;
}
.hero-tagline {
    font-size: 0.9rem;
    color: #6b6760;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* ── Upload zone ── */
.upload-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6760;
    margin-bottom: 0.5rem;
}
[data-testid="stFileUploader"] {
    background: #1a1916;
    border: 1px dashed #2e2c28;
    border-radius: 6px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #c8a96e;
}

/* ── Textarea ── */
[data-testid="stTextArea"] textarea {
    background: #1a1916 !important;
    border: 1px solid #2e2c28 !important;
    border-radius: 6px !important;
    color: #e8e4dc !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    resize: vertical;
}

/* ── Primary button ── */
.stButton > button {
    background: #c8a96e !important;
    color: #0f0e0c !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    transition: background 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #dbbf82 !important;
}

/* ── Urgency banner ── */
.urgency-banner {
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 500;
    font-size: 0.9rem;
}
.urgency-critical { background: #2d1414; border: 1px solid #7a2020; color: #f4a4a4; }
.urgency-high     { background: #2d1f0e; border: 1px solid #7a4a10; color: #f4c47a; }
.urgency-medium   { background: #1a1f0e; border: 1px solid #4a5a10; color: #c4d47a; }
.urgency-low      { background: #0e1a1f; border: 1px solid #104a5a; color: #7ac4d4; }

/* ── Analysis cards ── */
.analysis-card {
    background: #1a1916;
    border: 1px solid #2e2c28;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
}
.card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c8a96e;
    margin-bottom: 0.75rem;
    font-weight: 500;
}
.summary-text {
    font-size: 1rem;
    line-height: 1.75;
    color: #d4d0c8;
    font-weight: 300;
}

/* ── Rights list ── */
.right-item {
    display: flex;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #222018;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #c8c4bc;
    align-items: flex-start;
}
.right-item:last-child { border-bottom: none; }
.right-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #c8a96e;
    margin-top: 0.45rem;
    flex-shrink: 0;
}

/* ── Action steps ── */
.step-item {
    display: flex;
    gap: 1rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid #222018;
    align-items: flex-start;
}
.step-item:last-child { border-bottom: none; }
.step-num {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: #252320;
    border: 1px solid #3a3830;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    color: #c8a96e;
    flex-shrink: 0;
}
.step-body { flex: 1; }
.step-instruction {
    font-size: 0.88rem;
    line-height: 1.6;
    color: #c8c4bc;
}
.step-deadline {
    display: inline-block;
    margin-top: 0.35rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    background: #2d2008;
    color: #c8a96e;
    border: 1px solid #4a3810;
    border-radius: 3px;
    padding: 0.15rem 0.5rem;
    font-weight: 500;
}

/* ── Metadata row ── */
.meta-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.meta-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    padding: 0.3rem 0.75rem;
    border-radius: 3px;
    border: 1px solid #2e2c28;
    background: #1a1916;
    color: #8a8680;
    text-transform: uppercase;
}
.meta-pill.highlight {
    border-color: #4a3810;
    background: #2d2008;
    color: #c8a96e;
}

/* ── Sources ── */
.source-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 0.25rem 0.6rem;
    border: 1px solid #2e2c28;
    border-radius: 3px;
    color: #6b6760;
    margin: 0.2rem 0.2rem 0 0;
}

/* ── Disclaimer ── */
.disclaimer {
    margin-top: 2rem;
    padding: 1rem 1.25rem;
    border-left: 3px solid #2e2c28;
    font-size: 0.78rem;
    color: #4a4840;
    line-height: 1.7;
    font-style: italic;
}

/* ── Confidence badge ── */
.conf-high   { color: #7ac4a4; }
.conf-medium { color: #c4c47a; }
.conf-low    { color: #c47a7a; }

/* ── Divider ── */
hr { border-color: #2a2820; margin: 2rem 0; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #c8a96e !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper: render HTML safely ────────────────────────────────────────────────

def h(html: str):
    st.markdown(html, unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "processing" not in st.session_state:
    st.session_state.processing = False


# ── Header ────────────────────────────────────────────────────────────────────

h("""
<div class="hero">
  <p class="hero-wordmark">Haq<span>dar</span></p>
  <p class="hero-tagline">Pakistani Legal Document Assistant &nbsp;·&nbsp; Know Your Rights</p>
</div>
""")


# ── Layout: two columns ───────────────────────────────────────────────────────

col_input, col_output = st.columns([1, 1.4], gap="large")


# ── LEFT COLUMN — input ───────────────────────────────────────────────────────

with col_input:
    h('<p class="upload-label">Upload Document</p>')
    uploaded_file = st.file_uploader(
        label="upload",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        label_visibility="collapsed",
        help="Supports PDF, images (JPG/PNG), and plain text files",
        key="doc_uploader",
    )

    h('<p class="upload-label" style="margin-top:1.25rem">Or Paste Text</p>')
    pasted_text = st.text_area(
        label="paste",
        label_visibility="collapsed",
        placeholder="Paste the text of your legal document here...",
        height=220,
        key="doc_text",
    )

    # Jurisdiction hint
    h('<p class="upload-label" style="margin-top:1.25rem">Province (optional)</p>')
    jurisdiction_hint = st.selectbox(
        label="jurisdiction",
        label_visibility="collapsed",
        options=["Auto-detect", "Sindh", "Punjab", "KPK", "Balochistan", "Federal / Islamabad"],
        index=0,
        key="jurisdiction_select",
    )

    analyse_clicked = st.button("Analyse Document", key="btn_analyse")

    # Example documents
    with st.expander("Try an example"):
        example = st.selectbox(
            "Pick an example:",
            ["Eviction notice", "Employment termination", "Debt collection letter", "Court summons"],
            label_visibility="collapsed",
            key="example_select",
        )
        if st.button("Load example", key="btn_load_example"):
            examples = {
                "Eviction notice": """NOTICE TO PAY RENT OR QUIT

To: Muhammad Ali
Address: Flat 5, Block C, Gulshan-e-Iqbal, Karachi, Sindh

You are hereby notified that you are in arrears of rent amounting
to PKR 55,000 for the months of January and February 2025.

You are required to pay the said amount within 14 days of this notice,
failing which legal proceedings will be initiated against you under the
Sindh Rented Premises Ordinance 1979.

Issued by: Hassan Properties
Date: 1st March 2025""",

                "Employment termination": """TERMINATION LETTER

Dear Ahmed Khan,

This letter serves as formal notice that your employment with
Malik & Sons (Pvt) Ltd is terminated effective 31st March 2025.

Your last working day will be today. You will be paid your
remaining salary for March only. No notice period compensation
will be provided as you are terminated for poor performance.

HR Department
Malik & Sons (Pvt) Ltd
Lahore, Punjab""",

                "Debt collection letter": """LEGAL NOTICE FOR RECOVERY OF DEBT

To: Sara Baig
Address: House 12, Street 4, F-7, Islamabad

TAKE NOTICE that you owe the sum of Rs. 380,000 to our client
Atlas Finance Company being the outstanding loan amount as of
this date.

You are required to pay the full outstanding amount within
30 days of receipt of this notice, failing which our client
shall be constrained to initiate legal proceedings against you
without further notice.

Advocate Usman Khan
Islamabad""",

                "Court summons": """IN THE COURT OF CIVIL JUDGE, KARACHI

SUMMONS

Suit No. 245/2025

You are hereby summoned to appear before this court on
15th April 2025 in the above-mentioned suit filed against you
by Mr. Tariq Hussain for recovery of Rs. 120,000.

You are required to file your written statement within 30 days
of service of this summons.

Failure to appear will result in an ex-parte decree being
passed against you.

Civil Judge, Karachi
Dated: 10th March 2025"""
            }
            st.session_state["example_text"] = examples[example]
            st.rerun()

    # Load example into text area
    if "example_text" in st.session_state:
        pasted_text = st.session_state["example_text"]
        del st.session_state["example_text"]


# ── Analysis logic ────────────────────────────────────────────────────────────

def run_analysis(text: str, jurisdiction_hint: str):
    """Run the full pipeline: parse → RAG → LLM → analysis."""
    from ingestion.parser import DocumentParser
    from ingestion.retriever import RAGPipeline
    from generation.generator import LegalAnalysisGenerator
    from evaluation.evaluator import EvaluationAndSafetyPipeline

    # Override jurisdiction if user specified one
    jurisdiction_map = {
        "Sindh": "sindh",
        "Punjab": "punjab",
        "KPK": "kpk",
        "Balochistan": "balochistan",
        "Federal / Islamabad": "federal",
        "Auto-detect": None,
    }
    forced_jurisdiction = jurisdiction_map.get(jurisdiction_hint)

    # Step 1: Parse
    parser = DocumentParser()
    parsed = parser.parse_text(text)

    # Step 2-4: RAG pipeline
    rag = RAGPipeline()
    context = rag.run(parsed)

    # Override jurisdiction if user told us
    if forced_jurisdiction:
        context.jurisdiction = forced_jurisdiction
        context.facts.jurisdiction = forced_jurisdiction

    # Step 5: LLM generation
    generator = LegalAnalysisGenerator()
    analysis = generator.generate(context)

    # Step 7: Evaluation & safety
    safety_pipeline = EvaluationAndSafetyPipeline()
    safe = safety_pipeline.run(analysis, context)

    return safe.analysis, context, safe


# ── Trigger analysis ──────────────────────────────────────────────────────────

if analyse_clicked:
    # Determine input source
    input_text = ""

    if uploaded_file:
        with st.spinner("Reading document..."):
            from ingestion.parser import DocumentParser
            parser = DocumentParser()
            doc = parser.parse_bytes(uploaded_file.read(), uploaded_file.name)
            input_text = doc.clean_text

    elif pasted_text.strip():
        input_text = pasted_text.strip()

    if not input_text:
        with col_output:
            st.warning("Please upload a document or paste some text first.")
    else:
        with col_output:
            with st.spinner("Analysing your document..."):
                try:
                    analysis, context, safe_result = run_analysis(input_text, jurisdiction_hint)
                    st.session_state.analysis = analysis
                    st.session_state.context = context
                    st.session_state.safe = safe_result
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.session_state.analysis = None


# ── RIGHT COLUMN — output ─────────────────────────────────────────────────────

with col_output:

    if st.session_state.analysis is None:
        # Empty state
        h("""
        <div style="
            height: 420px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 1px dashed #2a2820;
            border-radius: 8px;
            color: #3a3830;
            gap: 0.75rem;
        ">
            <div style="font-size: 2rem; opacity: 0.3">⚖</div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
                        letter-spacing: 0.1em; text-transform: uppercase;">
                Upload a document to begin
            </div>
        </div>
        """)

    else:
        analysis = st.session_state.analysis
        context  = st.session_state.get("context")

        # ── Urgency banner ──
        urgency_config = {
            "critical": ("🔴", "critical", "Urgent — action required immediately"),
            "high":     ("🟠", "high",     "High priority — act within 14–30 days"),
            "medium":   ("🟡", "medium",   "Medium priority — review when possible"),
            "low":      ("🟢", "low",      "Low urgency — for your information"),
        }
        icon, cls, label = urgency_config.get(
            analysis.urgency, ("⚪", "medium", "Review required")
        )
        h(f'<div class="urgency-banner urgency-{cls}">{icon}&nbsp; {label}</div>')

        # ── Escalation warning ──
        safe = st.session_state.get("safe")
        if safe and safe.escalate:
            h('''<div style="background:#2d1414;border:1px solid #7a2020;
                border-radius:6px;padding:0.85rem 1.1rem;margin-bottom:1rem;
                font-size:0.85rem;color:#f4a4a4;line-height:1.6;">
                <strong>Legal advice strongly recommended</strong> — 
                This document involves a situation where professional legal 
                advice is important. Please consult a lawyer before acting.
            </div>''')

        # ── Meta pills ──
        doc_type_label = (context.doc_type.replace("_", " ").title()
                         if context else "Document")
        jurisdiction_label = (context.jurisdiction.title()
                              if context else "Pakistan")
        conf_cls = f"conf-{analysis.confidence}"

        h(f"""
        <div class="meta-row">
            <span class="meta-pill highlight">{doc_type_label}</span>
            <span class="meta-pill">{jurisdiction_label}</span>
            <span class="meta-pill">Confidence: <span class="{conf_cls}">
                {analysis.confidence.title()}</span></span>
        </div>
        """)

        # ── Summary ──
        h(f"""
        <div class="analysis-card">
            <div class="card-label">What this document means</div>
            <div class="summary-text">{analysis.summary}</div>
        </div>
        """)

        # ── Rights ──
        if analysis.rights:
            rights_html = "".join(
                f'<div class="right-item">'
                f'<div class="right-dot"></div>'
                f'<div>{right}</div>'
                f'</div>'
                for right in analysis.rights
            )
            h(f"""
            <div class="analysis-card">
                <div class="card-label">Your rights</div>
                {rights_html}
            </div>
            """)

        # ── Action steps ──
        if analysis.action_steps:
            steps_inner = []
            for step in analysis.action_steps:
                deadline = ""
                if step.deadline and str(step.deadline).lower() not in ("null", "none", ""):
                    deadline = f'<span class="step-deadline">{step.deadline}</span>'
                steps_inner.append(
                    f'<div class="step-item">'
                    f'<div class="step-num">{step.step:02d}</div>'
                    f'<div class="step-body">'
                    f'<div class="step-instruction">{step.instruction}</div>'
                    f'{deadline}'
                    f'</div>'
                    f'</div>'
                )
            h(
                '<div class="analysis-card">'
                '<div class="card-label">What to do next</div>'
                + "".join(steps_inner)
                + '</div>'
            )

        # ── Sources ──
        if analysis.sources_cited:
            sources_html = "".join(
                f'<span class="source-tag">{src}</span>'
                for src in analysis.sources_cited
            )
            h(f"""
            <div style="margin-bottom: 1rem;">
                <div class="upload-label" style="margin-bottom: 0.5rem;">
                    Legal sources referenced
                </div>
                {sources_html}
            </div>
            """)

        # ── Evaluation scores ──
        safe = st.session_state.get("safe")
        if safe:
            ev = safe.evaluation
            h(f'''<div style="margin-top:1rem;padding:0.85rem 1.1rem;
                background:var(--color-background-secondary);
                border-radius:var(--border-radius-md);
                border:0.5px solid var(--color-border-tertiary);">
                <div class="upload-label" style="margin-bottom:0.5rem;">Pipeline scores</div>
                <div style="display:flex;gap:1.5rem;font-size:0.78rem;color:var(--color-text-secondary);font-family:var(--font-mono);">
                    <span>Retrieval: {ev.retrieval_relevance:.2f}</span>
                    <span>Groundedness: {ev.answer_groundedness:.2f}</span>
                    <span>Completeness: {ev.completeness:.2f}</span>
                    <span>Overall: {ev.overall:.2f}</span>
                </div>
            </div>''')

        # ── Disclaimer ──
        h(f'<div class="disclaimer">{analysis.disclaimer}</div>')

        # ── Reset button ──
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Analyse another document", key="btn_reset"):
            st.session_state.analysis = None
            st.session_state.context  = None
            st.rerun()