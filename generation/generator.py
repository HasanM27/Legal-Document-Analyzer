"""
Step 5 — LLM Generation
=========================
Takes a RAGContext from Step 4, assembles a structured prompt,
calls the Groq API, and returns a clean LegalAnalysis object
with summary, rights, action_steps, and urgency level.

The LLMClient wrapper is provider-agnostic — swapping to a
different API is a one-line change in LLMClient.__init__().
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ingestion.retriever import RAGContext

log = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

@dataclass
class ActionStep:
    step:        int
    instruction: str
    deadline:    Optional[str] = None   # e.g. "within 14 days", "immediately"


@dataclass
class LegalAnalysis:
    """
    Final structured output returned to the user.
    Every field maps directly to a UI panel in Step 6.
    """
    summary:        str                  # plain-language explanation
    rights:         list[str]            # bullet list of user's rights
    action_steps:   list[ActionStep]     # numbered steps with optional deadlines
    urgency:        str                  # "critical", "high", "medium", "low"
    disclaimer:     str                  # always present — not legal advice
    sources_cited:  list[str]            # act titles referenced
    confidence:     str                  # "high", "medium", "low"
    raw_json:       dict = field(default_factory=dict)


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around the Groq API.
    Swap provider by changing model/client in __init__ only.
    """

    MODEL    = "llama-3.3-70b-versatile"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.1    # low = more consistent, deterministic output

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Groq API key found. Set the GROQ_API_KEY environment "
                "variable or pass api_key= to LLMClient()."
            )
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        except ImportError:
            raise RuntimeError(
                "groq package not installed. Run: pip install groq"
            )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM and return the raw text response."""
        response = self._client.chat.completions.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            temperature=self.TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


# ── Prompt builder ────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles the system and user prompts from a RAGContext.

    Design decisions:
    - System prompt defines the persona, output format, and hard rules
    - User prompt contains the document + retrieved legal chunks
    - We ask for JSON explicitly and give a concrete schema example
    - Temperature 0.1 keeps the output consistent across runs
    """

    # ── System prompt ─────────────────────────────────────────────────────────
    SYSTEM_PROMPT = """You are a Pakistani legal document assistant. Output JSON only.

ABSOLUTE OUTPUT RULES:
- Output a single raw JSON object. Nothing before it. Nothing after it.
- No markdown code fences. No backticks. No ```json.
- NO HTML TAGS OF ANY KIND. No <div>, <p>, <span>, <br>, no angle brackets.
- Every value must be plain text. Arrays must contain plain text strings only.

EXACT JSON STRUCTURE TO OUTPUT:

{
  "summary": "Plain text. 2-3 sentences explaining what this document means.",
  "rights": [
    "Plain text right 1",
    "Plain text right 2",
    "Plain text right 3"
  ],
  "action_steps": [
    {
      "step": 1,
      "instruction": "Plain text. What to do first. No HTML.",
      "deadline": "within 15 days"
    },
    {
      "step": 2,
      "instruction": "Plain text. What to do second.",
      "deadline": null
    }
  ],
  "urgency": "critical",
  "confidence": "high",
  "sources_cited": ["Act Name 1", "Act Name 2"]
}

URGENCY RULES — apply strictly:
- "critical" — deadline in the document is 15 days or less
- "high"     — deadline is 16 to 30 days
- "medium"   — no specific deadline mentioned
- "low"      — purely informational document

CONFIDENCE RULES — apply strictly:
- "high"   — retrieved law directly names this document type or situation
- "medium" — retrieved law is related but does not directly address this
- "low"    — little or no relevant law retrieved

CONTENT RULES:
1. Plain English only. A 15-year-old should understand every sentence.
2. Ground every right in the retrieved legal excerpts provided.
3. Step 1 must be the most urgent practical action the person can take today.
4. Always make the final step a recommendation to consult a lawyer.
5. If a step has a time limit, always fill in the deadline field."""

    def build(self, context: RAGContext) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt)."""
        user_prompt = self._build_user_prompt(context)
        return self.SYSTEM_PROMPT, user_prompt

    def _build_user_prompt(self, context: RAGContext) -> str:
        parts = []

        # Document metadata
        parts.append(
            f"DOCUMENT TYPE: {context.doc_type.replace('_', ' ').title()}\n"
            f"JURISDICTION: {context.jurisdiction.title()}\n"
        )

        # Extracted facts summary
        f = context.facts
        if f.monetary_amounts:
            parts.append(f"AMOUNTS MENTIONED: {', '.join(f.monetary_amounts)}")
        if f.deadlines:
            parts.append(f"DEADLINES MENTIONED: {', '.join(f.deadlines)}")
        if f.parties:
            parts.append(f"PARTIES INVOLVED: {', '.join(f.parties)}")
        if f.key_phrases:
            parts.append(f"KEY PHRASES: {', '.join(f.key_phrases)}")

        parts.append("")

        # The actual document
        parts.append("=" * 60)
        parts.append("DOCUMENT CONTENT:")
        parts.append("=" * 60)
        # Cap document at 3000 chars to leave room for legal context
        doc_text = context.document_text[:3000]
        if len(context.document_text) > 3000:
            doc_text += "\n[... document truncated for brevity ...]"
        parts.append(doc_text)
        parts.append("")

        # Retrieved legal knowledge
        if context.retrieved_chunks:
            parts.append("=" * 60)
            parts.append("RELEVANT PAKISTANI LAW:")
            parts.append("=" * 60)
            for i, chunk in enumerate(context.retrieved_chunks, 1):
                parts.append(
                    f"\n[Source {i}: {chunk.title} "
                    f"(relevance: {chunk.score:.2f})]"
                )
                parts.append(chunk.text[:600])  # cap each chunk
        else:
            parts.append("NOTE: No specific legal provisions retrieved. "
                        "Base your answer on general Pakistani civil law.")

        parts.append("")
        parts.append("Analyse this document and respond with the JSON object only.")

        return "\n".join(parts)


# ── Output parser ─────────────────────────────────────────────────────────────

class OutputParser:
    """
    Parses the LLM's JSON response into a LegalAnalysis object.
    Handles common failure modes: markdown fences, trailing text,
    missing fields, and malformed JSON.
    """

    DISCLAIMER = (
        "This analysis is for informational purposes only and does not "
        "constitute legal advice. Laws may have changed and individual "
        "circumstances vary. For important legal matters, please consult "
        "a qualified lawyer or visit your nearest legal aid office."
    )

    def parse(self, raw: str, context: RAGContext) -> LegalAnalysis:
        data = self._extract_json(raw)

        if data is None:
            log.warning("Failed to parse LLM JSON — using fallback response")
            return self._fallback(context, raw)

        return LegalAnalysis(
            summary=data.get("summary", "Unable to generate summary."),
            rights=data.get("rights", []),
            action_steps=self._parse_steps(data.get("action_steps", [])),
            urgency=self._validate_urgency(data.get("urgency", "medium")),
            disclaimer=self.DISCLAIMER,
            sources_cited=data.get("sources_cited", []),
            confidence=self._validate_confidence(data.get("confidence", "low")),
            raw_json=data,
        )

    def _extract_json(self, raw: str) -> Optional[dict]:
        """Try multiple strategies to extract valid JSON from LLM output."""
        # Strategy 1: direct parse
        try:
            return self._sanitise(json.loads(raw.strip()))
        except json.JSONDecodeError:
            pass

        # Strategy 2: strip markdown fences ```json ... ```
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            return self._sanitise(json.loads(cleaned))
        except json.JSONDecodeError:
            pass

        # Strategy 3: find first { ... } block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return self._sanitise(json.loads(match.group(0)))
            except json.JSONDecodeError:
                pass

        return None

    def _sanitise(self, data: dict) -> dict:
        """
        Walk the parsed JSON and strip HTML from every string value.
        Also fixes the case where action_steps is a single HTML string
        instead of a list of step objects.
        """
        if not isinstance(data, dict):
            return data

        # Fix action_steps if the LLM returned HTML string(s) instead of objects
        raw_steps = data.get("action_steps", [])
        if isinstance(raw_steps, list):
            fixed_steps = []
            for item in raw_steps:
                if isinstance(item, dict):
                    # Normal case — clean instruction and deadline fields
                    item["instruction"] = self._strip_html(
                        str(item.get("instruction", ""))
                    )
                    dl = item.get("deadline")
                    item["deadline"] = (
                        self._strip_html(str(dl))
                        if dl and str(dl).lower() not in ("null", "none", "")
                        else None
                    )
                    if item["instruction"]:
                        fixed_steps.append(item)
                elif isinstance(item, str):
                    # LLM returned HTML as a string — extract text only
                    text = self._strip_html(item)
                    if text:
                        # Try to split into individual steps by line
                        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 15]
                        for i, line in enumerate(lines, len(fixed_steps) + 1):
                            # Skip lines that look like step numbers ("01", "02", "1.", "2.")
                            if re.match(r"^\d{1,2}\.?$", line):
                                continue
                            fixed_steps.append({
                                "step": i,
                                "instruction": line,
                                "deadline": None,
                            })
            data["action_steps"] = fixed_steps

        # Strip HTML from summary and rights too
        if "summary" in data:
            data["summary"] = self._strip_html(str(data["summary"]))
        if "rights" in data and isinstance(data["rights"], list):
            data["rights"] = [
                self._strip_html(str(r))
                for r in data["rights"]
                if self._strip_html(str(r))
            ]

        return data

    def _parse_steps(self, raw_steps: list) -> list[ActionStep]:
        steps = []
        for i, s in enumerate(raw_steps, 1):
            if isinstance(s, dict):
                instruction = self._strip_html(s.get("instruction", ""))
                deadline    = s.get("deadline") or None
                if deadline:
                    deadline = self._strip_html(str(deadline))
                    if deadline.lower() in ("null", "none", ""):
                        deadline = None
                steps.append(ActionStep(
                    step=s.get("step", i),
                    instruction=instruction,
                    deadline=deadline,
                ))
            elif isinstance(s, str):
                clean = self._strip_html(s)
                clean = re.sub(r"\s{2,}", " ", clean).strip()
                if clean:
                    steps.append(ActionStep(step=i, instruction=clean))
        return steps

    def _strip_html(self, text: str) -> str:
        """Remove HTML tags and collapse leftover whitespace."""
        stripped = re.sub(r"<[^>]+>", " ", text)
        stripped = re.sub(r"\s{2,}", " ", stripped)
        return stripped.strip()

    def _validate_urgency(self, val: str) -> str:
        valid = {"critical", "high", "medium", "low"}
        return val.lower() if val.lower() in valid else "medium"

    def _validate_confidence(self, val: str) -> str:
        valid = {"high", "medium", "low"}
        return val.lower() if val.lower() in valid else "low"

    def _fallback(self, context: RAGContext, raw: str) -> LegalAnalysis:
        """
        When JSON parsing completely fails, return a safe fallback
        that tells the user something went wrong and to seek help.
        """
        return LegalAnalysis(
            summary=(
                f"We received your {context.doc_type.replace('_', ' ')} "
                "but encountered an issue generating the full analysis. "
                "Please try again or consult a legal professional."
            ),
            rights=[],
            action_steps=[
                ActionStep(
                    step=1,
                    instruction="Please consult a qualified lawyer or "
                                "legal aid office for assistance with "
                                "this document.",
                    deadline=None,
                )
            ],
            urgency="medium",
            disclaimer=self.DISCLAIMER,
            sources_cited=[],
            confidence="low",
            raw_json={"parse_error": True, "raw": raw[:200]},
        )


# ── Generator — main entry point ──────────────────────────────────────────────

class LegalAnalysisGenerator:
    """
    The single entry point for Step 5.

    Usage:
        generator = LegalAnalysisGenerator()
        analysis  = generator.generate(rag_context)
    """

    def __init__(
        self,
        llm:     Optional[LLMClient]    = None,
        builder: Optional[PromptBuilder] = None,
        parser:  Optional[OutputParser]  = None,
    ):
        self.llm     = llm     or LLMClient()
        self.builder = builder or PromptBuilder()
        self.parser  = parser  or OutputParser()

    def generate(self, context: RAGContext) -> LegalAnalysis:
        """
        Full pipeline:
          1. Build prompt from RAGContext
          2. Call Groq LLM
          3. Parse JSON response into LegalAnalysis
        """
        log.info(f"Generating analysis for: {context.doc_type}")

        # 1. Build prompts
        system_prompt, user_prompt = self.builder.build(context)

        log.info(f"  Prompt size: ~{len(user_prompt)} chars")
        log.info(f"  Retrieved chunks: {len(context.retrieved_chunks)}")

        # 2. Call LLM
        raw_response = self.llm.complete(system_prompt, user_prompt)
        log.info(f"  Response size: {len(raw_response)} chars")

        # 3. Parse and return
        analysis = self.parser.parse(raw_response, context)
        log.info(
            f"  Analysis complete: urgency={analysis.urgency}, "
            f"confidence={analysis.confidence}, "
            f"rights={len(analysis.rights)}, "
            f"steps={len(analysis.action_steps)}"
        )

        return analysis