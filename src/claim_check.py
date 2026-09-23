"""
Deterministic checks on AI output before it is shown (PRD FR-012, FR-013).

The AI's prompts ask it to cite sources and respect authority levels, but a
prompt is not an enforcement mechanism. These checks run in code on every
AI-generated text and flag, rather than silently pass:

  unsupported_claim        a regulatory-attributed mandatory claim ("FDA requires…",
                           "required by the QMSR…") with no A1/A2 citation
  guidance_as_requirement  a mandatory claim supported only by guidance (A5)
  superseded_citation      a former QS-regulation section (e.g. 21 CFR 820.30)
                           cited as if it were current
  unknown_citation         a bracketed section ID that is not in the library
  possible_standard_text   a long quotation attributed to ISO/IEC/CLSI/AAMI —
                           standards text must not be reproduced (PRD §8, C6)

The checks are heuristic by design: they favor flagging for human review over
missing a claim. They never rewrite the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from regulations import _read_snapshot

# Sections of the QS regulation that the QMSR removed on 2026-02-02.
SUPERSEDED_QS_SECTIONS = {
    "820.20", "820.22", "820.25", "820.30", "820.40", "820.50", "820.60", "820.65",
    "820.70", "820.72", "820.75", "820.80", "820.86", "820.90", "820.100", "820.120",
    "820.130", "820.140", "820.150", "820.160", "820.170", "820.180", "820.181",
    "820.184", "820.186", "820.198", "820.200", "820.250",
}

# A claim that attributes an obligation to a regulator or regulation.
_MANDATORY = [
    re.compile(
        r"\b(FDA|the agency|QMSR|regulations?|CFR|federal law|statute|ISO 13485)\b[^.;]{0,80}?"
        r"\b(requires?|required|mandates?|mandated|mandatory|obligates?|must)\b", re.I),
    re.compile(r"\b(required|mandated|mandatory)\s+(by|under|per|pursuant to)\b", re.I),
    re.compile(r"\b(is|are)\s+(a\s+)?(regulatory|legal)\s+requirements?\b", re.I),
]
_CFR = re.compile(r"(?:(?:21|42)\s*CFR\s*(?:§+\s*)?|§+\s*)(\d{3,4}\.\d+)")
_ISO_13485_CLAUSE = re.compile(r"ISO\s*13485[^.;]{0,30}?(?:§|clause|cl\.)?\s*\d+\.\d+", re.I)
_BRACKET = re.compile(r"\[([A-Za-z0-9][\w.:()\-]*?)\]")
_HISTORICAL = re.compile(r"\b(former|superseded|previous(ly)?|prior|old|QS regulation|QSR)\b", re.I)
_STANDARD = re.compile(r"\b(ISO|IEC|CLSI|AAMI)\b|\bclause\s+\d", re.I)
_QUOTE = re.compile(r"[\"“]([^\"”]{40,})[\"”]")
_GUIDANCE_WORD = re.compile(r"\bguidance\b", re.I)


@dataclass
class ClaimFinding:
    kind: str
    severity: str        # "error" | "warning"
    excerpt: str
    message: str


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(\"“])|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def _short(sentence: str, limit: int = 180) -> str:
    return sentence if len(sentence) <= limit else sentence[: limit - 1] + "…"


def check_output(text: str, sections: dict | None = None) -> list[ClaimFinding]:
    """Run all checks on one AI output. `sections` defaults to the pinned snapshot."""
    sections = sections if sections is not None else _read_snapshot().get("sections", {})
    findings: list[ClaimFinding] = []

    for sentence in _sentences(text):
        historical = bool(_HISTORICAL.search(sentence))
        cfr = _CFR.findall(sentence)
        brackets = [b for b in _BRACKET.findall(sentence) if ":" in b or re.match(r"\d", b)]

        # Superseded QS-regulation citations presented as current.
        for num in cfr:
            if num in SUPERSEDED_QS_SECTIONS and not historical:
                findings.append(ClaimFinding(
                    "superseded_citation", "error", _short(sentence),
                    f"21 CFR {num} was superseded by the QMSR on 2026-02-02. Cite the ISO 13485 "
                    "clause applied through 21 CFR 820.10 instead.",
                ))

        # Bracketed citations must resolve to the library.
        for sid in brackets:
            if sid not in sections:
                findings.append(ClaimFinding(
                    "unknown_citation", "warning", _short(sentence),
                    f"[{sid}] is not a section in the regulatory library.",
                ))
            elif sections[sid].get("status") == "superseded" and not historical:
                findings.append(ClaimFinding(
                    "superseded_citation", "error", _short(sentence),
                    f"[{sid}] is superseded text, shown for comparison only.",
                ))

        # Mandatory claims need A1/A2 support.
        if any(p.search(sentence) for p in _MANDATORY):
            binding = (
                any(n not in SUPERSEDED_QS_SECTIONS for n in cfr)
                or bool(_ISO_13485_CLAUSE.search(sentence))
                or any(sections.get(b, {}).get("authority_level") in {"A1", "A2"}
                       and sections.get(b, {}).get("status") != "superseded" for b in brackets)
            )
            if not binding:
                guidance = _GUIDANCE_WORD.search(sentence) or any(
                    sections.get(b, {}).get("authority_level") == "A5" for b in brackets)
                if guidance:
                    findings.append(ClaimFinding(
                        "guidance_as_requirement", "error", _short(sentence),
                        "Stated as a requirement but supported only by guidance. Guidance is a "
                        "recommendation (\"FDA recommends\") unless a regulation establishes it.",
                    ))
                else:
                    findings.append(ClaimFinding(
                        "unsupported_claim", "warning", _short(sentence),
                        "Regulatory requirement stated without a citation to a regulation or "
                        "incorporated standard. Verify before relying on it.",
                    ))

        # Long quotations attributed to a copyrighted standard.
        if _STANDARD.search(sentence):
            for quote in _QUOTE.findall(sentence):
                if len(quote.split()) >= 10:
                    findings.append(ClaimFinding(
                        "possible_standard_text", "error", _short(sentence),
                        "Possible quotation from a copyrighted standard. Cite the clause number "
                        "instead of reproducing its text.",
                    ))

    return findings
