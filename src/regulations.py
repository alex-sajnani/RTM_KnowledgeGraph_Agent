"""
Pinned, verbatim regulatory text used to ground every LLM prompt.

The text lives in data/regulatory_snapshot.json, a committed, version-pinned
snapshot built by scripts/refresh_regulatory_snapshot.py. Prompts are grounded
in exactly the text a human reviewed: the app never fetches regulations on its
own, and the Audit page's update check (regulatory_refresh.py) only writes
changes after a named reviewer approves them. The snapshot records each
source's URL, version date and SHA-256 for audit.

Callers use load_regulations() at call time (it is cached in memory), so an
approved update takes effect without restarting the app.

Snapshot contents:
  21 CFR Part 820 (QMSR, effective 2026-02-02) — 820.10, 820.35
  21 CFR 862.1215 / 807.81                  — classification (MMI, Class II); 510(k) change decision
  42 CFR 493.1253 / 493.1255                — CLIA performance verification, calibration
  89 FR 7496 (QMSR final rule preamble)     — curated excerpts mapping QS-reg concepts
                                               (DHF, DMR, design controls) to ISO 13485
  FDA "Deciding When to Submit a 510(k) for a Change to an Existing Device"
                                            — curated excerpts, incl. IVD-specific logic
  FDA Compliance Program 7382.850 (2026)    — QMSR inspection process; QMS Area element
                                               → ISO 13485 clause tables (Attachment A)
  FDA QMSR web page and FAQ                 — inspection change (QSIT retired), IDE
                                               design controls, records FDA may review,
                                               read-only access to ISO 13485 (Q12)

ISO 13485:2016 is incorporated by reference into the QMSR but is copyrighted,
so it is cited by clause number rather than quoted.

Which sections a prompt receives is decided deterministically — by team
(TEAM_GROUNDING) and by device class (PATHWAY_GROUNDING) — never by retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNAPSHOT_PATH = DATA_DIR / "regulatory_snapshot.json"
STANDARDS_PATH = DATA_DIR / "standards_registry.json"
DEVICE_DEFINITION_PATH = DATA_DIR / "device_definition.json"


def _read_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s (%s).", path, exc)
        return default


def load_standards(path: str | Path = STANDARDS_PATH) -> list[dict]:
    """
    Metadata records for copyrighted standards (content class L). These carry
    designation, edition, authority level, and FDA recognition number — never
    text from the standard itself.
    """
    return _read_json(Path(path), {}).get("standards", [])


def load_device_definition(path: str | Path = DEVICE_DEFINITION_PATH) -> dict:
    """The verified device definition (product code, regulation, class, pathway)."""
    return _read_json(Path(path), {})


# ---------------------------------------------------------------------------
# Device classification — selects the premarket pathway grounding
# ---------------------------------------------------------------------------

# The seeded device (product code MMI, 21 CFR 862.1215) is Class II / 510(k),
# and the regulatory library is built for that pathway only. Other classes get
# an explicit "not supported" pathway context rather than borrowed grounding.
DEVICE_CLASSES = ["Class II"]
# Default to the verified device definition's class (MMI / 862.1215 is Class II).
DEFAULT_DEVICE_CLASS = load_device_definition().get("device_class") or "Class II"

# Where FDA says the incorporated standards can be read free (QMSR FAQ Q12).
ISO_13485_READ_ONLY_URL = "https://ibr.ansi.org/Standards/iso1.aspx"

# Core QMSR text every design-control prompt receives.
CORE_QMSR = ["820.10", "qmsr-preamble:design-records"]

# Device class → the sections that govern whether a design change needs a new
# premarket submission.
PATHWAY_GROUNDING: dict[str, list[str]] = {
    "Class II": [
        "807.81",
        "510k-change:qmsr-note",
        "510k-change:risk-based-assessment",
        "510k-change:vv-role",
        "510k-change:ivd-operating-principle",
        "510k-change:ivd-risk-assessment",
        "510k-change:documentation",
    ],
}

PATHWAY_SUMMARY: dict[str, str] = {
    "Class II": (
        "Class II: a design change needs a new 510(k) when it could significantly affect "
        "safety or effectiveness (21 CFR 807.81(a)(3)); FDA's 510(k) change guidance sets out "
        "the IVD decision logic and the documentation expected when no 510(k) is filed."
    ),
}

# How FDA inspects under the QMSR (CP 7382.850) — used where inspection
# readiness is discussed (dashboard chat, Quality/RA briefing).
INSPECTION_GROUNDING = [
    "fda-qmsr:inspections",
    "cp7382850:qms-areas",
    "cp7382850:design-development",
    "cp7382850:change-control",
]

# SME team → the sections its briefing is grounded in (pathway text is added
# separately for Quality/RA based on the device class).
TEAM_GROUNDING: dict[str, list[str]] = {
    "Bioinformatics": ["493.1253"],
    "R&D": ["820.10", "493.1253"],
    "Pathology": ["493.1253", "qmsr-preamble:risk-management"],
    "Quality/RA": CORE_QMSR + ["820.35", "cp7382850:change-control", "493.1253", "493.1255"],
}

# Minimal stubs used only if the snapshot file is missing or unreadable.
_FALLBACK = {
    "820.10": "21 CFR §820.10: Manufacturers must document a quality management system that complies with ISO 13485, incorporated by reference.",
    "820.35": "21 CFR §820.35: In addition to ISO 13485 Clause 4.2.5, manufacturers must include specified information in certain records.",
    "493.1253": "42 CFR §493.1253: Laboratories must establish and verify performance specifications before reporting patient results.",
    "493.1255": "42 CFR §493.1255: Laboratories must perform calibration and calibration verification procedures.",
}

_snapshot_cache: dict | None = None


def _read_snapshot(path: Path = SNAPSHOT_PATH) -> dict:
    global _snapshot_cache
    if path == SNAPSHOT_PATH and _snapshot_cache is not None:
        return _snapshot_cache
    try:
        snapshot = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Regulatory snapshot unavailable (%s); using fallback stubs.", exc)
        snapshot = {}
    if path == SNAPSHOT_PATH:
        _snapshot_cache = snapshot
    return snapshot


def reload_snapshot() -> None:
    """Drop the in-memory snapshot so the next read picks up a rewritten file."""
    global _snapshot_cache
    _snapshot_cache = None


def load_regulations(snapshot_path: str | Path = SNAPSHOT_PATH) -> dict[str, str]:
    """
    Return a dict mapping section_id → verbatim regulatory text from the pinned
    snapshot. Falls back to minimal stubs (and logs a warning) if the snapshot
    is missing, so the app still starts.
    """
    sections = _read_snapshot(Path(snapshot_path)).get("sections", {})
    if not sections:
        return dict(_FALLBACK)
    return {sid: entry["text"] for sid, entry in sections.items()}


def grounding_status(snapshot_path: str | Path = SNAPSHOT_PATH) -> dict:
    """Snapshot provenance for the UI: generation time, sources, fallback flag."""
    snapshot = _read_snapshot(Path(snapshot_path))
    return {
        "using_fallback": not snapshot.get("sections"),
        "generated_at": snapshot.get("generated_at", ""),
        "last_update": snapshot.get("last_update", {}),
        "sources": snapshot.get("sources", {}),
        "section_count": len(snapshot.get("sections", {})),
    }


def section_label(section_id: str) -> str:
    """Human-readable label for a snapshot section, e.g. '21 CFR §820.10 — …'."""
    entry = _read_snapshot().get("sections", {}).get(section_id)
    return entry["label"] if entry else f"§{section_id}"


_label = section_label


def build_prompt_context(regulations: dict[str, str], section_ids: list[str]) -> str:
    """
    Formats the requested sections into a block suitable for injection into
    an LLM system prompt. Unknown section IDs are skipped.
    """
    lines = ["--- Applicable Regulatory Text (verbatim from pinned sources) ---"]
    sections = _read_snapshot().get("sections", {})
    for sid in dict.fromkeys(section_ids):  # de-duplicate, keep order
        text = regulations.get(sid)
        entry = sections.get(sid, {})
        # Copyright guard (PRD §8): only public-domain text may reach a prompt.
        if not text or entry.get("content_class", "P") != "P":
            continue
        tag = f"{entry.get('authority_level', '')}, {entry.get('status', 'current')}".strip(", ")
        lines.append(f"\n[{sid}] {_label(sid)} ({tag}):\n{text}")
    lines.append(
        "\nNote: ISO 13485:2016 is incorporated by reference into 21 CFR 820 (QMSR) and is cited "
        "by clause number only. The former QS regulation sections (e.g. 820.30, 820.40, 820.100, "
        "820.180) were superseded on 2026-02-02; where FDA guidance above cites them, apply the "
        "corresponding ISO 13485 clause under the QMSR."
    )
    lines.append("--- End Regulatory Text ---")
    return "\n".join(lines)


def pathway_context(regulations: dict[str, str], device_class: str) -> str:
    """Device-class-specific premarket pathway grounding for design changes."""
    if device_class not in PATHWAY_GROUNDING:
        # Never borrow another class's pathway: say plainly that none is grounded.
        return (
            f"Device classification: {device_class}. The regulatory library supports the Class II "
            "510(k) pathway only; no premarket pathway text is supplied for this class. Do not "
            "state a submission conclusion — flag it for Regulatory Affairs review."
        )
    return (
        f"Device classification: {device_class}. {PATHWAY_SUMMARY[device_class]}\n\n"
        + build_prompt_context(regulations, PATHWAY_GROUNDING[device_class])
    )


GROUNDING_INSTRUCTION = (
    "Ground every regulatory statement in the verbatim regulatory text supplied above — cite "
    "the bracketed section ID or CFR section supplied rather than relying on your own "
    "recollection of the regulations, and do not paraphrase or invent regulatory language. "
    "If a point is not supported by the supplied text, say so rather than asserting it. "
    "Each section is tagged with its authority level: say 'FDA requires' only for A1/A2 "
    "(regulation or incorporated standard); for A5 guidance say 'FDA recommends'; A4 and A6 "
    "describe FDA's interpretation or inspection practice, not requirements. "
    "Do not quote, reproduce, or closely paraphrase the text of ISO, IEC, CLSI, or AAMI "
    "standards, including text you may recall; cite the standard and clause number instead. "
    "If asked for a standard's text, decline, then give the clause number, any FDA "
    "interpretation supplied above, and FDA's read-only access route if supplied "
    "([fda-qmsr-faq:iso-access])."
)
