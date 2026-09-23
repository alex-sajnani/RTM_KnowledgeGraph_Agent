"""
Fetch the regulatory sources behind the pinned snapshot, check them for changes,
and apply changes only after a named reviewer approves.

The snapshot (data/regulatory_snapshot.json) is what every LLM prompt is
grounded in, so it is never overwritten silently. check_regulatory_updates()
fetches the live sources and reports two kinds of change without writing
anything:
  - section changes: a pinned section or excerpt's text differs (unified diff)
  - source changes:  a watched page or document changed, even if the pinned
                     excerpts did not (so a reviewer can look for new content)
apply_regulatory_updates() then writes the reviewed changes back, recording the
reviewer's name and a timestamp.

Every excerpt is located by start/end anchor phrases and copied verbatim. If an
anchor disappears, the fetcher raises SourceTextError and that source is
reported as needing manual review — text is never silently altered or truncated.

Sources are registered in FETCHERS. The Federal Register QMSR final rule is a
fixed 2024 publication, so it is fetched only when the snapshot is rebuilt
(scripts/refresh_regulatory_snapshot.py), not by the update check.
"""

from __future__ import annotations

import difflib
import gzip
import hashlib
import html
import io
import json
import re
import ssl
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.request import Request, urlopen

from regulations import SNAPSHOT_PATH, load_device_definition, reload_snapshot

ECFR = "https://www.ecfr.gov/api/versioner/v1"
FR_TEXT = "https://www.federalregister.gov/documents/full_text/text/2024/02/02/2024-01709.txt"
FR_PAGE = "https://www.federalregister.gov/d/2024-01709"
GUIDANCE_510K = "https://www.fda.gov/media/99812/download"
CP_7382_850 = "https://www.fda.gov/media/80195/download"
QMSR_PAGE = "https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr"
OPENFDA_CLASSIFICATION = "https://api.fda.gov/device/classification.json?search=product_code:{code}"
QMSR_FAQ = "https://www.fda.gov/medical-devices/quality-management-system-regulation-qmsr/quality-management-system-regulation-frequently-asked-questions"


# ---------------------------------------------------------------------------
# Snapshot specification
# ---------------------------------------------------------------------------

# eCFR sections: section_id -> (title, part, label, optional (start, end) anchors)
ECFR_SECTIONS = {
    "820.10":   ("21", "820", "21 CFR §820.10 — Requirements for a Quality Management System (QMSR)", None),
    "820.35":   ("21", "820", "21 CFR §820.35 — Control of Records (QMSR)", None),
    "807.81":   ("21", "807", "21 CFR §807.81 — When a Premarket Notification Submission Is Required", None),
    "862.1215": ("21", "862", "21 CFR §862.1215 — Creatine Kinase or Isoenzymes Test System (regulation under which FDA classifies troponin assays, product code MMI)", None),
    "493.1253": ("42", "493", "42 CFR §493.1253 — CLIA: Establishment and Verification of Performance Specifications", None),
    "493.1255": ("42", "493", "42 CFR §493.1255 — CLIA: Calibration and Calibration Verification", None),
}

# Anchored excerpts: excerpt_id -> (label, start anchor, end anchor)
_FR = "QMSR Final Rule Preamble (89 FR 7496)"
FR_EXCERPTS = {
    "qmsr-preamble:overview": (
        f"{_FR} — Summary of the Major Provisions",
        "We are amending part 820, primarily through incorporating by reference",
        "otherwise in compliance with the Federal Food, Drug, and Cosmetic Act",
    ),
    "qmsr-preamble:design-applicability": (
        f"{_FR} — Applicability of Design and Development (Clause 7.3)",
        "Manufacturers of class II and class III, and certain class I devices described in Sec. 820.10(c)",
        "compatibility testing).",
    ),
    "qmsr-preamble:design-records": (
        f"{_FR} — DHF/DMR Records Under ISO 13485",
        "Similarly, consistent with the former DHF, Clause 7.3.10",
        "will now be located in the manufacturer's MDF.",
    ),
    "qmsr-preamble:risk-management": (
        f"{_FR} — Risk Management Throughout the QMS",
        "FDA agrees that the embedded risk management concepts present in ISO 13485",
        "total product life-cycle risk management systems.",
    ),
    "qmsr-preamble:design-changes": (
        f"{_FR} — Documenting Design Changes (Clause 7.3.9)",
        "For all devices to which design and development requirements apply, FDA does not expect",
        "once the design has been released to production.",
    ),
    "qmsr-preamble:design-review": (
        f"{_FR} — Design Review and Validation Under Clauses 7.3.3 and 7.3.4",
        "FDA considers that a successful quality management system under Clause 7.3.3 and 7.3.4.",
        "tailor a design review that is appropriate to their individual needs.",
    ),
    "qmsr-preamble:clinical-evaluation": (
        f"{_FR} — Clinical or Performance Evaluation in Validation (Clause 7.3.7)",
        "Because the regulatory requirements that may apply to clinical evaluations are provided elsewhere",
        "Financial Disclosure by Clinical Investigators.",
    ),
    "qmsr-preamble:capa": (
        f"{_FR} — Corrective and Preventive Action (Clauses 8.5.2, 8.5.3)",
        "FDA continues to believe that it is essential that the manufacturer establish procedures for implementing corrective",
        "the methods that correct or prevent the problem from recurring.",
    ),
    "qmsr-preamble:capa-effectiveness": (
        f"{_FR} — Verifying CAPA Effectiveness",
        "FDA clarifies that consistent with the former QS regulation, as part of an effective quality system,",
        "do not adversely affect the finished device.",
    ),
}

# Superseded QS regulation (21 CFR 820 before the QMSR took effect on
# 2026-02-02), pinned paragraph by paragraph for comparison only. These are
# never used to ground prompts. id -> (section, label, start marker, next marker)
QS_REG_DATE = "2026-02-01"
_QS = "Former QS Regulation (superseded 2026-02-02)"
QS_REG_PARAGRAPHS = {
    "qsreg:820.30(b)": ("820.30", f"{_QS} — 21 CFR 820.30(b) Design and Development Planning",
                        "(b) Design and development planning.", "(c) Design input."),
    "qsreg:820.30(c)": ("820.30", f"{_QS} — 21 CFR 820.30(c) Design Input",
                        "(c) Design input.", "(d) Design output."),
    "qsreg:820.30(d)": ("820.30", f"{_QS} — 21 CFR 820.30(d) Design Output",
                        "(d) Design output.", "(e) Design review."),
    "qsreg:820.30(f)": ("820.30", f"{_QS} — 21 CFR 820.30(f) Design Verification",
                        "(f) Design verification.", "(g) Design validation."),
    "qsreg:820.30(g)": ("820.30", f"{_QS} — 21 CFR 820.30(g) Design Validation",
                        "(g) Design validation.", "(h) Design transfer."),
    "qsreg:820.30(i)": ("820.30", f"{_QS} — 21 CFR 820.30(i) Design Changes",
                        "(i) Design changes.", "(j) Design history file."),
    "qsreg:820.30(j)": ("820.30", f"{_QS} — 21 CFR 820.30(j) Design History File",
                        "(j) Design history file.", None),
    "qsreg:820.70(b)": ("820.70", f"{_QS} — 21 CFR 820.70(b) Production and Process Changes",
                        "(b) Production and process changes.", "(c) Environmental control."),
    "qsreg:820.100":   ("820.100", f"{_QS} — 21 CFR 820.100 Corrective and Preventive Action",
                        "§ 820.100 Corrective and preventive action.", None),
}

_G = "FDA Guidance, Deciding When to Submit a 510(k) for a Change to an Existing Device (2017)"
GUIDANCE_EXCERPTS = {
    "510k-change:qmsr-note": (
        f"{_G} — QMSR Notice",
        "This guidance document was issued prior to the effective date of the final rule.",
        "to ensure compliance with the relevant regulatory requirements.",
    ),
    "510k-change:risk-based-assessment": (
        f"{_G} — Guiding Principle 2, Initial Risk-Based Assessment",
        "2. Initial risk-based assessment",
        "lead to an initial decision whether or not submission of a new 510(k) is required.",
    ),
    "510k-change:vv-role": (
        f"{_G} — Guiding Principle 5, Role of Verification and Validation",
        "5. The role of testing",
        "must be conducted regardless of whether submission of a new 510(k) is required.",
    ),
    "510k-change:ivd-operating-principle": (
        f"{_G} — D1, IVD Changes to the Operating Principle",
        "D1 . Does the change alter the operating principle of the IVD? In most cases",
        "does not alter the operating principle of the IVD, proceed to D2.",
    ),
    "510k-change:ivd-risk-assessment": (
        f"{_G} — D3, IVD Risk-Based Assessment",
        "For IVDs, a manufacturer’s risk -based assessment identifies",
        "clinically significant in terms of clinical decision making.",
    ),
    "510k-change:documentation": (
        f"{_G} — Appendix B, Documentation",
        "If a manufacturer determines that the device change(s) does not require submission",
        "is not sufficient documentation.",
    ),
}

_CP = "FDA Compliance Program 7382.850, Inspection of Medical Device Manufacturers (2026)"
CP_EXCERPTS = {
    "cp7382850:qms-areas": (
        f"{_CP} — Risk-Based Inspection Process",
        "The risk-based inspection process aligns with the QMSR",
        "4 Other Applicable FDA Requirements (OAFRs).",
    ),
    "cp7382850:inspection-model-2": (
        f"{_CP} — Inspection Model 2, Minimum Elements",
        "Inspection Model 2 Identify product risks",
        "OAFR: Unique Device Identification",
    ),
    "cp7382850:change-control": (
        f"{_CP} — Attachment A, Change Control QMS Area",
        "Change Control QMS Area Purpose:",
        "Purchasing Changes Clauses 7.4.2, 7.4.3",
    ),
    "cp7382850:design-development": (
        f"{_CP} — Attachment A, Design and Development QMS Area",
        "Design and Development QMS Area Purpose:",
        "Design and Development Files Clause 7.3.10",
    ),
    "cp7382850:measurement-analysis-improvement": (
        f"{_CP} — Attachment A, Measurement, Analysis, and Improvement QMS Area",
        "Measurement, Analysis, and Improvement QMS Area Purpose:",
        "Preventive Action Clause 8.5.3",
    ),
}

_QP = "FDA Quality Management System Regulation (QMSR) web page"
QMSR_PAGE_EXCERPTS = {
    "fda-qmsr:inspections": (
        f"{_QP} — QMSR Inspections",
        "The FDA has a new inspection process that aligns with the requirements",
        "Medical Device PMA Preapproval and PMA Postmarket Inspections (7383.001).",
    ),
    "fda-qmsr:exemptions-ide": (
        f"{_QP} — CGMP Exemptions and IDE Devices",
        "Exemption from the CGMP requirements does not exempt manufacturers",
        "Design and Development, Clause 7 and its subclauses.",
    ),
}

_QF = "FDA QMSR Frequently Asked Questions"
QMSR_FAQ_EXCERPTS = {
    "fda-qmsr-faq:pre-qmsr-records": (
        f"{_QF} — Q7, Records Reviewed on or After February 2, 2026",
        "To help determine compliance with the QMSR, FDA investigators may review records",
        "records created prior to the QMSR effective date meet the QMSR requirements.",
    ),
    "fda-qmsr-faq:audit-records": (
        f"{_QF} — Q8, Management Review, Quality Audit, and Supplier Audit Records",
        "Yes. The QMSR gives the FDA the authority to inspect management review",
        "should be readily available upon inspection.",
    ),
    "fda-qmsr-faq:iso-access": (
        f"{_QF} — Q12, Access to ISO 13485:2016 and ISO 9000:2015 Clause 3",
        "The standards can be accessed in a read-only format",
        "https://ibr.ansi.org/Standards/iso1.aspx",
    ),
}


# Authority level (PRD §7) and status for every pinned source, keyed by source_id
# prefix; the first matching prefix wins. Everything pinned in the snapshot is
# public domain (content class "P", PRD §8) — copyrighted standards are held
# only as metadata in data/standards_registry.json, never here.
SOURCE_CLASSIFICATION: list[tuple[str, str, str]] = [
    ("ecfr:21-820-qsreg", "A1", "superseded"),   # former QS regulation, comparison only
    ("ecfr:", "A1", "current"),                  # statute/regulation
    ("fr:", "A4", "current"),                    # FDA regulatory interpretation (preamble)
    ("fda:510k-change", "A5", "current"),        # final FDA guidance
    ("fda:cp-", "A6", "current"),                # compliance program
    ("fda:qmsr-page", "A6", "current"),
    ("fda:qmsr-faq", "A6", "current"),
]


def classify_source(source_id: str) -> tuple[str, str]:
    """(authority_level, status) for a source_id; raises if unclassified."""
    for prefix, level, status in SOURCE_CLASSIFICATION:
        if source_id.startswith(prefix):
            return level, status
    raise ValueError(f"Source {source_id!r} has no authority classification")


def _tagged(fetch):
    """Stamp authority level, content class, and status on a fetcher's output."""
    def wrapper():
        sections, sources = fetch()
        for entry in sections.values():
            level, status = classify_source(entry["source"])
            entry.update(authority_level=level, content_class="P", status=entry.get("status", status))
        for sid, src in sources.items():
            level, status = classify_source(sid)
            src.update(authority_level=level, content_class="P", status=status)
        return sections, sources
    wrapper.__name__ = fetch.__name__
    wrapper.__wrapped__ = fetch
    return wrapper


class SourceTextError(RuntimeError):
    """A pinned section or excerpt anchor could not be found in the fetched source."""


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_get(url: str, timeout: int = 60) -> bytes:
    # eCFR rejects uncompressed requests (HTTP 406), so always ask for gzip.
    req = Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "Mozilla/5.0 (rtm-knowledge-graph)"})
    with urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        body = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return body


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def excerpt(text: str, start: str, end: str, source: str) -> str:
    i = text.find(start)
    if i < 0:
        raise SourceTextError(f"[{source}] start anchor not found: {start!r}")
    j = text.find(end, i)
    if j < 0:
        raise SourceTextError(f"[{source}] end anchor not found after start: {end!r}")
    return text[i:j + len(end)]


def paragraph(text: str, start: str, next_start: str | None, source: str) -> str:
    """Text from `start` up to (not including) `next_start`, or to the end."""
    i = text.find(start)
    if i < 0:
        raise SourceTextError(f"[{source}] paragraph marker not found: {start!r}")
    if next_start is None:
        return text[i:].strip()
    j = text.find(next_start, i + len(start))
    if j < 0:
        raise SourceTextError(f"[{source}] next paragraph marker not found: {next_start!r}")
    return text[i:j].strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _source(title: str, url: str, version_date: str, text: str) -> dict:
    # content_sha256 hashes the normalized text, not the raw bytes, so page
    # chrome (scripts, tracking tokens) never registers as a content change.
    return {
        "title": title,
        "url": url,
        "version_date": version_date,
        "content_sha256": sha256(text),
        "retrieved_at": _now(),
    }


def _excerpts(text: str, spec: dict, source_id: str, citation: str) -> dict[str, dict]:
    return {
        eid: {
            "label": label,
            "source": source_id,
            "citation": citation,
            "text": excerpt(text, start, end, source=eid),
        }
        for eid, (label, start, end) in spec.items()
    }


def _pdf_text(raw: bytes) -> str:
    from pypdf import PdfReader
    return normalize(" ".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages))


def _html_main_text(raw: bytes) -> tuple[str, str]:
    """Return (normalized <main> text, 'content current as of' date) for an fda.gov page."""
    page = raw.decode("utf-8", errors="replace")
    match = re.search(r"<main.*?</main>", page, re.S)
    body = match.group(0) if match else page
    body = re.sub(r"<(script|style).*?</\1>", " ", body, flags=re.S)
    text = normalize(html.unescape(re.sub(r"<[^>]+>", " ", body)))
    text = re.sub(r"\s+([,.;:])", r"\1", text)  # drop spaces left where inline links were
    stamp = re.search(r'<time datetime="(\d{4}-\d{2}-\d{2})', page)
    return text, stamp.group(1) if stamp else ""


# ---------------------------------------------------------------------------
# Source fetchers — each returns (sections, sources) in the snapshot schema
# ---------------------------------------------------------------------------

@_tagged
def fetch_ecfr() -> tuple[dict, dict]:
    """Every section in ECFR_SECTIONS at eCFR's current version."""
    titles = json.loads(http_get(f"{ECFR}/titles.json"))["titles"]
    as_of = {str(t["number"]): t["up_to_date_as_of"] for t in titles}
    parts: dict[tuple[str, str], ET.Element] = {}
    part_text: dict[tuple[str, str], list[str]] = {}
    sections: dict[str, dict] = {}
    for sid, (title, part, label, anchors) in ECFR_SECTIONS.items():
        key = (title, part)
        if key not in parts:
            parts[key] = ET.fromstring(http_get(f"{ECFR}/full/{as_of[title]}/title-{title}.xml?part={part}"))
            part_text[key] = []
        elem = next((e for e in parts[key].iter("DIV8") if e.get("N") == sid), None)
        if elem is None:
            raise SourceTextError(f"[eCFR] §{sid} not found in {title} CFR {part} as of {as_of[title]}")
        text = normalize(" ".join(elem.itertext()))
        part_text[key].append(text)
        if anchors:
            text = excerpt(text, *anchors, source=f"eCFR {sid}")
        sections[sid] = {
            "label": label,
            "source": f"ecfr:{title}-{part}",
            "citation": f"{title} CFR {sid}",
            "text": text,
        }
    # Hash only the pinned sections' text, so amendments elsewhere in a part
    # don't flag a change that cannot affect any prompt.
    sources = {
        f"ecfr:{title}-{part}": _source(
            f"{title} CFR Part {part} (eCFR)",
            f"https://www.ecfr.gov/current/title-{title}/part-{part}",
            as_of[title],
            " ".join(texts),
        )
        for (title, part), texts in part_text.items()
    }
    return sections, sources


@_tagged
def fetch_superseded_qs_regulation() -> tuple[dict, dict]:
    """The pre-QMSR QS regulation, as eCFR served it on QS_REG_DATE (comparison only)."""
    root = ET.fromstring(http_get(f"{ECFR}/full/{QS_REG_DATE}/title-21.xml?part=820"))
    texts = {e.get("N"): normalize(" ".join(e.itertext())) for e in root.iter("DIV8")}
    sections = {}
    for sid, (section, label, start, next_start) in QS_REG_PARAGRAPHS.items():
        if section not in texts:
            raise SourceTextError(f"[QS regulation] §{section} not found as of {QS_REG_DATE}")
        sections[sid] = {
            "label": label,
            "source": "ecfr:21-820-qsreg",
            "citation": f"21 CFR {sid.split(':', 1)[1]} (superseded)",
            "status": "superseded",
            "text": paragraph(texts[section], start, next_start, source=sid),
        }
    pinned = " ".join(s["text"] for s in sections.values())
    source = _source(
        "21 CFR Part 820, Quality System Regulation (superseded 2026-02-02; eCFR as of "
        f"{QS_REG_DATE})",
        f"https://www.ecfr.gov/on/{QS_REG_DATE}/title-21/part-820",
        QS_REG_DATE, pinned,
    )
    return sections, {"ecfr:21-820-qsreg": source}


@_tagged
def fetch_fr_preamble() -> tuple[dict, dict]:
    raw = http_get(FR_TEXT).decode("utf-8", errors="replace")
    text = re.sub(r"\[\[Page \d+\]\]", " ", raw)
    text = normalize(html.unescape(re.sub(r"<[^>]+>", " ", text)))
    source = _source(
        "Medical Devices; Quality System Regulation Amendments (Final Rule), 89 FR 7496",
        FR_PAGE, "2024-02-02", text,
    )
    return _excerpts(text, FR_EXCERPTS, "fr:89-FR-7496", "89 FR 7496"), {"fr:89-FR-7496": source}


@_tagged
def fetch_510k_guidance() -> tuple[dict, dict]:
    text = _pdf_text(http_get(GUIDANCE_510K))
    # Drop running page headers that land mid-sentence at page breaks.
    text = re.sub(r" \d+ Contains Nonbinding Recommendations", "", text)
    source = _source(_G, GUIDANCE_510K, "2017-10-25", text)
    return (
        _excerpts(text, GUIDANCE_EXCERPTS, "fda:510k-change-2017", "FDA 510(k) Change Guidance (2017)"),
        {"fda:510k-change-2017": source},
    )


@_tagged
def fetch_cp_7382_850() -> tuple[dict, dict]:
    text = _pdf_text(http_get(CP_7382_850))
    text = re.sub(r" PROGRAM 7382\.850 Page \d+ of \d+", "", text)  # page footers
    stamp = re.search(r"IMPLEMENTATION DATE (\d{2})/(\d{2})/(\d{4})", text)
    version = f"{stamp.group(3)}-{stamp.group(1)}-{stamp.group(2)}" if stamp else ""
    source = _source(_CP, CP_7382_850, version, text)
    return (
        _excerpts(text, CP_EXCERPTS, "fda:cp-7382.850", "FDA CP 7382.850"),
        {"fda:cp-7382.850": source},
    )


@_tagged
def fetch_qmsr_page() -> tuple[dict, dict]:
    text, version = _html_main_text(http_get(QMSR_PAGE))
    source = _source(_QP, QMSR_PAGE, version, text)
    return _excerpts(text, QMSR_PAGE_EXCERPTS, "fda:qmsr-page", "FDA QMSR web page"), {"fda:qmsr-page": source}


@_tagged
def fetch_qmsr_faq() -> tuple[dict, dict]:
    text, version = _html_main_text(http_get(QMSR_FAQ))
    source = _source(_QF, QMSR_FAQ, version, text)
    return _excerpts(text, QMSR_FAQ_EXCERPTS, "fda:qmsr-faq", "FDA QMSR FAQ"), {"fda:qmsr-faq": source}


Fetcher = Callable[[], tuple[dict, dict]]

# Sources watched by the Audit page's update check.
FETCHERS: dict[str, Fetcher] = {
    "eCFR": fetch_ecfr,
    "FDA QMSR web page": fetch_qmsr_page,
    "FDA QMSR FAQ": fetch_qmsr_faq,
    "FDA Compliance Program 7382.850": fetch_cp_7382_850,
    "FDA 510(k) change guidance": fetch_510k_guidance,
}

# Everything in the snapshot, including the fixed Federal Register publication.
# The Federal Register rule and the superseded QS regulation are fixed texts, so
# only a full rebuild (scripts/refresh_regulatory_snapshot.py) fetches them.
ALL_FETCHERS: dict[str, Fetcher] = {
    "Federal Register (QMSR final rule)": fetch_fr_preamble,
    "Superseded QS regulation": fetch_superseded_qs_regulation,
    **FETCHERS,
}


# ---------------------------------------------------------------------------
# Device definition verification (PRD §6.1)
# ---------------------------------------------------------------------------

_ROMAN = {"1": "Class I", "2": "Class II", "3": "Class III"}


def fetch_device_classification(product_code: str) -> dict:
    """FDA Product Classification record for a product code, via openFDA."""
    data = json.loads(http_get(OPENFDA_CLASSIFICATION.format(code=product_code)))
    record = data["results"][0]
    return {
        "product_code": record.get("product_code", ""),
        "device_name": record.get("device_name", ""),
        "regulation_number": record.get("regulation_number", ""),
        "device_class": _ROMAN.get(str(record.get("device_class", "")), str(record.get("device_class", ""))),
    }


def verify_device_definition(
    definition: dict | None = None,
    fetch: Callable[[str], dict] = fetch_device_classification,
) -> dict:
    """Compare the pinned device definition with FDA's current classification record."""
    definition = definition if definition is not None else load_device_definition()
    code = definition.get("product_code", "")
    try:
        live = fetch(code)
    except Exception as exc:
        return {"status": "error", "product_code": code, "error": str(exc)}
    differences = [
        f"{field}: pinned {definition.get(field)!r}, FDA {live.get(field)!r}"
        for field in ("regulation_number", "device_class")
        if str(definition.get(field, "")) != str(live.get(field, ""))
    ]
    return {"status": "mismatch" if differences else "match", "product_code": code,
            "live": live, "differences": differences}


# ---------------------------------------------------------------------------
# Check + apply
# ---------------------------------------------------------------------------

@dataclass
class SectionChange:
    section_id: str
    label: str
    status: str                 # "changed" | "new"
    diff: str                   # unified diff, one regulatory paragraph per line
    new_entry: dict


@dataclass
class SourceChange:
    source_id: str
    title: str
    url: str
    pinned_version: str
    live_version: str


@dataclass
class UpdateCheck:
    checked_at: str
    changes: list[SectionChange] = field(default_factory=list)
    source_changes: list[SourceChange] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)   # fetcher name -> error
    fetched_sources: dict = field(default_factory=dict)
    checked: list[str] = field(default_factory=list)       # fetcher names that succeeded
    device_definition: dict = field(default_factory=dict)  # verify_device_definition() result

    @property
    def up_to_date(self) -> bool:
        return not self.changes and not self.source_changes


def _paragraphs(text: str) -> list[str]:
    # One line per regulatory paragraph marker — (a), (1), (i) — so diffs read naturally.
    return re.split(r"\s+(?=\([a-z0-9]{1,4}\)\s)", text)


def diff_sections(pinned: dict[str, dict], fetched: dict[str, dict]) -> list[SectionChange]:
    """Compare fetched sections to the pinned snapshot; return only the differences."""
    changes = []
    for sid, entry in fetched.items():
        old = pinned.get(sid)
        if old is None:
            changes.append(SectionChange(sid, entry["label"], "new", "", entry))
        elif old["text"] != entry["text"]:
            diff = "\n".join(difflib.unified_diff(
                _paragraphs(old["text"]), _paragraphs(entry["text"]),
                fromfile=f"pinned {sid}", tofile=f"live {sid}", lineterm="",
            ))
            changes.append(SectionChange(sid, entry["label"], "changed", diff, entry))
    return changes


def diff_sources(pinned: dict[str, dict], fetched: dict[str, dict]) -> list[SourceChange]:
    """Sources whose content changed since the snapshot, even if no pinned excerpt did."""
    changes = []
    for sid, src in fetched.items():
        old = pinned.get(sid, {})
        if old.get("content_sha256") != src["content_sha256"]:
            changes.append(SourceChange(
                sid, src["title"], src["url"], old.get("version_date", ""), src["version_date"],
            ))
    return changes


def check_regulatory_updates(
    snapshot_path: str | Path = SNAPSHOT_PATH,
    fetchers: dict[str, Fetcher] | None = None,
    device_verifier: Callable[[], dict] | None = None,
) -> UpdateCheck:
    """
    Fetch the watched sources and diff them against the pinned snapshot.

    Writes nothing. A source that fails to fetch, or whose pinned anchor has
    disappeared, is reported in `errors` and does not block the others.
    """
    snapshot = json.loads(Path(snapshot_path).read_text())
    check = UpdateCheck(checked_at=_now())
    fetched_sections: dict[str, dict] = {}
    for name, fetch in (FETCHERS if fetchers is None else fetchers).items():
        try:
            sections, sources = fetch()
        except Exception as exc:  # network failure, changed page structure, missing anchor
            check.errors[name] = str(exc)
            continue
        fetched_sections.update(sections)
        check.fetched_sources.update(sources)
        check.checked.append(name)
    check.device_definition = (device_verifier or verify_device_definition)()
    check.changes = diff_sections(snapshot.get("sections", {}), fetched_sections)
    check.source_changes = diff_sources(snapshot.get("sources", {}), check.fetched_sources)
    return check


def apply_regulatory_updates(
    check: UpdateCheck,
    reviewer: str,
    snapshot_path: str | Path = SNAPSHOT_PATH,
) -> list[str]:
    """
    Write the reviewed changes into the snapshot and reload it in memory.

    Replaces only the changed sections, and refreshes provenance for every
    source that was fetched (which also acknowledges source-level changes).
    Returns the section IDs whose text changed.
    """
    if not reviewer.strip():
        raise ValueError("A reviewer name is required to apply regulatory updates.")
    path = Path(snapshot_path)
    snapshot = json.loads(path.read_text())
    applied = [c.section_id for c in check.changes]
    for change in check.changes:
        snapshot["sections"][change.section_id] = change.new_entry
    snapshot["sources"].update(check.fetched_sources)
    snapshot["last_update"] = {
        "reviewer": reviewer.strip(),
        "applied_at": _now(),
        "checked_at": check.checked_at,
        "sections": applied,
        "sources": [c.source_id for c in check.source_changes],
    }
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    reload_snapshot()
    return applied
