"""
Fetches and caches verbatim regulatory text from the eCFR public API.

Sections fetched:
  21 CFR Part 820 — 820.30 (design controls), 820.100 (CAPA),
                     820.180 (records), 820.40 (document controls)
  42 CFR Part 493 — 493.1253 (CLIA: performance specification verification),
                     493.1255 (CLIA: calibration and calibration verification)

Cache: regulations_cache.json at the project root, refreshed every 7 days.
Fallback: if eCFR is unreachable, minimal stub strings are returned so the
app never fails to start.
"""

import json
import ssl
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

_ECFR_BASE = "https://www.ecfr.gov/api/versioner/v1"
_CACHE_TTL_DAYS = 7

# Sections to fetch: section_id -> (title, part, label)
_SECTIONS = {
    "820.30":   ("21", "820", "21 CFR §820.30 — Design Controls"),
    "820.100":  ("21", "820", "21 CFR §820.100 — Corrective and Preventive Action (CAPA)"),
    "820.180":  ("21", "820", "21 CFR §820.180 — General Requirements (Records)"),
    "820.40":   ("21", "820", "21 CFR §820.40 — Document Controls"),
    "493.1253": ("42", "493", "42 CFR §493.1253 — CLIA: Establishment and Verification of Performance Specifications"),
    "493.1255": ("42", "493", "42 CFR §493.1255 — CLIA: Calibration and Calibration Verification"),
}

# Minimal stubs used when eCFR is unreachable.
_FALLBACK = {
    "820.30":   "21 CFR §820.30: Manufacturers shall establish and maintain design control procedures.",
    "820.100":  "21 CFR §820.100: Manufacturers shall establish CAPA procedures.",
    "820.180":  "21 CFR §820.180: All required records shall be maintained and accessible.",
    "820.40":   "21 CFR §820.40: Manufacturers shall establish document control procedures.",
    "493.1253": "42 CFR §493.1253: Laboratories must establish and verify performance specifications including accuracy, precision, reportable range, reference intervals, and for quantitative procedures, the limit of detection and limit of quantitation, before reporting patient results.",
    "493.1255": "42 CFR §493.1255: Laboratories must perform calibration and calibration verification procedures using the manufacturer's instructions to ensure accuracy of patient test results throughout the reportable range.",
}


def _get_latest_ecfr_date(title: str = "21") -> str:
    try:
        url = f"{_ECFR_BASE}/versions/title-{title}.json"
        with urlopen(url, timeout=10, context=_ssl_context()) as resp:
            data = json.loads(resp.read())
        dates = sorted(
            set(v["date"] for v in data.get("content_versions", [])),
            reverse=True,
        )
        return dates[0] if dates else "2025-10-27"
    except Exception:
        return "2025-10-27"


def _fetch_part_xml(title: str, part: str, date: str) -> str:
    url = f"{_ECFR_BASE}/full/{date}/title-{title}.xml?part={part}"
    with urlopen(url, timeout=30, context=_ssl_context()) as resp:
        return resp.read().decode("utf-8")


def _extract_section_text(xml_str: str, section_id: str) -> str | None:
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return None
    for elem in root.iter("DIV8"):
        if elem.get("N") == section_id:
            parts = []
            for child in elem.iter():
                if child.text and child.text.strip():
                    parts.append(child.text.strip())
                if child.tail and child.tail.strip():
                    parts.append(child.tail.strip())
            return " ".join(parts)
    return None


def load_regulations(cache_path: str = "regulations_cache.json") -> dict[str, str]:
    """
    Returns a dict mapping section_id -> full verbatim regulatory text.
    Reads from disk cache if fresh; otherwise fetches from eCFR and writes cache.
    Falls back to stub strings if eCFR is unreachable.
    """
    cache_file = Path(cache_path)
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            age = datetime.now() - datetime.fromisoformat(cached["fetched_at"])
            if age < timedelta(days=_CACHE_TTL_DAYS):
                return cached["sections"]
        except Exception:
            pass

    try:
        # Fetch the latest date per title, then deduplicate part fetches.
        dates_by_title: dict[str, str] = {}
        parts_xml: dict[tuple, str] = {}
        for section_id, (title, part, _) in _SECTIONS.items():
            if title not in dates_by_title:
                dates_by_title[title] = _get_latest_ecfr_date(title)
            key = (title, part)
            if key not in parts_xml:
                parts_xml[key] = _fetch_part_xml(title, part, dates_by_title[title])

        sections: dict[str, str] = {}
        for section_id, (title, part, _) in _SECTIONS.items():
            text = _extract_section_text(parts_xml[(title, part)], section_id)
            sections[section_id] = text if text else _FALLBACK[section_id]

        cache_file.write_text(json.dumps({
            "fetched_at": datetime.now().isoformat(),
            "ecfr_dates": dates_by_title,
            "sections": sections,
        }, indent=2))
        return sections

    except (URLError, Exception):
        return dict(_FALLBACK)


def build_prompt_context(regulations: dict[str, str], section_ids: list[str]) -> str:
    """
    Formats the requested sections into a block suitable for injection into
    an LLM system prompt.
    """
    lines = ["--- Applicable Regulatory Text (verbatim from eCFR) ---"]
    for sid in section_ids:
        entry = _SECTIONS.get(sid)
        label = entry[2] if entry else f"CFR §{sid}"
        text = regulations.get(sid, _FALLBACK.get(sid, ""))
        lines.append(f"\n{label}:\n{text}")
    lines.append("--- End Regulatory Text ---")
    return "\n".join(lines)
