"""
Rebuild data/regulatory_snapshot.json from scratch — the pinned, verbatim
regulatory text that every LLM prompt in the app is grounded in.

    python scripts/refresh_regulatory_snapshot.py

Review the git diff of the snapshot before committing it. For routine updates,
prefer the Audit page's "Check for regulatory updates" button, which shows a
diff and requires a named reviewer to apply changes.

Sources (all US-government works, public domain) and their excerpt anchors are
defined in src/regulatory_refresh.py:
  - eCFR: current QMSR 21 CFR 820, premarket pathway sections, CLIA
  - Federal Register 89 FR 7496: QMSR final rule preamble
  - FDA: 510(k) change guidance, Compliance Program 7382.850, QMSR page and FAQ

The script exits with an error if any source fails or any excerpt anchor is
missing, so a changed source can never silently produce altered text.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from regulations import SNAPSHOT_PATH  # noqa: E402
from regulatory_refresh import ALL_FETCHERS  # noqa: E402


def main() -> None:
    sections: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    for name, fetch in ALL_FETCHERS.items():
        try:
            fetched_sections, fetched_sources = fetch()
        except Exception as exc:
            sys.exit(f"{name}: {exc}")
        sections.update(fetched_sections)
        sources.update(fetched_sources)
        print(f"  fetched {name}: {len(fetched_sections)} sections")

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
        "sections": sections,
    }, indent=2, ensure_ascii=False) + "\n")

    total = sum(len(s["text"]) for s in sections.values())
    print(f"Wrote {SNAPSHOT_PATH.relative_to(ROOT)}: {len(sections)} sections, {total:,} chars")
    for sid, s in sections.items():
        print(f"  {sid:44s} {len(s['text']):6,d}  {s['label']}")


if __name__ == "__main__":
    main()
