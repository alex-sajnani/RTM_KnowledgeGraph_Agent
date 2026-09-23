"""
FDA inspection readiness, framed by Compliance Program 7382.850.

Since 2026-02-02, FDA inspects device manufacturers under CP 7382.850, which
organizes QMSR requirements into QMS Areas made of elements, each tied to ISO
13485 clauses (Attachment A). This module maps the RTM's own findings onto the
elements the RTM can actually evidence, so the Audit page can show readiness in
the terms an investigator uses.

The mapping is deterministic and uses only existing RTMGraph analytics. Element
names and clause lists are copied from CP 7382.850 Attachment A; a test checks
them against the pinned snapshot so they cannot drift from the source.
Elements the RTM does not model (e.g. design transfer, purchasing changes) are
reported as not assessed rather than implied to be clear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from graph import NodeStatus, NodeType, RTMGraph

CHANGE_CONTROL = "Change Control"
DESIGN_AND_DEVELOPMENT = "Design and Development"
MEASUREMENT_ANALYSIS_IMPROVEMENT = "Measurement, Analysis, and Improvement"

AREA_PURPOSE = {
    CHANGE_CONTROL: "To ensure changes are adequately evaluated for risk and impact on products "
                    "and processes prior to implementation.",
    DESIGN_AND_DEVELOPMENT: "To ensure the manufacturer’s design and development activities result "
                            "in safe and effective medical device that meets its intended use.",
    MEASUREMENT_ANALYSIS_IMPROVEMENT: "To ensure monitoring, measurement, analysis, and improvement "
                                      "activities are effective in identifying and reducing risks "
                                      "that impact the product and/or the QMS.",
}


@dataclass
class ElementAssessment:
    area: str
    element: str
    requirements: str           # clause list exactly as written in CP 7382.850 Attachment A
    status: str                 # "gap" | "clear" | "not_assessed"
    basis: str                  # what RTM evidence the assessment rests on
    findings: list[str] = field(default_factory=list)


def _label(graph: RTMGraph, node_id: str) -> str:
    try:
        node = graph.get_node(node_id)
        return f"`{node_id}` {node['title']}"
    except KeyError:
        return f"`{node_id}`"


# ---------------------------------------------------------------------------
# Checks — each returns a list of human-readable findings (empty = clear)
# ---------------------------------------------------------------------------

def _planning(graph: RTMGraph, _pending: list) -> list[str]:
    return [f"{_label(graph, n)} has no trace links" for n in graph.orphaned_nodes()]


def _inputs(graph: RTMGraph, _pending: list) -> list[str]:
    return [f"{_label(graph, n)} is not translated into any Design Input" for n in graph.unmet_user_needs()]


def _outputs(graph: RTMGraph, _pending: list) -> list[str]:
    return [f"{_label(graph, n)} has no Design Output" for n in graph.incomplete_design_inputs()]


def _verification(graph: RTMGraph, _pending: list) -> list[str]:
    findings = {n: f"{_label(graph, n)} has no verifying Test Result" for n in graph.missing_vv_links()}
    for gap in graph.chain_verification_gaps():
        if gap["node_type"] == NodeType.DESIGN_INPUT.value:
            findings[gap["id"]] = f"{_label(graph, gap['id'])} — {gap['issue']}"
    return list(findings.values())


def _validation(graph: RTMGraph, _pending: list) -> list[str]:
    return [
        f"{_label(graph, gap['id'])} — {gap['issue']}"
        for gap in graph.chain_verification_gaps()
        if gap["node_type"] == NodeType.USER_NEED.value
    ]


def _design_files(graph: RTMGraph, _pending: list) -> list[str]:
    waiting = {NodeStatus.PENDING_REVIEW.value: "awaiting review", NodeStatus.INVALIDATED.value: "invalidated"}
    return [
        f"{_label(graph, n['id'])} record is {waiting[n['status']]}"
        for n in graph.all_nodes()
        if n["status"] in waiting
    ]


def _changes(graph: RTMGraph, pending: list) -> list[str]:
    findings = []
    for report in pending:
        changed = getattr(report, "changed_node_id", "")
        risk = getattr(report, "risk_level", "")
        findings.append(
            f"Impact analysis for a change to {_label(graph, changed)} ({risk} risk) "
            f"has not been approved"
        )
        vv = getattr(report, "vv_invalidations", []) or []
        if vv:
            findings.append(f"V&V re-execution outstanding for: {', '.join(vv)}")
    invalidated = [n["id"] for n in graph.all_nodes() if n["status"] == NodeStatus.INVALIDATED.value]
    if invalidated:
        findings.append(
            f"Invalidated by an approved change, awaiting re-execution: {', '.join(invalidated)}"
        )
    return findings


def _corrective_action(graph: RTMGraph, pending: list) -> list[str]:
    closed = {NodeStatus.ACTIVE.value, NodeStatus.APPROVED.value}
    findings = [
        f"{_label(graph, n['id'])} is {n['status']}"
        for n in graph.all_nodes()
        if n["node_type"] == NodeType.CAPA.value and n["status"] not in closed
    ]
    for report in pending:
        for capa in getattr(report, "capa_triggers", []) or []:
            findings.append(f"{_label(graph, capa)} scope affected by an unapproved change")
    return findings


# (area, element, requirements from CP 7382.850 Attachment A, check, basis)
ELEMENTS: list[tuple[str, str, str, Callable | None, str]] = [
    (CHANGE_CONTROL, "Product and Process Changes", "Clauses 4.1.4, 7.2.2, 7.3.9, 7.3.10, 7.5.6, 7.5.7",
     _changes, "Unapproved impact analyses and artifacts invalidated by a change"),
    (CHANGE_CONTROL, "QMS Changes", "Clauses 4.1.4, 4.2.4, 4.2.5, 5.4.2, 5.6.1, 5.6.2, 5.6.3, 8.5.1", None, ""),
    (CHANGE_CONTROL, "Software Changes", "Clauses 4.1.6, 7.5.6, 7.6", None, ""),
    (CHANGE_CONTROL, "Purchasing Changes", "Clauses 7.4.2, 7.4.3", None, ""),
    (DESIGN_AND_DEVELOPMENT, "Design and Development Planning", "Clause 7.3.2",
     _planning, "Every artifact is linked into the trace matrix"),
    (DESIGN_AND_DEVELOPMENT, "Design and Development Inputs", "Clause 7.3.3",
     _inputs, "Every User Need drives at least one Design Input"),
    (DESIGN_AND_DEVELOPMENT, "Design and Development Outputs", "Clause 7.3.4",
     _outputs, "Every Design Input has a Design Output"),
    (DESIGN_AND_DEVELOPMENT, "Design and Development Verification", "Clause 7.3.6",
     _verification, "Every Design Input is closed by a completed verifying Test Result"),
    (DESIGN_AND_DEVELOPMENT, "Design and Development Validation", "Clause 7.3.7",
     _validation, "Every User Need's chain reaches a completed Test Result"),
    (DESIGN_AND_DEVELOPMENT, "Control of Design and Development Changes", "Clause 7.3.9",
     _changes, "Unapproved impact analyses and artifacts invalidated by a change"),
    (DESIGN_AND_DEVELOPMENT, "Design and Development Files", "Clause 7.3.10",
     _design_files, "Every record in the trace matrix is reviewed and current"),
    (DESIGN_AND_DEVELOPMENT, "Customer Related Processes", "21 CFR 820.10(b)(4) and Clauses 7.2.1, 7.2.2, 7.2.3", None, ""),
    (DESIGN_AND_DEVELOPMENT, "Design and Development General", "Clause 7.3.1", None, ""),
    (DESIGN_AND_DEVELOPMENT, "Design and Development Review", "Clause 7.3.5", None, ""),
    (DESIGN_AND_DEVELOPMENT, "Design and Development Software Validation", "Clause 7.3.7", None, ""),
    (DESIGN_AND_DEVELOPMENT, "Design and Development Transfer", "Clause 7.3.8", None, ""),
    (MEASUREMENT_ANALYSIS_IMPROVEMENT, "Corrective Action", "Clause 8.5.2",
     _corrective_action, "Every CAPA in the trace matrix is closed and unaffected by pending changes"),
]

AREAS = [CHANGE_CONTROL, DESIGN_AND_DEVELOPMENT, MEASUREMENT_ANALYSIS_IMPROVEMENT]

# Pinned CP 7382.850 Attachment A table for each area.
AREA_EXCERPT = {
    CHANGE_CONTROL: "cp7382850:change-control",
    DESIGN_AND_DEVELOPMENT: "cp7382850:design-development",
    MEASUREMENT_ANALYSIS_IMPROVEMENT: "cp7382850:measurement-analysis-improvement",
}

# Additional pinned FDA text (current, authoritative) that bears on an element:
# mostly FDA's own commentary on the clause in the QMSR final rule preamble.
_ELEMENT_EXTRAS = {
    "Design and Development Inputs": ["qmsr-preamble:design-review"],
    "Design and Development Outputs": ["qmsr-preamble:design-review"],
    "Design and Development Validation": ["qmsr-preamble:design-review", "qmsr-preamble:clinical-evaluation"],
    "Design and Development Files": ["qmsr-preamble:design-records", "820.35"],
    "Product and Process Changes": ["qmsr-preamble:design-changes", "fda-qmsr-faq:pre-qmsr-records"],
    "Control of Design and Development Changes": ["qmsr-preamble:design-changes", "fda-qmsr-faq:pre-qmsr-records"],
    "Corrective Action": ["qmsr-preamble:capa", "qmsr-preamble:capa-effectiveness"],
}

# The superseded QS regulation paragraph(s) covering the same ground, for
# comparison only. FDA judged the QS regulation and ISO 13485 "substantially
# similar" in totality (89 FR 7496), not clause-for-clause identical.
SUPERSEDED_EQUIVALENTS = {
    "Design and Development Planning": ["qsreg:820.30(b)"],
    "Design and Development Inputs": ["qsreg:820.30(c)"],
    "Design and Development Outputs": ["qsreg:820.30(d)"],
    "Design and Development Verification": ["qsreg:820.30(f)"],
    "Design and Development Validation": ["qsreg:820.30(g)"],
    "Control of Design and Development Changes": ["qsreg:820.30(i)"],
    "Design and Development Files": ["qsreg:820.30(j)"],
    "Product and Process Changes": ["qsreg:820.30(i)", "qsreg:820.70(b)"],
    "Corrective Action": ["qsreg:820.100"],
}


def superseded_references(assessment: ElementAssessment) -> list[str]:
    """Former QS regulation paragraphs to show beside an element, clearly labeled superseded."""
    return SUPERSEDED_EQUIVALENTS.get(assessment.element, [])


def element_references(assessment: ElementAssessment) -> list[str]:
    """
    Snapshot section IDs to show a reviewer for one element: the QMSR section
    that incorporates ISO 13485 (§820.10), FDA's CP 7382.850 table listing the
    element and its clauses, and any element-specific FDA text. ISO 13485's own
    clause text is copyrighted and is linked (read-only portal), not reproduced.
    """
    refs = ["820.10", AREA_EXCERPT[assessment.area]]
    if assessment.area == DESIGN_AND_DEVELOPMENT:
        refs.append("qmsr-preamble:design-applicability")
    refs += _ELEMENT_EXTRAS.get(assessment.element, [])
    return list(dict.fromkeys(refs))


def assess_inspection_readiness(graph: RTMGraph, pending_reports: list[Any] | None = None) -> list[ElementAssessment]:
    """
    Assess each CP 7382.850 element the RTM can evidence.

    pending_reports: impact analyses that have run but not been approved
    (objects with changed_node_id, risk_level, vv_invalidations, capa_triggers).
    """
    pending = pending_reports or []
    results = []
    for area, element, requirements, check, basis in ELEMENTS:
        if check is None:
            results.append(ElementAssessment(area, element, requirements, "not_assessed", "Not represented in the RTM"))
            continue
        findings = check(graph, pending)
        results.append(ElementAssessment(
            area, element, requirements, "gap" if findings else "clear", basis, findings,
        ))
    return results
