"""
graph.py — RTM Knowledge Graph Engine

NetworkX-based in-memory graph representing the FDA RTM dependency model.
Nodes represent compliance artifacts; edges encode the regulatory dependency
type between them (satisfies, verifies, mitigates, triggers, invalidates).

No external database required — runs entirely in memory with JSON persistence
for demo/portfolio use.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import networkx as nx


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    USER_NEED = "User Need"
    DESIGN_INPUT = "Design Input"
    DESIGN_OUTPUT = "Design Output"
    VV_PROTOCOL = "V&V Protocol"
    TEST_RESULT = "Test Result"
    HAZARD = "Hazard"
    RISK_CONTROL = "Risk Control"
    CAPA = "CAPA"
    PMA_SUPPLEMENT_TRIGGER = "PMA Supplement Trigger"


# Canonical hierarchy: nodes with a lower level are "upstream" (closer to User Need),
# nodes with a higher level are "downstream" (closer to Test Results / PMA filing).
# Used by the change-impact agent to classify predecessors by position in the RTM
# rather than by raw graph edge direction, so feedback edges (e.g. VERIFIES going
# backwards from a Test Result to a Design Input) don't misclassify downstream nodes
# as upstream requirements.
HIERARCHY_LEVEL: dict[str, int] = {
    NodeType.USER_NEED: 0,
    NodeType.HAZARD: 0,
    NodeType.DESIGN_INPUT: 1,
    NodeType.RISK_CONTROL: 1,
    NodeType.DESIGN_OUTPUT: 2,
    NodeType.VV_PROTOCOL: 3,
    NodeType.TEST_RESULT: 4,
    NodeType.CAPA: 5,
    NodeType.PMA_SUPPLEMENT_TRIGGER: 6,
}


class EdgeType(str, Enum):
    SATISFIES = "satisfies"
    VERIFIES = "verifies"
    MITIGATES = "mitigates"
    TRIGGERS = "triggers"
    INVALIDATES = "invalidates"
    LINKED_TO = "linked_to"


class NodeStatus(str, Enum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    INVALIDATED = "invalidated"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"



# ---------------------------------------------------------------------------
# RTM Knowledge Graph
# ---------------------------------------------------------------------------

class RTMGraph:
    """
    In-memory knowledge graph for FDA RTM dependency management.

    Uses a directed NetworkX graph where:
      - nodes carry compliance metadata (type, status, description, timestamps)
      - edges carry relationship type and creation audit metadata

    Supports versioned snapshots for audit trail.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._snapshots: list[dict] = []
        self._audit_log: list[dict] = []

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        title: str,
        description: str = "",
        status: NodeStatus = NodeStatus.ACTIVE,
        metadata: Optional[dict] = None,
    ) -> None:
        """Add or update an RTM node."""
        self._g.add_node(
            node_id,
            node_type=node_type.value,
            title=title,
            description=description,
            status=status.value,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        self._log("node_added", {"node_id": node_id, "node_type": node_type.value})

    def update_node_status(self, node_id: str, status: NodeStatus, reason: str = "") -> None:
        """Update a node's compliance status and record to audit log."""
        if node_id not in self._g:
            raise KeyError(f"Node '{node_id}' not found in graph.")
        old_status = self._g.nodes[node_id]["status"]
        self._g.nodes[node_id]["status"] = status.value
        self._g.nodes[node_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._log("status_changed", {
            "node_id": node_id,
            "old_status": old_status,
            "new_status": status.value,
            "reason": reason,
        })

    def get_node(self, node_id: str) -> dict:
        if node_id not in self._g:
            raise KeyError(f"Node '{node_id}' not found.")
        return {"id": node_id, **self._g.nodes[node_id]}

    def all_nodes(self) -> list[dict]:
        return [{"id": nid, **data} for nid, data in self._g.nodes(data=True)]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        confidence: float = 1.0,
        extracted_by: str = "manual",
    ) -> None:
        """Add a directed dependency edge between two RTM nodes."""
        if source not in self._g or target not in self._g:
            raise KeyError("Both source and target nodes must exist before adding an edge.")
        self._g.add_edge(
            source,
            target,
            edge_type=edge_type.value,
            confidence=confidence,
            extracted_by=extracted_by,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._log("edge_added", {
            "source": source,
            "target": target,
            "edge_type": edge_type.value,
            "confidence": confidence,
        })

    def all_edges(self) -> list[dict]:
        return [
            {"source": u, "target": v, **data}
            for u, v, data in self._g.edges(data=True)
        ]

    def remove_node(self, node_id: str) -> list[dict]:
        """Remove a node and all its connected edges. Returns the removed edges."""
        if node_id not in self._g:
            raise KeyError(f"Node '{node_id}' not found.")
        removed_edges = [
            {"source": u, "target": v, **data}
            for u, v, data in list(self._g.edges(node_id, data=True))
        ] + [
            {"source": u, "target": v, **data}
            for u, v, data in list(self._g.in_edges(node_id, data=True))
        ]
        node_type = self._g.nodes[node_id].get("node_type", "")
        self._g.remove_node(node_id)
        self._log("node_removed", {
            "node_id": node_id,
            "node_type": node_type,
            "edges_removed": len(removed_edges),
        })
        return removed_edges

    def remove_edge(self, source: str, target: str) -> None:
        """Remove a directed edge between two nodes."""
        if not self._g.has_edge(source, target):
            raise KeyError(f"Edge '{source}' → '{target}' not found.")
        edge_type = self._g.edges[source, target].get("edge_type", "")
        self._g.remove_edge(source, target)
        self._log("edge_removed", {"source": source, "target": target, "edge_type": edge_type})

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def downstream_nodes(self, node_id: str) -> list[str]:
        """Return all nodes reachable downstream from node_id (BFS)."""
        if node_id not in self._g:
            raise KeyError(f"Node '{node_id}' not found.")
        return list(nx.descendants(self._g, node_id))

    def upstream_nodes(self, node_id: str) -> list[str]:
        """Return all nodes that feed into node_id (reverse BFS)."""
        if node_id not in self._g:
            raise KeyError(f"Node '{node_id}' not found.")
        return list(nx.ancestors(self._g, node_id))

    def impact_path(self, source: str, target: str) -> list[str]:
        """Return shortest dependency path between two nodes, or []."""
        try:
            return nx.shortest_path(self._g, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    # ------------------------------------------------------------------
    # Audit & snapshots
    # ------------------------------------------------------------------

    def snapshot(self) -> str:
        """Save a versioned snapshot of the current graph state. Returns snapshot ID."""
        snap_id = str(uuid.uuid4())[:8]
        self._snapshots.append({
            "snapshot_id": snap_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nodes": self.all_nodes(),
            "edges": self.all_edges(),
        })
        return snap_id

    def audit_log(self) -> list[dict]:
        return list(self._audit_log)

    def _log(self, event_type: str, payload: dict) -> None:
        self._audit_log.append({
            "event_id": str(uuid.uuid4())[:8],
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        })

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"nodes": self.all_nodes(), "edges": self.all_edges()}

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "RTMGraph":
        g = cls()
        with open(path) as f:
            data = json.load(f)
        for node in data["nodes"]:
            nid = node.pop("id")
            g._g.add_node(nid, **node)
        for edge in data["edges"]:
            g._g.add_edge(edge["source"], edge["target"], **{
                k: v for k, v in edge.items() if k not in ("source", "target")
            })
        return g

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------

    def orphaned_nodes(self) -> list[str]:
        """Nodes with no edges at all — compliance gap indicator."""
        return [n for n in self._g.nodes if self._g.degree(n) == 0]

    def missing_vv_links(self) -> list[str]:
        """Design Inputs with no incoming 'verifies' edge from a Test Result."""
        missing = []
        for nid, data in self._g.nodes(data=True):
            if data["node_type"] == NodeType.DESIGN_INPUT.value:
                incoming_types = [
                    self._g.edges[u, nid].get("edge_type")
                    for u, _ in self._g.in_edges(nid)
                ]
                if EdgeType.VERIFIES.value not in incoming_types:
                    missing.append(nid)
        return missing

    def unmet_user_needs(self) -> list[str]:
        """User Needs with no outgoing linked_to edge to a Design Input."""
        unmet = []
        for nid, data in self._g.nodes(data=True):
            if data["node_type"] == NodeType.USER_NEED.value:
                neighbors = [v for _, v in self._g.out_edges(nid)]
                linked_dis = [
                    v for v in neighbors
                    if self._g.nodes[v].get("node_type") == NodeType.DESIGN_INPUT.value
                ]
                if not linked_dis:
                    unmet.append(nid)
        return unmet

    def incomplete_design_inputs(self) -> list[str]:
        """Design Inputs with no outgoing linked_to edge to a Design Output."""
        incomplete = []
        for nid, data in self._g.nodes(data=True):
            if data["node_type"] == NodeType.DESIGN_INPUT.value:
                neighbors = [v for _, v in self._g.out_edges(nid)]
                linked_dos = [
                    v for v in neighbors
                    if self._g.nodes[v].get("node_type") == NodeType.DESIGN_OUTPUT.value
                ]
                if not linked_dos:
                    incomplete.append(nid)
        return incomplete

    def has_path(self, source: str, target: str) -> bool:
        """Return True if there is a directed path from source to target."""
        try:
            return nx.has_path(self._g, source, target)
        except nx.NodeNotFound:
            return False

    # Per-status readiness weights used by completeness_score and completeness_breakdown.
    # not_started and pending_review are intentionally equal — both represent incomplete
    # artifacts. The score moves when either transitions to active/approved.
    _STATUS_READINESS: dict = {
        NodeStatus.ACTIVE.value: 1.0,
        NodeStatus.APPROVED.value: 1.0,
        NodeStatus.PENDING_REVIEW.value: 0.5,
        NodeStatus.NOT_STARTED.value: 0.5,
        NodeStatus.INVALIDATED.value: 0.0,
    }

    def completeness_score(self) -> float:
        """
        RTM completeness score (0–100), combining two weighted components:

        Structural score (40%): fraction of nodes free from topology gaps —
        orphaned nodes, missing V&V links, unmet User Needs, incomplete
        Design Inputs.  Captures linkage problems independent of status.

        Readiness score (60%): average per-node readiness across the graph.
        active/approved = 1.0, pending_review/not_started = 0.5 (equally
        incomplete), invalidated = 0.0.  Moves as statuses change.
        """
        total = self._g.number_of_nodes()
        if total == 0:
            return 0.0
        bd = self.completeness_breakdown()
        score = 0.40 * bd["structural_score"] + 0.60 * bd["readiness_score"]
        return round(score * 100, 1)

    def completeness_breakdown(self) -> dict:
        """
        Return the sub-scores that feed completeness_score().

        Keys:
          structural_score  — float 0–1, topology completeness
          readiness_score   — float 0–1, average per-node status readiness
          structural_issues — int, total count of topology gaps
          ready_count       — int, nodes with active or approved status
          partial_count     — int, nodes with pending_review or not_started
          blocked_count     — int, nodes with invalidated status
          total             — int, total node count
        """
        total = self._g.number_of_nodes()
        if total == 0:
            return {
                "structural_score": 0.0, "readiness_score": 0.0,
                "structural_issues": 0, "ready_count": 0,
                "partial_count": 0, "blocked_count": 0, "total": 0,
            }

        structural_issues = (
            len(self.orphaned_nodes())
            + len(self.missing_vv_links())
            + len(self.unmet_user_needs())
            + len(self.incomplete_design_inputs())
        )
        structural_score = max(0.0, 1.0 - structural_issues / total)

        ready_count = partial_count = blocked_count = 0
        readiness_sum = 0.0
        for _, d in self._g.nodes(data=True):
            s = d.get("status", NodeStatus.NOT_STARTED.value)
            w = self._STATUS_READINESS.get(s, 0.5)
            readiness_sum += w
            if w == 1.0:
                ready_count += 1
            elif w == 0.0:
                blocked_count += 1
            else:
                partial_count += 1
        readiness_score = readiness_sum / total

        return {
            "structural_score": structural_score,
            "readiness_score": readiness_score,
            "structural_issues": structural_issues,
            "ready_count": ready_count,
            "partial_count": partial_count,
            "blocked_count": blocked_count,
            "total": total,
        }


# ---------------------------------------------------------------------------
# Seed data loader — hs-cTnI immunoassay, PMA device (P240052)
# ---------------------------------------------------------------------------

def build_seed_graph() -> RTMGraph:
    """
    Build a representative RTM graph for a high-sensitivity cardiac Troponin I
    (hs-cTnI) immunoassay subject to PMA submission (P240052).

    Full dependency chain (edges flow in change-impact direction):
    User Needs → Design Inputs → Design Outputs → V&V Protocols → Test Results
                                                                  → CAPAs → PMA Supplement Triggers

    Pre-loaded change scenario: tightening DI-001 LoD from ≤ 2.0 pg/mL to ≤ 1.2 pg/mL
    triggers the max-severity chain: VP-001 invalidation → CAPA-018 review →
    PM-001 PMA supplement flag.

    Regulatory framework: FDA QMSR (21 CFR Part 820, effective Feb 2, 2026),
    ISO 13485:2016, ISO 14971:2019, 21 CFR Part 814 (PMA), CLSI EP17-A2/EP05-A3.
    """
    g = RTMGraph()

    # --- User Needs ---
    g.add_node("UN-001", NodeType.USER_NEED, "Sensitive cTnI Detection for AMI Rule-Out",
               "Assay shall detect cardiac Troponin I at concentrations relevant for 0h/1h "
               "AMI rule-out protocols in the emergency department setting.")
    g.add_node("UN-002", NodeType.USER_NEED, "Rapid Result Turnaround for Emergency Use",
               "Assay shall deliver a reportable result within a timeframe compatible with "
               "emergency department clinical decision-making workflows.")

    # --- Design Inputs ---
    g.add_node("DI-001", NodeType.DESIGN_INPUT, "Analytical Sensitivity — LoD ≤ 2.0 pg/mL",
               "Assay shall achieve a Limit of Detection (LoD) ≤ 2.0 pg/mL per CLSI EP17-A2, "
               "corresponding to the 99th percentile upper reference limit. "
               "Per QMSR §820.30(c), this is a verified design input traceable to UN-001.",
               metadata={"lod_pg_ml": 2.0, "method": "CLSI EP17-A2"})
    g.add_node("DI-002", NodeType.DESIGN_INPUT, "Turnaround Time ≤ 18 Minutes Sample-to-Result",
               "Time from sample aspiration to reportable result shall not exceed 18 minutes "
               "under normal operating conditions.",
               metadata={"max_tat_min": 18})

    # --- Design Outputs ---
    g.add_node("DO-001", NodeType.DESIGN_OUTPUT, "Capture/Detection Antibody Pair Spec v1.4",
               "Monoclonal antibody pair specification: epitope mapping, conjugation protocol, "
               "and lot release criteria. 48/50 production lots passed LoD verification at release. "
               "Per QMSR §820.30(d), this design output is traceable to DI-001.",
               metadata={"version": "1.4", "lots_qualified": 48, "lots_tested": 50})
    g.add_node("DO-002", NodeType.DESIGN_OUTPUT, "Signal Quantification Algorithm Spec v2.1",
               "4PL curve-fitting model for fluorescence signal-to-concentration conversion. "
               "Calibration range: 0.5–50,000 pg/mL. Algorithm version-controlled per QMSR §820.30(d).",
               metadata={"version": "2.1", "calibration_range_pg_ml": [0.5, 50000]})

    # --- V&V Protocols ---
    g.add_node("VP-001", NodeType.VV_PROTOCOL, "VP-001: LoD/LoQ Verification — CLSI EP17-A2",
               "Verification protocol: 50 replicates × 3 reagent lots × 3 days at 4 concentration levels. "
               "Pass criterion: LoD ≤ 2.0 pg/mL with 95% detection probability. "
               "Per QMSR §820.30(f), verification confirms DI-001 is met by DO-001.")
    g.add_node("VP-002", NodeType.VV_PROTOCOL, "VP-002: Precision Validation — CLSI EP05-A3",
               "Validation protocol: repeatability, within-run, between-run, and between-lot precision "
               "across 20 days / 3 lots. Pass criterion: CV ≤ 5% at all QC levels. "
               "Per QMSR §820.30(g), validation confirms fitness for intended clinical use.")

    # --- Test Results ---
    g.add_node("TR-001A", NodeType.TEST_RESULT, "VP-001 Results — Lot 3 Non-Conformance",
               "48/50 reagent lots passed LoD verification. Lot 3 LoD measured at 2.6 pg/mL "
               "(exceeds 2.0 pg/mL specification). Root cause under investigation: "
               "antibody conjugation batch variation suspected.",
               status=NodeStatus.PENDING_REVIEW,
               metadata={"pass_rate": 0.96, "failures": 2, "nonconforming_lot": "Lot 3",
                         "lot3_lod_pg_ml": 2.6})
    g.add_node("TR-002A", NodeType.TEST_RESULT, "VP-002 Results — PASS",
               "All precision criteria met across 20 days / 3 lots / 3 operators. "
               "Maximum observed CV: 3.8% at the low QC level. All lots within specification.",
               metadata={"pass_rate": 1.0, "max_cv_pct": 3.8, "days": 20})

    # --- Hazards ---
    g.add_node("H-001", NodeType.HAZARD, "False Negative Result — Missed AMI",
               "Hazard: assay returns a negative result for a patient with acute myocardial "
               "infarction, leading to inappropriate discharge from the emergency department. "
               "Root cause pathway: LoD insufficient to detect low-level troponin at 0h presentation.")
    g.add_node("H-002", NodeType.HAZARD, "Erroneous Result — Sample Interference",
               "Hazard: endogenous interferents (hemolysis, lipemia, icterus) suppress or elevate "
               "signal, producing a clinically incorrect troponin concentration. "
               "Root cause pathway: antibody pair susceptible to matrix interference.")

    # --- Risk Controls ---
    g.add_node("RC-001", NodeType.RISK_CONTROL, "ISO 14971 — False Negative Hazard Mitigation",
               "Risk control per ISO 14971:2019: false negative result hazard (missed AMI). "
               "Mitigation: LoD ≤ 2.0 pg/mL specification (DI-001) and mandatory VP-001 "
               "re-verification after any analytical sensitivity specification change.")
    g.add_node("RC-002", NodeType.RISK_CONTROL, "CLSI EP07 — Interference Susceptibility Control",
               "Interference testing per CLSI EP07-A3: hemolysis (H-index ≤ 200), "
               "lipemia (L-index ≤ 300), icterus (bilirubin ≤ 20 mg/dL). "
               "All interferents tested at clinically relevant concentrations.")

    # --- CAPAs ---
    g.add_node("CAPA-018", NodeType.CAPA, "CAPA-018: Lot 3 LoD Non-Conformance",
               "Root cause analysis for Lot 3 VP-001 non-conformance (LoD = 2.6 pg/mL). "
               "Suspected cause: antibody conjugation process variation. "
               "Corrective action: revised conjugation SOP and tightened incoming QC criteria.",
               status=NodeStatus.PENDING_REVIEW)

    # --- PMA Supplement Trigger ---
    g.add_node("PM-001", NodeType.PMA_SUPPLEMENT_TRIGGER, "PMA Supplement Trigger — LoD Spec Change",
               "If the approved LoD specification (DI-001) tightens by >20% from the PMA-approved "
               "value of 2.0 pg/mL, or if post-market LoD verification fails at any manufacturing "
               "site, a Prior Approval Supplement (PAS) is required per 21 CFR 814.39. "
               "A 30-Day Notice may apply for manufacturing process changes per 21 CFR 814.39(f).",
               metadata={"threshold_pct": 0.20, "metric": "LoD_pg_mL",
                         "approved_lod_pg_ml": 2.0, "regulation": "21 CFR 814.39"})

    # --- Edges ---
    g.add_edge("UN-001", "DI-001", EdgeType.LINKED_TO)
    g.add_edge("UN-002", "DI-002", EdgeType.LINKED_TO)

    g.add_edge("DI-001", "DO-001", EdgeType.LINKED_TO)
    g.add_edge("DI-002", "DO-002", EdgeType.LINKED_TO)

    g.add_edge("DO-001", "VP-001", EdgeType.LINKED_TO)
    g.add_edge("DO-002", "VP-002", EdgeType.LINKED_TO)

    g.add_edge("VP-001", "TR-001A", EdgeType.LINKED_TO)
    g.add_edge("VP-002", "TR-002A", EdgeType.LINKED_TO)

    g.add_edge("H-001", "RC-001", EdgeType.TRIGGERS)
    g.add_edge("RC-001", "DI-001", EdgeType.LINKED_TO)

    g.add_edge("H-002", "RC-002", EdgeType.TRIGGERS)
    g.add_edge("RC-002", "DO-001", EdgeType.LINKED_TO)

    g.add_edge("TR-001A", "CAPA-018", EdgeType.TRIGGERS)
    g.add_edge("CAPA-018", "PM-001", EdgeType.TRIGGERS)

    g.add_edge("TR-001A", "DI-001", EdgeType.VERIFIES)
    g.add_edge("TR-002A", "DI-002", EdgeType.VERIFIES)
    return g
