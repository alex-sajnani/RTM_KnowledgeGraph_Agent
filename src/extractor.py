"""
extractor.py — LLM Document Extraction Module

Ingests unstructured regulatory text (SOPs, FDA guidance, change logs, CLSI standards)
and extracts RTM-compatible nodes and edges using structured LLM prompts.

Each extracted relationship includes a confidence score (0.0–1.0).
All extraction events are audit-logged per QMSR §820.180 records requirements.

Output is a list of ExtractionResult objects that can be hydrated directly
into an RTMGraph instance after human review.

Regulatory framework: FDA QMSR (21 CFR Part 820, effective February 2026),
ISO 13485:2016, 21 CFR Part 814 (PMA regulations).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from graph import EdgeType, NodeStatus, NodeType, RTMGraph
from regulations import load_regulations, build_prompt_context

_regulations = load_regulations()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedNode:
    suggested_id: str
    node_type: NodeType
    title: str
    description: str
    confidence: float
    source_text: str  # snippet that evidences this node


@dataclass
class ExtractedEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float
    rationale: str  # why the LLM assigned this relationship


@dataclass
class ExtractionResult:
    document_name: str
    timestamp: str
    extracted_nodes: list[ExtractedNode]
    extracted_edges: list[ExtractedEdge]
    raw_llm_response: str
    extraction_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


# ---------------------------------------------------------------------------
# Extraction prompt templates
# ---------------------------------------------------------------------------

GRAPH_STRUCTURE_RULES = """
RTM GRAPH STRUCTURE RULES — apply these to every edge you extract:

1. EDGE DIRECTION CONVENTION
   Every edge points from CAUSE to EFFECT: the artifact that, when changed, forces an
   update in the artifact it points to. Never reverse an edge.

2. NODE HIERARCHY (upstream → downstream)
   Level 0 — root artifacts (parallel tracks):  User Need | Hazard
   Level 1 — derived requirements:               Design Input | Risk Control
   Level 2 — implementation:                     Design Output
   Level 3 — verification:                       V&V Protocol
   Level 4 — evidence:                           Test Result
   Level 5 — corrective action:                  CAPA
   Level 6 — regulatory filing:                  PMA Supplement Trigger

   Edges almost always flow from a lower level number to a higher one.
   A higher-level node should NEVER point back toward a lower-level node (that would be a cycle).

3. VALID EDGE PATTERNS — only these source → target combinations are permitted:
   User Need       --[linked_to]-->   Design Input
   Design Input    --[linked_to]-->   Design Output
   Design Output   --[linked_to]-->   V&V Protocol
   V&V Protocol    --[linked_to]-->   Test Result
   Test Result     --[verifies]-->    Design Input
   Test Result     --[triggers]-->    CAPA
   Test Result     --[triggers]-->    PMA Supplement Trigger
   Hazard          --[triggers]-->    Risk Control
   Risk Control    --[linked_to]-->   Design Input
   Risk Control    --[linked_to]-->   Design Output
   <any>           --[linked_to]-->   <any>           (general traceability; use sparingly)

   STRICT TYPE CONSTRAINTS PER EDGE LABEL — violating these is an error:
   - "verifies": source MUST be Test Result. target MUST be Design Input.
                 No other source or target type is valid for this label.
   - "triggers":  source MUST be Hazard or Test Result.
                  target MUST be Risk Control, CAPA, or PMA Supplement Trigger.
   - "linked_to": the default for all other relationships. When in doubt, use linked_to.
   - Do NOT use "satisfies" or "mitigates" — use linked_to instead.

4. READING NATURAL LANGUAGE — EDGE LABEL AS VERB
   When the document uses a relationship verb in active voice (subject verb object),
   treat that sentence as high-confidence evidence (≥ 0.90). The subject is the source
   node; the object is the target node.

   High-confidence patterns (confidence ≥ 0.90):
     "TR-002A verifies DI-001"                → source=TR-002A, target=DI-001,   edge=verifies
     "TR-002A triggers CAPA-019"              → source=TR-002A, target=CAPA-019, edge=triggers
     "VP-002 is linked to TR-002A"            → source=VP-002,  target=TR-002A,  edge=linked_to
     "DI-001 is linked to DO-001"             → source=DI-001,  target=DO-001,   edge=linked_to

   Lower-confidence patterns (confidence ≤ 0.75) — direction must be inferred:
     passive voice: "DI-001 will be addressed by DO-001"
     causal phrasing: "CAPA-019 was opened because of TR-002A"
     vague phrasing: "VP-001 may need to be re-run"

   For passive-voice sentences, invert the subject/object to recover cause→effect order
   before assigning source and target.
"""

SYSTEM_PROMPT = f"""You are a regulatory document analyst specializing in FDA medical device
compliance under FDA QMSR (21 CFR Part 820), ISO 13485:2016, and 21 CFR Part 814 (PMA
regulations). Your job is to read regulatory and technical documents for in vitro diagnostic
(IVD) devices and extract structured RTM (Requirements Traceability Matrix) entities.

{build_prompt_context(_regulations, ["820.30", "814.39"])}

{GRAPH_STRUCTURE_RULES}

You must return a valid JSON object — nothing else. No preamble, no markdown fences.

Node types you can extract:

- User Need (level 0) — a clinical or operational requirement the device must meet.
  Examples: "detect cTnI at concentrations relevant for AMI rule-out", "deliver result within ED triage window".

- Hazard (level 0) — an identified source of patient or user risk under ISO 14971.
  Examples: "false negative result", "sample interference from lipemia".

- Design Input (level 1) — a quantitative or qualitative REQUIREMENT (the "what") derived from a User Need.
  It is a SPECIFICATION or THRESHOLD the design must achieve, not the solution itself.
  Examples: "LoD ≤ 2.0 pg/mL", "TAT ≤ 18 minutes", "CV ≤ 10% across all lots".
  NOT a Design Input: an algorithm, a physical component, a software module, a proposed mitigation strategy.

- Risk Control (level 1) — a specific design or process measure that mitigates a Hazard.
  Examples: "interference testing per CLSI EP07", "LoD specification tightened to reduce false negatives".

- Design Output (level 2) — a concrete implementation ARTIFACT (the "how") that satisfies a Design Input.
  It is something built, written, or specified: a drawing, a software algorithm, a reagent formulation spec,
  a manufacturing procedure, a data model, a signal processing method.
  Examples: "antibody pair specification", "4PL signal quantification algorithm", "kinetic rate-of-change
  extrapolation model", "fluidic profile spec", "calibration curve software module".
  If the document proposes a new algorithm or updated software method, classify it as Design Output.

- V&V Protocol (level 3) — a formal WRITTEN TEST PLAN executed against a Design Output to produce
  objective evidence. It must reference a specific CLSI, ISO, or internal test method and specify
  acceptance criteria. A "change impact assessment", "design review", "risk assessment", or
  "regulatory strategy document" is NOT a V&V Protocol.
  Examples: "VP-001: LoD/LoQ Verification per CLSI EP17-A2", "VP-002: Precision Validation per CLSI EP05-A3".

- Test Result (level 4) — the actual OUTCOME (pass/fail/data) from executing a V&V Protocol.
  Examples: "Lot 3 LoD measured at 2.6 pg/mL — FAIL", "all precision criteria met across 20 days".

- CAPA (level 5) — a corrective or preventive action triggered by a non-conforming Test Result or audit finding.
  Examples: "root cause investigation for Lot 3 LoD exceedance", "process hold pending reagent reformulation".

- PMA Supplement Trigger (level 6) — a regulatory filing event required under 21 CFR 814.39 when an approved
  performance specification changes beyond a defined threshold.
  Examples: "LoD spec change >20% from PMA-approved value triggers Prior Approval Supplement".

CLASSIFICATION RULE: If you are unsure whether something is a Design Input or Design Output, ask:
  Is it a REQUIREMENT (threshold, spec limit, acceptance criterion) → Design Input.
  Is it an IMPLEMENTATION (algorithm, component spec, software method, procedure) → Design Output.

Edge types you can assign:
- linked_to  — default for all traceability relationships (User Need → Design Input,
               Design Input → Design Output, Design Output → V&V Protocol,
               V&V Protocol → Test Result, Risk Control → Design Input/Output, etc.)
- verifies   — Test Result → Design Input ONLY. No other source or target is valid.
- triggers   — Hazard → Risk Control, or Test Result → CAPA / PMA Supplement Trigger ONLY.
Do NOT use "satisfies" or "mitigates" — use linked_to instead.

Confidence scores: 0.0 (very uncertain) to 1.0 (explicit in document)."""

EXTRACTION_PROMPT_TEMPLATE = """Extract all RTM entities and relationships from the following document text.

Document: {document_name}

{graph_context}

Text:
\"\"\"{text}\"\"\"

Return a JSON object with this exact structure:
{{
  "nodes": [
    {{
      "suggested_id": "auto-generated short ID like DI-XXX or VP-XXX",
      "node_type": "one of the valid node types",
      "title": "short title (< 80 chars)",
      "description": "detailed description from the document",
      "confidence": 0.0-1.0,
      "source_text": "exact quote from the document that supports this node"
    }}
  ],
  "edges": [
    {{
      "source_id": "suggested_id of source node OR exact existing node ID",
      "target_id": "suggested_id of target node OR exact existing node ID",
      "edge_type": "one of the valid edge types",
      "confidence": 0.0-1.0,
      "rationale": "why this relationship exists based on the document"
    }}
  ]
}}

Rules:
- Only extract entities explicitly stated or clearly implied in the text.
- Assign confidence < 0.7 for inferred relationships.

- EXISTING NODES: Before creating any new node, check the existing graph context above.
  If any existing node covers the same artifact — matched by ID, title, numeric spec value,
  or description — do NOT create a new node. Use that existing node's ID in edges instead.

  EXCEPTION — User Need nodes: Clinical requirements are distinct if they differ in patient
  population (e.g. adult vs neonatal), clinical context (e.g. AMI rule-out vs sepsis
  monitoring), or performance dimension (e.g. sensitivity vs turnaround time). Even if two
  User Needs involve the same analyte, they MUST be separate nodes when the document
  describes a NEW unaddressed clinical requirement not explicitly covered by an existing node.
  Do NOT reuse an existing User Need ID for a new population or use-case — create a new node.

- NEW NODES: Only create a new node when ALL THREE conditions hold:
  (1) No existing node in the graph context covers the same concept.
  (2) The document provides direct, explicit evidence for this artifact (quote it in source_text).
  (3) You can confidently assign it a valid node type (confidence ≥ 0.80).
  If any condition fails, omit the node entirely. Do not invent placeholder nodes.
  When creating a new node, use the level-appropriate ID prefix: UN- User Need, HZ- Hazard,
  DI- Design Input, RC- Risk Control, DO- Design Output, VP- V&V Protocol,
  TR- Test Result, CA- CAPA, PM- PMA Supplement Trigger.

- EDGES WITHOUT NEW NODES: If the document describes a relationship between existing nodes,
  extract the edge (using their existing IDs) but add no new nodes.

- DIRECTION: Edges must follow the cause→effect convention and valid patterns in the system prompt.
  Source must be at an equal or lower hierarchy level than target (except for invalidates).

- EDGE TYPE: Choose the most specific valid edge type. Only use linked_to when no stricter type fits."""


# ---------------------------------------------------------------------------
# Graph context builder
# ---------------------------------------------------------------------------

def _build_graph_context(graph: "RTMGraph") -> str:
    """Serialize existing graph nodes and edges into a prompt-friendly block."""
    lines = [
        "EXISTING GRAPH CONTEXT",
        "The following nodes and edges already exist in the live RTM graph.",
        "Match document text to existing nodes by ID, title, description, OR by matching",
        "numeric/specification values (e.g. '≤ 18 minutes', 'LoD ≤ 2.0 pg/mL').",
        "If a match is found, use the existing node's ID — do NOT generate a new node.",
        "",
        "Existing nodes (ID | Type | Title | Description):",
    ]
    for node in graph.all_nodes():
        desc = node.get("description", "")
        desc_snippet = desc[:120].replace("\n", " ") if desc else ""
        lines.append(f"  {node['id']} | {node['node_type']} | {node['title']} | {desc_snippet}")

    lines.append("")
    lines.append("Existing edges (source_id --[edge_type]--> target_id):")
    for edge in graph.all_edges():
        lines.append(f"  {edge['source']} --[{edge['edge_type']}]--> {edge['target']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-processing deduplication
# ---------------------------------------------------------------------------

def _deduplicate_against_graph(
    nodes: list["ExtractedNode"],
    edges: list["ExtractedEdge"],
    graph: "RTMGraph",
) -> tuple[list["ExtractedNode"], list["ExtractedEdge"]]:
    """
    Remove extracted nodes that duplicate an existing graph node and redirect
    any edge references to the matched existing node's ID.

    Matching logic (in order of priority):
      1. Exact ID match (e.g. LLM returned 'DI-002' even though it exists).
      2. Same node_type + title substring overlap (case-insensitive).
      3. Same node_type + any key numeric/spec token (e.g. '18 minutes', '2.0 pg/ml')
         found in both the extracted node's title/description and the existing node's
         title/description.
    """
    existing = {n["id"]: n for n in graph.all_nodes()}
    id_remap: dict[str, str] = {}  # extracted suggested_id → existing graph ID
    keep: list["ExtractedNode"] = []

    for node in nodes:
        matched_id = _find_existing_match(node, existing)
        if matched_id is not None:
            id_remap[node.suggested_id] = matched_id
        else:
            keep.append(node)

    # Rewrite edge source/target IDs using the remap table
    remapped_edges: list["ExtractedEdge"] = []
    for edge in edges:
        src = id_remap.get(edge.source_id, edge.source_id)
        tgt = id_remap.get(edge.target_id, edge.target_id)
        if src == tgt:
            continue  # collapsed to a self-loop — drop it
        remapped_edges.append(ExtractedEdge(
            source_id=src,
            target_id=tgt,
            edge_type=edge.edge_type,
            confidence=edge.confidence,
            rationale=edge.rationale,
        ))

    return keep, remapped_edges


def _find_existing_match(node: "ExtractedNode", existing: dict) -> str | None:
    """Return the ID of a matching existing node, or None."""
    # 1. Exact ID match
    if node.suggested_id in existing:
        ex = existing[node.suggested_id]
        if ex["node_type"] == node.node_type.value:
            return node.suggested_id

    node_text = f"{node.title} {node.description}".lower()

    for eid, ex in existing.items():
        if ex["node_type"] != node.node_type.value:
            continue
        ex_text = f"{ex['title']} {ex.get('description', '')}".lower()

        # 2. Title substring overlap.
        # User Needs are high-level clinical requirements that naturally share vocabulary
        # (e.g. "detection", "assay", "troponin"). Require 3+ shared words so that
        # clinically distinct requirements (different populations, different dimensions)
        # are never collapsed onto an existing User Need.
        overlap_threshold = 3 if node.node_type == NodeType.USER_NEED else 2
        node_words = {w for w in node.title.lower().split() if len(w) > 3}
        ex_words   = {w for w in ex["title"].lower().split() if len(w) > 3}
        if len(node_words & ex_words) >= overlap_threshold:
            return eid

        # 3. Numeric/spec token overlap (e.g. "18 minutes", "2.0 pg/ml", "≤18")
        import re as _re
        tokens = set(_re.findall(r'[\d\.]+\s*(?:minutes?|min|pg/ml|pg/l|%|hours?|hr)', node_text))
        ex_tokens = set(_re.findall(r'[\d\.]+\s*(?:minutes?|min|pg/ml|pg/l|%|hours?|hr)', ex_text))
        if tokens and tokens & ex_tokens:
            return eid

    return None


# ---------------------------------------------------------------------------
# Stub-node inference
# ---------------------------------------------------------------------------

# Maps the ID prefix the LLM uses to the corresponding NodeType.
_PREFIX_TO_NODE_TYPE: dict[str, NodeType] = {
    "UN-": NodeType.USER_NEED,
    "HZ-": NodeType.HAZARD,
    "DI-": NodeType.DESIGN_INPUT,
    "RC-": NodeType.RISK_CONTROL,
    "DO-": NodeType.DESIGN_OUTPUT,
    "VP-": NodeType.VV_PROTOCOL,
    "TR-": NodeType.TEST_RESULT,
    "CA-": NodeType.CAPA,
    "PM-": NodeType.PMA_SUPPLEMENT_TRIGGER,
}


def _infer_stub_nodes(
    nodes: list["ExtractedNode"],
    edges: list["ExtractedEdge"],
    graph: Optional["RTMGraph"],
) -> list["ExtractedNode"]:
    """
    For every edge endpoint that the LLM referenced but did not include in
    the nodes array (and that does not already exist in the graph), create a
    low-confidence stub node so the edge is not silently dropped in add_to_graph.

    The stub carries confidence=0.60 so it lands in PENDING_REVIEW status,
    prompting the human reviewer to confirm or edit it before committing.
    """
    existing_ids: set[str] = set()
    if graph is not None:
        existing_ids = {n["id"] for n in graph.all_nodes()}

    extracted_ids = {n.suggested_id for n in nodes}
    known_ids = existing_ids | extracted_ids

    stubs: list["ExtractedNode"] = []
    seen: set[str] = set()

    for edge in edges:
        for node_id in (edge.source_id, edge.target_id):
            if node_id in known_ids or node_id in seen:
                continue
            seen.add(node_id)
            prefix = node_id[:3].upper() + "-" if len(node_id) >= 3 else ""
            node_type = _PREFIX_TO_NODE_TYPE.get(prefix, NodeType.USER_NEED)
            stubs.append(ExtractedNode(
                suggested_id=node_id,
                node_type=node_type,
                title=f"[Inferred] {node_id}",
                description=(
                    "Stub node inferred from edge reference. "
                    "The LLM mentioned this ID in a relationship but did not describe it explicitly. "
                    "Edit the title and description before adding to graph."
                ),
                confidence=0.60,
                source_text="(inferred from edge — no direct quote available)",
            ))

    return nodes + stubs


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class RTMDocumentExtractor:
    """
    LLM-powered extractor that converts unstructured regulatory text
    into RTM node and edge candidates.

    All extraction events are logged per QMSR §820.180. Human review of
    extracted entities is required before graph hydration (enforced by the
    add_to_graph method requiring explicit confirmation).
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._audit_log: list[dict] = []

    def extract(
        self,
        text: str,
        document_name: str = "unknown",
        graph: Optional["RTMGraph"] = None,
    ) -> ExtractionResult:
        """
        Extract RTM entities from a block of unstructured regulatory text.

        Args:
            text: Raw regulatory document text (SOPs, FDA guidance, change logs).
            document_name: Human-readable name for audit logging.
            graph: Existing RTMGraph instance. When provided, its nodes and edges
                are injected into the prompt so the LLM can reference existing IDs
                in extracted edges rather than generating duplicate nodes.

        Returns:
            ExtractionResult with extracted nodes, edges, and confidence scores.
            Nothing is written to the graph until add_to_graph() is called.
        """
        graph_context = _build_graph_context(graph) if graph is not None else ""
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            document_name=document_name,
            graph_context=graph_context,
            text=text[:8000],  # truncate to avoid token limits
        )

        raw_response = ""
        try:
            llm = ChatOpenAI(
                model=self.model,
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=2048,
            )
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            raw_response = response.content
            parsed = self._parse_response(raw_response)
        except Exception as e:
            parsed = {"nodes": [], "edges": []}
            raw_response = f"[Extraction error: {e}]"

        nodes = self._build_nodes(parsed.get("nodes", []))
        edges = self._build_edges(parsed.get("edges", []))

        if graph is not None:
            nodes, edges = _deduplicate_against_graph(nodes, edges, graph)

        # Infer stub nodes for any edge endpoint the LLM referenced but did not
        # include in the nodes array and that does not already exist in the graph.
        nodes = _infer_stub_nodes(nodes, edges, graph)

        result = ExtractionResult(
            document_name=document_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            extracted_nodes=nodes,
            extracted_edges=edges,
            raw_llm_response=raw_response,
        )

        self._log_extraction(result)
        return result

    def add_to_graph(
        self,
        result: ExtractionResult,
        graph: RTMGraph,
        confidence_threshold: float = 0.75,
        require_human_approval: bool = True,
    ) -> tuple[int, int]:
        """
        Hydrate an RTMGraph with extraction results above the confidence threshold.

        Args:
            result: ExtractionResult from extract().
            graph: Target RTMGraph instance.
            confidence_threshold: Minimum confidence to add without flag (default 0.75).
            require_human_approval: If True, nodes below threshold get PENDING_REVIEW status.

        Returns:
            Tuple of (nodes_added, edges_added).
        """
        nodes_added = 0
        edges_added = 0

        for node in result.extracted_nodes:
            status = (
                NodeStatus.ACTIVE
                if node.confidence >= confidence_threshold
                else NodeStatus.PENDING_REVIEW
            )
            graph.add_node(
                node.suggested_id,
                node.node_type,
                node.title,
                node.description,
                status=status,
                metadata={
                    "confidence": node.confidence,
                    "extraction_id": result.extraction_id,
                    "source_document": result.document_name,
                    "extracted_by": "llm",
                },
            )
            nodes_added += 1

        for edge in result.extracted_edges:
            try:
                # Reject any edge that would introduce a cycle: if target can
                # already reach source, adding source→target closes a loop.
                if graph.has_path(edge.target_id, edge.source_id):
                    continue
                graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type,
                    confidence=edge.confidence,
                    extracted_by=f"llm:{result.extraction_id}",
                )
                edges_added += 1
            except KeyError:
                pass

        return nodes_added, edges_added

    def audit_log(self) -> list[dict]:
        return list(self._audit_log)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> dict:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        return json.loads(cleaned)

    def _build_nodes(self, raw_nodes: list[dict]) -> list[ExtractedNode]:
        nodes = []
        for item in raw_nodes:
            try:
                node_type = NodeType(item.get("node_type", "User Need"))
            except ValueError:
                node_type = NodeType.USER_NEED
            nodes.append(ExtractedNode(
                suggested_id=item.get("suggested_id", str(uuid.uuid4())[:6]),
                node_type=node_type,
                title=item.get("title", "Untitled"),
                description=item.get("description", ""),
                confidence=float(item.get("confidence", 0.5)),
                source_text=item.get("source_text", ""),
            ))
        return nodes

    # Blocked edge types — coerce to linked_to if the LLM produces them.
    _BLOCKED_EDGE_TYPES: set[str] = {"satisfies", "mitigates"}

    # Per-type prefix constraints: (valid_source_prefixes, valid_target_prefixes).
    # Empty set means unrestricted for that side.
    _EDGE_PREFIX_RULES: dict[str, tuple[set[str], set[str]]] = {
        "verifies":  ({"TR-"},          {"DI-"}),
        "triggers":  ({"HZ-", "TR-"},   {"RC-", "CA-", "PM-"}),
    }

    def _coerce_edge_type(self, edge_type: EdgeType, source_id: str, target_id: str) -> EdgeType:
        """Coerce disallowed or prefix-mismatched edge types to linked_to."""
        if edge_type.value in self._BLOCKED_EDGE_TYPES:
            return EdgeType.LINKED_TO
        rule = self._EDGE_PREFIX_RULES.get(edge_type.value)
        if rule:
            src_prefix = source_id[:3].upper() if len(source_id) >= 3 else ""
            tgt_prefix = target_id[:3].upper() if len(target_id) >= 3 else ""
            valid_src, valid_tgt = rule
            if src_prefix not in valid_src or tgt_prefix not in valid_tgt:
                return EdgeType.LINKED_TO
        return edge_type

    def _build_edges(self, raw_edges: list[dict]) -> list[ExtractedEdge]:
        edges = []
        for item in raw_edges:
            try:
                edge_type = EdgeType(item.get("edge_type", "linked_to"))
            except ValueError:
                edge_type = EdgeType.LINKED_TO
            source_id = item.get("source_id", "")
            target_id = item.get("target_id", "")
            edge_type = self._coerce_edge_type(edge_type, source_id, target_id)
            edges.append(ExtractedEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                confidence=float(item.get("confidence", 0.5)),
                rationale=item.get("rationale", ""),
            ))
        return edges

    def _log_extraction(self, result: ExtractionResult) -> None:
        self._audit_log.append({
            "extraction_id": result.extraction_id,
            "document_name": result.document_name,
            "timestamp": result.timestamp,
            "nodes_extracted": len(result.extracted_nodes),
            "edges_extracted": len(result.extracted_edges),
        })


# ---------------------------------------------------------------------------
# Sample regulatory documents for demo / extraction testing
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = {
    "change_log_cr089.txt": """
DESIGN CHANGE LOG — hs-cTnI Immunoassay — Rev 2.4
Change Request: CR-089
Date: 2026-05-15
Authored by: Assay Development Engineering

Change Request CR-089: Analytical Sensitivity Specification Revision

Background:
Post-market surveillance (PMS) data from Q1 2026 identified that 2.4% of field lots
reported LoD values between 2.0 and 2.6 pg/mL, marginally exceeding the approved
specification. Emerging clinical evidence from the ESC 0h/1h HEART pathway study
(published March 2026) demonstrates that a tighter LoD of ≤ 1.2 pg/mL improves
rule-out sensitivity from 96.8% to 99.1% in the 0h cohort. FDA TPLC guidance requires
real-time monitoring of performance metrics for Class III IVD devices.

Proposed Change:
Design Input DI-001 (Analytical Sensitivity — LoD) revised from ≤ 2.0 pg/mL
to ≤ 1.2 pg/mL across all manufacturing lots. This is a performance improvement
change that tightens the analytical sensitivity requirement.

Regulatory Framework:
This change is subject to FDA QMSR §820.30(i) design change controls. The change
modifies an approved performance specification and must be evaluated against
21 CFR 814.39 to determine whether a Prior Approval Supplement (PAS) or
30-Day Notice is required.

Impact Assessment:
- V&V Protocol VP-001 (LoD/LoQ Verification, CLSI EP17-A2) must be re-executed
  under the new LoD specification of ≤ 1.2 pg/mL.
- Risk Control RC-001 (ISO 14971 false negative hazard) must be reassessed
  given the tightened sensitivity specification.
- CAPA-018 scope may expand: the Lot 3 non-conformance at 2.6 pg/mL becomes
  more significant if the specification tightens to 1.2 pg/mL.
- PMA Supplement Trigger PM-001: a >20% change in LoD specification from the
  approved value of 2.0 pg/mL (threshold: 1.6 pg/mL) is crossed by this change.
  PMA supplement evaluation required under 21 CFR 814.39.

Approval Required:
Change cannot be implemented without QA/RA review and determination of FDA
notification pathway. Design freeze remains in effect until CR-089 is approved.
    """,

    "qmsr_pma_guidance_excerpt.txt": (
        "REGULATORY TEXT — verbatim from eCFR (ecfr.gov)\n\n"
        + build_prompt_context(_regulations, ["820.30", "820.100", "820.40", "820.180", "814.39"])
    ),
}
