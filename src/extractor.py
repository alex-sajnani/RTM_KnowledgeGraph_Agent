"""
extractor.py — LLM Document Extraction Module

Ingests unstructured regulatory text (SOPs, FDA guidance, change logs, CLSI standards)
and extracts RTM-compatible nodes and edges using structured LLM prompts.

Each extracted relationship includes a confidence score (0.0–1.0).
All extraction events are audit-logged per QMSR §820.180 records requirements.

Output is a list of ExtractionResult objects that can be hydrated directly
into an RTMGraph instance after human review.

Regulatory framework: FDA QMSR (21 CFR Part 820, effective February 2026),
ISO 13485:2016.
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

from graph import EdgeType, HIERARCHY_LEVEL, NodeStatus, NodeType, RTMGraph
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
    is_required: bool = False  # placeholder the team must fill in; forces NOT_STARTED status
    is_in_review: bool = False  # exists but being revised/reviewed; forces PENDING_REVIEW status


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

   Edges almost always flow from a lower level number to a higher one.
   A higher-level node should NEVER point back toward a lower-level node (that would be a cycle).

3. VALID EDGE PATTERNS — only these source → target combinations are permitted:
   User Need       --[linked_to]-->   Design Input
   Design Input    --[linked_to]-->   Design Output
   Design Output   --[linked_to]-->   V&V Protocol
   V&V Protocol    --[linked_to]-->   Test Result
   Test Result     --[verifies]-->    Design Input
   Test Result     --[triggers]-->    CAPA
   Hazard          --[triggers]-->    Risk Control
   Risk Control    --[linked_to]-->   Design Input
   Risk Control    --[linked_to]-->   Design Output
   <any>           --[linked_to]-->   <any>           (general traceability; use sparingly)

   STRICT TYPE CONSTRAINTS PER EDGE LABEL — violating these is an error:
   - "verifies": source MUST be Test Result. target MUST be Design Input.
                 No other source or target type is valid for this label.
   - "triggers":  source MUST be Hazard or Test Result.
                  target MUST be Risk Control or CAPA.
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
compliance under FDA QMSR (21 CFR Part 820) and ISO 13485:2016. Your job is to read
structured RTM (Requirements Traceability Matrix) entities.

{build_prompt_context(_regulations, ["820.30"])}

Ground all regulatory reasoning in the verbatim eCFR text above — rely on the exact section text
supplied rather than your own training-knowledge recollection of the regulation, and never
paraphrase or invent regulatory language.

{GRAPH_STRUCTURE_RULES}

You must return a valid JSON object — nothing else. No preamble, no markdown fences.

Node types you can extract:

- User Need (level 0) — a clinical or operational requirement the device must meet.
  Examples: "detect cTnI at concentrations relevant for AMI rule-out", "deliver result within ED triage window".
  TRIGGER PHRASES — extract a new User Need node whenever the document contains any of:
    "new requirement", "new unaddressed requirement", "new clinical requirement", "the device must",
    "the system shall", "distinct population", "not covered by existing", "new use case", "new patient population".
  When multiple User Needs are present (e.g. different patient populations, different analytes, different clinical
  contexts), EACH must become its own node. Never collapse two distinct clinical requirements into one node.
  If the document says this requirement is "not covered by any existing specification" or is "unaddressed",
  you MUST create a new User Need node — do NOT reuse an existing one.

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
  ONLY extract a Test Result when the document explicitly states that results exist or were changed —
  e.g. measured values, pass/fail verdicts, lot-specific data, or a report that has already been produced.
  Do NOT extract a Test Result for future/planned reports ("will produce", "will generate", "to be completed").

- CAPA (level 5) — a corrective or preventive action triggered by a non-conforming Test Result or audit finding.
  Examples: "root cause investigation for Lot 3 LoD exceedance", "process hold pending reagent reformulation".

CLASSIFICATION RULE: If you are unsure whether something is a Design Input or Design Output, ask:
  Is it a REQUIREMENT (threshold, spec limit, acceptance criterion) → Design Input.
  Is it an IMPLEMENTATION (algorithm, component spec, software method, procedure) → Design Output.

Edge types you can assign:
- linked_to  — default for all traceability relationships (User Need → Design Input,
               Design Input → Design Output, Design Output → V&V Protocol,
               V&V Protocol → Test Result, Risk Control → Design Input/Output, etc.)
- verifies   — Test Result → Design Input ONLY. No other source or target is valid.
- triggers   — Hazard → Risk Control, or Test Result → CAPA ONLY.
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
      "source_text": "exact quote from the document that supports this node",
      "planned": true or false,
      "in_review": true or false
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

- PLANNED VS EXISTING (`planned` field): Set "planned": true when the document
  describes the artifact as future work the team WILL create — phrased as "will
  write/author/produce", "will need to", "to be developed", "needs to be defined",
  etc. The artifact does not exist yet; it is a placeholder the team must fill in.
  Set "planned": false when the artifact already exists — it has concrete values,
  was produced/measured, or is referenced as a current specification.
  A User Need is ALWAYS "planned": false — a clinical requirement exists the moment
  it is raised, even though everything it drives downstream may still be planned.

- IN REVIEW (`in_review` field): Set "in_review": true when the artifact EXISTS but is
  not finished — it is being revised, updated, or is under review/approval ("to be
  revised", "under revision", "pending review", "needs updating"). It is incomplete
  and must not be treated as a closed/verified artifact. Set false when finished and
  approved. If "planned" is true, leave "in_review" false (it does not exist yet).

- EXISTING NODES: Before creating any new node, check the existing graph context above.
  If any existing node covers the same artifact — matched by ID, title, numeric spec value,
  or description — do NOT create a new node. Use that existing node's ID in edges instead.

  EXCEPTION — User Need nodes: Clinical requirements are distinct if they differ in patient
  population (e.g. adult vs neonatal), clinical context (e.g. AMI rule-out vs sepsis
  monitoring), or performance dimension (e.g. sensitivity vs turnaround time). Even if two
  User Needs involve the same analyte, they MUST be separate nodes when the document
  describes a NEW unaddressed clinical requirement not explicitly covered by an existing node.
  Do NOT reuse an existing User Need ID for a new population or use-case — create a new node.

  MANDATORY USER NEED RULE: If the document contains ANY of the following phrases, you MUST
  extract a new User Need node regardless of what existing nodes are in the graph context:
    - "new requirement", "new unaddressed requirement", "unaddressed requirement"
    - "distinct population", "new population"
    - "not covered by any existing specification", "not covered by existing"
    - "the device must be capable of", "the device must detect"
    - "new clinical requirement", "new use case"
  These phrases are direct evidence of a User Need. Failure to extract a User Need node
  when these phrases appear is an extraction error.

- NEW NODES: Only create a new node when ALL THREE conditions hold:
  (1) No existing node in the graph context covers the same concept.
  (2) The document provides direct, explicit evidence for this artifact (quote it in source_text).
  (3) You can confidently assign it a valid node type (confidence ≥ 0.80).
  If any condition fails, omit the node entirely. Do not invent placeholder nodes.
  When creating a new node, use the level-appropriate ID prefix: UN- User Need, HZ- Hazard,
  DI- Design Input, RC- Risk Control, DO- Design Output, VP- V&V Protocol,
  TR- Test Result, CA- CAPA.

- EDGES WITHOUT NEW NODES: If the document describes a relationship between existing nodes,
  extract the edge (using their existing IDs) but add no new nodes.

- CROSS-CHAIN EDGE PROHIBITION (hard rule): Never create an edge that mixes a newly
  extracted node with an existing graph node. Two and only two patterns are permitted:
    (a) Both source and target are newly extracted nodes in this document — allowed.
    (b) Both source and target are existing nodes in the graph context above — allowed.
  A mixed edge (one endpoint new, one endpoint existing) is always an extraction error.
  If the document text implies such a connection, omit the edge entirely.
  Example of a PROHIBITED edge: new DI-003 → existing DO-001. This must never appear.

- CHAIN COMPLETENESS: When you extract a Design Input node, check whether the document
  also describes a concrete implementation artifact (an output spec, reagent formulation,
  algorithm, procedure, or software module) that satisfies it. If so, you MUST also
  extract the corresponding Design Output node and a linked_to edge from the Design Input
  to that Design Output. Do not leave a chain that has a DI but no DO when the document
  explicitly mentions an output artifact. Similarly, if you extract a Design Output and the
  document describes a verification plan, extract the V&V Protocol node as well.

- USER NEED → DESIGN INPUT EDGES: Whenever you extract a new User Need node AND the document
  describes a Design Input that "satisfies", "addresses", or "is derived from" that requirement,
  you MUST also extract a linked_to edge from the User Need to that Design Input.

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
# Cross-chain edge filter
# ---------------------------------------------------------------------------

def _filter_cross_chain_edges(
    nodes: list["ExtractedNode"],
    edges: list["ExtractedEdge"],
    graph: "RTMGraph",
) -> list["ExtractedEdge"]:
    """
    Drop edges that bridge a newly extracted node to an existing graph node.

    Permitted:
      - Both endpoints are newly extracted (same document chain).
      - Both endpoints already exist in the graph (existing relationship update).
    Dropped:
      - One endpoint is new, the other is existing (cross-chain hallucination).
    """
    new_ids = {n.suggested_id for n in nodes}
    existing_ids = {n["id"] for n in graph.all_nodes()}

    filtered = []
    for edge in edges:
        src_new = edge.source_id in new_ids
        tgt_new = edge.target_id in new_ids
        src_existing = edge.source_id in existing_ids
        tgt_existing = edge.target_id in existing_ids

        if src_new and tgt_new:
            filtered.append(edge)
        elif src_existing and tgt_existing:
            filtered.append(edge)
        # else: mixed (one new, one existing) → drop silently

    return filtered


# ---------------------------------------------------------------------------
# Skip-level edge filter
# ---------------------------------------------------------------------------

def _filter_skip_level_edges(
    nodes: list["ExtractedNode"],
    edges: list["ExtractedEdge"],
    graph: Optional["RTMGraph"],
) -> list["ExtractedEdge"]:
    """
    Drop `linked_to` edges that skip a hierarchy level on the verification spine
    (e.g. Design Input → V&V Protocol, which bypasses the Design Output).

    The RTM spine advances one level at a time: UN → DI → DO → VP → TR. A
    `linked_to` edge whose target sits two or more levels below its source jumps
    over at least one intermediate artifact. Chain-completeness inference
    (`_infer_missing_chain_links`) rebuilds the proper step-by-step path, so the
    skip edge is redundant and would double-link the chain. This runs BEFORE that
    inference so the missing intermediate (e.g. the Design Output) gets created.

    Only `linked_to` is filtered. `verifies` (TR → DI back-edge) and `triggers`
    (Hazard → Risk Control, Test Result → CAPA cross-track) are intentionally
    non-adjacent and exempt.
    """
    levels: dict[str, int] = {n.suggested_id: HIERARCHY_LEVEL.get(n.node_type, -1) for n in nodes}
    if graph is not None:
        for gn in graph.all_nodes():
            try:
                levels[gn["id"]] = HIERARCHY_LEVEL[NodeType(gn["node_type"])]
            except (ValueError, KeyError):
                pass

    kept: list["ExtractedEdge"] = []
    for edge in edges:
        if edge.edge_type == EdgeType.LINKED_TO:
            src_lvl = levels.get(edge.source_id)
            tgt_lvl = levels.get(edge.target_id)
            if src_lvl is not None and tgt_lvl is not None and tgt_lvl - src_lvl >= 2:
                continue  # skip-level edge — the intermediate artifact is missing
        kept.append(edge)
    return kept


# ---------------------------------------------------------------------------
# Chain completeness inference
# ---------------------------------------------------------------------------

def _infer_missing_chain_links(
    nodes: list["ExtractedNode"],
    edges: list["ExtractedEdge"],
) -> tuple[list["ExtractedNode"], list["ExtractedEdge"]]:
    """
    Enforce DI → DO → VP chain completeness within a single extraction result.

    Pass 1: For each Design Input that has no outgoing edge to a Design Output
            among the extracted nodes, create a stub DO node (confidence=0.70,
            PENDING_REVIEW) and a linked_to edge from the DI to it.

    Pass 2: For each Design Output (original or just-created stub) that has no
            outgoing edge to a V&V Protocol among the extracted nodes, wire in
            the first unconnected VP node found in the extraction. If no VP
            exists in the extraction, skip — VP may be out of scope for this doc.

    Stubs are flagged with confidence=0.70 so the human reviewer sees them as
    PENDING_REVIEW and can supply the correct title/description before committing.
    """
    extra_nodes: list["ExtractedNode"] = []
    extra_edges: list["ExtractedEdge"] = []

    def current_nodes() -> list["ExtractedNode"]:
        return nodes + extra_nodes

    def current_edges() -> list["ExtractedEdge"]:
        return edges + extra_edges

    # --- Pass 1: DI → DO ---
    di_nodes = [n for n in nodes if n.node_type == NodeType.DESIGN_INPUT]
    for di in di_nodes:
        has_do_edge = any(
            e.source_id == di.suggested_id
            and any(
                n.suggested_id == e.target_id and n.node_type == NodeType.DESIGN_OUTPUT
                for n in current_nodes()
            )
            for e in current_edges()
        )
        if has_do_edge:
            continue

        suffix = di.suggested_id.split("-", 1)[-1] if "-" in di.suggested_id else "NEW"
        do_id = f"DO-{suffix}"
        # Avoid collision with IDs already in this extraction
        taken = {n.suggested_id for n in current_nodes()}
        if do_id in taken:
            do_id = f"DO-{suffix}b"

        extra_nodes.append(ExtractedNode(
            suggested_id=do_id,
            node_type=NodeType.DESIGN_OUTPUT,
            title=f"Design Output for {di.title}",
            description=(
                f"Implementation artifact satisfying Design Input {di.suggested_id}. "
                "Inferred by chain completeness — edit title and description to match "
                "the specific output artifact described in the source document before committing."
            ),
            confidence=0.70,
            source_text="(inferred from chain completeness — no direct quote)",
            # A Design Output inferred from a planned Design Input is itself planned,
            # so it inherits the required-placeholder (NOT_STARTED) treatment.
            is_required=di.is_required,
        ))
        extra_edges.append(ExtractedEdge(
            source_id=di.suggested_id,
            target_id=do_id,
            edge_type=EdgeType.LINKED_TO,
            confidence=0.70,
            rationale=f"Chain completeness: Design Input {di.suggested_id} requires a Design Output.",
        ))

    # --- Pass 2: DO → VP ---
    # Collect VP nodes that are not yet the target of any DO→VP edge
    def connected_vp_targets() -> set[str]:
        return {
            e.target_id
            for e in current_edges()
            if any(
                n.suggested_id == e.source_id and n.node_type == NodeType.DESIGN_OUTPUT
                for n in current_nodes()
            )
        }

    do_nodes = [n for n in current_nodes() if n.node_type == NodeType.DESIGN_OUTPUT]
    unconnected_vps = [
        n for n in current_nodes()
        if n.node_type == NodeType.VV_PROTOCOL
        and n.suggested_id not in connected_vp_targets()
    ]

    for do_node in do_nodes:
        already_has_vp = any(
            e.source_id == do_node.suggested_id
            and any(
                n.suggested_id == e.target_id and n.node_type == NodeType.VV_PROTOCOL
                for n in current_nodes()
            )
            for e in current_edges()
        )
        if already_has_vp or not unconnected_vps:
            continue

        vp = unconnected_vps.pop(0)
        extra_edges.append(ExtractedEdge(
            source_id=do_node.suggested_id,
            target_id=vp.suggested_id,
            edge_type=EdgeType.LINKED_TO,
            confidence=0.70,
            rationale=f"Chain completeness: Design Output {do_node.suggested_id} linked to V&V Protocol {vp.suggested_id}.",
        ))

    return nodes + extra_nodes, edges + extra_edges


# ---------------------------------------------------------------------------
# Required Test Result inference (open V&V loop placeholders)
# ---------------------------------------------------------------------------

def _infer_required_test_results(
    nodes: list["ExtractedNode"],
    edges: list["ExtractedEdge"],
) -> tuple[list["ExtractedNode"], list["ExtractedEdge"]]:
    """
    Enforce VP → TR chain completeness within a single extraction result.

    Every V&V Protocol ultimately requires a Test Result as objective evidence
    (QMSR §820.30(f)/(g)). When a document describes a protocol the team plans to
    author but the result report does not yet exist — "the lab WILL produce a test
    result" — no Test Result is extracted (the system prompt forbids extracting
    TR nodes for future/planned reports). That leaves the V&V loop open and
    invisible in the RTM.

    For each V&V Protocol with no outgoing linked_to edge to a Test Result among
    the extracted nodes, this pass creates a 'required' Test Result placeholder and
    a linked_to edge from the protocol to it. Required placeholders are flagged with
    is_required=True so add_to_graph lands them in NOT_STARTED status, and carry the
    description "To be defined by the team" — they are explicit placeholders the
    team must fill in before the RTM is submission-ready, not inferred content.
    """
    extra_nodes: list["ExtractedNode"] = []
    extra_edges: list["ExtractedEdge"] = []
    taken = {n.suggested_id for n in nodes}

    vp_nodes = [n for n in nodes if n.node_type == NodeType.VV_PROTOCOL]
    for vp in vp_nodes:
        has_tr_edge = any(
            e.source_id == vp.suggested_id
            and any(
                n.suggested_id == e.target_id and n.node_type == NodeType.TEST_RESULT
                for n in nodes + extra_nodes
            )
            for e in edges + extra_edges
        )
        if has_tr_edge:
            continue

        suffix = vp.suggested_id.split("-", 1)[-1] if "-" in vp.suggested_id else "NEW"
        tr_id = f"TR-{suffix}"
        if tr_id in taken:
            tr_id = f"TR-{suffix}b"
        taken.add(tr_id)

        extra_nodes.append(ExtractedNode(
            suggested_id=tr_id,
            node_type=NodeType.TEST_RESULT,
            title=f"[Required] Test Result for {vp.suggested_id}",
            description="To be defined by the team",
            confidence=0.0,
            source_text="(required placeholder — V&V protocol exists but no result report yet)",
            is_required=True,
        ))
        extra_edges.append(ExtractedEdge(
            source_id=vp.suggested_id,
            target_id=tr_id,
            edge_type=EdgeType.LINKED_TO,
            confidence=0.0,
            rationale=f"Chain completeness: V&V Protocol {vp.suggested_id} requires a Test Result as objective evidence.",
        ))

        # Close the V&V loop: the Test Result verifies the Design Input that the
        # protocol's Design Output implements (QMSR §820.30(f)). Trace VP ← DO ← DI
        # through the extracted edges and add a `verifies` back-edge TR → DI. This
        # is the documented accepted cycle, so add_to_graph exempts it from the
        # cycle check below.
        do_ids = {
            e.source_id for e in edges + extra_edges
            if e.target_id == vp.suggested_id
            and any(
                n.suggested_id == e.source_id and n.node_type == NodeType.DESIGN_OUTPUT
                for n in nodes + extra_nodes
            )
        }
        di_ids = {
            e.source_id for e in edges + extra_edges
            if e.target_id in do_ids
            and any(
                n.suggested_id == e.source_id and n.node_type == NodeType.DESIGN_INPUT
                for n in nodes + extra_nodes
            )
        }
        for di_id in di_ids:
            extra_edges.append(ExtractedEdge(
                source_id=tr_id,
                target_id=di_id,
                edge_type=EdgeType.VERIFIES,
                confidence=0.0,
                rationale=f"V&V loop closure: Test Result {tr_id} verifies Design Input {di_id} (QMSR §820.30(f)).",
            ))

    return nodes + extra_nodes, edges + extra_edges


# ---------------------------------------------------------------------------
# Planned-artifact detection
# ---------------------------------------------------------------------------

# Future/planning language: phrases describing an artifact as work the team WILL
# perform, rather than something already produced. Nodes whose evidence reads this
# way are required placeholders (NOT_STARTED), not existing artifacts.
_PLANNED_LANGUAGE = re.compile(
    r"\bwill\s+(?:\w+\s+){0,3}(?:write|author|produce|draft|develop|create|generate|"
    r"need|drive|define|prepare|conduct|perform|execute|run|build|design|formulate)"
    r"|\bto\s+be\s+(?:written|authored|produced|developed|created|defined|determined|"
    r"drafted|generated|completed|conducted|performed|formulated|established)"
    r"|\bneeds?\s+to\s+be\b|\bwill\s+need\s+to\b|\bplans?\s+to\b|\bintends?\s+to\b",
    re.IGNORECASE,
)


def _is_planned_artifact(node: "ExtractedNode") -> bool:
    """
    True when the document describes this artifact as planned/future work — something
    the team WILL write, author, or produce — rather than an existing artifact.

    Such nodes are required placeholders: they land in NOT_STARTED status with the
    required-placeholder styling, like the inferred Test Result.

    User Needs are excluded: a clinical requirement exists the moment it is raised,
    even though every artifact it drives downstream is still to be built.
    """
    if node.node_type == NodeType.USER_NEED:
        return False
    text = f"{node.source_text} {node.description}"
    return bool(_PLANNED_LANGUAGE.search(text))


# Revision/review language: phrases describing an artifact that EXISTS but is not
# finished — it is being revised, updated, or is under review. Unlike a planned
# artifact (which does not exist yet), this one has a current version, so it lands
# in PENDING_REVIEW rather than NOT_STARTED. Either way it is incomplete and surfaces
# as a gap — only active/approved artifacts close a V&V loop.
_IN_REVIEW_LANGUAGE = re.compile(
    r"\bto\s+be\s+(?:revised|updated|amended|reviewed|finalized|finalised|confirmed|approved)"
    r"|\b(?:under|pending|awaiting|in)\s+(?:revision|review|approval)"
    r"|\bbeing\s+(?:revised|updated|reviewed|amended)"
    r"|\bneeds?\s+(?:revision|updating|review|approval)"
    r"|\brequires?\s+(?:revision|updating|review|approval)",
    re.IGNORECASE,
)


def _is_in_review_artifact(node: "ExtractedNode") -> bool:
    """
    True when the document describes this artifact as existing but incomplete — being
    revised, updated, or under review. It lands in PENDING_REVIEW (it exists, unlike a
    planned artifact), but is still considered incomplete and surfaces as a gap.

    User Needs are excluded, as in _is_planned_artifact.
    """
    if node.node_type == NodeType.USER_NEED:
        return False
    text = f"{node.source_text} {node.description}"
    return bool(_IN_REVIEW_LANGUAGE.search(text))


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
                title=node_id,
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
            edges = _filter_cross_chain_edges(nodes, edges, graph)

        # Drop skip-level linked_to edges (e.g. DI → VP) before chain inference
        # rebuilds the proper step-by-step path through the missing intermediate.
        edges = _filter_skip_level_edges(nodes, edges, graph)

        # Enforce DI → DO → VP chain completeness; stubs land in PENDING_REVIEW
        nodes, edges = _infer_missing_chain_links(nodes, edges)

        # Enforce VP → TR completeness: every V&V Protocol gets a required Test
        # Result placeholder (NOT_STARTED) when its result report does not yet exist.
        nodes, edges = _infer_required_test_results(nodes, edges)

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
            if node.is_required:
                # Required placeholders are not evidence — they mark an open V&V loop
                # the team must close, so they land in NOT_STARTED regardless of
                # confidence and are tagged for distinct rendering in Graph Explorer.
                status = NodeStatus.NOT_STARTED
            elif node.is_in_review:
                # Exists but unfinished (being revised / under review) — incomplete,
                # so it never lands in a completed status regardless of confidence.
                status = NodeStatus.PENDING_REVIEW
            else:
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
                    "required": node.is_required,
                },
            )
            nodes_added += 1

        for edge in result.extracted_edges:
            try:
                # Reject any edge that would introduce a cycle: if target can
                # already reach source, adding source→target closes a loop.
                # EXCEPTION: `verifies` (Test Result → Design Input) is the
                # documented V&V loop-closure back-edge (QMSR §820.30(f)) — it is
                # meant to close the DI→DO→VP→TR→DI loop, so it is exempt.
                if (
                    edge.edge_type != EdgeType.VERIFIES
                    and graph.has_path(edge.target_id, edge.source_id)
                ):
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
            node = ExtractedNode(
                suggested_id=item.get("suggested_id", str(uuid.uuid4())[:6]),
                node_type=node_type,
                title=item.get("title", "Untitled"),
                description=item.get("description", ""),
                confidence=float(item.get("confidence", 0.5)),
                source_text=item.get("source_text", ""),
            )
            # A node the document describes as planned/future work the team WILL
            # create — rather than an existing artifact — is a required placeholder.
            # Trust an explicit LLM `planned` flag, and fall back to detecting
            # forward-looking language in the evidence quote.
            node.is_required = bool(item.get("planned", False)) or _is_planned_artifact(node)
            # Otherwise, an existing-but-unfinished artifact (being revised / under
            # review) is incomplete too — it lands in PENDING_REVIEW and is a gap.
            if not node.is_required:
                node.is_in_review = bool(item.get("in_review", False)) or _is_in_review_artifact(node)
            nodes.append(node)
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
This change is subject to FDA QMSR §820.30(i) design change controls.

Impact Assessment:
- V&V Protocol VP-001 (LoD/LoQ Verification, CLSI EP17-A2) must be re-executed
  under the new LoD specification of ≤ 1.2 pg/mL.
- Risk Control RC-001 (ISO 14971 false negative hazard) must be reassessed
  given the tightened sensitivity specification.
- CAPA-018 scope may expand: the Lot 3 non-conformance at 2.6 pg/mL becomes
  more significant if the specification tightens to 1.2 pg/mL.

Approval Required:
Change cannot be implemented without QA/RA review. Design freeze remains in
effect until CR-089 is approved.
    """,

    "qmsr_guidance_excerpt.txt": (
        "REGULATORY TEXT — verbatim from eCFR (ecfr.gov)\n\n"
        + build_prompt_context(_regulations, ["820.30", "820.100", "820.40", "820.180"])
    ),
}
