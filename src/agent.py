"""
agent.py — LangGraph Change Impact Analysis Agent (Change Impact Sub-Agent)

When a node in the RTM changes, this agent:
  1. Traverses all downstream dependency edges from the changed node
  2. Classifies each impacted node and flags regulatory categories
     (V&V invalidations, PMA supplement triggers, CAPA triggers)
  3. Returns a structured ImpactReport — no compliance status is updated
     without human approval (human-in-the-loop gate enforced in supervisor)

Architecture: LangGraph state machine with three nodes:
  traverse → classify → report

The graph (RTMGraph) is NOT stored in LangGraph state. Instead, node functions
close over the graph instance passed to build_impact_agent(), keeping AgentState
fully serializable for checkpointing and human-in-the-loop resumption.

The LLM is called once in the 'report' node to generate a plain-English
compliance summary. Risk scoring and the escalation interrupt live in the
supervisor (src/supervisor.py), which has the full picture after this agent
completes.

Regulatory framework: FDA QMSR (21 CFR Part 820, effective Feb 2, 2026),
ISO 13485:2016, ISO 14971:2019, 21 CFR Part 814 (PMA), CLSI EP17-A2/EP05-A3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from graph import (
    EdgeType,
    HIERARCHY_LEVEL,
    NodeStatus,
    NodeType,
    RTMGraph,
)
from regulations import load_regulations, build_prompt_context

_regulations = load_regulations()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ImpactedNode:
    node_id: str
    node_type: str
    title: str
    current_status: str
    edge_path: list[str]
    edge_types_on_path: list[str]
    required_action: str
    direction: str = "downstream"  # "downstream" | "upstream"


@dataclass
class ImpactReport:
    changed_node_id: str
    changed_node_title: str
    change_description: str
    timestamp: str
    impacted_nodes: list[ImpactedNode]
    vv_invalidations: list[str]
    pma_supplement_flags: list[str]
    capa_triggers: list[str]
    llm_summary: str
    sme_notifications: list = field(default_factory=list)
    team_briefings: dict = field(default_factory=dict)
    # Risk assessment — populated by the supervisor after this agent completes
    risk_level: str = "low"          # "low" | "high" | "critical"
    risk_rationale: str = ""
    immediate_concerns: list = field(default_factory=list)
    # Escalation record — populated by the supervisor's escalation gate
    escalation_required: bool = False
    escalation_reviewer: str = ""
    escalation_notes: str = ""
    # Human approval gate (set by app.py after reviewer confirms)
    approved: bool = False


# ---------------------------------------------------------------------------
# LangGraph state — fully serializable (no live objects)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    changed_node_id: str
    change_description: str
    downstream_ids: list[str]
    upstream_ids: list[str]
    impacted_nodes: list[dict]
    vv_invalidations: list[str]
    pma_supplement_flags: list[str]
    capa_triggers: list[str]
    llm_summary: str


# ---------------------------------------------------------------------------
# Agent factory — graph captured in closures, not passed through state
# ---------------------------------------------------------------------------

def build_impact_agent(graph: RTMGraph) -> Any:
    """
    Compile and return the Change Impact sub-agent.

    All node functions close over the RTMGraph instance. AgentState contains
    only serializable primitives, making checkpointing and interrupt/resume safe.

    Args:
        graph: The live RTMGraph instance to traverse.

    Returns:
        Compiled LangGraph state machine: traverse → classify → report.
    """

    # ------------------------------------------------------------------
    # Node 1: Traverse
    # ------------------------------------------------------------------

    def traverse_node(state: AgentState) -> AgentState:
        """
        Step 1 — Traverse downstream descendants and immediate upstream
        predecessors from the changed node.

        Downstream: full BFS via RTMGraph.downstream_nodes().
        Upstream: one level (immediate predecessors only) to surface the
        requirements this node is supposed to satisfy — bidirectional
        traceability per QMSR §820.30(b). Each item is tagged with
        direction="downstream" or direction="upstream".
        """
        changed_id = state["changed_node_id"]
        downstream_ids = graph.downstream_nodes(changed_id)
        impacted = []

        for target_id in downstream_ids:
            try:
                node_data = graph.get_node(target_id)
                path = graph.impact_path(changed_id, target_id)

                edge_types = []
                for i in range(len(path) - 1):
                    edge_data = graph._g.edges[path[i], path[i + 1]]
                    edge_types.append(edge_data.get("edge_type", "linked_to"))

                impacted.append({
                    "node_id": target_id,
                    "node_type": node_data["node_type"],
                    "title": node_data["title"],
                    "current_status": node_data["status"],
                    "edge_path": path,
                    "edge_types_on_path": edge_types,
                    "direction": "downstream",
                })
            except Exception:
                continue

        # Upstream = direct predecessors whose hierarchy level is <= the changed
        # node's level (i.e. they are closer to User Need in the RTM chain).
        # Predecessors at a *higher* level are feedback edges (e.g. a Test Result
        # VALIDATES a Design Input) and are excluded — they belong downstream.
        changed_level = HIERARCHY_LEVEL.get(
            graph.get_node(changed_id).get("node_type", ""), 0
        )
        upstream_ids = [
            p for p in graph._g.predecessors(changed_id)
            if HIERARCHY_LEVEL.get(graph.get_node(p).get("node_type", ""), 0) <= changed_level
        ]
        for pred_id in upstream_ids:
            try:
                node_data = graph.get_node(pred_id)
                edge_data = graph._g.edges[pred_id, changed_id]
                impacted.append({
                    "node_id": pred_id,
                    "node_type": node_data["node_type"],
                    "title": node_data["title"],
                    "current_status": node_data["status"],
                    "edge_path": [pred_id, changed_id],
                    "edge_types_on_path": [edge_data.get("edge_type", "linked_to")],
                    "direction": "upstream",
                })
            except Exception:
                continue

        return {**state, "downstream_ids": downstream_ids, "upstream_ids": upstream_ids, "impacted_nodes": impacted}

    # ------------------------------------------------------------------
    # Node 2: Classify
    # ------------------------------------------------------------------

    def classify_node(state: AgentState) -> AgentState:
        """
        Step 2 — Deterministic regulatory classification of each impacted node.

        Assigns a required_action string based on node type and flags regulatory
        categories (V&V invalidations, PMA supplement triggers, CAPAs).

        Intentionally rule-based for auditability: every action string maps 1:1
        to a specific regulatory citation. Risk reasoning across the full impact
        chain is handled by score_risk_node in the supervisor.
        """
        vv_invalidations = []
        pma_supplement_flags = []
        capa_triggers = []
        scored = []

        for item in state["impacted_nodes"]:
            # Upstream requirements are surfaced for human review only — they
            # do not trigger regulatory flags (those are downstream obligations).
            if item.get("direction") == "upstream":
                action = (
                    "Verify that the revised node still satisfies this upstream requirement. "
                    "Check bidirectional traceability per QMSR §820.30(b)."
                )
                scored.append({**item, "required_action": action})
                continue

            node_type_str = item["node_type"]
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                node_type = NodeType.USER_NEED

            if node_type == NodeType.VV_PROTOCOL:
                action = (
                    "Re-execute V&V protocol — change to upstream design output may invalidate "
                    "the verification or validation basis per QMSR §820.30(f)/(g)."
                )
                vv_invalidations.append(item["node_id"])
            elif node_type == NodeType.TEST_RESULT:
                action = (
                    "Review test result validity — upstream V&V protocol or design output has changed. "
                    "Assess whether existing data remains valid under the new specification."
                )
            elif node_type == NodeType.HAZARD:
                action = (
                    "Review hazard analysis under ISO 14971:2019 — an upstream change may alter "
                    "the probability or severity estimate for this hazard. Update risk management "
                    "file if residual risk acceptability is affected."
                )
            elif node_type == NodeType.RISK_CONTROL:
                action = (
                    "Reassess risk control adequacy under ISO 14971:2019 — linked design input "
                    "has changed. Confirm residual risk acceptability."
                )
            elif node_type == NodeType.CAPA:
                action = (
                    "Review CAPA scope per QMSR §820.100 — root cause evidence chain has been "
                    "modified by an upstream design change. Update corrective action plan if warranted."
                )
                capa_triggers.append(item["node_id"])
            elif node_type == NodeType.PMA_SUPPLEMENT_TRIGGER:
                action = (
                    "PMA supplement trigger review required — performance specification chain has "
                    "changed. Assess whether a Prior Approval Supplement (PAS) or 30-Day Notice "
                    "is required under 21 CFR 814.39."
                )
                pma_supplement_flags.append(item["node_id"])
            elif node_type == NodeType.DESIGN_OUTPUT:
                action = (
                    "Review design output specification per QMSR §820.30(d) — upstream design "
                    "input dependency has changed."
                )
            elif node_type == NodeType.DESIGN_INPUT:
                action = (
                    "Review design input specification per QMSR §820.30(c) for consistency "
                    "with the changed upstream requirement."
                )
            else:
                action = "Review node for impact — downstream dependency has changed."

            scored.append({**item, "required_action": action})

        return {
            **state,
            "impacted_nodes": scored,
            "vv_invalidations": vv_invalidations,
            "pma_supplement_flags": pma_supplement_flags,
            "capa_triggers": capa_triggers,
        }

    # ------------------------------------------------------------------
    # Node 3: Report
    # ------------------------------------------------------------------

    def report_node(state: AgentState) -> AgentState:
        """
        Step 3 — LLM call to generate a plain-English compliance impact summary.

        The LLM receives the changed node details, the classified impact list,
        and the regulatory flag counts. It produces a concise 3-5 sentence
        summary for the human reviewer.

        SME team briefings are generated separately by the SME Router sub-agent.
        Risk scoring and escalation decisions are made by the supervisor.
        """
        changed_id = state["changed_node_id"]
        try:
            changed_node = graph.get_node(changed_id)
        except KeyError:
            changed_node = {"title": changed_id, "node_type": "Unknown"}

        impact_lines = [
            f"- [{item['node_type']}] {item['node_id']}: {item['title']} "
            f"— {item['required_action']}"
            for item in state["impacted_nodes"]
        ]
        impact_text = "\n".join(impact_lines) if impact_lines else "No downstream impacts detected."

        reg_context = build_prompt_context(_regulations, ["820.30", "820.100", "814.39", "493.1253", "493.1255"])
        system_prompt = (
            "You are a regulatory affairs assistant helping a QA team understand the downstream "
            "compliance impact of a change to an IVD assay Requirements Traceability Matrix (RTM). "
            "ISO 13485:2016, ISO 14971:2019, and CLSI analytical performance standards "
            "(EP17-A2, EP05-A3) also apply.\n\n"
            f"{reg_context}\n\n"
            "Your role is to summarize the impact for a human reviewer who must decide whether "
            "to approve compliance status updates. Be specific, factual, and cite applicable "
            "regulatory sections where relevant. "
            "Do not make compliance decisions — only surface what needs human review."
        )

        user_prompt = (
            f"A change was made to the following RTM node:\n\n"
            f"Node: {changed_node.get('title', changed_id)} ({changed_id})\n"
            f"Type: {changed_node.get('node_type', 'Unknown')}\n"
            f"Change description: {state['change_description']}\n\n"
            f"Downstream impact analysis identified {len(state['impacted_nodes'])} affected nodes:\n\n"
            f"{impact_text}\n\n"
            f"V&V protocols requiring re-execution: {state['vv_invalidations'] or 'None'}\n"
            f"PMA supplement triggers flagged: {state['pma_supplement_flags'] or 'None'}\n"
            f"CAPAs requiring review: {state['capa_triggers'] or 'None'}\n\n"
            f"Write a 3–5 sentence plain-English compliance summary for the QA/RA team. "
            f"Focus on what needs to be acted on first and whether a PMA supplement submission "
            f"may be required under 21 CFR 814.39."
        )

        llm_summary = ""
        try:
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=512,
            )
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            llm_summary = response.content
        except Exception:
            llm_summary = (
                f"[LLM unavailable — check OPENAI_API_KEY] "
                f"Manual review required for {len(state['impacted_nodes'])} downstream nodes. "
                f"V&V invalidations: {state['vv_invalidations']}. "
                f"PMA supplement flags: {state['pma_supplement_flags']}."
            )

        return {**state, "llm_summary": llm_summary}

    # ------------------------------------------------------------------
    # Compile the state machine
    # ------------------------------------------------------------------

    builder = StateGraph(AgentState)
    builder.add_node("traverse", traverse_node)
    builder.add_node("classify", classify_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "traverse")
    builder.add_edge("traverse", "classify")
    builder.add_edge("classify", "report")
    builder.add_edge("report", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Standalone entry point (skips supervisor — no SME routing, no escalation)
# ---------------------------------------------------------------------------

def run_impact_analysis(
    graph: RTMGraph,
    changed_node_id: str,
    change_description: str,
) -> ImpactReport:
    """
    Run the change impact sub-agent in isolation.

    For the full multi-agent pipeline (impact + risk scoring + escalation gate
    + SME briefings), use supervisor.run_full_analysis() instead.

    Args:
        graph: The live RTMGraph instance.
        changed_node_id: The ID of the node that changed.
        change_description: Human-readable description of what changed and why.

    Returns:
        ImpactReport with downstream obligations surfaced.
        report.approved is always False — human must approve.
        report.sme_notifications, team_briefings, risk_level, and
        escalation fields will be at their defaults; populate via supervisor.
    """
    agent = build_impact_agent(graph)

    initial_state: AgentState = {
        "changed_node_id": changed_node_id,
        "change_description": change_description,
        "downstream_ids": [],
        "upstream_ids": [],
        "impacted_nodes": [],
        "vv_invalidations": [],
        "pma_supplement_flags": [],
        "capa_triggers": [],
        "llm_summary": "",
    }

    result = agent.invoke(initial_state)

    impacted = [
        ImpactedNode(
            node_id=item["node_id"],
            node_type=item["node_type"],
            title=item["title"],
            current_status=item["current_status"],
            edge_path=item["edge_path"],
            edge_types_on_path=item["edge_types_on_path"],
            required_action=item["required_action"],
            direction=item.get("direction", "downstream"),
        )
        for item in result["impacted_nodes"]
    ]

    try:
        changed_title = graph.get_node(changed_node_id)["title"]
    except KeyError:
        changed_title = changed_node_id

    return ImpactReport(
        changed_node_id=changed_node_id,
        changed_node_title=changed_title,
        change_description=change_description,
        timestamp=datetime.now(timezone.utc).isoformat(),
        impacted_nodes=impacted,
        vv_invalidations=result["vv_invalidations"],
        pma_supplement_flags=result["pma_supplement_flags"],
        capa_triggers=result["capa_triggers"],
        llm_summary=result["llm_summary"],
        approved=False,
    )
