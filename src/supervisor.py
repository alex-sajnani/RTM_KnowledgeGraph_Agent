"""
supervisor.py — Top-Level LangGraph Multi-Agent Orchestrator

Coordinates two sub-agents in sequence, then adds risk scoring and an
optional escalation gate before assembling the final report:

  1. run_impact_agent   — Change Impact sub-agent (traverse → classify → report)
  2. score_risk         — Deterministic risk level (reviewer-attested change type
                          + structural ceiling) + LLM explanation
  3. [conditional]      — "critical" → escalation_gate (interrupt); else → run_sme_agent
  4. escalation_gate    — LangGraph interrupt(): pauses for human review
                          resume approved  → run_sme_agent
                          resume rejected  → END (nothing stored)
  5. run_sme_agent      — SME Router sub-agent (map_teams → generate_briefings)
  6. assemble_report    — Builds final ImpactReport from all sub-agent results

Design principles:
  - The supervisor owns its sub-agent dependencies: build_supervisor() compiles
    both sub-agents internally so the caller never needs to know which agents exist.
  - Risk level is purely deterministic (structural flags); the LLM only produces
    explanatory text — it does not control routing.
  - The checkpointer lives here so interrupt/resume state is persisted across
    Streamlit reruns.

Public interface:
    run_full_analysis(graph, node_id, description, checkpointer, thread_id, resume_payload)
    → tuple[ImpactReport | None, dict | None, str]
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt, Command
from pydantic import BaseModel
from typing_extensions import TypedDict

from agent import (
    ImpactedNode,
    ImpactReport,
    AgentState,
    build_impact_agent,
)
from sme_agent import (
    SMEState,
    build_sme_agent,
    notifications_from_dicts,
)
from graph import RTMGraph, NodeType

__all__ = [
    "build_supervisor",
    "run_full_analysis",
    "CHANGE_TYPES",
    "NON_SUBSTANTIVE_CHANGE_TYPES",
    "DEFAULT_CHANGE_TYPE",
]


# ---------------------------------------------------------------------------
# Supervisor state — fully serializable
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    changed_node_id: str
    change_description: str
    change_type: str             # reviewer-attested type; drives risk downgrade
    impact_result: dict          # raw AgentState dict from the impact sub-agent
    sme_result: dict             # raw SMEState dict from the SME router sub-agent
    risk_level: str              # "low" | "high" | "critical"
    risk_rationale: str
    immediate_concerns: list[str]
    escalation_decision: dict    # {"approved": bool, "reviewer": str, "notes": str}
    assembled_report: Any        # ImpactReport | None


# ---------------------------------------------------------------------------
# Risk scoring — deterministic level, LLM explanation only
# ---------------------------------------------------------------------------

# Change-type classification — drives the deterministic risk downgrade.
# The reviewer attests the change type at submit time; a non-substantive type
# (documentation-only, no change) downgrades the structural ceiling to "low",
# while substantive types leave the ceiling intact. This replaces an LLM
# inference of "is this substantive?" with an auditable human attestation.
CHANGE_TYPES = [
    "Functional change",
    "Corrective / CAPA action",
    "Documentation only",
    "No change",
]
NON_SUBSTANTIVE_CHANGE_TYPES = {"Documentation only", "No change"}
DEFAULT_CHANGE_TYPE = "Functional change"


class RiskAssessment(BaseModel):
    """Internal container for the scored result: a code-decided level plus prose."""
    risk_level: Literal["low", "high", "critical"]
    rationale: str
    immediate_concerns: list[str]


class RiskExplanation(BaseModel):
    """LLM structured output — prose only. The risk level is decided in code."""
    rationale: str
    immediate_concerns: list[str]


def _structural_risk_ceiling(impact_result: dict) -> str:
    """
    Compute the maximum risk level the structural topology allows.

    This is a ceiling — the LLM may only assess risk at or below this level.
    critical: Any V&V invalidation.
    high:     Any CAPA trigger OR Hazard/Risk Control node in the impact chain.
    low:      Everything else.
    """
    if impact_result.get("vv_invalidations"):
        return "critical"
    high_types = {NodeType.HAZARD.value, NodeType.RISK_CONTROL.value}
    if impact_result.get("capa_triggers") or any(
        n["node_type"] in high_types
        for n in impact_result.get("impacted_nodes", [])
    ):
        return "high"
    return "low"


def _risk_level_for_change(change_type: str, impact_result: dict) -> str:
    """
    Decide the risk level: the structural ceiling, downgraded to 'low' for a
    non-substantive change type.

    This is a pure function of (change_type, topology) — no model opinion, fully
    reproducible, and auditable: the reviewer attests the change type and the
    topology is a verifiable fact. A non-substantive change (documentation-only,
    no change) is 'low' even when V&V nodes are topologically downstream, because
    nothing was actually altered to invalidate them.
    """
    if change_type in NON_SUBSTANTIVE_CHANGE_TYPES:
        return "low"
    return _structural_risk_ceiling(impact_result)


def _assess_risk(
    change_type: str,
    change_description: str,
    impact_result: dict,
) -> RiskAssessment:
    """
    Produce the scored risk result: a code-decided level plus a plain-English
    explanation.

    The level comes from `_risk_level_for_change` — the LLM never decides or
    changes it. The LLM is handed the already-decided level and writes only the
    rationale and the immediate actions the review team must take. If the LLM is
    unavailable, a deterministic fallback explanation is returned. This keeps the
    safety-relevant decision auditable while still producing readable guidance.
    """
    risk_level = _risk_level_for_change(change_type, impact_result)
    ceiling = _structural_risk_ceiling(impact_result)
    downgraded = risk_level == "low" and ceiling != "low"
    node_summary = "\n".join(
        f"- [{n['node_type']}] {n['node_id']}: {n['title']}"
        for n in impact_result.get("impacted_nodes", [])
    )

    system_prompt = (
        "You are a regulatory affairs specialist for an IVD medical device company. "
        "A change impact analysis has been run on a Requirements Traceability Matrix, "
        "and the risk level has ALREADY been determined. Your job is to explain that "
        "determination in plain English and list the immediate actions the review team "
        "must take — you do NOT decide or change the risk level.\n\n"
        "IMPORTANT: Only reference node IDs and artifacts explicitly listed in the user message."
    )

    user_prompt = (
        f"Change type: {change_type}\n"
        f"Change description: {change_description}\n\n"
        f"Determined risk level: {risk_level.upper()}\n"
        f"Structural ceiling (maximum risk from topology): {ceiling.upper()}\n\n"
        f"Structural triggers:\n"
        f"  V&V invalidations: {impact_result.get('vv_invalidations') or 'none'}\n"
        f"  CAPA triggers: {impact_result.get('capa_triggers') or 'none'}\n\n"
        f"Nodes in impact chain (only these exist in scope):\n"
        f"{node_summary or '(none)'}\n\n"
        + (
            "Note: the topology would allow a higher risk, but the reviewer classified "
            "this as a non-substantive change, so the level is LOW. Explain why the "
            "structural triggers do not apply given the change type.\n\n"
            if downgraded else ""
        )
        + f"Write a 2-3 sentence rationale for the {risk_level.upper()} rating, and list "
        f"the top 3 immediate actions the review team must take. Only cite node IDs listed above."
    )

    try:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=400,
        ).with_structured_output(RiskExplanation)
        explanation = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return RiskAssessment(
            risk_level=risk_level,
            rationale=explanation.rationale,
            immediate_concerns=explanation.immediate_concerns,
        )
    except Exception:
        return RiskAssessment(
            risk_level=risk_level,
            rationale=(
                f"[LLM unavailable] {risk_level.upper()} risk for a '{change_type}' change. "
                f"Structural ceiling '{ceiling}': "
                f"V&V invalidations={impact_result.get('vv_invalidations')}, "
                f"CAPA triggers={impact_result.get('capa_triggers')}."
            ),
            immediate_concerns=["Check OPENAI_API_KEY and rerun for detailed guidance."],
        )


# ---------------------------------------------------------------------------
# Supervisor builder — owns sub-agent compilation
# ---------------------------------------------------------------------------

def build_supervisor(graph: RTMGraph, checkpointer: Any) -> Any:
    """
    Compile and return the top-level supervisor LangGraph.

    Both sub-agents are compiled here and captured in node closures.
    The caller (run_full_analysis) does not need to know which sub-agents exist.

    Args:
        graph: RTMGraph instance, passed through to the impact sub-agent closure.
        checkpointer: MemorySaver (or compatible) for interrupt/resume state.

    Returns:
        Compiled LangGraph supervisor with checkpointer attached.
    """
    # Compile sub-agents once at supervisor build time — not inside running nodes
    impact_agent = build_impact_agent(graph)
    sme_agent = build_sme_agent()

    # ------------------------------------------------------------------
    # Node: run_impact_agent
    # ------------------------------------------------------------------

    def run_impact_agent_node(state: SupervisorState) -> SupervisorState:
        """Invoke the Change Impact sub-agent and store its full result dict."""
        initial: AgentState = {
            "changed_node_id": state["changed_node_id"],
            "change_description": state["change_description"],
            "downstream_ids": [],
            "upstream_ids": [],
            "impacted_nodes": [],
            "vv_invalidations": [],
            "capa_triggers": [],
            "llm_summary": "",
        }
        result = impact_agent.invoke(initial)
        return {**state, "impact_result": result}

    # ------------------------------------------------------------------
    # Node: score_risk
    # ------------------------------------------------------------------

    def score_risk_node(state: SupervisorState) -> SupervisorState:
        """
        Decide the risk level deterministically, then have the LLM explain it.

        The level is a pure function of the reviewer-attested change type and the
        graph topology (`_risk_level_for_change`); a non-substantive change
        (documentation-only, no change) yields 'low' even when V&V nodes are
        topologically downstream. The LLM only writes the rationale and immediate
        actions. Risk level controls routing (critical → escalation gate).
        """
        assessment = _assess_risk(
            state.get("change_type", DEFAULT_CHANGE_TYPE),
            state["change_description"],
            state["impact_result"],
        )
        return {
            **state,
            "risk_level": assessment.risk_level,
            "risk_rationale": assessment.rationale,
            "immediate_concerns": assessment.immediate_concerns,
        }

    # ------------------------------------------------------------------
    # Conditional edge: route after risk scoring
    # ------------------------------------------------------------------

    def route_after_risk(state: SupervisorState) -> Literal["escalation_gate", "run_sme_agent"]:
        """Route critical-risk changes to the escalation gate; others proceed directly."""
        return "escalation_gate" if state["risk_level"] == "critical" else "run_sme_agent"

    # ------------------------------------------------------------------
    # Node: escalation_gate  (LangGraph interrupt)
    # ------------------------------------------------------------------

    def escalation_gate_node(state: SupervisorState) -> SupervisorState:
        """
        Pause execution for human review of a critical-risk change.

        Calls LangGraph interrupt() with the risk payload. Execution suspends
        here and resumes only when the caller invokes the supervisor with
        Command(resume={"approved": bool, "reviewer": str, "notes": str}).

        If approved → continues to run_sme_agent.
        If rejected → END is returned via a second conditional edge.
        """
        decision = interrupt({
            "risk_level": state["risk_level"],
            "risk_rationale": state["risk_rationale"],
            "immediate_concerns": state["immediate_concerns"],
            "vv_invalidations": state["impact_result"].get("vv_invalidations", []),
            "capa_triggers": state["impact_result"].get("capa_triggers", []),
        })
        return {**state, "escalation_decision": decision}

    def route_after_escalation(state: SupervisorState) -> str:
        """Proceed only if the human approved. Rejection routes to END."""
        return "run_sme_agent" if state.get("escalation_decision", {}).get("approved") else END

    # ------------------------------------------------------------------
    # Node: run_sme_agent
    # ------------------------------------------------------------------

    def run_sme_agent_node(state: SupervisorState) -> SupervisorState:
        """Invoke the SME Router sub-agent with the scored impacted nodes.

        Passes the change description and changed-node identity through so each
        team briefing reacts to the specific change, not just the impacted nodes.
        """
        changed_id = state["changed_node_id"]
        try:
            changed_title = graph.get_node(changed_id)["title"]
        except KeyError:
            changed_title = changed_id
        initial: SMEState = {
            "impacted_nodes": state["impact_result"].get("impacted_nodes", []),
            "change_description": state["change_description"],
            "changed_node_id": changed_id,
            "changed_node_title": changed_title,
            "sme_notifications": [],
            "team_briefings": {},
        }
        result = sme_agent.invoke(initial)
        return {**state, "sme_result": result}

    # ------------------------------------------------------------------
    # Node: assemble_report
    # ------------------------------------------------------------------

    def assemble_report_node(state: SupervisorState) -> SupervisorState:
        """Assemble the final ImpactReport from both sub-agent results."""
        impact = state["impact_result"]
        sme = state["sme_result"]
        escalation = state.get("escalation_decision", {})

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
            for item in impact.get("impacted_nodes", [])
        ]

        changed_id = state["changed_node_id"]
        try:
            changed_title = graph.get_node(changed_id)["title"]
        except KeyError:
            changed_title = changed_id

        sme_notifications = notifications_from_dicts(sme.get("sme_notifications", []))

        report = ImpactReport(
            changed_node_id=changed_id,
            changed_node_title=changed_title,
            change_description=state["change_description"],
            change_type=state.get("change_type", DEFAULT_CHANGE_TYPE),
            timestamp=datetime.now(timezone.utc).isoformat(),
            impacted_nodes=impacted,
            vv_invalidations=impact.get("vv_invalidations", []),
            capa_triggers=impact.get("capa_triggers", []),
            llm_summary=impact.get("llm_summary", ""),
            sme_notifications=sme_notifications,
            team_briefings=sme.get("team_briefings", {}),
            risk_level=state["risk_level"],
            risk_rationale=state["risk_rationale"],
            immediate_concerns=state["immediate_concerns"],
            escalation_required=bool(escalation),
            escalation_reviewer=escalation.get("reviewer", ""),
            escalation_notes=escalation.get("notes", ""),
            approved=False,
        )

        return {**state, "assembled_report": report}

    # ------------------------------------------------------------------
    # Build and compile
    # ------------------------------------------------------------------

    builder = StateGraph(SupervisorState)

    builder.add_node("run_impact_agent", run_impact_agent_node)
    builder.add_node("score_risk", score_risk_node)
    builder.add_node("escalation_gate", escalation_gate_node)
    builder.add_node("run_sme_agent", run_sme_agent_node)
    builder.add_node("assemble_report", assemble_report_node)

    builder.add_edge(START, "run_impact_agent")
    builder.add_edge("run_impact_agent", "score_risk")
    builder.add_conditional_edges("score_risk", route_after_risk)
    builder.add_conditional_edges("escalation_gate", route_after_escalation)
    builder.add_edge("run_sme_agent", "assemble_report")
    builder.add_edge("assemble_report", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_full_analysis(
    graph: RTMGraph,
    changed_node_id: str,
    change_description: str,
    checkpointer: Any = None,
    thread_id: str | None = None,
    resume_payload: dict | None = None,
    supervisor: Any = None,
    change_type: str = DEFAULT_CHANGE_TYPE,
) -> tuple[ImpactReport | None, dict | None, str]:
    """
    Run the full multi-agent change impact pipeline.

    On a fresh run (resume_payload=None):
      - Runs impact agent → scores risk → escalation gate (if critical) → SME agent → report
      - If the escalation gate fires, returns (None, interrupt_payload, thread_id)
        so the caller can show the escalation UI and resume later.

    On resume (resume_payload provided):
      - Resumes the paused graph from the escalation gate with the human decision.
      - approved=True  → SME agent runs, full report returned.
      - approved=False → pipeline halts, (None, None, thread_id) returned.

    Args:
        graph: Live RTMGraph instance.
        changed_node_id: ID of the changed node (ignored on resume).
        change_description: Description of the change (ignored on resume).
        checkpointer: MemorySaver from st.session_state; persists interrupt state
                      across Streamlit reruns.
        thread_id: Used for resume; a new UUID is generated on fresh runs.
        resume_payload: dict with "approved", "reviewer", "notes" keys.
        supervisor: Pre-compiled supervisor graph. Pass st.session_state.supervisor
                    to avoid rebuilding on every call. If None, build_supervisor()
                    is called (useful for standalone/test use).
        change_type: Reviewer-attested change type (one of CHANGE_TYPES). A
                     non-substantive type downgrades the risk level to "low";
                     ignored on resume. Defaults to "Functional change".

    Returns:
        (ImpactReport, None, thread_id)       — analysis complete
        (None, interrupt_payload, thread_id)  — paused at escalation gate
        (None, None, thread_id)               — rejected (pipeline halted)
    """
    effective_checkpointer = checkpointer or MemorySaver()
    effective_thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": effective_thread_id}}

    if supervisor is None:
        supervisor = build_supervisor(graph, effective_checkpointer)

    if resume_payload is not None:
        result = supervisor.invoke(Command(resume=resume_payload), config=config)
    else:
        initial: SupervisorState = {
            "changed_node_id": changed_node_id,
            "change_description": change_description,
            "change_type": change_type,
            "impact_result": {},
            "sme_result": {},
            "risk_level": "",
            "risk_rationale": "",
            "immediate_concerns": [],
            "escalation_decision": {},
            "assembled_report": None,
        }
        result = supervisor.invoke(initial, config=config)

    # Check for interrupt (escalation gate fired)
    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = interrupts[0]["value"] if isinstance(interrupts[0], dict) else interrupts[0].value
        return None, payload, effective_thread_id

    # Check for completed report
    report = result.get("assembled_report")
    if report is not None:
        return report, None, effective_thread_id

    # Pipeline halted (escalation rejected — route_after_escalation → END)
    return None, None, effective_thread_id
