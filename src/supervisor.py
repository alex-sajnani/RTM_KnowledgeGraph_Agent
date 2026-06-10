"""
supervisor.py — Top-Level LangGraph Multi-Agent Orchestrator

Coordinates two sub-agents in sequence, then adds risk scoring and an
optional escalation gate before assembling the final report:

  1. run_impact_agent   — Change Impact sub-agent (traverse → classify → report)
  2. score_risk         — Deterministic risk level + LLM explanation
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

__all__ = ["build_supervisor", "run_full_analysis"]


# ---------------------------------------------------------------------------
# Supervisor state — fully serializable
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    changed_node_id: str
    change_description: str
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

class RiskExplanation(BaseModel):
    rationale: str
    immediate_concerns: list[str]


def _compute_risk_level(impact_result: dict) -> str:
    """
    Determine risk level from structural signals alone — no LLM involved.

    critical: Any V&V invalidation OR any PMA supplement flag.
    high:     Any CAPA trigger OR any Hazard/Risk Control node in the impact chain.
    low:      Everything else.
    """
    if impact_result.get("vv_invalidations") or impact_result.get("pma_supplement_flags"):
        return "critical"
    high_types = {NodeType.HAZARD.value, NodeType.RISK_CONTROL.value}
    if impact_result.get("capa_triggers") or any(
        n["node_type"] in high_types
        for n in impact_result.get("impacted_nodes", [])
    ):
        return "high"
    return "low"


def _generate_risk_explanation(
    risk_level: str,
    change_description: str,
    impact_result: dict,
) -> RiskExplanation:
    """
    Call LLM to produce a plain-English rationale and ordered concern list.

    The risk_level is already determined deterministically — the LLM only
    explains it and surfaces what the reviewer should act on first.
    """
    node_summary = "\n".join(
        f"- [{n['node_type']}] {n['node_id']}: {n['title']}"
        for n in impact_result.get("impacted_nodes", [])
    )

    system_prompt = (
        "You are a regulatory affairs specialist for an IVD medical device company. "
        "A change impact analysis has been run on the Requirements Traceability Matrix "
        "for a high-sensitivity cardiac Troponin I immunoassay (PMA P240052). "
        "Explain the risk level in plain English and list the most urgent actions, "
        "ordered by regulatory priority. Be concise and specific."
    )

    user_prompt = (
        f"Change description: {change_description}\n\n"
        f"Risk level (already determined): {risk_level.upper()}\n\n"
        f"Structural triggers:\n"
        f"  V&V invalidations: {impact_result.get('vv_invalidations') or 'none'}\n"
        f"  PMA supplement flags: {impact_result.get('pma_supplement_flags') or 'none'}\n"
        f"  CAPA triggers: {impact_result.get('capa_triggers') or 'none'}\n\n"
        f"Impacted nodes:\n{node_summary or '(none)'}\n\n"
        f"Write a 2-3 sentence rationale explaining why this risk level applies, "
        f"then list the top 3 immediate actions the review team must take."
    )

    try:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=400,
        ).with_structured_output(RiskExplanation)
        return llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    except Exception:
        return RiskExplanation(
            rationale=(
                f"[LLM unavailable] Risk level '{risk_level}' determined from structural signals: "
                f"V&V invalidations={impact_result.get('vv_invalidations')}, "
                f"PMA flags={impact_result.get('pma_supplement_flags')}."
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
            "pma_supplement_flags": [],
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
        Determine risk level deterministically, then generate LLM explanation.

        Risk level controls routing (critical → escalation gate). The LLM only
        produces human-readable rationale and concern list — it cannot change
        the routing outcome.
        """
        risk_level = _compute_risk_level(state["impact_result"])
        explanation = _generate_risk_explanation(
            risk_level,
            state["change_description"],
            state["impact_result"],
        )
        return {
            **state,
            "risk_level": risk_level,
            "risk_rationale": explanation.rationale,
            "immediate_concerns": explanation.immediate_concerns,
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
            "pma_flags": state["impact_result"].get("pma_supplement_flags", []),
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
        """Invoke the SME Router sub-agent with the scored impacted nodes."""
        initial: SMEState = {
            "impacted_nodes": state["impact_result"].get("impacted_nodes", []),
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
            timestamp=datetime.now(timezone.utc).isoformat(),
            impacted_nodes=impacted,
            vv_invalidations=impact.get("vv_invalidations", []),
            pma_supplement_flags=impact.get("pma_supplement_flags", []),
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
