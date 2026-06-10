"""
sme_agent.py — SME Router Sub-Agent (Map-Reduce, Parallel Briefings)

Given a list of scored impacted nodes from the Change Impact Agent, this
sub-agent:
  1. Maps each impacted node type to the appropriate SME team(s) using
     SME_NOTIFICATION_MAP (map_teams_node)
  2. Fans out one brief_team_node per unique team via LangGraph Send —
     each team's LLM briefing call runs in parallel (map_to_teams edge)
  3. Collects results via a _merge_dicts reducer on team_briefings, then
     writes llm_briefing back into each notification (finalize_notifications_node)

Teams:
  - Bioinformatics: performance re-analysis framing (CLSI, statistical)
  - R&D: study design and re-validation framing (CLSI protocols, timelines)
  - Pathology: clinical risk and patient safety framing (ISO 14971, lab director)
  - Quality/RA: regulatory submission framing (21 CFR 814.39, QMSR §820.30(i))

Architecture: LangGraph state machine with map-reduce pattern:
  map_teams → [Send × N teams] → brief_team (×N parallel) → finalize_notifications

Called by src/supervisor.py as a compiled subgraph node. The supervisor's
interface is unchanged: it passes impacted_nodes in and reads sme_notifications
and team_briefings from the result.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from graph import NodeType
from regulations import load_regulations, build_prompt_context

_regulations = load_regulations()


# ---------------------------------------------------------------------------
# SME notification data structure
# ---------------------------------------------------------------------------

@dataclass
class SMENotification:
    team: str                  # "Bioinformatics" | "R&D" | "Pathology" | "Quality/RA"
    trigger_node_id: str
    trigger_node_type: str
    trigger_node_title: str
    review_obligation: str     # short action statement surfaced in the UI table
    llm_briefing: str = ""     # team-specific LLM summary, populated by brief_team_node


# ---------------------------------------------------------------------------
# SME mapping: node type → list of (team, review_obligation) pairs
# ---------------------------------------------------------------------------

SME_NOTIFICATION_MAP: dict[str, list[tuple[str, str]]] = {
    NodeType.VV_PROTOCOL.value: [
        ("Bioinformatics",
         "Re-analyze performance dataset against updated analytical specification; "
         "assess statistical validity of prior LoD/LoQ claims."),
        ("R&D",
         "Initiate re-validation study under revised acceptance criteria per QMSR §820.30(g); "
         "identify applicable CLSI protocols and timeline."),
    ],
    NodeType.TEST_RESULT.value: [
        ("Bioinformatics",
         "Review test result validity; assess whether existing performance dataset "
         "remains statistically valid under the modified upstream specification."),
    ],
    NodeType.HAZARD.value: [
        ("Quality/RA",
         "Re-evaluate hazard probability and severity estimates under ISO 14971:2019; "
         "confirm residual risk remains acceptable given the upstream design change."),
    ],
    NodeType.RISK_CONTROL.value: [
        ("Pathology",
         "Assess clinical impact of modified performance specification on patient risk "
         "per ISO 14971:2019; evaluate reference interval and clinical decision threshold implications."),
    ],
    NodeType.CAPA.value: [
        ("Quality/RA",
         "Review CAPA scope per QMSR §820.100; update corrective action plan in light "
         "of changed upstream design artifact."),
    ],
    NodeType.PMA_SUPPLEMENT_TRIGGER.value: [
        ("Quality/RA",
         "Evaluate whether a Prior Approval Supplement (PAS) or 30-Day Notice is required "
         "under 21 CFR 814.39; assess QMSR §820.30(i) design change documentation obligations."),
    ],
    NodeType.DESIGN_INPUT.value: [
        ("R&D",
         "Assess whether additional bench studies are required to support the revised "
         "design input specification; identify any CLSI protocol updates needed."),
    ],
    NodeType.DESIGN_OUTPUT.value: [
        ("R&D",
         "Review design output specification per QMSR §820.30(d) for consistency "
         "with the changed upstream design input requirement."),
    ],
}

# Team-specific system prompts for LLM briefing generation
TEAM_SYSTEM_PROMPTS: dict[str, str] = {
    "Bioinformatics": (
        "You are a bioinformatics scientist reviewing the analytical performance implications "
        "of a design specification change on a high-sensitivity cardiac troponin I (hs-cTnI) "
        "immunoassay. Frame your response around: re-analysis of existing performance datasets, "
        "statistical re-evaluation of LoD/LoQ claims per CLSI EP17-A2, and what new data "
        "collection is needed before the change can be validated. Note that clinical laboratories "
        "using this device must re-verify the updated performance specifications under CLIA before "
        "reporting patient results. Be concise and technical.\n\n"
        + build_prompt_context(_regulations, ["493.1253"])
    ),
    "R&D": (
        "You are a senior assay development scientist assessing whether new bench validation "
        "studies are required following a design specification change on a hs-cTnI immunoassay. "
        "Frame your response around: which CLSI protocols must be re-executed (EP17-A2 for LoD, "
        "EP05-A3 for precision), estimated study scope and timeline, and any reagent or "
        "manufacturing process changes that must be assessed. Be concise and actionable."
    ),
    "Pathology": (
        "You are a clinical pathologist and laboratory medical director evaluating how a "
        "change to a hs-cTnI immunoassay's performance specification affects clinical "
        "decision-making and patient safety in the emergency department. Frame your response "
        "around: impact on the 99th percentile upper reference limit, 0h/1h AMI rule-out "
        "protocol validity, clinical risk of false negatives per ISO 14971, and whether "
        "physician notification or protocol updates are required. Be clinically direct."
    ),
    "Quality/RA": (
        "You are a regulatory affairs specialist assessing PMA submission obligations "
        "following a design specification change on a Class III IVD device (hs-cTnI immunoassay, "
        "PMA P240052). Frame your response around: 21 CFR 814.39 supplement type determination "
        "(Prior Approval Supplement vs. 30-Day Notice), QMSR §820.30(i) design change controls "
        "and documentation requirements, Design History File (DHF) update obligations, FDA "
        "communication strategy, and the downstream CLIA obligation for customer laboratories "
        "to re-verify performance specifications before reporting patient results. "
        "Be precise about regulatory citations.\n\n"
        + build_prompt_context(_regulations, ["820.30", "814.39", "493.1253", "493.1255"])
    ),
}


# ---------------------------------------------------------------------------
# State reducer
# ---------------------------------------------------------------------------

def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer for team_briefings: merge parallel brief_team outputs into one dict."""
    return {**a, **b}


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class SMEState(TypedDict):
    impacted_nodes: list[dict]                          # scored nodes from the change impact agent
    sme_notifications: list[dict]                       # assembled SMENotification dicts
    team_briefings: Annotated[dict, _merge_dicts]       # team_name → LLM briefing; merged across parallel brief_team nodes
    # Send-payload fields — populated by map_to_teams, consumed by brief_team_node
    team: str
    team_notifications: list[dict]


# ---------------------------------------------------------------------------
# Node 1: map_teams
# ---------------------------------------------------------------------------

def map_teams_node(state: SMEState) -> dict:
    """
    Step 1 — Map each impacted node to SME team(s) using SME_NOTIFICATION_MAP.

    No LLM call in this step — pure deterministic mapping.
    Initializes team_briefings to an empty dict; brief_team_node fills it in.
    """
    notifications = []

    for node in state["impacted_nodes"]:
        node_type = node.get("node_type", "")
        assignments = SME_NOTIFICATION_MAP.get(node_type, [])
        for team, obligation in assignments:
            notifications.append({
                "team": team,
                "trigger_node_id": node["node_id"],
                "trigger_node_type": node_type,
                "trigger_node_title": node.get("title", ""),
                "review_obligation": obligation,
                "llm_briefing": "",
            })

    return {"sme_notifications": notifications, "team_briefings": {}}


# ---------------------------------------------------------------------------
# Conditional edge: fan out one Send per unique team
# ---------------------------------------------------------------------------

def map_to_teams(state: SMEState) -> list[Send]:
    """
    Fan out one brief_team_node invocation per unique SME team.

    Groups sme_notifications by team and returns a list of Send objects —
    one per team — so LangGraph runs them in parallel. Each Send carries
    the team name and that team's notification list as its state payload.
    """
    team_nodes: dict[str, list[dict]] = {}
    for n in state["sme_notifications"]:
        team_nodes.setdefault(n["team"], []).append(n)

    return [
        Send("brief_team", {"team": team, "team_notifications": notifs})
        for team, notifs in team_nodes.items()
    ]


# ---------------------------------------------------------------------------
# Node 2: brief_team (parallel, one invocation per team)
# ---------------------------------------------------------------------------

def brief_team_node(state: SMEState) -> dict:
    """
    Step 2 — Generate one LLM briefing for a single SME team.

    Receives the team name and that team's notifications via the Send payload.
    Makes one ChatOpenAI call using the team-specific system prompt.
    Returns only team_briefings — the _merge_dicts reducer accumulates results
    from all parallel invocations into a single dict.
    """
    team = state["team"]
    team_notifs = state["team_notifications"]

    system_prompt = TEAM_SYSTEM_PROMPTS.get(
        team,
        "You are a subject matter expert reviewing the impact of a design change on an IVD assay."
    )

    node_lines = [
        f"- [{n['trigger_node_type']}] {n['trigger_node_id']}: {n['trigger_node_title']}\n"
        f"  Your obligation: {n['review_obligation']}"
        for n in team_notifs
    ]
    user_prompt = (
        f"The following RTM nodes in your area of responsibility have been flagged "
        f"by the change impact analysis:\n\n{chr(10).join(node_lines)}\n\n"
        f"Write a 3–4 sentence briefing describing what your team needs to do, "
        f"in what order, and what the key risk is if action is delayed."
    )

    briefing = ""
    try:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=300,
        )
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        briefing = response.content
    except Exception:
        briefing = (
            f"[LLM unavailable — check OPENAI_API_KEY] "
            f"Manual review required for {len(team_notifs)} flagged node(s)."
        )

    return {"team_briefings": {team: briefing}}


# ---------------------------------------------------------------------------
# Node 3: finalize_notifications
# ---------------------------------------------------------------------------

def finalize_notifications_node(state: SMEState) -> dict:
    """
    Step 3 — Write llm_briefing back into each notification.

    Runs once after all parallel brief_team invocations complete and
    team_briefings is fully populated. Joins the briefing strings into
    the sme_notifications list so downstream consumers see a complete record.
    """
    updated = [
        {**n, "llm_briefing": state["team_briefings"].get(n["team"], "")}
        for n in state["sme_notifications"]
    ]
    return {"sme_notifications": updated}


# ---------------------------------------------------------------------------
# Build the LangGraph sub-agent
# ---------------------------------------------------------------------------

def build_sme_agent() -> Any:
    """
    Compile and return the LangGraph SME router sub-agent.

    Graph: map_teams → [Send × N teams] → brief_team (parallel) → finalize_notifications
    """
    builder = StateGraph(SMEState)

    builder.add_node("map_teams", map_teams_node)
    builder.add_node("brief_team", brief_team_node)
    builder.add_node("finalize_notifications", finalize_notifications_node)

    builder.add_edge(START, "map_teams")
    builder.add_conditional_edges("map_teams", map_to_teams, ["brief_team"])
    builder.add_edge("brief_team", "finalize_notifications")
    builder.add_edge("finalize_notifications", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Convert raw dicts back to SMENotification dataclasses
# ---------------------------------------------------------------------------

def notifications_from_dicts(raw: list[dict]) -> list[SMENotification]:
    return [
        SMENotification(
            team=n["team"],
            trigger_node_id=n["trigger_node_id"],
            trigger_node_type=n["trigger_node_type"],
            trigger_node_title=n["trigger_node_title"],
            review_obligation=n["review_obligation"],
            llm_briefing=n.get("llm_briefing", ""),
        )
        for n in raw
    ]
