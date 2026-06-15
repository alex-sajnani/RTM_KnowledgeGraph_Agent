"""
app.py — RTM Knowledge Graph Agent — Streamlit Dashboard

Run with:
    streamlit run app.py

Requires: OPENAI_API_KEY in .env or Streamlit secrets.

Pages (sidebar navigation):
  Dashboard        — AI prompt bar, quick-action cards, My Workbench
  Change Impact    — multi-agent analysis with SME notification assignments
  Graph Explorer   — interactive vis.js RTM dependency graph (raw HTML via st.components.v1)
  Document Extract — LLM entity extraction from regulatory text
  Audit            — RTM completeness score, orphan detection, V&V gaps, and event log
"""

import os
import sys
import json
import io
import csv
import html
import tempfile
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# Support Streamlit Cloud secrets as fallback to .env
if "OPENAI_API_KEY" not in os.environ:
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

# Add src/ to path (bare imports inside src/)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph import RTMGraph, NodeType, NodeStatus, EdgeType, HIERARCHY_LEVEL, build_seed_graph
from supervisor import build_supervisor, run_full_analysis, CHANGE_TYPES
from langgraph.checkpoint.memory import MemorySaver
from agent import ImpactReport
from extractor import RTMDocumentExtractor, SAMPLE_DOCUMENTS
from sme_agent import SME_NOTIFICATION_MAP
from regulations import load_regulations

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RTM Knowledge Graph Agent",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Vitanostics/Collate aesthetic
# ---------------------------------------------------------------------------

st.html("""
<style>
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Sidebar: always visible, never collapsible ── */

/* Hide the chevron button that collapses the sidebar */
button[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* Hide the expand button shown when sidebar is collapsed */
div[data-testid="collapsedControl"]           { display: none !important; }

/* Override the translateX Streamlit uses to slide the sidebar off-screen */
section[data-testid="stSidebar"] {
    transform:   none !important;
    min-width:   244px !important;
    position:    sticky !important;
    top:         0 !important;
    height:      100dvh !important;
    overflow-y:  auto !important;
}

/* Center the graph-stat metrics and completeness caption in the sidebar */
section[data-testid="stSidebar"] [data-testid="stMetric"] { text-align: center; }
section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    display: flex !important;
    justify-content: center !important;
    text-align: center !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { text-align: center; }
</style>
""")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "graph" not in st.session_state:
    st.session_state.graph = build_seed_graph()
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "impact_reports" not in st.session_state:
    st.session_state.impact_reports = []          # stored/approved reports (dashboard reads these)
if "current_impact_report" not in st.session_state:
    st.session_state.current_impact_report = None  # latest run's working report (display only)
if "extraction_results" not in st.session_state:
    st.session_state.extraction_results = []
if "notified_teams" not in st.session_state:
    st.session_state.notified_teams = {}
if "dashboard_query_result" not in st.session_state:
    st.session_state.dashboard_query_result = None
if "extraction_sme_state" not in st.session_state:
    st.session_state.extraction_sme_state = {}
if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"
if "prefill_change" not in st.session_state:
    st.session_state.prefill_change = ""
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
if "pending_escalation" not in st.session_state:
    st.session_state.pending_escalation = None   # interrupt payload dict while awaiting review
if "escalation_thread_id" not in st.session_state:
    st.session_state.escalation_thread_id = None
if "regulations" not in st.session_state:
    st.session_state.regulations = load_regulations()
if "supervisor" not in st.session_state:
    st.session_state.supervisor = build_supervisor(
        st.session_state.graph, st.session_state.checkpointer
    )

g: RTMGraph = st.session_state.graph

# Warn immediately when no API key is configured — avoids silent failures later.
if not os.environ.get("OPENAI_API_KEY"):
    st.warning(
        "**OPENAI_API_KEY not set.** "
        "Get a key at [platform.openai.com](https://platform.openai.com), "
        "then add `OPENAI_API_KEY=sk-...` to your `.env` file and restart the app. "
        "Impact Analysis and Document Extract will not work without it.",
        icon="🔑",
    )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def status_badge(status: str) -> None:
    color_map = {
        "not_started": "#FFCDD2",   # light red
        "active": "#C8E6C9",        # light green
        "pending_review": "#FFF9C4", # light yellow
        "invalidated": "#EF9A9A",   # red
        "approved": "#BBDEFB",      # blue
    }
    label_map = {
        "not_started": "Not Started",
        "active": "Active",
        "pending_review": "Pending Review",
        "invalidated": "Invalidated",
        "approved": "Approved",
    }
    color = color_map.get(status, "#E0E0E0")
    label = label_map.get(status, status.replace("_", " ").title())
    st.badge(label, color=color)


_EDITABLE_STATUSES = ["not_started", "pending_review", "active"]
_STATUS_LABELS = {"not_started": "Not Started", "pending_review": "Pending Review", "active": "Active"}


def navigate_to(page: str, prefill: str = "") -> None:
    st.session_state.current_page = page
    if prefill:
        st.session_state.prefill_change = prefill


def _query_graph(question: str, graph: RTMGraph, audit_log: list, impact_reports: list) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    nodes = graph.all_nodes()
    edges = graph.all_edges()
    node_by_id = {n["id"]: n for n in nodes}

    def _node_line(n: dict) -> str:
        line = (
            f"[{n['node_type']}] {n['id']}: {n['title']} (status: {n['status']})\n"
            f"  {n.get('description','')[:300]}"
        )
        # Chain-scoped closure verdict for User Needs and Design Inputs, so a status
        # question about one artifact cannot borrow a Test Result from another chain.
        closure = graph.closure_status(n["id"])
        if closure:
            trs = ", ".join(closure["test_results"]) or "none"
            line += (
                f"\n  V&V STATUS: {closure['verdict']} "
                f"Relevant Test Results for {n['id']}: {trs}."
            )
        # Full connected chain: every node a change here could propagate to (the
        # downstream closure). This is the authoritative membership list for "what
        # is in <node>'s chain" — so a chain/status answer surfaces ALL impacted
        # artifacts (e.g. an open CAPA triggered off a downstream Test Result), not
        # just the V&V-loop endpoints.
        downstream = graph.downstream_nodes(n["id"])
        if downstream:
            members = ", ".join(
                f"{d} [{node_by_id[d]['node_type']}] ({node_by_id[d]['status']})"
                for d in downstream
                if d in node_by_id
            )
            line += (
                f"\n  CONNECTED CHAIN (downstream nodes a change to {n['id']} may "
                f"impact): {members}"
            )
        return line

    node_text = "\n".join(_node_line(n) for n in nodes)
    edge_text = "\n".join(
        f"  {e['source']} --{e['edge_type']}--> {e['target']}"
        for e in edges
    )
    combined_log = graph.audit_log() + audit_log
    audit_text = "\n".join(
        f"{a.get('event_type', a.get('event',''))} | {a.get('timestamp','')[:19]} | "
        + ", ".join(f"{k}={v}" for k, v in a.items() if k not in ("event_type","event","timestamp","event_id"))
        for a in combined_log[-25:]
    )
    impact_text = ""
    for r in reversed(impact_reports[-3:]):
        impacted_ids = ", ".join(
            getattr(n, "node_id", "") for n in r.impacted_nodes
        ) or "none"
        concerns = "; ".join(r.immediate_concerns) if r.immediate_concerns else "none"
        escalation = (
            f"escalated to {r.escalation_reviewer or 'reviewer'}"
            f"{' (approved)' if r.approved else ' (pending/declined)'}"
            if r.escalation_required else "no escalation required"
        )
        state_label = "approved & stored" if r.approved else "working (pending final approval)"
        impact_text += (
            f"\n- Impact run on {r.changed_node_id} ({r.changed_node_title[:40]}) "
            f"at {r.timestamp[:19]} [{state_label}]:"
            f"\n    Change description: {r.change_description}"
            f"\n    Risk level: {r.risk_level.upper()} — {r.risk_rationale or 'no rationale recorded'}"
            f"\n    Immediate concerns: {concerns}"
            f"\n    {len(r.impacted_nodes)} nodes affected: {impacted_ids}"
            f"\n    V&V invalidations={r.vv_invalidations or 'none'}, "
            f"CAPAs={r.capa_triggers or 'none'}"
            f"\n    Escalation: {escalation}"
        )

    # V&V closure gaps — Design Inputs / User Needs whose chain is an open loop
    # (no Test Result closes verification/validation). Surfaced so any status
    # question correctly flags unverified/unvalidated chains.
    gaps = graph.chain_verification_gaps()
    if gaps:
        gap_text = "\n".join(
            f"  [{g['node_type']}] {g['id']}: {g['title']} (status: {g['status']}) — {g['issue']}"
            for g in gaps
        )
    else:
        gap_text = "  None — every User Need and Design Input has closing Test Result evidence."

    system_prompt = (
        "You are an expert regulatory affairs analyst for a medical device company. "
        "You have full access to the Requirements Traceability Matrix (RTM) for an "
        "hs-cTnI immunoassay (PMA P240052) including all nodes, edges, audit history, "
        "recent change impact analyses, and a V&V closure analysis. "
        "Answer questions about specific nodes, their upstream/downstream dependencies, "
        "status history, compliance flags, and relationships. "
        "When asked about the status of a specific User Need or Design Input, the "
        "AUTHORITATIVE verdict is the per-node 'V&V STATUS' line "
        "attached to that node in the RTM NODES list. Use it verbatim. It is scoped "
        "to that node's OWN design-control chain. CRITICAL: never attribute a Test "
        "Result to a node unless that Test Result is listed in that node's own "
        "V&V STATUS line. Different User Needs have separate chains — e.g. a pending "
        "Test Result in UN-001's chain says nothing about UN-002's validation. Do "
        "NOT pull a Test Result from another chain to explain a node's status. "
        "A design-control chain is only 'closed' when a completed Test Result "
        "verifies its Design Input (QMSR §820.30(f)) and validates its User Need "
        "(QMSR §820.30(g)). The V&V CLOSURE GAPS section below is a GLOBAL list of "
        "open loops across ALL chains — use it only for graph-wide questions, never "
        "to infer a specific node's status (defer to that node's V&V STATUS line). "
        "CONNECTED CHAIN COMPLETENESS: when a question asks about a node or its "
        "chain, the chain is that node's CONNECTED CHAIN line — every downstream "
        "node a change there could impact, NOT just the Design Input / Test Result "
        "V&V endpoints. You MUST account for every connected node that is in a "
        "material status and surface it as its own bullet: any CAPA, any node that "
        "is invalidated / pending_review / not_started, and any node flagged by a "
        "recent impact analysis. An existing open CAPA hanging off a downstream "
        "Test Result (e.g. a non-conformance corrective action) is one of the most "
        "important facts about a chain's status — never omit it. Only nodes in a "
        "fully completed status (active/approved) with nothing else to report may "
        "be folded out. "
        "RECENT IMPACT ANALYSES are part of a chain's current status, not separate "
        "trivia. When a question asks about a node or a chain, you MUST check the "
        "RECENT IMPACT ANALYSES section for any run whose changed node OR affected "
        "nodes fall in that chain, and surface it: name the change, its risk level, "
        "the affected nodes, and any V&V invalidations or CAPAs. A working "
        "(not-yet-approved) analysis is still a finding the reviewer needs to see — "
        "report it and note its approval state. Do not omit a relevant impact "
        "analysis just because the question used the word 'status'. "
        "Cite node IDs and edge types directly. "
        "Use regulatory terminology (QMSR §820.30, ISO 14971, 21 CFR Part 814) where relevant.\n\n"
        "OUTPUT FORMAT — keep it tight: aim for under 120 words. No opening "
        "preamble ('The status for ... is as follows') and no closing summary "
        "sentence ('Therefore, the chain remains...'). Lead with a one-line verdict "
        "(e.g. 'UN-001 chain: OPEN — not validated'). Then at most one short bullet "
        "per relevant node: '`ID` — <fact>' in a single line, no nested sub-bullets. "
        "Fold any relevant impact analysis into one bullet, not a paragraph. State "
        "each fact once; do not restate a Test Result's status under every node. "
        "Omit only nodes in a fully completed (active/approved) status with nothing "
        "else to report; always keep connected chain members in a material status "
        "(any CAPA, or anything invalidated / pending_review / not_started).\n\n"
        "STRICT GROUNDING RULE: You may ONLY refer to nodes that appear in the RTM NODES "
        "list provided in the user message. Those node IDs and titles are the complete and "
        "authoritative set of artifacts in the current graph. Never invent, assume, or "
        "reference any node, ID, title, or edge that is not explicitly present in that list. "
        "Do not fabricate node IDs (e.g. do not guess that a 'DI-002' or 'TR-005' exists "
        "unless it is listed). If the answer requires a node that is not in the list, state "
        "plainly that no such node exists in the current graph rather than making one up."
    )
    user_prompt = (
        f"RTM NODES:\n{node_text}\n\n"
        f"RTM EDGES:\n{edge_text}\n\n"
        f"V&V CLOSURE GAPS (open-loop chains missing Test Result evidence):\n{gap_text}\n\n"
        f"AUDIT LOG (last 25 events):\n{audit_text}\n\n"
        f"RECENT IMPACT ANALYSES:{impact_text or ' none'}\n\n"
        f"QUESTION: {question}"
    )

    try:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=700,
        )
        return llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    except Exception as exc:
        return f"Query failed — check OPENAI_API_KEY. Detail: {exc}"


def _extraction_team_briefing(team: str, nodes: list, doc_name: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    node_lines = "\n".join(
        f"- [{n.node_type.value}] {n.suggested_id}: {n.title} (confidence {n.confidence:.2f})"
        for n in nodes
    )
    system_prompt = (
        f"You are briefing the {team} team at a medical device company. "
        "They must review proposed RTM entities extracted from a regulatory document "
        "before those entities are committed to the live Requirements Traceability Matrix. "
        "Write a concise 2–3 sentence briefing explaining what is being proposed, "
        "what they should verify, and any risks or gaps they should flag before approving."
    )
    from sme_agent import QUANTITATIVE_GUARDRAIL
    user_prompt = (
        f"Source document: {doc_name}\n\n"
        f"Proposed RTM additions for {team} review:\n{node_lines}\n\n"
        f"Brief the {team} team.\n\n"
        f"{QUANTITATIVE_GUARDRAIL}"
    )
    try:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=200,
        )
        return llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]).content
    except Exception:
        return (
            f"Review {len(nodes)} proposed node(s) and confirm they are consistent "
            "with existing RTM structure before approving addition to the live graph."
        )


def _render_wrapped_table(rows: list[dict], wrap_columns: set[str] | None = None) -> None:
    """Render list-of-dict rows as an HTML table whose long cells wrap.

    st.dataframe (Streamlit 1.57) truncates cells with an ellipsis and exposes no
    text-wrap option, so columns of long free-text (e.g. Required Action, Review
    Obligation) get cut off. This renders a styled HTML table instead, wrapping
    only the columns named in ``wrap_columns`` and keeping the rest on one line.

    Args:
        rows: List of row dicts; keys of the first row define the column order.
        wrap_columns: Column names whose cells should wrap onto multiple lines.
    """
    if not rows:
        return
    columns = list(rows[0].keys())
    wrap = wrap_columns or set()

    header_html = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body_html = ""
    for row in rows:
        cells = ""
        for c in columns:
            cls = "wrap" if c in wrap else "nowrap"
            cells += f'<td class="{cls}">{html.escape(str(row.get(c, "")))}</td>'
        body_html += f"<tr>{cells}</tr>"

    st.markdown(
        f"""
        <style>
        .rtm-table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; margin-bottom: 1rem; }}
        .rtm-table th, .rtm-table td {{
            text-align: left; padding: 0.5rem 0.75rem; vertical-align: top;
            border-bottom: 1px solid rgba(128,128,128,0.2);
        }}
        .rtm-table th {{ font-weight: 600; background: rgba(128,128,128,0.06); white-space: nowrap; }}
        .rtm-table td.wrap {{ white-space: normal; overflow-wrap: anywhere; }}
        .rtm-table td.nowrap {{ white-space: nowrap; }}
        </style>
        <table class="rtm-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{body_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


COLOR_MAP = {
    "User Need":        "#5e81ac",  # Nord frost — dark blue
    "Design Input":     "#81a1c1",  # Nord frost — blue
    "Design Output":    "#88c0d0",  # Nord frost — cyan
    "V&V Protocol":     "#a3be8c",  # Nord aurora — green
    "Test Result":      "#ebcb8b",  # Nord aurora — yellow
    "Hazard":           "#bf616a",  # Nord aurora — red (risk)
    "Risk Control":     "#b48ead",  # Nord aurora — purple
    "CAPA":             "#d08770",  # Nord aurora — orange
}

EDGE_COLORS = {
    "verifies":    "#88c0d0",  # cyan  — Test Result → Design Input
    "triggers":    "#d08770",  # orange
    "invalidates": "#bf616a",  # red
    "linked_to":   "#4c566a",  # muted
}

# RTM hierarchy levels — pinned to vis.js node.level in Hierarchy layout
NODE_TYPE_LEVEL = {
    "User Need":              0,
    "Hazard":                 0,
    "Design Input":           1,
    "Risk Control":           1,
    "Design Output":          2,
    "V&V Protocol":           3,
    "Test Result":            4,
    "CAPA":                   5,
}

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### :material/biotech: RTM Agent")
    st.caption("hs-cTnI Immunoassay · P240052")

    st.space("small")

    nav_items = [
        ("dashboard",       ":material/home:",         "Dashboard"),
        ("change_impact",   ":material/bolt:",          "Change Impact"),
        ("graph_explorer",  ":material/hub:",           "Graph Explorer"),
        ("doc_extract",     ":material/description:",   "Document Extract"),
        ("audit",           ":material/fact_check:",    "Audit"),
    ]

    for page_key, icon, label in nav_items:
        is_active = st.session_state.current_page == page_key
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{page_key}", icon=icon, type=btn_type):
            navigate_to(page_key)
            st.rerun()

    st.space("medium")

    # Graph stats
    all_nodes = g.all_nodes()
    pending = sum(1 for n in all_nodes if n["status"] == "pending_review")
    not_started = sum(1 for n in all_nodes if n["status"] == "not_started")
    st.metric("Nodes", len(all_nodes))
    st.metric("Pending", pending)
    st.metric("Not Started", not_started)
    st.caption(f"Completeness: **{g.completeness_score()}%**")

# ===========================================================================
# PAGE: DASHBOARD
# ===========================================================================

if st.session_state.current_page == "dashboard":
    st.title("How can the RTM Agent help you today?")
    st.caption("Multi-agent change impact analysis, document extraction, and PMA readiness for the hs-cTnI immunoassay.")

    # AI prompt bar
    with st.container(border=True):
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            prompt_text = st.text_input(
                "prompt_bar",
                placeholder="Ask about your RTM",
                label_visibility="collapsed",
            )
        with col_btn:
            prompt_submit = st.button("Send", type="primary", icon=":material/send:")

    if prompt_submit and prompt_text:
        # Include the current working report (run but not yet persisted through the
        # final approval gate) so a freshly-run analysis is queryable from the chat
        # without waiting on approval. Deduped by identity against the persisted list.
        reports_for_chat = list(st.session_state.impact_reports)
        working = st.session_state.current_impact_report
        if working is not None and working not in reports_for_chat:
            reports_for_chat.append(working)
        with st.spinner("Querying RTM..."):
            st.session_state.dashboard_query_result = {
                "question": prompt_text,
                "answer": _query_graph(
                    prompt_text, g,
                    st.session_state.audit_log,
                    reports_for_chat,
                ),
            }
        st.rerun()

    if st.session_state.dashboard_query_result:
        qr = st.session_state.dashboard_query_result
        with st.container(border=True):
            st.caption(f"**Q:** {qr['question']}")
            st.write(qr["answer"])
            col_clear, col_impact = st.columns([1, 5])
            with col_clear:
                if st.button("Clear", key="clear_query", icon=":material/close:"):
                    st.session_state.dashboard_query_result = None
                    st.rerun()
            with col_impact:
                if st.button("Run as change impact →", key="query_to_impact", icon=":material/bolt:"):
                    navigate_to("change_impact", prefill=qr["question"])
                    st.session_state.dashboard_query_result = None
                    st.rerun()

    # Quick-action cards
    st.space("small")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            st.markdown(":material/bolt: **Run Impact Analysis**")
            st.caption("Trigger the multi-agent change impact pipeline on any RTM node.")
            if st.button("Open", key="card_impact", icon=":material/arrow_forward:"):
                navigate_to("change_impact")
                st.rerun()

    with c2:
        with st.container(border=True):
            st.markdown(":material/description: **Extract from Document**")
            st.caption("Paste regulatory text and extract RTM entities with LLM.")
            if st.button("Open", key="card_extract", icon=":material/arrow_forward:"):
                navigate_to("doc_extract")
                st.rerun()

    with c3:
        with st.container(border=True):
            st.markdown(":material/fact_check: **Audit**")
            st.caption("RTM completeness score, orphan detection, V&V gaps, and event log.")
            if st.button("Open", key="card_audit", icon=":material/arrow_forward:"):
                navigate_to("audit")
                st.rerun()

    with c4:
        with st.container(border=True):
            st.markdown(":material/hub: **Graph Explorer**")
            st.caption("Visualize the full RTM dependency network. Filter, search, and inspect nodes.")
            if st.button("Open", key="card_graph", icon=":material/arrow_forward:"):
                navigate_to("graph_explorer")
                st.rerun()

    # My Workbench
    st.space("small")
    st.subheader("My Workbench")

    wb_tab1, wb_tab2, wb_tab3 = st.tabs(["Change Records", "RTM Artifacts", "V&V Status"])

    with wb_tab1:
        if st.session_state.impact_reports:
            for r in reversed(st.session_state.impact_reports[-5:]):
                cols = st.columns([5, 2])
                with cols[0]:
                    st.markdown(f"**{r.changed_node_id}** — {r.changed_node_title[:50]}")
                    st.caption(r.timestamp[:19].replace("T", " ") + " UTC")
                with cols[1]:
                    st.markdown(f"{len(r.impacted_nodes)} nodes affected")
                st.divider()
        else:
            st.info("No impact analyses run yet. Use Change Impact to get started.")

    with wb_tab2:
        _status_changed = False

        def _render_artifact_row(node: dict, indent: bool = False) -> bool:
            """Render one artifact row with an editable status; return True if the status changed."""
            cols = st.columns([4, 2, 2])
            with cols[0]:
                prefix = "&nbsp;&nbsp;&nbsp;&nbsp;↳ " if indent else ""
                st.markdown(f"{prefix}**{node['id']}** {node['title'][:45]}", unsafe_allow_html=True)
            with cols[1]:
                st.caption(node["node_type"])
            with cols[2]:
                current_status = node["status"]
                if current_status in _EDITABLE_STATUSES:
                    new_status = st.selectbox(
                        "status",
                        options=_EDITABLE_STATUSES,
                        index=_EDITABLE_STATUSES.index(current_status),
                        format_func=lambda x: _STATUS_LABELS[x],
                        label_visibility="collapsed",
                        key=f"wb2_status_{node['id']}",
                    )
                    if new_status != current_status:
                        g.update_node_status(node["id"], NodeStatus(new_status), reason="Manual update via dashboard")
                        st.session_state.audit_log.append({
                            "event": "manual_status_update",
                            "node_id": node["id"],
                            "old_status": current_status,
                            "new_status": new_status,
                            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                        })
                        return True
                else:
                    status_badge(current_status)
            return False

        # Group artifacts under their parent User Need, with descendants nested below.
        # A node is assigned to a User Need if it is downstream of that need, OR — for
        # parallel root tracks like Hazard/Risk Control that have no User Need ancestor —
        # if the node's own downstream chain converges into that need's chain. Each node
        # is assigned once (first matching User Need); truly disconnected nodes fall into
        # "Ungrouped".
        all_nodes_by_id = {n["id"]: n for n in g.all_nodes()}
        user_needs = sorted(
            (n for n in all_nodes_by_id.values() if n["node_type"] == NodeType.USER_NEED.value),
            key=lambda x: x["id"],
        )

        def _descendant_sort_key(node: dict) -> tuple[int, str]:
            return (HIERARCHY_LEVEL.get(node["node_type"], 99), node["id"])

        # Pass 1 — each User Need's membership set (itself + everything downstream of it).
        un_members: dict[str, set[str]] = {}
        assigned: dict[str, str] = {}  # node_id -> user_need_id
        for un in user_needs:
            members = {un["id"]} | {nid for nid in g.downstream_nodes(un["id"]) if nid in all_nodes_by_id}
            un_members[un["id"]] = members
            for nid in members:
                assigned.setdefault(nid, un["id"])

        # Pass 2 — attach unassigned feeder nodes (e.g. Hazard/Risk Control roots) to the
        # first User Need whose chain their downstream chain converges into.
        for nid, node in all_nodes_by_id.items():
            if nid in assigned:
                continue
            chain = {nid} | {d for d in g.downstream_nodes(nid) if d in all_nodes_by_id}
            for un in user_needs:
                if chain & un_members[un["id"]]:
                    assigned[nid] = un["id"]
                    break

        for un in user_needs:
            group_ids = [nid for nid, owner in assigned.items() if owner == un["id"] and nid != un["id"]]
            st.markdown(f"#### {un['id']} — {un['title'][:60]}")
            if _render_artifact_row(un):
                _status_changed = True
            for node in sorted((all_nodes_by_id[nid] for nid in group_ids), key=_descendant_sort_key):
                if _render_artifact_row(node, indent=True):
                    _status_changed = True
            st.divider()

        ungrouped = sorted(
            (n for nid, n in all_nodes_by_id.items() if nid not in assigned),
            key=_descendant_sort_key,
        )
        if ungrouped:
            st.markdown("#### Ungrouped")
            for node in ungrouped:
                if _render_artifact_row(node):
                    _status_changed = True

        if _status_changed:
            st.rerun()
        st.components.v1.html("""
<script>
(function() {
  var COLOR_MAP = {
    "Active":         "#dcfce7",
    "Pending Review": "#fef9c3",
    "Not Started":    "#dbeafe",
  };
  function applyColors() {
    try {
      var doc = window.parent.document;
      doc.querySelectorAll('[data-testid="stSelectbox"]').forEach(function(sb) {
        var inner = sb.querySelector('[data-baseweb="select"] > div > div');
        if (!inner) return;
        var text = inner.textContent.trim();
        var color = COLOR_MAP[text];
        if (!color) return;
        var ctrl = sb.querySelector('[data-baseweb="select"] > div');
        if (ctrl) {
          ctrl.style.backgroundColor = color;
          ctrl.style.borderRadius = "6px";
          ctrl.style.transition = "background-color 0.2s";
        }
      });
    } catch(e) {}
  }
  applyColors();
  var obs = new MutationObserver(applyColors);
  try {
    obs.observe(window.parent.document.body, {subtree: true, childList: true, characterData: true});
  } catch(e) {}
})();
</script>
""", height=0)

    with wb_tab3:
        vv_nodes = [n for n in g.all_nodes() if n["node_type"] == NodeType.VV_PROTOCOL.value]
        _vv_status_changed = False
        for node in vv_nodes:
            cols = st.columns([4, 2, 2])
            with cols[0]:
                st.markdown(f"**{node['id']}** {node['title'][:45]}")
            with cols[1]:
                st.caption(node["node_type"])
            with cols[2]:
                current_status = node["status"]
                if current_status in _EDITABLE_STATUSES:
                    new_status = st.selectbox(
                        "status",
                        options=_EDITABLE_STATUSES,
                        index=_EDITABLE_STATUSES.index(current_status),
                        format_func=lambda x: _STATUS_LABELS[x],
                        label_visibility="collapsed",
                        key=f"wb3_status_{node['id']}",
                    )
                    if new_status != current_status:
                        g.update_node_status(node["id"], NodeStatus(new_status), reason="Manual update via dashboard")
                        st.session_state.audit_log.append({
                            "event": "manual_status_update",
                            "node_id": node["id"],
                            "old_status": current_status,
                            "new_status": new_status,
                            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                        })
                        _vv_status_changed = True
                else:
                    status_badge(current_status)
        if _vv_status_changed:
            st.rerun()

# ===========================================================================
# PAGE: CHANGE IMPACT
# ===========================================================================

elif st.session_state.current_page == "change_impact":
    st.header("Change Impact Analysis")
    st.caption("Multi-agent pipeline: Change Impact Agent → SME Router Agent → assembled report.")

    node_options = {
        f"{n['id']} — {n['title'][:55]}": n["id"]
        for n in g.all_nodes()
    }

    col_left, col_right = st.columns([1, 2])
    with col_left:
        selected_label = st.selectbox("Select changed RTM node", list(node_options.keys()))
        selected_node_id = node_options[selected_label]
        change_type = st.selectbox(
            "Change type",
            CHANGE_TYPES,
            help="The reviewer's attestation of what kind of change this is. "
                 "Documentation-only and No-change downgrade the risk to LOW even when "
                 "verification evidence is topologically downstream — an auditable, "
                 "reproducible decision the model never overrides.",
        )

    with col_right:
        prefill = st.session_state.pop("prefill_change", "") if "prefill_change" in st.session_state else ""
        change_desc = st.text_area(
            "Describe the change",
            value=prefill,
            placeholder="e.g. Tightening LoD specification from ≤ 2.0 pg/mL to ≤ 1.2 pg/mL "
                        "based on new clinical evidence from the ESC 0h/1h HEART pathway study.",
            height=100,
        )

    run_btn = st.button("Run Impact Analysis", type="primary", icon=":material/bolt:")

    if run_btn and change_desc:
        # Each run replaces the displayed output. The working report lives in
        # current_impact_report; it is only persisted to impact_reports (which the
        # dashboard reads) once a human approves it via the approval gate below.
        st.session_state.current_impact_report = None
        st.session_state.pending_escalation = None
        st.session_state.escalation_thread_id = None
        with st.spinner("Running multi-agent analysis (Change Impact + Risk Scoring + SME Router)..."):
            report, interrupt_payload, tid = run_full_analysis(
                g, selected_node_id, change_desc,
                checkpointer=st.session_state.checkpointer,
                supervisor=st.session_state.supervisor,
                change_type=change_type,
            )
        if interrupt_payload:
            st.session_state.pending_escalation = interrupt_payload
            st.session_state.escalation_thread_id = tid
            st.rerun()
        elif report:
            # Hold the result for display only. Nothing is logged or stored until a
            # human approves it at the approval gate, so abandoned/unapproved runs
            # leave no queryable trace.
            st.session_state.current_impact_report = report

    # ── Escalation gate UI ────────────────────────────────────────────────────
    if st.session_state.pending_escalation:
        esc = st.session_state.pending_escalation
        st.error(
            f"**Critical Risk Detected — Human Review Required**\n\n"
            f"{esc.get('risk_rationale', '')}",
            icon=":material/warning:",
        )

        if esc.get("immediate_concerns"):
            st.markdown("**Immediate concerns:**")
            for concern in esc["immediate_concerns"]:
                st.markdown(f"- {concern}")

        flag_cols = st.columns(3)
        with flag_cols[0]:
            vv = esc.get("vv_invalidations", [])
            if vv:
                st.error("V&V Invalidations\n" + "\n".join(f"• {v}" for v in vv), icon=":material/science:")
        with flag_cols[1]:
            capa = esc.get("capa_triggers", [])
            if capa:
                st.warning("CAPA Reviews\n" + "\n".join(f"• {c}" for c in capa), icon=":material/build:")

        st.divider()
        esc_reviewer = st.text_input(
            "Reviewer name / ID",
            placeholder="e.g., J.Smith / RA-Director",
            key="esc_reviewer",
        )
        esc_notes = st.text_area(
            "Review notes",
            placeholder="Describe your rationale for approving or rejecting this change.",
            height=80,
            key="esc_notes",
        )

        col_approve, col_reject = st.columns(2)
        with col_approve:
            if st.button(
                "Approve & continue analysis",
                type="primary",
                icon=":material/check_circle:",
                disabled=not esc_reviewer,
            ):
                with st.spinner("Resuming pipeline — running SME briefings..."):
                    report, _, _ = run_full_analysis(
                        g, "", "",
                        checkpointer=st.session_state.checkpointer,
                        thread_id=st.session_state.escalation_thread_id,
                        resume_payload={
                            "approved": True,
                            "reviewer": esc_reviewer,
                            "notes": esc_notes,
                        },
                        supervisor=st.session_state.supervisor,
                    )
                st.session_state.pending_escalation = None
                st.session_state.escalation_thread_id = None
                if report:
                    # Escalation approval only continues the pipeline; the assembled
                    # report is still a working result. It is persisted to
                    # impact_reports only when approved at the report approval gate.
                    st.session_state.current_impact_report = report
                    st.session_state.audit_log.append({
                        "event": "escalation_approved",
                        "node": report.changed_node_id,
                        "reviewer": esc_reviewer,
                        "timestamp": report.timestamp,
                        "risk_level": report.risk_level,
                    })
                st.rerun()

        with col_reject:
            if st.button(
                "Reject — halt pipeline",
                type="secondary",
                icon=":material/cancel:",
                disabled=not esc_reviewer,
            ):
                with st.spinner("Recording rejection..."):
                    run_full_analysis(
                        g, "", "",
                        checkpointer=st.session_state.checkpointer,
                        thread_id=st.session_state.escalation_thread_id,
                        resume_payload={
                            "approved": False,
                            "reviewer": esc_reviewer,
                            "notes": esc_notes,
                        },
                        supervisor=st.session_state.supervisor,
                    )
                st.session_state.audit_log.append({
                    "event": "escalation_rejected",
                    "reviewer": esc_reviewer,
                    "notes": esc_notes,
                    "risk_level": esc.get("risk_level", "critical"),
                })
                st.session_state.pending_escalation = None
                st.session_state.escalation_thread_id = None
                st.warning("Pipeline halted. The change has been rejected. Re-submit after addressing the risk.", icon=":material/block:")
                st.rerun()

    # ── Impact report ─────────────────────────────────────────────────────────
    if st.session_state.current_impact_report:
        report = st.session_state.current_impact_report

        st.subheader("Impact report")

        # Risk level badge
        risk_color = {"critical": "red", "high": "orange", "low": "green"}.get(report.risk_level, "gray")
        risk_label = {"critical": "Critical", "high": "High", "low": "Low"}.get(report.risk_level, report.risk_level.title())
        col_risk, col_meta = st.columns([1, 4])
        with col_risk:
            st.badge(f"Risk: {risk_label}", color=risk_color)
        with col_meta:
            st.caption(f"Change type attested as **{report.change_type}**")
            if report.escalation_required and report.escalation_reviewer:
                st.caption(f"Escalation reviewed by **{report.escalation_reviewer}**")

        # Compliance LLM summary
        if report.llm_summary:
            st.info(f"**Compliance Summary**\n\n{report.llm_summary}")

        # V&V / CAPA flags
        flag_cols = st.columns(2)
        with flag_cols[0]:
            if report.vv_invalidations:
                st.error("V&V Invalidations\n" + "\n".join(f"• {v}" for v in report.vv_invalidations), icon=":material/science:")
            else:
                st.success("No V&V Invalidations", icon=":material/science:")
        with flag_cols[1]:
            if report.capa_triggers:
                st.warning("CAPA Reviews\n" + "\n".join(f"• {c}" for c in report.capa_triggers), icon=":material/build:")
            else:
                st.success("No CAPA Triggers", icon=":material/build:")

        # Downstream obligations table
        downstream_nodes = [n for n in report.impacted_nodes if n.direction == "downstream"]
        upstream_nodes_list = [n for n in report.impacted_nodes if n.direction == "upstream"]
        if downstream_nodes:
            st.subheader(f"Downstream obligations ({len(downstream_nodes)} nodes)")
            rows = []
            for n in downstream_nodes:
                rows.append({
                    "Node ID": n.node_id,
                    "Type": n.node_type,
                    "Title": n.title,
                    "Status": n.current_status,
                    "Required Action": n.required_action,
                })
            _render_wrapped_table(rows, wrap_columns={"Title", "Required Action"})
        else:
            st.success("No downstream dependencies found — this node has no impact chain.")

        if upstream_nodes_list:
            st.subheader(f"Upstream requirements ({len(upstream_nodes_list)} nodes)")
            st.caption("Verify that the changed node still satisfies these parent requirements per QMSR §820.30(b).")
            up_rows = []
            for n in upstream_nodes_list:
                up_rows.append({
                    "Node ID": n.node_id,
                    "Type": n.node_type,
                    "Title": n.title,
                    "Status": n.current_status,
                    "Required Action": n.required_action,
                })
            _render_wrapped_table(up_rows, wrap_columns={"Title", "Required Action"})

        # Agent traversal trace
        with st.expander("Agent traversal trace", icon=":material/route:"):
            st.caption(
                "Exact traversal path the Change Impact agent followed. "
                "Each row shows the dependency chain from the changed node to the impacted node "
                "and the deterministic classification rule that triggered the required action."
            )
            if report.impacted_nodes:
                for n in report.impacted_nodes:
                    path_str = " → ".join(n.edge_path) if n.edge_path else report.changed_node_id
                    edge_str = " → ".join(n.edge_types_on_path) if n.edge_types_on_path else "—"
                    st.markdown(f"**`{n.node_id}`** [{n.node_type}] — *{n.title}*")
                    st.caption(f"Path: {path_str}")
                    st.caption(f"Edge types: {edge_str}")
                    st.caption(f"Rule: {n.required_action}")
                    st.divider()
            else:
                st.info("No downstream nodes traversed.")

        # SME Notification Assignments
        if report.sme_notifications:
            st.subheader("SME notification assignments")
            st.caption(
                "Teams automatically identified based on impact chain. "
                "Notify before updating compliance status."
            )

            sorted_notifs = sorted(report.sme_notifications, key=lambda n: n.team)

            sme_rows = []
            for n in sorted_notifs:
                sme_rows.append({
                    "Team": n.team,
                    "Trigger Node": f"{n.trigger_node_id} ({n.trigger_node_type})",
                    "Review Obligation": n.review_obligation,
                })
            _render_wrapped_table(sme_rows, wrap_columns={"Review Obligation"})

            # CSV export
            csv_buffer = io.StringIO()
            writer = csv.DictWriter(csv_buffer, fieldnames=["Team", "Trigger Node", "Trigger Type", "Trigger Title", "Review Obligation"])
            writer.writeheader()
            for n in sorted_notifs:
                writer.writerow({
                    "Team": n.team,
                    "Trigger Node": n.trigger_node_id,
                    "Trigger Type": n.trigger_node_type,
                    "Trigger Title": n.trigger_node_title,
                    "Review Obligation": n.review_obligation,
                })
            st.download_button(
                "Export notification list as CSV",
                icon=":material/download:",
                data=csv_buffer.getvalue(),
                file_name=f"sme_notifications_{report.changed_node_id}_{report.timestamp[:10]}.csv",
                mime="text/csv",
            )

        # Team Briefings
        if report.team_briefings:
            st.subheader("Team briefings")
            st.caption("LLM-generated briefings tailored to each team's domain vocabulary.")

            team_icons = {
                "Bioinformatics": ":material/genetics:",
                "R&D": ":material/science:",
                "Pathology": ":material/biotech:",
                "Quality/RA": ":material/policy:",
            }
            for team, briefing in report.team_briefings.items():
                icon = team_icons.get(team, ":material/group:")
                with st.expander(f"{team} briefing", icon=icon):
                    st.write(briefing)

        # Human approval gate
        st.subheader("Human approval gate")
        st.caption("Per 21 CFR Part 11 and QMSR §820.40, compliance status cannot be updated without documented human approval.")

        if report.approved:
            st.success("Report approved and stored. Downstream V&V/Test Result obligations and upstream traceability requirements marked PENDING_REVIEW.", icon=":material/check_circle:")
            approver = None
            approve_btn = False
        else:
            approver = st.text_input("Approver name / ID", placeholder="e.g., J.Smith / RA-Lead")
            approve_btn = st.button("Approve impact report & update statuses", type="secondary", icon=":material/check_circle:")

        if approve_btn and approver:
            for n in report.impacted_nodes:
                is_upstream = getattr(n, "direction", "downstream") == "upstream"
                # Two classes of artifact move to PENDING_REVIEW on approval:
                #   - downstream V&V Protocol / Test Result obligations (data generated
                #     against a now-superseded spec), and
                #   - ALL upstream requirements, which must be re-verified for
                #     bidirectional traceability per QMSR §820.30(b) before they can be
                #     considered current again. Leaving them 'active' would mislead the
                #     dashboard into showing them as still-verified.
                if (
                    n.node_type in [NodeType.VV_PROTOCOL.value, NodeType.TEST_RESULT.value]
                    or is_upstream
                ):
                    reason = (
                        f"Bidirectional traceability re-verification required per QMSR "
                        f"§820.30(b) after change to {report.changed_node_id}; "
                        f"approved by {approver}"
                        if is_upstream
                        else f"Change impact analysis approved by {approver}"
                    )
                    try:
                        g.update_node_status(
                            n.node_id,
                            NodeStatus.PENDING_REVIEW,
                            reason=reason,
                        )
                    except Exception:
                        pass
            report.approved = True
            # Persist the approved report so the dashboard chat and workbench can
            # reference it. Storage happens only here, on explicit human approval.
            st.session_state.impact_reports.append(report)
            st.session_state.audit_log.append({
                "event": "impact_report_approved",
                "node": report.changed_node_id,
                "approver": approver,
                "timestamp": report.timestamp,
                "impacted_count": len(report.impacted_nodes),
                "risk_level": report.risk_level,
                "sme_teams": list(report.team_briefings.keys()),
            })
            st.success(f"Report approved by {approver}. Downstream V&V/Test Result obligations and upstream traceability requirements marked PENDING_REVIEW.")
            st.rerun()

    elif not st.session_state.pending_escalation:
        st.info("Select a node and describe the change, then click **Run Impact Analysis**.")

# ===========================================================================
# PAGE: GRAPH EXPLORER
# ===========================================================================

elif st.session_state.current_page == "graph_explorer":
    import networkx as nx

    st.header("RTM Dependency Graph")
    st.caption("Interactive dependency network for the hs-cTnI immunoassay PMA device (P240052). Drag nodes, zoom, and hover for details.")

    import json

    if True:
        all_nodes_list = g.all_nodes()
        all_edges_list = g.all_edges()
        all_types = list(COLOR_MAP.keys())

        col_graph, col_ctrl = st.columns([3, 1], gap="medium")

        # ── RIGHT PANEL ──────────────────────────────────────────────────────
        with col_ctrl:

            # DATA MODEL
            st.markdown(
                "<span style='font-weight:600;font-size:13px;'>Data model</span> "
                "<span style='color:#888;font-size:11px;'>entire graph</span>",
                unsafe_allow_html=True,
            )
            schema_parts = []
            for nt, color in COLOR_MAP.items():
                schema_parts.append(
                    f'<span style="background:{color};color:#fff;border-radius:4px;'
                    f'padding:2px 6px;margin:2px;font-size:10px;display:inline-block;">{nt}</span>'
                )
            edge_parts = []
            for et, color in EDGE_COLORS.items():
                edge_parts.append(
                    f'<span style="color:{color};font-size:10px;margin:2px;display:inline-block;">→ {et}</span>'
                )
            st.markdown(
                f'<div style="background:#1a1a2e;border-radius:8px;padding:10px;">'
                f'{"".join(schema_parts)}'
                f'<hr style="border-color:#333;margin:6px 0;">'
                f'{"".join(edge_parts)}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.space("small")

            # FLOW MAP
            st.markdown(
                "<span style='font-weight:600;font-size:13px;'>Flow Map</span> "
                "<span style='color:#888;font-size:11px;'>FDA design control</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background:#fff;border-radius:8px;padding:6px;border:1px solid #e0e0e0;">'
                '<svg width="100%" viewBox="0 0 470 385" xmlns="http://www.w3.org/2000/svg">'
                '<defs>'
                '<marker id="fah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
                '<path d="M0,0 L0,8 L8,4 z" fill="#555"/>'
                '</marker>'
                '<marker id="fahr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
                '<path d="M0,0 L0,7 L7,3.5 z" fill="#888"/>'
                '</marker>'
                '</defs>'
                # Review (top-right, shaded, large bold-italic R)
                '<rect x="318" y="12" width="142" height="62" rx="2" fill="#d0d0d0" stroke="#222" stroke-width="2.5"/>'
                '<text text-anchor="middle" font-family="Georgia,Times New Roman,serif" fill="#000">'
                '<tspan x="389" y="52" font-size="28" font-style="italic" font-weight="bold">R</tspan>'
                '<tspan font-size="21">eview</tspan>'
                '</text>'
                # User Needs
                '<rect x="79" y="37" width="76" height="55" rx="2" fill="#fff" stroke="#444" stroke-width="1.5"/>'
                '<text text-anchor="middle" font-family="Arial,sans-serif" fill="#111" font-size="12">'
                '<tspan x="117" y="61">User</tspan><tspan x="117" dy="15">Needs</tspan>'
                '</text>'
                # Design Input
                '<rect x="160" y="110" width="72" height="58" rx="2" fill="#fff" stroke="#444" stroke-width="1.5"/>'
                '<text text-anchor="middle" font-family="Arial,sans-serif" fill="#111" font-size="12">'
                '<tspan x="196" y="134">Design</tspan><tspan x="196" dy="15">Input</tspan>'
                '</text>'
                # Design Process
                '<rect x="243" y="180" width="80" height="58" rx="2" fill="#fff" stroke="#444" stroke-width="1.5"/>'
                '<text text-anchor="middle" font-family="Arial,sans-serif" fill="#111" font-size="12">'
                '<tspan x="283" y="204">Design</tspan><tspan x="283" dy="15">Process</tspan>'
                '</text>'
                # Design Output
                '<rect x="321" y="248" width="80" height="50" rx="2" fill="#fff" stroke="#444" stroke-width="1.5"/>'
                '<text text-anchor="middle" font-family="Arial,sans-serif" fill="#111" font-size="12">'
                '<tspan x="361" y="269">Design</tspan><tspan x="361" dy="15">Output</tspan>'
                '</text>'
                # Medical Device
                '<rect x="358" y="306" width="86" height="44" rx="2" fill="#fff" stroke="#444" stroke-width="1.5"/>'
                '<text text-anchor="middle" font-family="Arial,sans-serif" fill="#111" font-size="12">'
                '<tspan x="401" y="325">Medical</tspan><tspan x="401" dy="15">Device</tspan>'
                '</text>'
                # Verification (shaded, large italic V)
                '<rect x="88" y="245" width="132" height="46" rx="2" fill="#d0d0d0" stroke="#444" stroke-width="1.5"/>'
                '<text text-anchor="middle" font-family="Georgia,Times New Roman,serif" fill="#000">'
                '<tspan x="154" y="275" font-size="20" font-style="italic" font-weight="bold">V</tspan>'
                '<tspan font-size="14">erification</tspan>'
                '</text>'
                # Validation (shaded, large italic V)
                '<rect x="14" y="304" width="132" height="54" rx="2" fill="#d0d0d0" stroke="#444" stroke-width="1.5"/>'
                '<text text-anchor="middle" font-family="Georgia,Times New Roman,serif" fill="#000">'
                '<tspan x="80" y="337" font-size="22" font-style="italic" font-weight="bold">V</tspan>'
                '<tspan font-size="17">alidation</tspan>'
                '</text>'
                # 1. User Needs → Design Input
                '<line x1="117" y1="92" x2="183" y2="110" stroke="#555" stroke-width="1.5" marker-end="url(#fah)"/>'
                # 2. Design Input → Design Process (gentle curve)
                '<path d="M 205 168 Q 224 184 241 196" stroke="#555" stroke-width="1.5" fill="none" marker-end="url(#fah)"/>'
                # 3. Design Process → Design Output
                '<line x1="307" y1="238" x2="330" y2="248" stroke="#555" stroke-width="1.5" marker-end="url(#fah)"/>'
                # 4. Design Output → Verification (left)
                '<line x1="321" y1="268" x2="222" y2="268" stroke="#555" stroke-width="1.5" marker-end="url(#fah)"/>'
                # 5. Design Output → Medical Device (down)
                '<line x1="368" y1="298" x2="382" y2="304" stroke="#555" stroke-width="1.5" marker-end="url(#fah)"/>'
                # 6. Medical Device → Validation (long left)
                '<line x1="358" y1="328" x2="148" y2="328" stroke="#555" stroke-width="1.5" marker-end="url(#fah)"/>'
                # 7. Validation → User Needs (left-side path up)
                '<path d="M 22 304 L 22 64 L 79 64" stroke="#555" stroke-width="2" fill="none" marker-end="url(#fah)"/>'
                # 8. Verification → Design Input (upward feedback)
                '<line x1="191" y1="245" x2="193" y2="170" stroke="#555" stroke-width="1.5" marker-end="url(#fah)"/>'
                # 9. Review → User Needs (lighter)
                '<line x1="318" y1="43" x2="155" y2="64" stroke="#888" stroke-width="1.2" marker-end="url(#fahr)"/>'
                # 10. Review → Design Input (lighter)
                '<line x1="323" y1="65" x2="218" y2="110" stroke="#888" stroke-width="1.2" marker-end="url(#fahr)"/>'
                # 11. Review → Design Output (lighter, straight down)
                '<line x1="357" y1="74" x2="356" y2="248" stroke="#888" stroke-width="1.2" marker-end="url(#fahr)"/>'
                # 12. Review → Medical Device (lighter, right-side path down)
                '<line x1="452" y1="74" x2="441" y2="306" stroke="#888" stroke-width="1.2" marker-end="url(#fahr)"/>'
                '</svg>'
                '<div style="font-size:8px;color:#aaa;text-align:right;margin-top:2px;">'
                '<a href="https://web.archive.org/web/20230201083208/https://www.fda.gov/media/116573/download"'
                ' target="_blank" style="color:#aaa;text-decoration:none;">Source: FDA Design Control Guidance</a>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.space("small")

            # COUNTERS
            st.markdown("**Counters**")
            c1, c2 = st.columns(2)
            c1.markdown(
                f'<div style="background:#2d2d2d;color:#fff;border-radius:6px;'
                f'padding:6px 10px;font-size:11px;text-align:center;">'
                f'<div style="font-size:9px;color:#aaa;"># Nodes</div>'
                f'<div style="font-size:18px;font-weight:700;">{len(all_nodes_list)}</div></div>',
                unsafe_allow_html=True,
            )
            c2.markdown(
                f'<div style="background:#2d2d2d;color:#fff;border-radius:6px;'
                f'padding:6px 10px;font-size:11px;text-align:center;">'
                f'<div style="font-size:9px;color:#aaa;"># Links</div>'
                f'<div style="font-size:18px;font-weight:700;">{len(all_edges_list)}</div></div>',
                unsafe_allow_html=True,
            )

            st.space("small")

            # FILTERS
            st.markdown("**Filters**")
            st.markdown('<span style="font-size:11px;color:#555;">Node Types</span>', unsafe_allow_html=True)
            selected_types = []
            for node_type in all_types:
                dot_color = COLOR_MAP[node_type]
                count = sum(1 for n in all_nodes_list if n["node_type"] == node_type)
                cb_col, badge_col = st.columns([1, 5])
                with cb_col:
                    checked = st.checkbox("", value=True, key=f"filter_{node_type}", label_visibility="collapsed")
                with badge_col:
                    st.markdown(
                        f'<span style="background:{dot_color};color:#fff;border-radius:4px;'
                        f'padding:2px 7px;font-size:10px;">{node_type}</span>',
                        unsafe_allow_html=True,
                    )
                if checked:
                    selected_types.append(node_type)

            st.space("small")

            # CONTROLS
            st.markdown("**Controls**")
            layout_choice = "Hierarchy"
            size_by_indegree = st.checkbox("Size by in-degree", value=False, key="size_indegree")
            show_edge_labels  = st.checkbox("Show edge labels",  value=True,  key="show_edge_labels")
            show_details      = st.checkbox("Show details",      value=True,  key="show_details")
            lock_positions    = st.checkbox("Lock positions",    value=False, key="lock_pos")

            st.space("small")

            # SEARCH
            search_query = st.text_input(
                "Search for a node", placeholder="Node ID or title…",
                key="search_node",
            )

            st.space("small")

            # GRAPH PARAMETERS
            st.markdown("**Graph Parameters**")
            all_node_ids = [n["id"] for n in all_nodes_list]
            root_node = st.selectbox(
                "Root node", ["(all)"] + sorted(all_node_ids),
                key="root_node",
            )

            st.divider()

            # ADD NODE
            with st.expander("Add node"):
                with st.form("add_node_form", clear_on_submit=True):
                    new_id    = st.text_input("Node ID", placeholder="e.g. RC-003")
                    new_type  = st.selectbox("Type", [nt.value for nt in NodeType])
                    new_title = st.text_input("Title")
                    new_desc  = st.text_area("Description", height=70)
                    new_status = st.selectbox("Status", [ns.value for ns in NodeStatus])
                    node_submitted = st.form_submit_button("Add node", type="primary", use_container_width=True)
                if node_submitted:
                    if not new_id.strip():
                        st.error("Node ID required.")
                    elif new_id.strip() in {n["id"] for n in g.all_nodes()}:
                        st.error(f"'{new_id.strip()}' already exists.")
                    elif not new_title.strip():
                        st.error("Title required.")
                    else:
                        g.add_node(
                            node_id=new_id.strip(),
                            node_type=NodeType(new_type),
                            title=new_title.strip(),
                            description=new_desc.strip(),
                            status=NodeStatus(new_status),
                        )
                        st.success(f"Node **{new_id.strip()}** added.")
                        st.rerun()

            # ADD DEPENDENCY
            with st.expander("Add dependency"):
                existing_ids = sorted(n["id"] for n in g.all_nodes())
                with st.form("add_edge_form", clear_on_submit=True):
                    src   = st.selectbox("Source", existing_ids, key="new_edge_src")
                    tgt   = st.selectbox("Target", existing_ids, key="new_edge_tgt")
                    etype = st.selectbox("Relationship", [et.value for et in EdgeType])
                    edge_submitted = st.form_submit_button("Add dependency", type="primary", use_container_width=True)
                if edge_submitted:
                    if src == tgt:
                        st.error("Source and target must differ.")
                    else:
                        try:
                            g.add_edge(
                                source=src,
                                target=tgt,
                                edge_type=EdgeType(etype),
                                extracted_by="manual",
                            )
                            st.success(f"**{src} → {tgt}** added.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

            # DELETE NODE
            with st.expander("Delete node"):
                del_node_ids = sorted(n["id"] for n in g.all_nodes())
                if del_node_ids:
                    with st.form("del_node_form", clear_on_submit=True):
                        del_node_id = st.selectbox("Node to delete", del_node_ids, key="del_node_sel")
                        connected_count = len(g._g.edges(del_node_id)) + len(list(g._g.in_edges(del_node_id)))
                        if connected_count:
                            st.caption(f":warning: Removes {connected_count} connected edge(s).")
                        del_node_submitted = st.form_submit_button(
                            "Delete node", type="primary", use_container_width=True
                        )
                    if del_node_submitted:
                        removed_edges = g.remove_node(del_node_id)
                        st.session_state.audit_log.append({
                            "event": "node_deleted",
                            "node_id": del_node_id,
                            "edges_removed": len(removed_edges),
                            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                        })
                        st.success(f"**{del_node_id}** deleted ({len(removed_edges)} edge(s) removed).")
                        st.rerun()
                else:
                    st.caption("No nodes in graph.")

            # DELETE EDGE
            with st.expander("Delete edge"):
                edge_options = [
                    f"{e['source']} → {e['target']} ({e.get('edge_type', '')})"
                    for e in g.all_edges()
                ]
                if edge_options:
                    with st.form("del_edge_form", clear_on_submit=True):
                        del_edge_label = st.selectbox("Edge to delete", edge_options, key="del_edge_sel")
                        del_edge_submitted = st.form_submit_button(
                            "Delete edge", type="primary", use_container_width=True
                        )
                    if del_edge_submitted:
                        # Parse back source/target from the label
                        parts = del_edge_label.split(" → ")
                        del_src = parts[0].strip()
                        del_tgt = parts[1].split(" (")[0].strip()
                        try:
                            g.remove_edge(del_src, del_tgt)
                            st.session_state.audit_log.append({
                                "event": "edge_deleted",
                                "source": del_src,
                                "target": del_tgt,
                                "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
                            })
                            st.success(f"**{del_src} → {del_tgt}** removed.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.caption("No edges in graph.")

        # ── LEFT PANEL — GRAPH ───────────────────────────────────────────────
        with col_graph:

            # In-degree map
            in_deg: dict[str, int] = {}
            for e in all_edges_list:
                in_deg[e["target"]] = in_deg.get(e["target"], 0) + 1

            # Root-node subgraph filter
            reachable: set[str] | None = None
            if root_node != "(all)":
                G_reach = nx.DiGraph()
                for e in all_edges_list:
                    G_reach.add_edge(e["source"], e["target"])
                try:
                    reachable = nx.descendants(G_reach, root_node) | {root_node}
                except Exception:
                    reachable = {root_node}

            # Build vis.js node/edge dicts
            visible_ids: set[str] = set()
            vis_nodes = []
            for n in all_nodes_list:
                if n["node_type"] not in selected_types:
                    continue
                if reachable is not None and n["id"] not in reachable:
                    continue
                if search_query:
                    sq = search_query.lower()
                    if sq not in n["id"].lower() and sq not in n["title"].lower():
                        continue
                visible_ids.add(n["id"])

                color = COLOR_MAP.get(n["node_type"], "#888888")

                size = 20
                if size_by_indegree:
                    size = max(15, min(55, 15 + in_deg.get(n["id"], 0) * 8))

                is_required = bool(n.get("metadata", {}).get("required"))
                label = f"{n['id']}\n{n['title'][:30]}" if show_details else n["id"]
                required_suffix = " · required — to be defined" if is_required else ""
                tooltip = f"{n['id']}: {n['title']} ({n['node_type']} · {n['status']}{required_suffix})"

                # Required placeholders (open V&V loops) render as a faded, dashed-border
                # node so they are visually distinct from committed artifacts.
                if is_required:
                    node_color = {
                        "background": "#f5f5f5",
                        "border": color,
                        "highlight": {"background": "#ededed", "border": "#5b5bd6"},
                    }
                else:
                    node_color = {
                        "background": color, "border": color,
                        "highlight": {"background": "#5b5bd6", "border": "#5b5bd6"},
                    }

                vis_nodes.append({
                    "id": n["id"],
                    "label": label,
                    "color": node_color,
                    "level": NODE_TYPE_LEVEL.get(n["node_type"], 4),
                    "size": size,
                    "title": tooltip,
                    "font": {"color": "#111111", "size": 12},
                    "shape": "dot",
                    "borderWidth": 3 if is_required else 1,
                    "shapeProperties": {"borderDashes": [4, 4]} if is_required else {"borderDashes": False},
                    "node_type": n["node_type"],
                    "node_status": n["status"],
                    "node_title": n["title"],
                    "node_description": n.get("description", ""),
                })

            node_level_map = {n["id"]: NODE_TYPE_LEVEL.get(n["node_type"], 4) for n in all_nodes_list}

            # Pre-index same-level outgoing edges per source so each gets a unique curve
            from collections import defaultdict
            same_level_out: dict = defaultdict(list)
            for e in all_edges_list:
                if e["source"] not in visible_ids or e["target"] not in visible_ids:
                    continue
                if node_level_map.get(e["source"]) == node_level_map.get(e["target"]):
                    same_level_out[e["source"]].append(e["target"])

            vis_edges = []
            for e in all_edges_list:
                if e["source"] not in visible_ids or e["target"] not in visible_ids:
                    continue
                edge_color = EDGE_COLORS.get(e.get("edge_type", ""), "#AAAAAA")
                src_level = node_level_map.get(e["source"], 0)
                tgt_level = node_level_map.get(e["target"], 0)
                same_level = src_level == tgt_level
                if e.get("edge_type") == "verifies":
                    smooth = False
                elif same_level:
                    siblings = same_level_out[e["source"]]
                    idx = siblings.index(e["target"]) if e["target"] in siblings else 0
                    roundness = 0.2 + (idx // 2) * 0.1
                    curve_type = "curvedCW" if idx % 2 == 0 else "curvedCCW"
                    smooth = {"type": curve_type, "roundness": roundness}
                elif src_level > tgt_level:
                    # back-edge (feedback): curve left to stay visually separate from forward flow
                    smooth = {"type": "curvedCCW", "roundness": 0.5}
                else:
                    smooth = False
                vis_edges.append({
                    "id": f"{e['source']}_{e['target']}",
                    "from": e["source"],
                    "to": e["target"],
                    "label": e.get("edge_type", "") if show_edge_labels else "",
                    "color": {"color": edge_color, "highlight": "#5b5bd6", "inherit": False},
                    "arrows": "to",
                    "font": {"size": 13, "color": "#000000", "background": "white", "strokeWidth": 3, "strokeColor": "white", "align": "middle"},
                    "smooth": smooth,
                })

            nodes_json = json.dumps(vis_nodes)
            edges_json = json.dumps(vis_edges)

            graph_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #ffffff; overflow: hidden; }}
  #graph {{ width: 100vw; height: 820px; background: #ffffff; }}

  #controls {{
    position: fixed;
    right: 18px;
    bottom: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    z-index: 999;
  }}

  /* ── Joystick ── */
  #joy-base {{
    width: 76px; height: 76px;
    background: rgba(255,255,255,0.08);
    border: 1.5px solid rgba(255,255,255,0.22);
    border-radius: 50%;
    position: relative;
    touch-action: none;
    cursor: grab;
    user-select: none;
  }}
  #joy-base:active {{ cursor: grabbing; }}
  #joy-thumb {{
    width: 30px; height: 30px;
    background: rgba(255,255,255,0.55);
    border-radius: 50%;
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    box-shadow: 0 0 6px rgba(0,0,0,0.4);
    transition: background 0.1s;
  }}

  /* ── Zoom wrap ── */
  #zoom-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }}
  .zoom-btn {{
    width: 26px; height: 26px;
    background: rgba(255,255,255,0.08);
    border: 1.5px solid rgba(255,255,255,0.22);
    border-radius: 6px;
    color: #ddd;
    font-size: 15px;
    line-height: 1;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s;
  }}
  .zoom-btn:hover {{ background: rgba(255,255,255,0.2); }}

  #zoom-slider {{
    -webkit-appearance: slider-vertical;
    writing-mode: vertical-lr;
    direction: rtl;
    width: 26px;
    height: 110px;
    cursor: pointer;
    accent-color: rgba(255,255,255,0.65);
    background: transparent;
  }}

  /* ── Node detail panel ── */
  #node-detail {{
    display: none;
    position: absolute;
    bottom: 16px;
    left: 16px;
    right: 88px;
    background: rgba(255,255,255,0.97);
    border: 1px solid #dde1e7;
    border-radius: 8px;
    padding: 12px 36px 12px 14px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.13);
    z-index: 20;
    max-height: 180px;
    overflow-y: auto;
    font-family: sans-serif;
  }}
  #node-detail-id {{
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #888;
    text-transform: uppercase;
    margin: 0 0 2px 0;
  }}
  #node-detail-meta {{
    font-size: 10px;
    color: #999;
    margin: 0 0 6px 0;
  }}
  #node-detail-title {{
    font-size: 13px;
    font-weight: 600;
    color: #111;
    margin: 0 0 6px 0;
    line-height: 1.35;
  }}
  #node-detail-desc {{
    font-size: 12px;
    color: #444;
    margin: 0;
    line-height: 1.55;
    white-space: pre-wrap;
  }}
  #node-detail-close {{
    position: absolute;
    top: 8px; right: 10px;
    background: none;
    border: none;
    cursor: pointer;
    color: #aaa;
    font-size: 16px;
    line-height: 1;
    padding: 0;
  }}
  #node-detail-close:hover {{ color: #333; }}
</style>
</head>
<body>
<div id="graph"></div>

<div id="controls">
  <div id="joy-base"><div id="joy-thumb"></div></div>
  <div id="zoom-wrap">
    <button class="zoom-btn" id="btn-in">+</button>
    <input type="range" id="zoom-slider" min="0.05" max="3" step="0.02" value="1">
    <button class="zoom-btn" id="btn-out">−</button>
  </div>
</div>

<div id="node-detail">
  <button id="node-detail-close">&#x2715;</button>
  <p id="node-detail-id"></p>
  <p id="node-detail-meta"></p>
  <p id="node-detail-title"></p>
  <p id="node-detail-desc"></p>
</div>

<script>
const nodesData = new vis.DataSet({nodes_json});
const edgesData = new vis.DataSet({edges_json});

const network = new vis.Network(
  document.getElementById('graph'),
  {{ nodes: nodesData, edges: edgesData }},
  {{
    layout: {{
      hierarchical: {{
        enabled: true,
        levelSeparation: 280,
        nodeSpacing: 220,
        treeSpacing: 550,
        sortMethod: 'directed',
        direction: 'UD',
        edgeMinimization: true,
        blockShifting: true,
        parentCentralization: true,
        improvedLayout: true,
      }}
    }},
    physics: {{ enabled: false }},
    interaction: {{ dragNodes: true, dragView: true, zoomView: false, hover: true, navigationButtons: false, keyboard: false }},
    nodes: {{ borderWidth: 1, shape: 'dot', font: {{ color: '#111111', size: 12, multi: false }}, widthConstraint: {{ maximum: 130 }} }},
    edges: {{ arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }}, font: {{ size: 13, color: '#000000', background: 'white', strokeWidth: 3, strokeColor: 'white', align: 'middle' }} }},
  }}
);

// ── Fixed grid layout — 250 px horizontal, 280 px vertical spacing ──
// Row 0 (y=0):   UN-001 | H-001 | H-002        UN-002
// Row 1 (y=280): DI-001 | RC-001 | RC-002       DI-002
// Row 2 (y=560): TR-001A | VP-001 | DO-001      TR-002A | VP-002 | DO-002
// Row 3 (y=840): CAPA-018
// TR is left of VP, VP is left of DO — all three share the same y row.
// Hierarchical layout disabled so positions are permanent.
network.once('afterDrawing', function() {{
  var pos = network.getPositions();
  var grid = {{
    'UN-001':   {{ x: -650, y:    0 }},
    'H-001':    {{ x: -400, y:    0 }},
    'H-002':    {{ x: -150, y:    0 }},
    'DI-001':   {{ x: -650, y:  280 }},
    'RC-001':   {{ x: -400, y:  280 }},
    'RC-002':   {{ x: -150, y:  280 }},
    'TR-001A':  {{ x: -650, y:  560 }},
    'VP-001':   {{ x: -400, y:  560 }},
    'DO-001':   {{ x: -150, y:  560 }},
    'CAPA-018': {{ x: -650, y:  840 }},
    'UN-002':   {{ x:  500, y:    0 }},
    'DI-002':   {{ x:  500, y:  280 }},
    'TR-002A':  {{ x:  250, y:  560 }},
    'VP-002':   {{ x:  500, y:  560 }},
    'DO-002':   {{ x:  750, y:  560 }},
  }};
  var updates = [];
  var gridIds = new Set(Object.keys(grid));

  // Apply hardcoded seed positions.
  Object.keys(grid).forEach(function(id) {{
    if (pos[id] !== undefined) updates.push({{ id: id, x: grid[id].x, y: grid[id].y }});
  }});

  // Place any non-seed nodes to the right in grid-aligned rows.
  // Group them into connected components, then assign each component a column
  // block starting to the right of the last seed node (x=750).
  var allNodes   = nodesData.get();
  var allEdges   = edgesData.get();
  var nonSeed    = allNodes.filter(function(n) {{ return !gridIds.has(n.id); }});

  if (nonSeed.length > 0) {{
    var nonSeedSet = new Set(nonSeed.map(function(n) {{ return n.id; }}));
    var nodeLvl    = {{}};
    allNodes.forEach(function(n) {{ nodeLvl[n.id] = (n.level !== undefined ? n.level : 4); }});

    // Build undirected adjacency for component detection (non-seed nodes only).
    var adj = {{}};
    nonSeed.forEach(function(n) {{ adj[n.id] = []; }});
    allEdges.forEach(function(e) {{
      if (nonSeedSet.has(e.from) && nonSeedSet.has(e.to)) {{
        adj[e.from].push(e.to);
        adj[e.to].push(e.from);
      }}
    }});

    // BFS to collect connected components.
    var visited    = new Set();
    var components = [];
    nonSeed.forEach(function(node) {{
      if (!visited.has(node.id)) {{
        var comp  = [];
        var queue = [node.id];
        visited.add(node.id);
        while (queue.length) {{
          var cur = queue.shift();
          comp.push(cur);
          adj[cur].forEach(function(nb) {{
            if (!visited.has(nb)) {{ visited.add(nb); queue.push(nb); }}
          }});
        }}
        components.push(comp);
      }}
    }});

    var LEV_SEP  = 280;   // vertical gap between levels — matches seed grid
    var COL_SEP  = 250;   // horizontal gap between sibling nodes
    var TREE_GAP = 380;   // horizontal gap between separate chains
    var curX     = 750 + TREE_GAP;   // start to the right of seed rightmost node

    // Mirror the seed grid's row collapse: the V&V triad — Design Output (L2),
    // V&V Protocol (L3), Test Result (L4) — shares a single row instead of
    // stacking, so a one-node-per-level chain forms a visible triangle and the
    // straight `verifies` back-edge (TR → DI) reads as the closing leg rather
    // than overlapping a vertical column.
    var ROW_OF_LEVEL = {{ 0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3 }};

    components.forEach(function(comp) {{
      // Group node IDs by display row (not raw level).
      var byRow = {{}};
      comp.forEach(function(id) {{
        var lv  = nodeLvl[id] || 0;
        var row = (ROW_OF_LEVEL[lv] !== undefined) ? ROW_OF_LEVEL[lv] : lv;
        if (!byRow[row]) byRow[row] = [];
        byRow[row].push(id);
      }});

      var rows     = Object.keys(byRow).map(Number).sort(function(a,b){{return a-b;}});
      var maxWidth = Math.max.apply(null, rows.map(function(r){{return byRow[r].length;}}));

      rows.forEach(function(row) {{
        // Within a shared row, order by descending level so Test Result (L4)
        // sits left, V&V Protocol (L3) center, Design Output (L2) right —
        // matching the seed grid's TR | VP | DO ordering.
        var ids = byRow[row].slice().sort(function(a, b){{
          return (nodeLvl[b] || 0) - (nodeLvl[a] || 0);
        }});
        var rowWidth = (ids.length - 1) * COL_SEP;
        var startX   = curX - rowWidth / 2;
        ids.forEach(function(id, i) {{
          updates.push({{ id: id, x: startX + i * COL_SEP, y: row * LEV_SEP }});
        }});
      }});

      curX += maxWidth * COL_SEP + TREE_GAP;
    }});
  }}

  network.setOptions({{ layout: {{ hierarchical: false }}, physics: {{ enabled: false }} }});
  if (updates.length) nodesData.update(updates);
}});

// ── Edge label visibility (hide labels for off-canvas edges) ──
const edgeLabelMap = {{}};
edgesData.forEach(e => {{ edgeLabelMap[e.id] = e.label || ''; }});

function refreshEdgeLabels() {{
  const pos = network.getPositions();
  const vp  = network.getViewPosition();
  const sc  = network.getScale();
  const el  = document.getElementById('graph');
  const hw  = el.offsetWidth  / sc / 2;
  const hh  = el.offsetHeight / sc / 2;
  const pad = 60 / sc;
  const l = vp.x - hw - pad, r = vp.x + hw + pad;
  const t = vp.y - hh - pad, b = vp.y + hh + pad;
  const updates = [];
  edgesData.forEach(e => {{
    const fp = pos[e.from], tp = pos[e.to];
    if (!fp || !tp) return;
    const both = fp.x > l && fp.x < r && fp.y > t && fp.y < b
              && tp.x > l && tp.x < r && tp.y > t && tp.y < b;
    const hDist = Math.abs(fp.x - tp.x);
    const vDist = Math.abs(fp.y - tp.y);
    const isCrossBranch = vDist > 0 && (hDist / vDist) > 1.5;
    const want = (both && !isCrossBranch) ? edgeLabelMap[e.id] : '';
    if (e.label !== want) updates.push({{ id: e.id, label: want }});
  }});
  if (updates.length) edgesData.update(updates);
}}

// ── Fit on load ────────────────────────────────────────────
const slider = document.getElementById('zoom-slider');
network.once('afterDrawing', () => {{
  network.fit({{ animation: false }});
  slider.value = Math.min(3, Math.max(0.05, network.getScale()));
  refreshEdgeLabels();
}});
network.on('zoom',    refreshEdgeLabels);
network.on('dragEnd', refreshEdgeLabels);

// ── Joystick ───────────────────────────────────────────────
const joyBase  = document.getElementById('joy-base');
const joyThumb = document.getElementById('joy-thumb');
const BASE_R = 38, THUMB_R = 15;
let jdx = 0, jdy = 0, jActive = false, rafId = null;

function panLoop() {{
  if (jdx !== 0 || jdy !== 0) {{
    const pos = network.getViewPosition();
    const s   = network.getScale();
    network.moveTo({{ position: {{ x: pos.x + jdx * 5 / s, y: pos.y + jdy * 5 / s }}, animation: false }});
  }}
  rafId = requestAnimationFrame(panLoop);
}}

joyBase.addEventListener('pointerdown', e => {{
  jActive = true;
  joyBase.setPointerCapture(e.pointerId);
  joyThumb.style.background = 'rgba(255,255,255,0.88)';
  rafId = requestAnimationFrame(panLoop);
  e.preventDefault();
}});

joyBase.addEventListener('pointermove', e => {{
  if (!jActive) return;
  const r  = joyBase.getBoundingClientRect();
  const cx = r.left + BASE_R, cy = r.top + BASE_R;
  let ox = e.clientX - cx, oy = e.clientY - cy;
  const d = Math.hypot(ox, oy), maxD = BASE_R - THUMB_R;
  if (d > maxD) {{ ox = ox / d * maxD; oy = oy / d * maxD; }}
  joyThumb.style.transform = `translate(calc(-50% + ${{ox}}px), calc(-50% + ${{oy}}px))`;
  jdx = ox / maxD;
  jdy = oy / maxD;
}});

function joyRelease() {{
  jActive = false; jdx = 0; jdy = 0;
  joyThumb.style.transform = 'translate(-50%, -50%)';
  joyThumb.style.background = 'rgba(255,255,255,0.55)';
  if (rafId) {{ cancelAnimationFrame(rafId); rafId = null; }}
}}
joyBase.addEventListener('pointerup',     joyRelease);
joyBase.addEventListener('pointercancel', joyRelease);

// ── Zoom slider ────────────────────────────────────────────
function setZoom(s) {{
  s = Math.min(3, Math.max(0.05, s));
  network.moveTo({{ scale: s, animation: {{ duration: 180, easingFunction: 'easeInOutQuad' }} }});
  slider.value = s;
}}

network.on('zoom', p => {{ slider.value = Math.min(3, Math.max(0.05, p.scale)); }});
slider.addEventListener('input', () => {{
  network.moveTo({{ scale: parseFloat(slider.value), animation: false }});
}});
document.getElementById('btn-in') .addEventListener('click', () => setZoom(network.getScale() * 1.25));
document.getElementById('btn-out').addEventListener('click', () => setZoom(network.getScale() / 1.25));

// ── Node detail panel (double-click) ───────────────────────
const detailPanel = document.getElementById('node-detail');
network.on('doubleClick', params => {{
  if (params.nodes.length === 0) {{ detailPanel.style.display = 'none'; return; }}
  const n = nodesData.get(params.nodes[0]);
  if (!n) return;
  document.getElementById('node-detail-id').textContent   = n.id;
  document.getElementById('node-detail-meta').textContent = n.node_type + ' · ' + n.node_status;
  document.getElementById('node-detail-title').textContent = n.node_title || '';
  document.getElementById('node-detail-desc').textContent  = n.node_description || '(no description)';
  detailPanel.style.display = 'block';
}});
document.getElementById('node-detail-close').addEventListener('click', () => {{
  detailPanel.style.display = 'none';
}});
</script>
</body>
</html>"""

        with col_graph:
            st.components.v1.html(graph_html, height=820, scrolling=False)


# ===========================================================================
# PAGE: DOCUMENT EXTRACT
# ===========================================================================

elif st.session_state.current_page == "doc_extract":
    st.header("LLM Document Extraction")
    st.caption(
        "Paste unstructured regulatory text (SOPs, FDA guidance, change logs, CLSI standards). "
        "The LLM extracts RTM nodes and edges with confidence scores. "
        "High-confidence entities can be added to the live graph after review."
    )

    sample_choice = st.selectbox(
        "Load a sample document (or paste your own below)",
        ["— paste your own —"] + list(SAMPLE_DOCUMENTS.keys()),
    )

    default_text = SAMPLE_DOCUMENTS.get(sample_choice, "") if sample_choice != "— paste your own —" else ""
    doc_text = st.text_area("Document text", value=default_text, height=220)
    doc_name = st.text_input("Document name", value=sample_choice if sample_choice != "— paste your own —" else "custom_doc.txt")

    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.75, 0.05,
                                     help="Entities below this threshold get PENDING_REVIEW status")

    extract_btn = st.button("Extract entities", type="primary", icon=":material/search:")

    if extract_btn and doc_text:
        with st.spinner("Extracting RTM entities..."):
            extractor = RTMDocumentExtractor()
            result = extractor.extract(doc_text, doc_name, graph=st.session_state.graph)
            st.session_state.extraction_results.append(result)
            st.session_state.audit_log.append({
                "event": "document_extracted",
                "document": doc_name,
                "extraction_id": result.extraction_id,
                "nodes": len(result.extracted_nodes),
                "edges": len(result.extracted_edges),
                "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            })

    if st.session_state.extraction_results:
        result = st.session_state.extraction_results[-1]

        st.subheader(f"Extraction Results — {result.extraction_id}")
        st.caption(f"Document: {result.document_name} | {result.timestamp[:19]} UTC")

        col_n, col_e = st.columns(2)
        with col_n:
            st.metric("Nodes Extracted", len(result.extracted_nodes))
        with col_e:
            st.metric("Edges Extracted", len(result.extracted_edges))

        if result.extracted_nodes:
            st.subheader("Extracted Nodes")
            hcols = st.columns([1.2, 1.5, 3, 0.8, 1.2, 0.6])
            for label in ("ID", "Type", "Title", "Confidence", "Status", ""):
                hcols[("ID", "Type", "Title", "Confidence", "Status", "").index(label)].caption(label)
            for i, n in enumerate(list(result.extracted_nodes)):
                cols = st.columns([1.2, 1.5, 3, 0.8, 1.2, 0.6])
                cols[0].write(n.suggested_id)
                cols[1].write(n.node_type.value)
                cols[2].write(n.title)
                cols[3].write(f"{n.confidence:.2f}")
                if getattr(n, "is_required", False):
                    cols[4].write("Not Started")
                elif getattr(n, "is_in_review", False):
                    cols[4].write("Pending Review")
                else:
                    cols[4].write("Active" if n.confidence >= confidence_threshold else "Pending Review")
                if cols[5].button("✕", key=f"del_node_{result.extraction_id}_{i}", help="Remove this node"):
                    result.extracted_nodes.pop(i)
                    st.rerun()

        if result.extracted_edges:
            st.subheader("Extracted Edges")
            hcols = st.columns([1.5, 1.5, 1.2, 0.8, 3, 0.6])
            for label, col in zip(("Source", "Target", "Type", "Confidence", "Rationale", ""), hcols):
                col.caption(label)
            for i, e in enumerate(list(result.extracted_edges)):
                cols = st.columns([1.5, 1.5, 1.2, 0.8, 3, 0.6])
                cols[0].write(e.source_id)
                cols[1].write(e.target_id)
                cols[2].write(e.edge_type.value)
                cols[3].write(f"{e.confidence:.2f}")
                cols[4].write(e.rationale)
                if cols[5].button("✕", key=f"del_edge_{result.extraction_id}_{i}", help="Remove this edge"):
                    result.extracted_edges.pop(i)
                    st.rerun()

        if not result.extracted_nodes and not result.extracted_edges:
            with st.expander("No entities extracted — show LLM response for diagnosis", icon=":material/warning:"):
                st.code(result.raw_llm_response or "(empty)", language=None)

        # ── SME Review ─────────────────────────────────────────────────────
        if result.extracted_nodes:
            st.divider()
            st.subheader("SME review")
            st.caption(
                "Route extracted entities to relevant teams for sign-off before committing to the live graph."
            )

            eid = result.extraction_id
            sme_state = st.session_state.extraction_sme_state.get(eid)

            if not sme_state:
                if st.button("Request SME review", icon=":material/send:", key=f"req_sme_{eid}"):
                    team_nodes: dict[str, list] = {}
                    for n in result.extracted_nodes:
                        for team, obligation in SME_NOTIFICATION_MAP.get(n.node_type.value, []):
                            team_nodes.setdefault(team, []).append((n, obligation))

                    with st.spinner("Generating team briefings..."):
                        briefings = {}
                        for team, pairs in team_nodes.items():
                            briefings[team] = {
                                "nodes": pairs,
                                "obligation": pairs[0][1],
                                "briefing": _extraction_team_briefing(
                                    team, [p[0] for p in pairs], result.document_name
                                ),
                            }
                    st.session_state.extraction_sme_state[eid] = {
                        "briefings": briefings,
                        "approvals": set(),
                    }
                    st.rerun()
            else:
                briefings = sme_state["briefings"]
                approvals  = sme_state["approvals"]
                team_icons = {
                    "Bioinformatics": ":material/genetics:",
                    "R&D":            ":material/science:",
                    "Pathology":      ":material/biotech:",
                    "Quality/RA":     ":material/policy:",
                }
                all_approved = bool(briefings) and all(t in approvals for t in briefings)

                for team, data in briefings.items():
                    is_approved = team in approvals
                    with st.container(border=True):
                        col_hd, col_btn = st.columns([5, 1])
                        with col_hd:
                            st.markdown(f"**{team_icons.get(team, ':material/group:')} {team}**")
                            node_summary = "  ·  ".join(
                                f"`{p[0].suggested_id}` ({p[0].node_type.value})"
                                for p in data["nodes"][:4]
                            )
                            st.caption(f"Reviewing: {node_summary}")
                            st.caption(data["obligation"][:130])
                        with col_btn:
                            if is_approved:
                                st.success("Approved")
                            else:
                                if st.button("Approve", key=f"sme_approve_{eid}_{team}", type="primary"):
                                    sme_state["approvals"].add(team)
                                    st.rerun()
                        with st.expander("Briefing", expanded=not is_approved):
                            st.write(data["briefing"])

                st.space("small")
                if not all_approved:
                    pending = [t for t in briefings if t not in approvals]
                    st.warning(f"Pending sign-off from: {', '.join(pending)}", icon=":material/pending:")

        add_btn = st.button("Add to graph (after human review)", type="secondary", icon=":material/add_circle:")
        if add_btn:
            extractor = RTMDocumentExtractor()
            nodes_added, edges_added = extractor.add_to_graph(result, g, confidence_threshold)
            st.success(f"Added {nodes_added} nodes and {edges_added} edges to the live RTM graph.")
            st.session_state.audit_log.append({
                "event": "entities_added_to_graph",
                "extraction_id": result.extraction_id,
                "nodes_added": nodes_added,
                "edges_added": edges_added,
                "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            })
            st.rerun()
    else:
        st.info("Load a sample document or paste regulatory text, then click **Extract Entities**.")

# ===========================================================================
# PAGE: AUDIT TRAIL
# ===========================================================================

elif st.session_state.current_page == "audit":
    st.header("Audit")
    st.caption("RTM completeness per QMSR §820.30 and immutable event log per §820.180.")

    st.subheader("Readiness assessment")

    score = g.completeness_score()
    bd = g.completeness_breakdown()
    orphans = g.orphaned_nodes()
    missing_vv = g.missing_vv_links()
    open_loops = g.chain_verification_gaps()

    # Score card
    score_col, detail_col = st.columns([1, 2])
    with score_col:
        with st.container(border=True):
            gap = 90 - score
            st.metric(
                "RTM Completeness",
                f"{score}%",
                delta="Audit ready" if score >= 90 else f"{gap}% gap to threshold",
                delta_color="normal" if score >= 90 else "inverse",
            )
            st.caption("40% structural · 60% artifact readiness")

    with detail_col:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total RTM nodes", bd["total"])
        m2.metric(
            "Structural score",
            f"{round(bd['structural_score'] * 100, 1)}%",
            delta=f"-{bd['structural_issues']} gaps" if bd["structural_issues"] else "No gaps",
            delta_color="inverse" if bd["structural_issues"] else "normal",
            help="Fraction of nodes free from topology gaps: orphaned nodes, missing V&V links, unmet User Needs, incomplete Design Inputs.",
        )
        m3.metric(
            "Artifact readiness",
            f"{round(bd['readiness_score'] * 100, 1)}%",
            delta=f"{bd['ready_count']} active · {bd['partial_count']} pending · {bd['blocked_count']} not started",
            delta_color="normal" if bd["ready_count"] == bd["total"] else "off",
            help="Average per-node readiness: active/approved = 100%, pending review/not started = 50%, invalidated = 0%.",
        )

    if score >= 90:
        st.success("RTM is audit-ready. All nodes have traceability links and V&V coverage.", icon=":material/check_circle:")
    else:
        if orphans:
            st.error(f"**Orphaned Nodes ({len(orphans)})** — No traceability links:")
            for nid in orphans:
                try:
                    node = g.get_node(nid)
                    st.write(f"- `{nid}` [{node['node_type']}] {node['title']}")
                except Exception:
                    st.write(f"- `{nid}`")

        if missing_vv:
            st.warning(f"**Design Outputs Missing V&V Links ({len(missing_vv)})**:")
            for nid in missing_vv:
                try:
                    node = g.get_node(nid)
                    st.write(f"- `{nid}` {node['title']} — No 'verifies' edge from any Test Result")
                except Exception:
                    st.write(f"- `{nid}`")

        # Status-aware open loops: a 'verifies' edge may exist, but if its Test
        # Result is a required placeholder (not_started) or still pending, the
        # design-control loop is NOT closed. These gaps are invisible to the
        # structural missing_vv check above, so surface them explicitly.
        if open_loops:
            st.warning(f"**Open Verification/Validation Loops ({len(open_loops)})** — links exist but no completed Test Result closes them:")
            for gap in open_loops:
                st.write(f"- `{gap['id']}` [{gap['node_type']}] {gap['title']} — {gap['issue']}")

    # ── RTM Export ────────────────────────────────────────────────────────────
    st.subheader("Export RTM")
    st.caption("One row per dependency link with source and target node attributes inlined.")

    node_lookup = {n["id"]: n for n in g.all_nodes()}
    _COLS = [
        "source_id", "source_type", "source_title", "source_status",
        "edge_type",
        "target_id", "target_type", "target_title", "target_status",
    ]
    _buf = io.StringIO()
    _writer = csv.DictWriter(_buf, fieldnames=_COLS)
    _writer.writeheader()
    for _e in g.all_edges():
        _src = node_lookup.get(_e["source"], {})
        _tgt = node_lookup.get(_e["target"], {})
        _writer.writerow({
            "source_id":     _e["source"],
            "source_type":   _src.get("node_type", ""),
            "source_title":  _src.get("title", ""),
            "source_status": _src.get("status", ""),
            "edge_type":     _e.get("edge_type", ""),
            "target_id":     _e["target"],
            "target_type":   _tgt.get("node_type", ""),
            "target_title":  _tgt.get("title", ""),
            "target_status": _tgt.get("status", ""),
        })
    st.download_button(
        "Export RTM as CSV",
        icon=":material/download:",
        data=_buf.getvalue(),
        file_name="rtm_dependency_graph.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # ── Event log ─────────────────────────────────────────────────────────────
    st.subheader("Event log")
    st.caption("Immutable log of all agent actions, approvals, and graph mutations. Per QMSR §820.180.")

    graph_log = g.audit_log()
    all_events = list(reversed(st.session_state.audit_log)) + list(reversed(graph_log))

    if all_events:
        for event in all_events[:50]:
            cols = st.columns([2, 6])
            with cols[0]:
                ts = event.get("timestamp", "")
                st.caption(ts[:19].replace("T", " ") if ts else "—")
            with cols[1]:
                event_type = event.get("event_type") or event.get("event", "unknown")
                st.markdown(f"**{event_type}**")
                details = {k: v for k, v in event.items()
                           if k not in ("event_type", "event", "timestamp", "event_id")}
                if details:
                    st.caption(" · ".join(f"{k}: {v}" for k, v in details.items()))

        st.space("small")
        export_data = json.dumps(all_events, indent=2, default=str)
        st.download_button(
            "Export audit log as JSON",
            icon=":material/download:",
            data=export_data,
            file_name="rtm_audit_log.json",
            mime="application/json",
        )
    else:
        st.info("No audit events recorded yet.")
