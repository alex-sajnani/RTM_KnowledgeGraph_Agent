# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# One-time setup
cp .env.example .env        # then set OPENAI_API_KEY=sk-...
pip install -r requirements.txt

# Launch
streamlit run app.py        # opens at http://localhost:8501
```

The only required credential is `OPENAI_API_KEY`. Set `OPENAI_MODEL` to override the default (`gpt-4o-mini`).

## Architecture

```
app.py (Streamlit entry point)
  └── sys.path.insert(0, "src/")   ← all src imports are bare (no src. prefix)
       ├── supervisor.py   — top-level LangGraph orchestrator; run_full_analysis() is the public entry point
       ├── agent.py        — Change Impact sub-agent (traverse → classify → report); build_impact_agent(graph)
       ├── sme_agent.py    — SME Router sub-agent (map_teams → Send × N → brief_team parallel → finalize_notifications); build_sme_agent()
       ├── graph.py        — RTMGraph class, node/edge enums, build_seed_graph()
       ├── extractor.py    — LLM document extraction, ExtractionResult → RTMGraph
       └── regulations.py  — eCFR fetch + cache; load_regulations() / build_prompt_context()
```

**Multi-agent pipeline** (`src/supervisor.py`):
1. `run_impact_agent` — invokes compiled Change Impact subgraph; result stored in `impact_result`
2. `score_risk` — deterministic risk level from structural flags + LLM rationale; sets `risk_level`, `risk_rationale`, `immediate_concerns` in state
3. `[conditional edge]` — `"critical"` → `escalation_gate`; `"high"` / `"low"` → `run_sme_agent`
4. `escalation_gate` — LangGraph `interrupt()`: suspends execution, persists state in `MemorySaver`; on resume `approved=True` → `run_sme_agent`, `approved=False` → `END`
5. `run_sme_agent` — feeds `impacted_nodes` into SME Router subgraph; result stored in `sme_result`
6. `assemble_report` — builds `ImpactReport` dataclass from both sub-agent results + escalation decision

**Supervisor factory pattern** (`src/supervisor.py`):
`build_supervisor(graph, checkpointer)` compiles both sub-agents internally and captures them in node closures. `run_full_analysis` does not construct sub-agents — it only passes the graph and checkpointer.

**`run_full_analysis` signature:**
```python
run_full_analysis(
    graph: RTMGraph,
    changed_node_id: str,
    change_description: str,
    checkpointer=None,           # MemorySaver from st.session_state.checkpointer
    thread_id: str | None = None,
    resume_payload: dict | None = None,  # {"approved": bool, "reviewer": str, "notes": str}
    supervisor: Any = None,      # pass st.session_state.supervisor to avoid rebuild each call
) -> tuple[ImpactReport | None, dict | None, str]
# (report, interrupt_payload, thread_id)
# Interrupted: report=None, interrupt_payload carries risk details
# Complete:    interrupt_payload=None
# Rejected:    report=None, interrupt_payload=None
```

**Change Impact sub-agent** (`src/agent.py`):
`build_impact_agent(graph: RTMGraph)` — all node functions close over the graph argument. The graph is NOT in `AgentState`; state is fully serializable (strings, lists, dicts only). Three nodes: `traverse → classify → report`. Classify is intentionally deterministic (regulatory action strings mapped to node types for auditability). Risk intelligence lives in supervisor's `score_risk_node`.

`AgentState` fields (all serializable):
- `downstream_ids: list[str]` — BFS descendants of the changed node
- `upstream_ids: list[str]` — immediate predecessors (direct parents) of the changed node
- `impacted_nodes: list[dict]` — all nodes with `direction` field: `"downstream"` or `"upstream"`

`ImpactedNode` carries a `direction: str = "downstream"` field. Upstream nodes surface for QMSR §820.30(b) bidirectional traceability review only — they never trigger `vv_invalidations`, `pma_supplement_flags`, or `capa_triggers`.

`traverse_node` collects both `nx.descendants(graph, changed_id)` (downstream) and `graph._g.predecessors(changed_id)` (upstream). `classify_node` early-continues for upstream nodes, assigning a fixed §820.30(b) bidirectional review action instead of the regulatory classification tree.

**Risk level determination** (`src/supervisor.py` — `_compute_risk_level`):
Purely deterministic from structural signals — no LLM controls routing:
- `"critical"`: any V&V invalidations OR any PMA supplement flags
- `"high"`: any CAPA triggers OR Hazard/Risk Control nodes in impact chain
- `"low"`: everything else

The LLM call that follows `_compute_risk_level` only produces human-readable `rationale` and `immediate_concerns` (structured output via `RiskExplanation` Pydantic model). It cannot change the routing outcome.

**RTMGraph** (`src/graph.py`) wraps a `networkx.DiGraph`. `build_seed_graph()` loads the hs-cTnI immunoassay PMA device demo dataset (16 nodes, 18 edges).

Analytics helpers on `RTMGraph`:
- `orphaned_nodes()` — nodes with no edges
- `missing_vv_links()` — Design Inputs with no incoming `validates` edge from a Test Result
- `unmet_user_needs()` — User Needs with no out-neighbors of type Design Input
- `incomplete_design_inputs()` — Design Inputs with no out-neighbors of type Design Output
- `has_path(source, target)` — returns True if a directed path exists; used by `extractor.py` for cycle detection
- `completeness_score()` — 0–100 score combining structural topology completeness (40%) and average per-node status readiness (60%); readiness weights: active/approved = 1.0, pending_review/not_started = 0.5, invalidated = 0.0

**Node types and hierarchy levels** (used by vis.js in Graph Explorer):
- Level 0: `User Need`, `Hazard` (parallel tracks)
- Level 1: `Design Input`, `Risk Control` (parallel tracks)
- Level 2: `Design Output`
- Level 3: `V&V Protocol`
- Level 4: `Test Result`
- Level 5: `CAPA`
- Level 6: `PMA Supplement Trigger`

**Seed graph dependency chain:**
```
User Need → Hazard (LINKED_TO) → Risk Control (TRIGGERS) → Design Input/Output (LINKED_TO)
User Need → Design Input (LINKED_TO) → Design Output (LINKED_TO)
Design Output → V&V Protocol (LINKED_TO)
V&V Protocol → Test Result (LINKED_TO)
Test Result → CAPA (TRIGGERS) → PMA Supplement Trigger (TRIGGERS)
Test Result → Design Input (VALIDATES)   ← feedback/validation edge
```
Edge direction rule: every edge points from the artifact that, when changed, forces updates in the artifact it points to. A change to a Design Output forces re-execution of its V&V protocol, so the edge is `DO→VP (LINKED_TO)`. Following this, the full change-impact chain from `DI-001` is: `DI-001 → DO-001 → VP-001 → TR-001A → CAPA-018 → PM-001`.

**SME Router** (`src/sme_agent.py`) — `SME_NOTIFICATION_MAP` is the single source of truth mapping `NodeType` values to (team, obligation) pairs. Four teams: Bioinformatics, R&D, Pathology, Quality/RA.

Graph: `map_teams → [Send × N teams] → brief_team (×N parallel) → finalize_notifications`

- `map_teams_node` — pure deterministic mapping; populates `sme_notifications` list and initializes `team_briefings: {}`
- `map_to_teams` — conditional edge function; groups `sme_notifications` by team, returns one `Send("brief_team", {"team": ..., "team_notifications": [...]})` per unique team
- `brief_team_node` — one LLM call per invocation using a team-specific system prompt from `TEAM_SYSTEM_PROMPTS`; returns `{"team_briefings": {team: briefing_text}}` only
- `finalize_notifications_node` — runs once after all parallel invocations complete; joins `team_briefings` into each notification's `llm_briefing` field

`SMEState.team_briefings` carries `Annotated[dict, _merge_dicts]` so LangGraph merges parallel writes from concurrent `brief_team` nodes rather than overwriting. `team` and `team_notifications` are Send-payload fields in `SMEState` consumed by `brief_team_node`.

Also imported by `app.py` for the Document Extract SME review path.

**ImpactReport** (`src/agent.py`) — includes `sme_notifications: list[SMENotification]` and `team_briefings: dict` populated by the supervisor. The `pma_supplement_flags` field flags `PMA_SUPPLEMENT_TRIGGER` nodes in the impact chain. Risk and escalation fields: `risk_level`, `risk_rationale`, `immediate_concerns`, `escalation_required`, `escalation_reviewer`, `escalation_notes`. `approved: bool = False` — set to `True` only when a human explicitly approves the report through the UI; the graph is never updated otherwise.

**UI navigation** (`app.py`) — sidebar-driven; `st.session_state.current_page` controls which page renders. Five pages: `dashboard`, `change_impact`, `graph_explorer`, `doc_extract`, `audit`. The `audit` page combines RTM completeness assessment (top) and the event log (bottom).

**Key session state keys:**
- `graph` — live RTMGraph instance
- `audit_log` — app-level event list (separate from graph's internal audit log)
- `impact_reports` — list of ImpactReport objects
- `extraction_results` — list of ExtractionResult objects
- `extraction_sme_state` — dict: extraction_id → {briefings, approvals} for doc extract SME review
- `notified_teams` — dict for change impact SME notification tracking
- `dashboard_query_result` — dict: {question, answer} for the dashboard LLM query bar
- `checkpointer` — `MemorySaver` instance; persists LangGraph interrupt/resume state across Streamlit reruns
- `pending_escalation` — interrupt payload dict (risk details) while awaiting human review; `None` when idle
- `escalation_thread_id` — LangGraph thread ID for the paused supervisor graph; used to resume after escalation
- `regulations` — dict of section_id → verbatim eCFR text; loaded once at session start via `load_regulations()`
- `supervisor` — compiled LangGraph supervisor graph; built once at session start via `build_supervisor(graph, checkpointer)` and passed to all `run_full_analysis` calls to avoid recompilation on every button click

**Dashboard query bar** (`app.py`) — submitting a query calls `_query_graph()` which passes all nodes, edges, audit log, and recent impact reports as context to the LLM and returns a plain-English answer. A "Run as change impact →" button hands the query text off to the Change Impact page.

**Graph Explorer** (`app.py`) — renders via raw vis.js HTML in `st.components.v1.html`. Same-level edges use `curvedCW` smooth type to prevent visual overlap; cross-level edges use `cubicBezier` with `forceDirection: vertical`. Double-clicking a node opens a detail panel showing full description, type, and status.

**Document Extract SME review** (`app.py`) — after extraction, "Request SME review" groups extracted nodes by team via `SME_NOTIFICATION_MAP`, generates an LLM briefing per team via `_extraction_team_briefing()`, and shows per-team approve cards. "Add to graph" is always available but shows a warning if reviews are pending.

## Key Constraints

- `app.py` inserts `src/` into `sys.path` — imports inside `src/` use bare module names (`from graph import ...`), not `from src.graph import ...`.
- `regulations.py` calls `load_regulations()` at module import time in `extractor.py`, `agent.py`, and `sme_agent.py` — verbatim eCFR text is fetched from `ecfr.gov` on first run and cached in `regulations_cache.json` (7-day TTL). If eCFR is unreachable, fallback stubs are used so the app still starts. `regulations_cache.json` is gitignored.
- `NodeType.PMA_SUPPLEMENT_TRIGGER = "PMA Supplement Trigger"` — the old `PCCP_TRIGGER` enum member has been renamed to match the display label everywhere. All downstream references (`agent.py`, `sme_agent.py`, `extractor.py`, `app.py` color/level maps) use the new name.
- The compiled supervisor (`st.session_state.supervisor`) is built once and passed as `supervisor=` to `run_full_analysis`. Do not call `build_supervisor` per-invocation; pass the cached instance instead.
- The Change Impact page includes an "Agent traversal trace" expander after the obligations table, showing the exact node path, edge types, and classification rule for each impacted node. The page renders downstream and upstream impacted nodes in separate dataframes.
- The RTMGraph is NOT passed through LangGraph state. `build_impact_agent(graph)` captures the graph in closures; `AgentState` contains only serializable primitives. This is required for checkpointing and interrupt/resume.
- The graph is never mutated by the agent — only `app.py` (via human approval UI) calls `update_node_status()`. `report.approved` is set to `True` immediately before `st.rerun()` in the approval handler.
- `extractor.py` truncates input text to 8,000 characters before sending to the LLM.
- `extractor.py` runs cycle detection before adding any extracted edge: `if graph.has_path(edge.target_id, edge.source_id): continue`. This prevents extracted edges from closing cycles in the DAG.
- LLM provider: OpenAI (`langchain_openai.ChatOpenAI`). API key: `OPENAI_API_KEY`. Default model: `gpt-4o-mini`.
- Streamlit Cloud deployment: `app.py` reads `st.secrets["OPENAI_API_KEY"]` as fallback when env var is not set.
- Edge direction convention: all edges point from "cause" to "effect" — the artifact that, when changed, forces work on the artifact it points to. Risk control edges: `H → RC (TRIGGERS)` and `RC → DI/DO (LINKED_TO)`. V&V chain: `DO → VP (LINKED_TO)` and `VP → TR (LINKED_TO)` — not the reverse. The `TR → DI (VALIDATES)` feedback edge goes upstream (Test Result validates the Design Input), which is why `missing_vv_links()` checks Design Inputs for incoming `validates` edges. This ensures all change-impact traversal is a simple BFS/DFS downstream.
- The `MemorySaver` in `st.session_state.checkpointer` must be passed to `run_full_analysis` on every call. Creating a new `MemorySaver` on resume will lose the saved thread state and break the interrupt/resume flow.
- Dashboard metric card c4 is **Graph Explorer** (`:material/hub:` icon, navigates to `graph_explorer`). The fourth card is not a duplicate of the Document Extract card.
- Tests live in `tests/`. Run with `python3 -m pytest tests/` from the project root. `tests/conftest.py` sets CWD to project root and inserts `src/` into sys.path so bare imports work. Five test files (117 tests total):
  - `test_graph.py` (40 tests — seed shape, traversal, completeness metrics, remove_node/remove_edge, save/load roundtrip, cycle detection)
  - `test_agent.py` (14 tests — LLM mocked via `unittest.mock.patch`)
  - `test_supervisor.py` (12 tests — all `_compute_risk_level` branches)
  - `test_extractor.py` (25 tests — `_parse_response`, `_build_nodes/edges`, `_find_existing_match` all three matching strategies, `_deduplicate_against_graph`, `add_to_graph` cycle rejection and confidence threshold)
  - `test_sme_agent.py` (26 tests — team routing per node type, `map_to_teams` Send fan-out, `brief_team_node` LLM and fallback, `finalize_notifications_node` join, `notifications_from_dicts`, `SME_NOTIFICATION_MAP` completeness)
