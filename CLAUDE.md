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
2. `score_risk` — deterministic risk level (structural ceiling, downgraded to `"low"` for a non-substantive `change_type`) + LLM explanation only; sets `risk_level`, `risk_rationale`, `immediate_concerns` in state
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
    change_type: str = "Functional change",  # reviewer attestation; non-substantive → "low"
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
- `upstream_ids: list[str]` — all BFS ancestors (via `nx.ancestors`) at a strictly lower hierarchy level than the changed node
- `impacted_nodes: list[dict]` — all nodes with `direction` field: `"downstream"` or `"upstream"`

`ImpactedNode` carries a `direction: str = "downstream"` field. Upstream nodes surface for QMSR §820.30(b) bidirectional traceability review only — they never trigger `vv_invalidations` or `capa_triggers`.

`traverse_node` collects downstream nodes via `graph.downstream_nodes(changed_id)` (`nx.descendants`) filtered to nodes at or below the changed node's hierarchy level, and upstream nodes via `graph.upstream_nodes(changed_id)` (`nx.ancestors`) filtered to nodes at a strictly lower hierarchy level. This hierarchy filter prevents feedback edges (e.g. `VERIFIES` from a Test Result back to a Design Input) from misclassifying downstream nodes as upstream requirements. `classify_node` early-continues for upstream nodes, assigning a fixed §820.30(b) bidirectional review action instead of the regulatory classification tree.

**Risk level determination** (`src/supervisor.py`):
The risk level is decided **deterministically in code** — the LLM never controls it. Two auditable inputs:
1. `_structural_risk_ceiling(impact_result)` — deterministic upper bound from topology:
   - `"critical"`: any V&V invalidations
   - `"high"`: any CAPA triggers OR Hazard/Risk Control nodes in impact chain
   - `"low"`: everything else
2. `_risk_level_for_change(change_type, impact_result)` — applies the reviewer-attested change type: returns `"low"` if `change_type in NON_SUBSTANTIVE_CHANGE_TYPES` (`"Documentation only"`, `"No change"`), else the structural ceiling. A non-substantive change is `"low"` even when V&V nodes are topologically downstream. The decision is a pure function of (change_type, topology) — reproducible, no model opinion in the path.

`_assess_risk(change_type, change_description, impact_result)` calls `_risk_level_for_change` for the level, then makes one LLM call (`RiskExplanation` Pydantic model — `rationale`, `immediate_concerns` only, **no `risk_level` field**) to explain the already-decided level. The LLM is told the level is final and that it does not decide or change it; when the topology would allow higher but the change is non-substantive, the prompt asks it to explain why the structural triggers do not apply. `RiskAssessment` remains the internal container (`risk_level` + prose) returned by `_assess_risk`. **There is no post-LLM clamp** — the LLM never produces a level to clamp.

**Change type** (`src/supervisor.py`): `CHANGE_TYPES = ["Functional change", "Corrective / CAPA action", "Documentation only", "No change"]`; `NON_SUBSTANTIVE_CHANGE_TYPES = {"Documentation only", "No change"}`; `DEFAULT_CHANGE_TYPE = "Functional change"` (all exported). The app renders a "Change type" selectbox on the Change Impact page (left column, under the node selector) and passes the choice as `change_type=` to `run_full_analysis`. It flows through `SupervisorState.change_type` into `score_risk_node` and onto `ImpactReport.change_type` (dataclass default `"Functional change"`, so `agent.run_impact_analysis` standalone reports work unchanged). The report header shows a "Change type attested as **{type}**" caption. `change_type` is ignored on resume (state is already checkpointed).

**RTMGraph** (`src/graph.py`) wraps a `networkx.DiGraph`. `build_seed_graph()` loads the hs-cTnI immunoassay demo dataset (15 nodes, 15 edges).

Analytics helpers on `RTMGraph`:
- `orphaned_nodes()` — nodes with no edges
- `missing_vv_links()` — Design Inputs with no incoming `verifies` edge from a Test Result
- `unmet_user_needs()` — User Needs with no out-neighbors of type Design Input
- `incomplete_design_inputs()` — Design Inputs with no out-neighbors of type Design Output
- `is_verified(node_id)` — True if a Design Input has an incoming `verifies` edge from a Test Result (QMSR §820.30(f) loop closure)
- `is_validated(node_id)` — True if a User Need's downstream chain (`downstream_nodes`) contains at least one Test Result (QMSR §820.30(g) objective evidence)
- `chain_verification_gaps()` — open-loop chains: Design Inputs that fail `is_verified` and User Needs that fail `is_validated`. Returns `[{id, node_type, title, status, issue}]`. Injected into the dashboard chat context (`_query_graph`) AND rendered on the Audit page's readiness assessment as "Open Verification/Validation Loops" so status questions and the audit view flag chains with missing test results as not yet verified/validated. This is the status-aware counterpart to the structural `missing_vv_links()`: a `verifies` edge from a required placeholder (not_started) or pending TR satisfies `missing_vv_links()` but is still flagged here because the loop is not closed.
- `has_path(source, target)` — returns True if a directed path exists; used by `extractor.py` for cycle detection
- `completeness_score()` — 0–100 score combining structural topology completeness (40%) and average per-node status readiness (60%); readiness weights: active/approved = 1.0, pending_review/not_started = 0.5, invalidated = 0.0
- `completeness_breakdown()` — returns sub-scores feeding `completeness_score()`: `structural_score`, `readiness_score`, `structural_issues`, `ready_count`, `partial_count`, `blocked_count`, `total`

**Node types and hierarchy levels** (used by vis.js in Graph Explorer):
- Level 0: `User Need`, `Hazard` (parallel tracks)
- Level 1: `Design Input`, `Risk Control` (parallel tracks)
- Level 2: `Design Output`
- Level 3: `V&V Protocol`
- Level 4: `Test Result`
- Level 5: `CAPA`

**Seed graph dependency chain:**
```
Hazard → Risk Control (TRIGGERS) → Design Input/Output (LINKED_TO)   ← parallel risk track (no edge from User Need)
User Need → Design Input (LINKED_TO) → Design Output (LINKED_TO)
Design Output → V&V Protocol (LINKED_TO)
V&V Protocol → Test Result (LINKED_TO)
Test Result → CAPA (TRIGGERS)
Test Result → Design Input (VERIFIES)   ← feedback/verification edge
```
Edge direction rule: every edge points from the artifact that, when changed, forces updates in the artifact it points to. A change to a Design Output forces re-execution of its V&V protocol, so the edge is `DO→VP (LINKED_TO)`. Following this, the full change-impact chain from `DI-001` is: `DI-001 → DO-001 → VP-001 → TR-001A → CAPA-018`.

**SME Router** (`src/sme_agent.py`) — `SME_NOTIFICATION_MAP` is the single source of truth mapping `NodeType` values to (team, obligation) pairs. Four teams: Bioinformatics, R&D, Pathology, Quality/RA. Seven node types are mapped; User Need is intentionally excluded (root artifact, not a downstream obligation).

Graph: `map_teams → [Send × N teams] → brief_team (×N parallel) → finalize_notifications`

- `map_teams_node` — pure deterministic mapping; populates `sme_notifications` list and initializes `team_briefings: {}`
- `map_to_teams` — conditional edge function; groups `sme_notifications` by team, returns one `Send("brief_team", {"team": ..., "team_notifications": [...]})` per unique team
- `brief_team_node` — one LLM call per invocation using a team-specific system prompt from `TEAM_SYSTEM_PROMPTS`; returns `{"team_briefings": {team: briefing_text}}` only
- `finalize_notifications_node` — runs once after all parallel invocations complete; joins `team_briefings` into each notification's `llm_briefing` field

`SMEState.team_briefings` carries `Annotated[dict, _merge_dicts]` so LangGraph merges parallel writes from concurrent `brief_team` nodes rather than overwriting. `team` and `team_notifications` are Send-payload fields in `SMEState` consumed by `brief_team_node`.

Also imported by `app.py` for the Document Extract SME review path.

**ImpactReport** (`src/agent.py`) — includes `sme_notifications: list[SMENotification]` and `team_briefings: dict` populated by the supervisor. Risk and escalation fields: `risk_level`, `risk_rationale`, `immediate_concerns`, `escalation_required`, `escalation_reviewer`, `escalation_notes`. `approved: bool = False` — set to `True` only when a human explicitly approves the report through the UI; the graph is never updated otherwise.

**UI navigation** (`app.py`) — sidebar-driven; `st.session_state.current_page` controls which page renders. Five pages: `dashboard`, `change_impact`, `graph_explorer`, `doc_extract`, `audit`. The `audit` page combines RTM completeness assessment (top) and the event log (bottom). The sidebar footer shows three graph-stat metrics — **Nodes** (total), **Pending** (`pending_review` count), **Not Started** (`not_started` count) — plus a completeness-score caption.

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
- The compiled supervisor (`st.session_state.supervisor`) is built once and passed as `supervisor=` to `run_full_analysis`. Do not call `build_supervisor` per-invocation; pass the cached instance instead.
- The Change Impact page includes an "Agent traversal trace" expander after the obligations table, showing the exact node path, edge types, and classification rule for each impacted node. The page renders downstream and upstream impacted nodes in separate dataframes.
- The RTMGraph is NOT passed through LangGraph state. `build_impact_agent(graph)` captures the graph in closures; `AgentState` contains only serializable primitives. This is required for checkpointing and interrupt/resume.
- The graph is never mutated by the agent — only `app.py` (via human approval UI) calls `update_node_status()`. `report.approved` is set to `True` immediately before `st.rerun()` in the approval handler.
- `extractor.py` truncates input text to 8,000 characters before sending to the LLM.
- `extractor.py` runs cycle detection before adding any extracted edge: `if graph.has_path(edge.target_id, edge.source_id): continue`. This prevents extracted edges from closing cycles in the DAG.
- `extractor.py` post-processing pipeline (runs after LLM call, in order): (1) `_deduplicate_against_graph()` — removes extracted nodes that match existing graph nodes and remaps edge IDs; (2) `_filter_cross_chain_edges()` — drops edges that bridge a newly extracted node to an existing graph node (both endpoints must be all-new or all-existing); (3) `_infer_missing_chain_links()` — enforces DI→DO→VP chain completeness within an extraction, creating 0.70-confidence stub nodes for gaps; (4) `_infer_required_test_results()` — enforces VP→TR completeness: for any V&V Protocol with no extracted Test Result, creates a "required" TR placeholder (`is_required=True`, description "To be defined by the team") plus a `linked_to` edge `VP→TR`, then closes the V&V loop by tracing `VP ← DO ← DI` and adding a `verifies` back-edge `TR→DI` (QMSR §820.30(f)); (5) `_infer_stub_nodes()` — creates 0.60-confidence stubs for any edge endpoint the LLM referenced but did not include in the nodes array.
- Required placeholders (`ExtractedNode.is_required=True`) mark an artifact that does not exist yet — planned future work the team must produce, or an open V&V loop to close. `add_to_graph` lands them in `NodeStatus.NOT_STARTED` regardless of confidence and tags `metadata["required"]=True`. The Graph Explorer renders required placeholders with a faded fill + dashed border; the doc-extract review table shows their status as "Not Started". Three sources set `is_required`:
  1. `_infer_required_test_results()` — the inferred ghost Test Result for an open V&V loop.
  2. `_build_nodes()` — any LLM-extracted node the document describes as planned/future work. Driven by an explicit LLM `planned` boolean (added to the extraction schema/prompt) OR a `_is_planned_artifact()` regex fallback that detects forward-looking language ("will need to write", "will author", "to be developed") in the evidence quote. **User Needs are never marked planned** — a raised clinical requirement exists immediately, even though everything it drives downstream is still to be built.
  3. `_infer_missing_chain_links()` — an inferred Design Output stub inherits `is_required` from its parent Design Input (a planned DI yields a planned DO).
  Net effect for a document describing only future work (e.g. "engineers will write a spec, the lab will produce a result"): the root User Need is `active`, and the entire downstream DI→DO→VP→TR chain is `not_started`.
  The `[Required] ` title prefix is reserved for the inferred ghost Test Result alone — the one artifact the document never describes (`_infer_required_test_results()` sets it directly at creation). LLM-extracted planned nodes (DI/VP) keep their natural document titles, and the chain-inferred Design Output keeps its `[Inferred] ` marker; their planned status is conveyed by the `not_started` Status column, not a title bracket. (There is no separate title-marking pass — a node's `not_started`/`is_required` status, not its title, signals it is planned.)
- In-review artifacts (`ExtractedNode.is_in_review=True`) are existing-but-unfinished — described as being revised, updated, or under review ("to be revised", "under revision", "pending review"). `_build_nodes()` sets this via an LLM `in_review` boolean OR the `_is_in_review_artifact()` regex fallback. They land in `NodeStatus.PENDING_REVIEW` (they exist, so not `not_started`) regardless of confidence — never `active`. Both `not_started` and `pending_review` are incomplete and surface as gaps: only `active`/`approved` close a V&V loop. `is_required` takes precedence over `is_in_review` (a planned artifact does not exist yet). Distinct from the confidence-threshold path, which also yields `pending_review` for low-confidence extractions.
- `RTMDocumentExtractor._coerce_edge_type()` — sanitizes LLM edge types: coerces blocked types (`satisfies`, `mitigates`) to `linked_to`, and enforces prefix constraints for `verifies` (source must be `TR-`, target must be `DI-`) and `triggers` (source must be `HZ-` or `TR-`, target must be `RC-` or `CA-`).
- LLM provider: OpenAI (`langchain_openai.ChatOpenAI`). API key: `OPENAI_API_KEY`. Default model: `gpt-4o-mini`.
- Streamlit Cloud deployment: `app.py` reads `st.secrets["OPENAI_API_KEY"]` as fallback when env var is not set.
- Edge direction convention: all edges point from "cause" to "effect" — the artifact that, when changed, forces work on the artifact it points to. Risk control edges: `H → RC (TRIGGERS)` and `RC → DI/DO (LINKED_TO)`. V&V chain: `DO → VP (LINKED_TO)` and `VP → TR (LINKED_TO)` — not the reverse. The `TR → DI (VERIFIES)` feedback edge goes upstream (Test Result verifies the Design Input), which is why `missing_vv_links()` checks Design Inputs for incoming `verifies` edges. This ensures all change-impact traversal is a simple BFS/DFS downstream.
- The `MemorySaver` in `st.session_state.checkpointer` must be passed to `run_full_analysis` on every call. Creating a new `MemorySaver` on resume will lose the saved thread state and break the interrupt/resume flow.
- Dashboard metric card c4 is **Graph Explorer** (`:material/hub:` icon, navigates to `graph_explorer`). The fourth card is not a duplicate of the Document Extract card.
- Tests live in `tests/`. Run with `python3 -m pytest tests/` from the project root. `tests/conftest.py` sets CWD to project root and inserts `src/` into sys.path so bare imports work. Single consolidated test file (`test_all.py`, 150 tests total):
  - Section 1: RTMGraph — 52 tests (seed shape, traversal, completeness metrics, V&V closure / open-loop chain detection including required-placeholder TR not closing the loop, remove_node/remove_edge, save/load roundtrip, cycle detection)
  - Section 2: Change Impact agent — 12 tests (LLM mocked via `unittest.mock.patch`)
  - Section 3: Supervisor risk routing — 16 tests (all `_structural_risk_ceiling` branches + `_risk_level_for_change` change-type downgrade: substantive types preserve the ceiling, `"Documentation only"`/`"No change"` downgrade to `"low"`, no-op on an already-low ceiling, `NON_SUBSTANTIVE_CHANGE_TYPES` membership)
  - Section 4: Document extractor — 45 tests (`_parse_response`, `_build_nodes/edges`, `_find_existing_match` all three matching strategies, `_deduplicate_against_graph`, `_is_planned_artifact` future-tense detection + `_build_nodes` planned→required marking + inferred-DO planned inheritance, `_is_in_review_artifact` revision detection → PENDING_REVIEW with planned-precedence, `[Required]` prefix reserved for the inferred ghost Test Result only (planned DI/VP keep natural titles, inferred DO keeps `[Inferred]`), `_infer_required_test_results` VP→TR placeholder creation + `verifies` loop-closure back-edge, `add_to_graph` cycle rejection with `verifies` exemption, confidence threshold, and required-placeholder NOT_STARTED status)
  - Section 5: SME Router — 25 tests (team routing per node type, `map_to_teams` Send fan-out, `brief_team_node` LLM and fallback, `finalize_notifications_node` join, `notifications_from_dicts`, `SME_NOTIFICATION_MAP` completeness)
