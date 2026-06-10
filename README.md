# RTM Knowledge Graph Agent
### hs-cTnI Immunoassay — PMA Change Impact Analysis via Multi-Agent LangGraph

A multi-agent LangGraph system that transforms a static Requirements Traceability Matrix (RTM) into a live dependency graph for an FDA Class III IVD device. When any compliance artifact changes, the agent pipeline traverses the full downstream chain, scores the risk, pauses for human review on critical changes, surfaces every obligation that needs action, and routes team-specific briefings to the right subject matter experts.

---

## What This Builds

FDA PMA device development requires bidirectional traceability across a chain like:

```
User Needs → Hazards → Risk Controls → Design Inputs → Design Outputs → V&V Protocols → Test Results → CAPA / PMA Supplement Triggers
```

Every edge points from the artifact that, when changed, forces work on the artifact it points to — so a change impact analysis is a straightforward graph traversal downstream. Upstream predecessors are also surfaced for QMSR §820.30(b) bidirectional traceability review.

Most teams manage this in spreadsheets. When a design input changes, someone has to manually trace every downstream obligation, assess the regulatory risk, and figure out who to notify. This project replaces that manual process with a multi-agent LLM pipeline that includes a guardrail: changes that invalidate V&V evidence or trigger PMA supplement review cannot proceed without explicit documented sign-off.

**Six core capabilities:**

1. **RTM Query Bar** — ask plain-English questions about any node, its history, dependencies, or compliance status directly from the dashboard; the LLM answers against the full live graph context
2. **Multi-Agent Change Impact** — select any RTM node, describe the change, and the supervisor runs: Change Impact Agent (traverse → classify → report) → risk scoring → escalation gate (if critical) → SME Router Agent (team-specific briefings)
3. **Critical-Risk Escalation Gate** — changes that invalidate V&V protocols or trigger PMA supplement review are automatically classified as critical; the pipeline pauses and requires a named reviewer to approve or reject before SME briefings are generated
4. **SME Outreach Flow** — each affected team receives a card with an LLM briefing in their domain vocabulary and an approve button; the human approval gate prevents any status update without documented sign-off
5. **Interactive Graph Explorer** — vis.js hierarchical dependency graph with double-click node detail panels, same-level edge curving to prevent overlap, and root-node subgraph filtering
6. **Audit Readiness Dashboard** — live completeness score, orphan detection, V&V gap report, and PMA supplement flag monitoring

---

## Setup (one API key, that's it)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/rtm-knowledge-graph-agent
cd rtm-knowledge-graph-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501`. No database, no Docker, no external services.

**Optional:** Override the model with `OPENAI_MODEL=gpt-4o` in `.env` (default: `gpt-4o-mini`).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Streamlit Dashboard                          │
│   Dashboard │ Change Impact │ Graph Explorer │ Extract │ Audit   │
└────────────────────────────┬─────────────────────────────────────┘
                             │ run_full_analysis(graph, node_id, desc,
                             │                  checkpointer, thread_id)
             ┌───────────────▼──────────────────────────┐
             │              Supervisor                   │
             │          (src/supervisor.py)              │
             │                                           │
             │  ① run_impact_agent                       │
             │  ┌──────────────────────────────────────┐ │
             │  │       Change Impact Agent            │ │
             │  │       (src/agent.py)                 │ │
             │  │  traverse → classify → report        │ │
             │  │  graph captured in closures          │ │
             │  │  [1 LLM call — compliance summary]   │ │
             │  └─────────────────┬────────────────────┘ │
             │                    │ impact_result         │
             │  ② score_risk      │                       │
             │    deterministic risk level               │
             │    + LLM rationale/concerns               │
             │                    │                       │
             │  ③ [conditional] ──┘                       │
             │    critical ──→ escalation_gate           │
             │                  LangGraph interrupt()    │
             │                  ↓ approved               │
             │    high/low ──→ ④ run_sme_agent           │
             │  ┌──────────────────────────────────────┐ │
             │  │        SME Router Agent              │ │
             │  │        (src/sme_agent.py)            │ │
             │  │  map_teams → Send × N teams          │ │
             │  │    → brief_team (parallel)           │ │
             │  │    → finalize_notifications          │ │
             │  │  [N LLM calls — run in parallel]     │ │
             │  └─────────────────┬────────────────────┘ │
             │                    │ sme_result            │
             │  ⑤ assemble_report │                       │
             └────────────────────┴──────────────────────┘
                             │ ImpactReport
                             ▼
             ┌────────────────────────────────────────────────┐
             │              RTMGraph (NetworkX)               │
             │   16 nodes · 18 edges (DAG; no cycles)         │
             │   Chain: UN→DI→DO→VP→TR→CAPA/PM               │
             │   Read via closure in agent.py                 │
             └────────────────────────────────────────────────┘
             ┌───────────────────────────────────┐
             │     LLM (OpenAI gpt-4o-mini)      │
             │  Compliance summaries             │
             │  Risk rationale (structured out.) │
             │  Team-specific briefings          │
             │  Document entity extraction       │
             │  Graph Q&A (dashboard)            │
             └───────────────────────────────────┘
```

| Layer | Technology | Role |
|-------|-----------|------|
| UI | Streamlit + vis.js (via st.components) | Dashboard, interactive graph visualization |
| Supervisor Agent | LangGraph | Sequences sub-agents, scores risk, owns escalation interrupt |
| Change Impact Agent | LangGraph (closure over RTMGraph) | traverse → classify → report; serializable state |
| SME Router Agent | LangGraph (`Send` map-reduce) | `map_teams → Send × N → brief_team` (parallel) `→ finalize_notifications`; LLM calls run concurrently |
| Risk Scoring | Deterministic + LLM structured output | Classifies risk level; LLM produces rationale only |
| Escalation Gate | LangGraph `interrupt()` / `Command(resume=...)` | Pauses critical-risk pipeline for human review |
| Regulatory Source | eCFR API (ecfr.gov) | Verbatim 21 CFR text fetched at startup; 7-day disk cache |
| Graph Engine | NetworkX | In-memory RTM dependency graph |
| LLM | OpenAI gpt-4o-mini | Compliance summaries, risk rationale, team briefings, doc extraction, Q&A |
| Persistence | MemorySaver (session) + JSON (export) | Interrupt/resume state; audit log export |

---

## Project Structure

```
rtm-knowledge-graph-agent/
├── app.py                    # Streamlit dashboard (entry point)
├── src/
│   ├── supervisor.py         # Top-level LangGraph orchestrator + escalation gate
│   ├── agent.py              # Change Impact sub-agent
│   ├── sme_agent.py          # SME Router sub-agent
│   ├── graph.py              # RTMGraph class, node/edge types, seed data
│   ├── extractor.py          # LLM document extraction module
│   └── regulations.py        # eCFR fetch + cache; injects verbatim CFR text into LLM prompts
├── tests/
│   ├── conftest.py           # pytest: sets CWD to project root, adds src/ to sys.path
│   ├── test_graph.py         # 40 tests: seed shape, traversal, completeness, remove/save/load, cycle detection
│   ├── test_agent.py         # 14 tests: Change Impact agent (LLM mocked via unittest.mock.patch)
│   ├── test_supervisor.py    # 12 tests: _compute_risk_level — all low/high/critical branches
│   ├── test_extractor.py     # 25 tests: parse, deduplication, cycle rejection, add_to_graph confidence threshold
│   └── test_sme_agent.py     # 26 tests: team routing, Send fan-out, brief_team_node, finalize_notifications
├── regulations_cache.json    # Auto-generated; verbatim eCFR sections, refreshed every 7 days
├── .streamlit/
│   └── config.toml           # Streamlit theme config
├── requirements.txt
├── .env.example
└── .gitignore
```

Run the test suite (no API key needed — all LLM calls are mocked):

```bash
python3 -m pytest tests/
```

---

## Key Design Decisions

**Why multi-agent?**
The change impact pipeline, the risk assessment, and the SME notification pipeline have different concerns and different LLM prompting strategies. Separating them into sub-agents makes each independently testable. The supervisor is the only place with awareness of the full pipeline — sub-agents do the work, the supervisor makes the decisions.

**Why does the supervisor own the escalation gate, not the impact agent?**
Escalation is a coordination decision, not a technical one. The supervisor has the full picture after the impact agent completes — only then can it assess whether the combination of flagged nodes rises to critical risk. The checkpointer (required for interrupt/resume) lives in the supervisor graph. Each sub-agent can be called independently without knowledge of escalation.

**Why is risk level deterministic, not LLM-driven?**
Routing decisions in a regulatory system should be auditable. V&V invalidations and PMA supplement flags have a clear, documented meaning under 21 CFR 814.39 and QMSR §820.30 — there is no ambiguity about whether they require escalation. The LLM's role is to explain the risk in plain language for the reviewer, not to make the routing call. This means: "why was this escalated?" always has a crisp answer traceable to a structural graph property.

**Why do sub-agents close over the graph instead of receiving it through state?**
LangGraph state is meant to be serializable (for checkpointing and interrupt/resume). A live NetworkX object cannot be serialized. Passing the graph through closures keeps `AgentState` as plain dicts/strings, which makes the interrupt/resume flow work correctly across Streamlit reruns.

**Human-in-the-loop gate (compliance status updates)**
The agent never updates compliance status autonomously. `ImpactReport.approved` is always `False` until a human approves it through the UI. This mirrors 21 CFR Part 11 requirements for documented human approval before any compliance record changes.

**Human-in-the-loop gate (escalation)**
For critical-risk changes, the pipeline uses LangGraph's `interrupt()` to actually pause mid-execution. The graph state is persisted in a `MemorySaver` stored in `st.session_state`. The reviewer's name, decision, and notes are recorded in the `ImpactReport`. Rejection routes the graph to `END` — no SME briefings are generated and nothing is stored. This is not a UI-layer flag check; the pipeline cannot proceed without a `Command(resume=...)` from the human.

**Confidence scoring on extraction**
Every LLM-extracted entity has a `confidence` score. Entities below the threshold (default 0.75) get `PENDING_REVIEW` status and are flagged in the UI. Humans decide what to accept.

**Upstream surfacing alongside downstream traversal**
The Change Impact agent collects both downstream descendants and immediate upstream predecessors of the changed node. Upstream nodes are tagged `direction="upstream"` and rendered in a separate table — they receive a fixed QMSR §820.30(b) bidirectional traceability action and are explicitly excluded from the V&V/CAPA/PMA regulatory flag lists. This matters because a change to a Design Input may also need to be verified against the User Need that drove it.

**Deterministic completeness scoring**
`RTMGraph.completeness_score()` penalizes six independently weighted categories: orphaned nodes, Design Outputs with no V&V protocol, User Needs with no satisfying Design Input, Design Inputs with no satisfying Design Output, any INVALIDATED node, and any PENDING_REVIEW on a V&V Protocol / CAPA / PMA Supplement Trigger. Each category is independently auditable — the score is not a black box.

**Why parallel SME briefings via `Send`?**
The original `generate_briefings_node` called each team's LLM sequentially — Bioinformatics, then R&D, then Pathology, then Quality/RA. These calls are completely independent. Replacing the loop with LangGraph's `Send` API fans out one `brief_team_node` per team in parallel. A `_merge_dicts` reducer on `team_briefings` in `SMEState` (annotated with `Annotated[dict, _merge_dicts]`) lets each parallel node write its one key without overwriting others. A final `finalize_notifications_node` runs once after all parallel invocations complete and joins the briefing strings back into each notification dict. The result is N LLM calls → 1 round-trip instead of N sequential round-trips, with no change to the supervisor's interface.

**Why 117 tests with LLM mocking?**
The regulatory routing logic (`_compute_risk_level`), graph analytics, upstream/downstream classification, extractor deduplication, and SME team routing are the highest-stakes code paths. They are also fully deterministic — no LLM call should control whether a change is escalated or not. Mocking `ChatOpenAI` lets these decision paths be tested without an API key and without flaky LLM responses. The extractor test suite covers the three deduplication matching strategies (`_find_existing_match`) and cycle rejection in `add_to_graph`. The SME test suite covers the `Send` fan-out logic, `brief_team_node` fallback behavior, and `finalize_notifications_node` join correctness independently.

---

## Seed Dataset

The app loads a representative RTM for a **high-sensitivity cardiac Troponin I (hs-cTnI) immunoassay** — a Class III IVD device under PMA P240052 — covering:

- 2 User Needs (AMI detection sensitivity, emergency TAT)
- 2 Hazards (false negative result — missed AMI; erroneous result — sample interference)
- 2 Design Inputs (LoD ≤ 2.0 pg/mL per CLSI EP17-A2; TAT ≤ 18 min)
- 2 Design Outputs (antibody pair spec v1.4; signal quantification algorithm v2.1)
- 2 V&V Protocols (VP-001: LoD/LoQ verification; VP-002: precision validation)
- 2 Test Results (VP-001: Lot 3 non-conformance at 2.6 pg/mL; VP-002: PASS)
- 2 Risk Controls (ISO 14971 false negative hazard mitigation; CLSI EP07 interference control)
- 1 CAPA (CAPA-018: Lot 3 LoD non-conformance)
- 1 PMA Supplement Trigger (PM-001: >20% LoD spec change triggers 21 CFR 814.39 review)

**Pre-loaded scenario:** Tightening LoD from ≤ 2.0 pg/mL to ≤ 1.2 pg/mL triggers a critical-risk impact chain: VP-001 re-execution required (V&V invalidation) + PM-001 flagged (PMA supplement trigger) → escalation gate fires → reviewer must approve before SME notifications go to all four teams.

---

## Regulatory Context

The following regulations are actively queried at startup via the [eCFR public API](https://www.ecfr.gov) (`ecfr.gov/api/versioner/v1`). Verbatim section text is injected into every LLM prompt at runtime and cached locally for 7 days (`regulations_cache.json`).

| Section | Source | Scope |
|---------|--------|-------|
| 21 CFR §820.30 | FDA QMSR (21 CFR Part 820) | Design controls — bidirectional traceability requirement |
| 21 CFR §820.40 | FDA QMSR (21 CFR Part 820) | Document controls |
| 21 CFR §820.100 | FDA QMSR (21 CFR Part 820) | Corrective and preventive action (CAPA) |
| 21 CFR §820.180 | FDA QMSR (21 CFR Part 820) | General records requirements |
| 21 CFR §814.39 | 21 CFR Part 814 | PMA supplement requirements |
| 42 CFR §493.1253 | CLIA (42 CFR Part 493) | Establishment and verification of performance specifications (LoD/LoQ) |
| 42 CFR §493.1255 | CLIA (42 CFR Part 493) | Calibration and calibration verification |
