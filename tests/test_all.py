"""
test_all.py — Consolidated unit tests for the RTM Knowledge Graph Agent.

Sections:
  1. RTMGraph (graph.py)          — 40 tests, fully deterministic
  2. Change Impact agent (agent.py) — 14 tests, LLM mocked
  3. Supervisor risk routing       — 12 tests, structural ceiling fully deterministic
  4. Document extractor (extractor.py) — 25 tests, LLM mocked
  5. SME Router (sme_agent.py)     — 26 tests, LLM mocked where needed
  6. Regulatory grounding (regulations.py) — pinned snapshot, device-class pathway
  7. Regulatory update check (regulatory_refresh.py) — fetch mocked, reviewer-gated apply
  8. FDA inspection readiness (inspection.py) — CP 7382.850 element mapping
  9. PRD MVP — authority tags, standards metadata, claim check, device definition
"""

import json
import re
import pytest
from unittest.mock import patch, MagicMock

from graph import RTMGraph, NodeType, NodeStatus, EdgeType, build_seed_graph
from agent import build_impact_agent, AgentState
from supervisor import (
    _structural_risk_ceiling,
    _risk_level_for_change,
    _assess_risk,
    NON_SUBSTANTIVE_CHANGE_TYPES,
    DEFAULT_CHANGE_TYPE,
)
from extractor import (
    RTMDocumentExtractor,
    ExtractionResult,
    ExtractedNode,
    ExtractedEdge,
    _find_existing_match,
    _deduplicate_against_graph,
    _infer_required_test_results,
    _infer_missing_chain_links,
    _filter_skip_level_edges,
    _is_planned_artifact,
    _is_in_review_artifact,
)
from sme_agent import (
    SMEState,
    SMENotification,
    SME_NOTIFICATION_MAP,
    map_teams_node,
    map_to_teams,
    brief_team_node,
    finalize_notifications_node,
    build_sme_agent,
    notifications_from_dicts,
    team_system_prompt,
)
from regulations import (
    CORE_QMSR,
    DEFAULT_DEVICE_CLASS,
    DEVICE_CLASSES,
    PATHWAY_GROUNDING,
    TEAM_GROUNDING,
    TOPIC_GROUNDING,
    build_prompt_context,
    grounding_status,
    load_regulations,
    pathway_context,
    sections_for_query,
)
from regulatory_refresh import (
    SourceTextError,
    apply_regulatory_updates,
    check_regulatory_updates,
    diff_sections,
    excerpt,
)
from inspection import AREAS, ELEMENTS, assess_inspection_readiness, element_references, superseded_references


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def g():
    return build_seed_graph()


@pytest.fixture
def seed():
    return build_seed_graph()


@pytest.fixture
def extractor():
    return RTMDocumentExtractor()


@pytest.fixture
def empty_graph():
    return RTMGraph()


# ===========================================================================
# 1. RTMGraph — graph.py
# ===========================================================================

# Seed graph shape

def test_seed_graph_node_count(g):
    assert len(g.all_nodes()) == 15


def test_seed_graph_edge_count(g):
    assert len(g.all_edges()) == 15


def test_seed_graph_node_ids(g):
    ids = {n["id"] for n in g.all_nodes()}
    expected = {
        "UN-001", "UN-002",
        "DI-001", "DI-002",
        "DO-001", "DO-002",
        "VP-001", "VP-002",
        "TR-001A", "TR-002A",
        "H-001", "H-002",
        "RC-001", "RC-002",
        "CAPA-018",
    }
    assert ids == expected


# Traversal

def test_downstream_from_di001_contains_do001(g):
    downstream = g.downstream_nodes("DI-001")
    assert "DO-001" in downstream


def test_downstream_from_di001_includes_full_regulatory_chain(g):
    downstream = set(g.downstream_nodes("DI-001"))
    assert "DO-001" in downstream
    assert "VP-001" in downstream
    assert "TR-001A" in downstream
    assert "CAPA-018" in downstream


def test_downstream_from_tr001a_includes_capa(g):
    downstream = set(g.downstream_nodes("TR-001A"))
    assert "CAPA-018" in downstream
    assert "DI-001" in downstream  # reached via VERIFIES back-edge


def test_downstream_from_terminal_node_is_empty(g):
    assert g.downstream_nodes("CAPA-018") == []


def test_upstream_from_di001_contains_un001(g):
    upstream = set(g.upstream_nodes("DI-001"))
    assert "UN-001" in upstream


def test_impact_path_di001_to_do001(g):
    path = g.impact_path("DI-001", "DO-001")
    assert path[0] == "DI-001"
    assert path[-1] == "DO-001"
    assert len(path) == 2


def test_impact_path_di001_to_capa018(g):
    path = g.impact_path("DI-001", "CAPA-018")
    assert path[0] == "DI-001"
    assert path[-1] == "CAPA-018"
    assert len(path) >= 4


def test_impact_path_tr001a_to_capa018(g):
    path = g.impact_path("TR-001A", "CAPA-018")
    assert path[0] == "TR-001A"
    assert path[-1] == "CAPA-018"


def test_impact_path_no_path_returns_empty(g):
    assert g.impact_path("CAPA-018", "UN-001") == []


# has_path

def test_has_path_positive(g):
    assert g.has_path("UN-001", "DI-001") is True
    assert g.has_path("DI-001", "DO-001") is True
    assert g.has_path("DI-001", "VP-001") is True
    assert g.has_path("DI-001", "CAPA-018") is True
    assert g.has_path("TR-001A", "CAPA-018") is True


def test_has_path_negative(g):
    assert g.has_path("CAPA-018", "UN-001") is False
    assert g.has_path("TR-001A", "VP-001") is True  # via VERIFIES feedback edge


def test_has_path_unknown_node(g):
    assert g.has_path("NONEXISTENT", "DI-001") is False


# Completeness metrics

def test_no_orphans_in_seed(g):
    assert g.orphaned_nodes() == []


def test_no_missing_vv_in_seed(g):
    assert g.missing_vv_links() == []


def test_no_unmet_user_needs_in_seed(g):
    assert g.unmet_user_needs() == []


def test_no_incomplete_design_inputs_in_seed(g):
    assert g.incomplete_design_inputs() == []


def test_completeness_score_below_100_due_to_pending_capa(g):
    score = g.completeness_score()
    assert score < 100.0


def test_completeness_score_drops_on_invalidation(g):
    baseline = g.completeness_score()
    g.update_node_status("VP-001", NodeStatus.INVALIDATED, reason="test")
    assert g.completeness_score() < baseline


def test_unmet_user_need_detected(g):
    g.add_node("UN-999", NodeType.USER_NEED, "Unmet test need")
    assert "UN-999" in g.unmet_user_needs()


def test_incomplete_design_input_detected(g):
    g.add_node("DI-999", NodeType.DESIGN_INPUT, "Dangling DI")
    g.add_edge("UN-001", "DI-999", EdgeType.SATISFIES)
    assert "DI-999" in g.incomplete_design_inputs()


# V&V closure (open-loop chain detection)

def _add_open_loop_chain(g):
    """UN-003 → DI-003 → DO-003 → VP-003 with no Test Result (open loop)."""
    g.add_node("UN-003", NodeType.USER_NEED, "Detect Sub-Femtomolar Troponin")
    g.add_node("DI-003", NodeType.DESIGN_INPUT, "Sensitivity Input Specification")
    g.add_node("DO-003", NodeType.DESIGN_OUTPUT, "Optimized Capture Antibody")
    g.add_node("VP-003", NodeType.VV_PROTOCOL, "Protocol to Verify New Sensitivity")
    g.add_edge("UN-003", "DI-003", EdgeType.LINKED_TO)
    g.add_edge("DI-003", "DO-003", EdgeType.LINKED_TO)
    g.add_edge("DO-003", "VP-003", EdgeType.LINKED_TO)


def test_seed_design_input_verified_only_when_test_result_complete(g):
    # DI-002 is verified by TR-002A (active/complete).
    assert g.is_verified("DI-002") is True
    # DI-001 is verified by TR-001A, but TR-001A is PENDING_REVIEW (Lot 3
    # non-conformance under investigation), so the loop is not yet closed.
    assert g.is_verified("DI-001") is False


def test_open_loop_design_input_not_verified(g):
    _add_open_loop_chain(g)
    assert g.is_verified("DI-003") is False


def test_is_verified_false_for_non_design_input(g):
    assert g.is_verified("UN-002") is False
    assert g.is_verified("VP-002") is False


def test_seed_user_need_validated_only_when_test_result_complete(g):
    # UN-002 chain reaches TR-002A (active) downstream.
    assert g.is_validated("UN-002") is True
    # UN-001 chain only reaches TR-001A (pending_review) downstream.
    assert g.is_validated("UN-001") is False


def test_open_loop_user_need_not_validated(g):
    _add_open_loop_chain(g)
    assert g.is_validated("UN-003") is False


def test_is_validated_false_for_non_user_need(g):
    assert g.is_validated("DI-002") is False


def test_pending_test_result_does_not_close_loop(g):
    # Setting TR-001A to active should close DI-001 / UN-001; pending should not.
    assert g.is_verified("DI-001") is False
    g.update_node_status("TR-001A", NodeStatus.ACTIVE, reason="test")
    assert g.is_verified("DI-001") is True
    assert g.is_validated("UN-001") is True
    g.update_node_status("TR-001A", NodeStatus.INVALIDATED, reason="test")
    assert g.is_verified("DI-001") is False


def test_seed_has_open_loops_from_pending_test_result(g):
    gap_ids = {entry["id"] for entry in g.chain_verification_gaps()}
    # DI-001 / UN-001 are open because TR-001A is pending_review.
    assert "DI-001" in gap_ids
    assert "UN-001" in gap_ids
    # DI-002 / UN-002 are closed because TR-002A is active.
    assert "DI-002" not in gap_ids
    assert "UN-002" not in gap_ids


def test_gap_issue_distinguishes_pending_from_missing(g):
    _add_open_loop_chain(g)
    by_id = {e["id"]: e["issue"] for e in g.chain_verification_gaps()}
    # DI-003 has no verifying Test Result at all.
    assert "no Test Result has a 'verifies' edge" in by_id["DI-003"]
    # DI-001's verifying Test Result exists but is not completed.
    assert "not in a completed status" in by_id["DI-001"]
    assert "TR-001A" in by_id["DI-001"]


def test_chain_verification_gaps_flags_open_loop(g):
    _add_open_loop_chain(g)
    gap_ids = {entry["id"] for entry in g.chain_verification_gaps()}
    assert "UN-003" in gap_ids
    assert "DI-003" in gap_ids
    # The chain's Design Output and V&V Protocol are not themselves loop-closure
    # artifacts, so they are not flagged.
    assert "DO-003" not in gap_ids
    assert "VP-003" not in gap_ids


def test_chain_verification_gap_closes_when_complete_test_result_added(g):
    _add_open_loop_chain(g)
    gap_ids = {e["id"] for e in g.chain_verification_gaps()}
    assert "UN-003" in gap_ids and "DI-003" in gap_ids
    # Close the loop with a completed (active) Test Result that verifies DI-003.
    g.add_node("TR-003A", NodeType.TEST_RESULT, "Sensitivity test result",
               status=NodeStatus.ACTIVE)
    g.add_edge("VP-003", "TR-003A", EdgeType.LINKED_TO)
    g.add_edge("TR-003A", "DI-003", EdgeType.VERIFIES)
    assert g.is_verified("DI-003") is True
    assert g.is_validated("UN-003") is True
    remaining = {e["id"] for e in g.chain_verification_gaps()}
    assert "UN-003" not in remaining and "DI-003" not in remaining


def test_required_placeholder_tr_does_not_close_loop(g):
    # A required Test Result placeholder (not_started) with a verifies edge must
    # NOT mark the Design Input as verified — the loop stays open and is flagged.
    _add_open_loop_chain(g)
    g.add_node("TR-003", NodeType.TEST_RESULT, "[Required] Test Result for VP-003",
               "To be defined by the team", status=NodeStatus.NOT_STARTED,
               metadata={"required": True})
    g.add_edge("VP-003", "TR-003", EdgeType.LINKED_TO)
    g.add_edge("TR-003", "DI-003", EdgeType.VERIFIES)
    assert g.is_verified("DI-003") is False
    assert g.is_validated("UN-003") is False
    gaps = {e["id"]: e["issue"] for e in g.chain_verification_gaps()}
    assert "DI-003" in gaps and "UN-003" in gaps
    # The gap explanation must name the not_started placeholder, not claim "missing".
    assert "not_started" in gaps["DI-003"]


# Audit log

def test_status_change_logged(g):
    before = len(g.audit_log())
    g.update_node_status("DI-001", NodeStatus.INVALIDATED, reason="unit test")
    log = g.audit_log()
    assert len(log) == before + 1
    last = log[-1]
    assert last["event_type"] == "status_changed"
    assert last["node_id"] == "DI-001"
    assert last["new_status"] == NodeStatus.INVALIDATED.value


def test_status_change_raises_on_unknown_node(g):
    with pytest.raises(KeyError):
        g.update_node_status("GHOST", NodeStatus.ACTIVE)


# Snapshots

def test_snapshot_returns_id(g):
    snap_id = g.snapshot()
    assert isinstance(snap_id, str)
    assert len(snap_id) > 0


# Cycle detection

def test_has_path_detects_would_be_cycle(g):
    # RC-001 → DI-001 exists; adding DI-001 → RC-001 would close a cycle
    assert g.has_path("RC-001", "DI-001") is True


# remove_node / remove_edge

def test_remove_node_reduces_count(g):
    before = len(g.all_nodes())
    g.remove_node("CAPA-018")
    assert len(g.all_nodes()) == before - 1
    assert all(n["id"] != "CAPA-018" for n in g.all_nodes())


def test_remove_node_also_removes_edges(g):
    # TR-001A → CAPA-018 exists; removing CAPA-018 must drop that edge
    removed = g.remove_node("CAPA-018")
    edge_targets = {e["target"] for e in g.all_edges()}
    assert "CAPA-018" not in edge_targets
    assert any(e["target"] == "CAPA-018" for e in removed)


def test_remove_node_unknown_raises(g):
    with pytest.raises(KeyError):
        g.remove_node("GHOST-999")


def test_remove_edge_reduces_count(g):
    before = len(g.all_edges())
    g.remove_edge("TR-001A", "CAPA-018")
    assert len(g.all_edges()) == before - 1


def test_remove_edge_unknown_raises(g):
    with pytest.raises(KeyError):
        g.remove_edge("CAPA-018", "TR-001A")  # reversed — does not exist


def test_add_edge_missing_node_raises(g):
    with pytest.raises(KeyError):
        g.add_edge("DI-001", "NONEXISTENT", EdgeType.LINKED_TO)


def test_get_node_unknown_raises(g):
    with pytest.raises(KeyError):
        g.get_node("GHOST-999")


# save / load roundtrip

def test_save_load_roundtrip(g, tmp_path):
    path = tmp_path / "rtm_test.json"
    g.save(path)
    g2 = RTMGraph.load(path)
    assert len(g2.all_nodes()) == len(g.all_nodes())
    assert len(g2.all_edges()) == len(g.all_edges())
    node = g2.get_node("DI-001")
    assert node["node_type"] == "Design Input"


# completeness_score edge cases

def test_completeness_score_empty_graph():
    empty = RTMGraph()
    assert empty.completeness_score() == 0.0


def test_completeness_score_multiple_penalties_additive(g):
    g.add_node("ORPHAN-999", NodeType.USER_NEED, "orphan node")
    g.update_node_status("VP-001", NodeStatus.INVALIDATED)
    score = g.completeness_score()
    assert score < 90.0


def test_completeness_score_fully_clean():
    clean = RTMGraph()
    clean.add_node("UN-1", NodeType.USER_NEED, "need", status=NodeStatus.ACTIVE)
    clean.add_node("DI-1", NodeType.DESIGN_INPUT, "di", status=NodeStatus.ACTIVE)
    clean.add_node("DO-1", NodeType.DESIGN_OUTPUT, "do", status=NodeStatus.ACTIVE)
    clean.add_node("VP-1", NodeType.VV_PROTOCOL, "vp", status=NodeStatus.ACTIVE)
    clean.add_node("TR-1", NodeType.TEST_RESULT, "tr", status=NodeStatus.ACTIVE)
    clean.add_edge("UN-1", "DI-1", EdgeType.SATISFIES)
    clean.add_edge("DI-1", "DO-1", EdgeType.SATISFIES)
    clean.add_edge("DO-1", "VP-1", EdgeType.VERIFIES)
    clean.add_edge("TR-1", "DI-1", EdgeType.VERIFIES)
    assert clean.completeness_score() == 100.0


# snapshot content

def test_snapshot_captures_state(g):
    snap_id = g.snapshot()
    snap = next(s for s in g._snapshots if s["snapshot_id"] == snap_id)
    assert len(snap["nodes"]) == len(g.all_nodes())
    assert len(snap["edges"]) == len(g.all_edges())


# Accepted cycle via VERIFIES back-edge (ISO 13485 §7.3.6 (QMSR §820.10) design decision)

def test_vp001_has_verifies_back_edge_to_di001(g):
    assert g.has_path("VP-001", "DI-001") is True
    assert g.has_path("DI-001", "VP-001") is True


# ===========================================================================
# 2. Change Impact agent — agent.py
# ===========================================================================

def _base_agent_state(node_id: str = "DI-001") -> AgentState:
    return {
        "changed_node_id": node_id,
        "change_description": "unit test change",
        "downstream_ids": [],
        "upstream_ids": [],
        "impacted_nodes": [],
        "vv_invalidations": [],
        "capa_triggers": [],
        "llm_summary": "",
    }


def _run_agent(node_id: str = "DI-001") -> dict:
    g = build_seed_graph()
    agent = build_impact_agent(g)
    mock_response = MagicMock()
    mock_response.content = "mocked compliance summary"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("agent.ChatOpenAI", return_value=mock_llm):
        return agent.invoke(_base_agent_state(node_id))


# Downstream traversal

def test_downstream_vv_flagged():
    result = _run_agent("DI-001")
    assert "VP-001" in result["vv_invalidations"]


def test_downstream_capa_flagged():
    result = _run_agent("DI-001")
    assert "CAPA-018" in result["capa_triggers"]


def test_downstream_nodes_have_direction():
    result = _run_agent("DI-001")
    downstream = [n for n in result["impacted_nodes"] if n.get("direction") == "downstream"]
    assert len(downstream) >= 4


def test_di001_downstream_includes_regulatory_chain():
    result = _run_agent("DI-001")
    downstream_ids = {n["node_id"] for n in result["impacted_nodes"] if n.get("direction") == "downstream"}
    assert {"DO-001", "VP-001", "TR-001A", "CAPA-018"}.issubset(downstream_ids)


# Upstream traversal

def test_upstream_predecessor_surfaced():
    result = _run_agent("DI-001")
    upstream = [n for n in result["impacted_nodes"] if n.get("direction") == "upstream"]
    assert len(upstream) > 0


def test_un001_is_upstream_of_di001():
    result = _run_agent("DI-001")
    upstream_ids = [n["node_id"] for n in result["impacted_nodes"] if n.get("direction") == "upstream"]
    assert "UN-001" in upstream_ids


def test_upstream_nodes_not_in_vv_invalidations():
    result = _run_agent("DI-001")
    assert "UN-001" not in result["vv_invalidations"]
    assert "UN-001" not in result["capa_triggers"]


def test_upstream_ids_populated():
    result = _run_agent("DI-001")
    assert "UN-001" in result["upstream_ids"]


# Terminal node

def test_terminal_node_no_downstream_flags():
    result = _run_agent("CAPA-018")
    assert result["vv_invalidations"] == []
    assert result["capa_triggers"] == []
    downstream = [n for n in result["impacted_nodes"] if n.get("direction") == "downstream"]
    assert downstream == []


# LLM summary (mocked)

def test_llm_summary_populated():
    result = _run_agent("DI-001")
    assert result["llm_summary"] == "mocked compliance summary"


# Regulatory citations in action strings

def test_vv_action_cites_qmsr():
    result = _run_agent("DI-001")
    vv_nodes = [n for n in result["impacted_nodes"] if n["node_id"] == "VP-001"]
    assert vv_nodes
    assert "ISO 13485 §7.3.6/§7.3.7 (QMSR §820.10)" in vv_nodes[0]["required_action"]


def test_upstream_action_cites_design_traceability_clause():
    result = _run_agent("DI-001")
    upstream = [n for n in result["impacted_nodes"] if n.get("direction") == "upstream"]
    assert upstream
    assert "ISO 13485 §7.3.2 (QMSR §820.10)" in upstream[0]["required_action"]


# ===========================================================================
# 3. Supervisor risk routing — supervisor.py
# ===========================================================================

def _impact(vv=None, capa=None, nodes=None):
    return {
        "vv_invalidations": vv or [],
        "capa_triggers": capa or [],
        "impacted_nodes": nodes or [],
    }


# "low" branch

def test_risk_low_on_empty_impact():
    assert _structural_risk_ceiling(_impact()) == "low"


def test_risk_low_with_only_design_output():
    assert _structural_risk_ceiling(_impact(
        nodes=[{"node_type": "Design Output", "node_id": "DO-001", "title": "spec"}]
    )) == "low"


def test_risk_low_with_only_test_result():
    assert _structural_risk_ceiling(_impact(
        nodes=[{"node_type": "Test Result", "node_id": "TR-001A", "title": "result"}]
    )) == "low"


# "critical" branch — V&V invalidations

def test_risk_critical_on_vv_invalidation():
    assert _structural_risk_ceiling(_impact(vv=["VP-001"])) == "critical"


def test_risk_critical_on_multiple_vv():
    assert _structural_risk_ceiling(_impact(vv=["VP-001", "VP-002"])) == "critical"


# "high" branch — CAPA triggers

def test_risk_high_on_capa_trigger():
    assert _structural_risk_ceiling(_impact(capa=["CAPA-018"])) == "high"


# "high" branch — Hazard / Risk Control nodes

def test_risk_high_on_hazard_node():
    assert _structural_risk_ceiling(_impact(
        nodes=[{"node_type": "Hazard", "node_id": "H-001", "title": "missed AMI"}]
    )) == "high"


def test_risk_high_on_risk_control_node():
    assert _structural_risk_ceiling(_impact(
        nodes=[{"node_type": "Risk Control", "node_id": "RC-001", "title": "RC"}]
    )) == "high"


# Priority: critical beats high

def test_critical_beats_high_when_both_present():
    result = _structural_risk_ceiling(_impact(
        vv=["VP-001"],
        capa=["CAPA-018"],
        nodes=[{"node_type": "Hazard", "node_id": "H-001", "title": "h"}],
    ))
    assert result == "critical"


# Deterministic risk level — change-type downgrade (_risk_level_for_change)

def test_substantive_change_keeps_critical_ceiling():
    assert _risk_level_for_change(
        "Substantive change", _impact(vv=["VP-001"])
    ) == "critical"


def test_substantive_change_keeps_high_ceiling():
    assert _risk_level_for_change(
        "Substantive change", _impact(capa=["CAPA-018"])
    ) == "high"


def test_default_change_type_keeps_ceiling():
    assert _risk_level_for_change(
        DEFAULT_CHANGE_TYPE, _impact(vv=["VP-001"])
    ) == "critical"


def test_documentation_only_downgrades_critical_to_low():
    assert _risk_level_for_change(
        "Documentation only", _impact(vv=["VP-001"])
    ) == "low"


def test_documentation_only_downgrades_high_to_low():
    assert _risk_level_for_change(
        "Documentation only", _impact(capa=["CAPA-018"])
    ) == "low"


def test_change_types_are_substantive_or_documentation_only():
    from supervisor import CHANGE_TYPES
    assert CHANGE_TYPES == ["Substantive change", "Documentation only"]


def test_non_substantive_set_membership():
    # Guard: documentation-only is the one downgrade type; the default is substantive.
    assert NON_SUBSTANTIVE_CHANGE_TYPES == {"Documentation only"}
    assert DEFAULT_CHANGE_TYPE not in NON_SUBSTANTIVE_CHANGE_TYPES


def test_non_substantive_on_low_ceiling_stays_low():
    # Downgrade is a no-op when the ceiling is already low.
    assert _risk_level_for_change("Documentation only", _impact()) == "low"


# ===========================================================================
# 4. Document extractor — extractor.py
# ===========================================================================

def _make_extracted_node(
    suggested_id="NEW-001",
    node_type=NodeType.DESIGN_INPUT,
    title="New input spec",
    description="Some spec description",
    confidence=0.9,
    source_text="quote",
) -> ExtractedNode:
    return ExtractedNode(
        suggested_id=suggested_id,
        node_type=node_type,
        title=title,
        description=description,
        confidence=confidence,
        source_text=source_text,
    )


def _make_extracted_edge(
    source_id="NEW-001",
    target_id="DO-001",
    edge_type=EdgeType.SATISFIES,
    confidence=0.85,
    rationale="test",
) -> ExtractedEdge:
    return ExtractedEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        confidence=confidence,
        rationale=rationale,
    )


# _parse_response

def test_parse_response_valid_json(extractor):
    raw = '{"nodes": [], "edges": []}'
    result = extractor._parse_response(raw)
    assert result == {"nodes": [], "edges": []}


def test_parse_response_strips_markdown_fence(extractor):
    raw = "```json\n{\"nodes\": [], \"edges\": []}\n```"
    result = extractor._parse_response(raw)
    assert result == {"nodes": [], "edges": []}


def test_parse_response_strips_bare_fence(extractor):
    raw = "```\n{\"nodes\": [], \"edges\": []}\n```"
    result = extractor._parse_response(raw)
    assert result == {"nodes": [], "edges": []}


def test_parse_response_invalid_json_raises(extractor):
    with pytest.raises(json.JSONDecodeError):
        extractor._parse_response("not json at all")


# _build_nodes

def test_build_nodes_valid(extractor):
    raw = [{
        "suggested_id": "DI-999",
        "node_type": "Design Input",
        "title": "Test spec",
        "description": "desc",
        "confidence": 0.9,
        "source_text": "quote",
    }]
    nodes = extractor._build_nodes(raw)
    assert len(nodes) == 1
    assert nodes[0].suggested_id == "DI-999"
    assert nodes[0].node_type == NodeType.DESIGN_INPUT
    assert nodes[0].confidence == 0.9


def test_build_nodes_unknown_type_falls_back_to_user_need(extractor):
    raw = [{"suggested_id": "X-001", "node_type": "Completely Unknown", "title": "t",
            "description": "", "confidence": 0.5, "source_text": ""}]
    nodes = extractor._build_nodes(raw)
    assert nodes[0].node_type == NodeType.USER_NEED


def test_build_nodes_missing_fields_use_defaults(extractor):
    nodes = extractor._build_nodes([{}])
    assert len(nodes) == 1
    assert nodes[0].title == "Untitled"
    assert nodes[0].confidence == 0.5


# _build_edges

def test_build_edges_valid(extractor):
    # "satisfies" is in _BLOCKED_EDGE_TYPES and gets coerced to linked_to
    raw = [{"source_id": "DI-001", "target_id": "DO-001", "edge_type": "satisfies",
            "confidence": 0.8, "rationale": "because"}]
    edges = extractor._build_edges(raw)
    assert len(edges) == 1
    assert edges[0].edge_type == EdgeType.LINKED_TO


def test_build_edges_unknown_type_falls_back_to_linked_to(extractor):
    raw = [{"source_id": "A", "target_id": "B", "edge_type": "INVENTED_TYPE",
            "confidence": 0.5, "rationale": ""}]
    edges = extractor._build_edges(raw)
    assert edges[0].edge_type == EdgeType.LINKED_TO


# _find_existing_match

def test_find_existing_match_exact_id(seed):
    existing = {n["id"]: n for n in seed.all_nodes()}
    node = _make_extracted_node(suggested_id="DI-001", node_type=NodeType.DESIGN_INPUT,
                                title="Some other title")
    result = _find_existing_match(node, existing)
    assert result == "DI-001"


def test_find_existing_match_exact_id_wrong_type_not_matched(seed):
    existing = {n["id"]: n for n in seed.all_nodes()}
    node = _make_extracted_node(suggested_id="DI-001", node_type=NodeType.VV_PROTOCOL,
                                title="Some other title")
    result = _find_existing_match(node, existing)
    assert result != "DI-001" or result is None


def test_find_existing_match_title_word_overlap(seed):
    existing = {n["id"]: n for n in seed.all_nodes()}
    node = _make_extracted_node(suggested_id="DI-NEW",
                                node_type=NodeType.DESIGN_INPUT,
                                title="Analytical Sensitivity Specification")
    result = _find_existing_match(node, existing)
    assert result == "DI-001"


def test_find_existing_match_numeric_token_overlap(seed):
    existing = {n["id"]: n for n in seed.all_nodes()}
    node = _make_extracted_node(suggested_id="DI-NEW",
                                node_type=NodeType.DESIGN_INPUT,
                                title="Turnaround 18 minutes requirement",
                                description="TAT shall not exceed 18 minutes")
    result = _find_existing_match(node, existing)
    assert result == "DI-002"


def test_find_existing_match_no_match_returns_none(seed):
    existing = {n["id"]: n for n in seed.all_nodes()}
    node = _make_extracted_node(suggested_id="UN-999",
                                node_type=NodeType.USER_NEED,
                                title="Completely novel unrelated requirement",
                                description="nothing similar")
    result = _find_existing_match(node, existing)
    assert result is None


# _deduplicate_against_graph

def test_deduplicate_removes_duplicate_node(seed):
    node = _make_extracted_node(suggested_id="DI-001",
                                node_type=NodeType.DESIGN_INPUT,
                                title="Analytical Sensitivity — LoD ≤ 2.0 pg/mL")
    nodes, edges = _deduplicate_against_graph([node], [], seed)
    assert len(nodes) == 0


def test_deduplicate_keeps_novel_node(seed):
    node = _make_extracted_node(suggested_id="DI-999",
                                node_type=NodeType.DESIGN_INPUT,
                                title="Completely novel spec with no overlap",
                                description="no match possible")
    nodes, edges = _deduplicate_against_graph([node], [], seed)
    assert len(nodes) == 1


def test_deduplicate_remaps_edge_source_to_existing_id(seed):
    dup_node = _make_extracted_node(suggested_id="DI-001",
                                    node_type=NodeType.DESIGN_INPUT,
                                    title="Analytical Sensitivity — LoD ≤ 2.0 pg/mL")
    edge = _make_extracted_edge(source_id="DI-001", target_id="DO-001")
    nodes, edges = _deduplicate_against_graph([dup_node], [edge], seed)
    assert len(nodes) == 0
    assert len(edges) == 1
    assert edges[0].source_id == "DI-001"


def test_deduplicate_drops_self_loop_edge(seed):
    dup_node = _make_extracted_node(suggested_id="DI-NEW",
                                    node_type=NodeType.DESIGN_INPUT,
                                    title="Analytical Sensitivity Specification")  # matches DI-001
    edge = _make_extracted_edge(source_id="DI-NEW", target_id="DI-001")  # self-loop after remap
    nodes, edges = _deduplicate_against_graph([dup_node], [edge], seed)
    assert len(edges) == 0


# add_to_graph — confidence threshold

def test_add_to_graph_high_confidence_node_is_active(seed):
    node = _make_extracted_node(suggested_id="UN-NEW", node_type=NodeType.USER_NEED,
                                title="New clinical need", confidence=0.9)
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[node],
        extracted_edges=[],
        raw_llm_response="{}",
    )
    RTMDocumentExtractor().add_to_graph(result, seed, confidence_threshold=0.75)
    added = seed.get_node("UN-NEW")
    assert added["status"] == NodeStatus.ACTIVE.value


def test_add_to_graph_low_confidence_node_is_pending(seed):
    node = _make_extracted_node(suggested_id="UN-LOW", node_type=NodeType.USER_NEED,
                                title="Low confidence need", confidence=0.5)
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[node],
        extracted_edges=[],
        raw_llm_response="{}",
    )
    RTMDocumentExtractor().add_to_graph(result, seed, confidence_threshold=0.75)
    added = seed.get_node("UN-LOW")
    assert added["status"] == NodeStatus.PENDING_REVIEW.value


# Planned-artifact detection

def test_is_planned_artifact_detects_future_work():
    di = _make_extracted_node(suggested_id="DI-003", node_type=NodeType.DESIGN_INPUT,
                              title="Sensitivity Input Specification for Neonates")
    di.source_text = "the engineers will need to write a new sensitivity input specification"
    vp = _make_extracted_node(suggested_id="VP-003", node_type=NodeType.VV_PROTOCOL,
                              title="Verification Protocol")
    vp.source_text = "The validation group will author a dedicated protocol to verify performance"
    assert _is_planned_artifact(di) is True
    assert _is_planned_artifact(vp) is True


def test_is_planned_artifact_false_for_existing_artifact():
    di = _make_extracted_node(suggested_id="DI-007", node_type=NodeType.DESIGN_INPUT,
                              title="LoD spec")
    di.source_text = "Design Input DI-007 specifies LoD <= 1.2 pg/mL across all lots"
    assert _is_planned_artifact(di) is False


def test_is_planned_artifact_false_for_user_need_even_when_future_language():
    # A raised clinical requirement exists immediately, even if downstream work is planned.
    un = _make_extracted_node(suggested_id="UN-003", node_type=NodeType.USER_NEED,
                              title="Neonatal detection")
    un.source_text = "the device must be capable of detecting... engineers will need to write a spec"
    assert _is_planned_artifact(un) is False


def test_build_nodes_marks_planned_node_required():
    ex = RTMDocumentExtractor()
    raw = [
        {"suggested_id": "UN-003", "node_type": "User Need", "title": "Need",
         "description": "d", "confidence": 1.0, "source_text": "the device must detect X"},
        {"suggested_id": "DI-003", "node_type": "Design Input", "title": "Spec",
         "description": "d", "confidence": 1.0,
         "source_text": "engineers will need to write a new sensitivity input specification"},
    ]
    nodes = {n.suggested_id: n for n in ex._build_nodes(raw)}
    assert nodes["UN-003"].is_required is False
    assert nodes["DI-003"].is_required is True


def test_build_nodes_honors_explicit_llm_planned_flag():
    ex = RTMDocumentExtractor()
    raw = [{"suggested_id": "DO-003", "node_type": "Design Output", "title": "Output",
            "description": "d", "confidence": 1.0, "source_text": "antibody formulation",
            "planned": True}]
    nodes = ex._build_nodes(raw)
    assert nodes[0].is_required is True


def test_inferred_design_output_inherits_planned_from_design_input():
    di = _make_extracted_node(suggested_id="DI-003", node_type=NodeType.DESIGN_INPUT,
                              title="Sensitivity Input Specification")
    di.is_required = True  # planned DI
    nodes, edges = _infer_missing_chain_links([di], [])
    do = next(n for n in nodes if n.node_type == NodeType.DESIGN_OUTPUT)
    assert do.is_required is True


def test_inferred_design_output_not_required_for_existing_design_input():
    di = _make_extracted_node(suggested_id="DI-007", node_type=NodeType.DESIGN_INPUT,
                              title="Existing spec")  # is_required defaults False
    nodes, edges = _infer_missing_chain_links([di], [])
    do = next(n for n in nodes if n.node_type == NodeType.DESIGN_OUTPUT)
    assert do.is_required is False


# Skip-level edge filter: a linked_to edge that jumps over an intermediate
# hierarchy level (e.g. Design Input → V&V Protocol, skipping Design Output) is
# dropped before chain inference rebuilds the proper step-by-step path.

def test_filter_skip_level_drops_design_input_to_vv_protocol():
    di = _make_extracted_node(suggested_id="DI-003", node_type=NodeType.DESIGN_INPUT,
                              title="Sensitivity spec")
    vp = _make_extracted_node(suggested_id="VP-003", node_type=NodeType.VV_PROTOCOL,
                              title="Verification protocol")
    edges = [ExtractedEdge("DI-003", "VP-003", EdgeType.LINKED_TO, 0.9, "x")]
    kept = _filter_skip_level_edges([di, vp], edges, None)
    assert kept == []


def test_filter_skip_level_keeps_adjacent_link():
    di = _make_extracted_node(suggested_id="DI-003", node_type=NodeType.DESIGN_INPUT,
                              title="Sensitivity spec")
    do = _make_extracted_node(suggested_id="DO-003", node_type=NodeType.DESIGN_OUTPUT,
                              title="Output artifact")
    edges = [ExtractedEdge("DI-003", "DO-003", EdgeType.LINKED_TO, 0.9, "x")]
    kept = _filter_skip_level_edges([di, do], edges, None)
    assert len(kept) == 1


def test_filter_skip_level_exempts_verifies_back_edge():
    # TR (level 4) → DI (level 1) is a verifies back-edge — never a skip-level drop.
    tr = _make_extracted_node(suggested_id="TR-003", node_type=NodeType.TEST_RESULT,
                              title="Result")
    di = _make_extracted_node(suggested_id="DI-003", node_type=NodeType.DESIGN_INPUT,
                              title="Spec")
    edges = [ExtractedEdge("TR-003", "DI-003", EdgeType.VERIFIES, 0.9, "x")]
    kept = _filter_skip_level_edges([tr, di], edges, None)
    assert len(kept) == 1


def test_filter_skip_level_inference_rebuilds_path_via_design_output():
    # After dropping DI → VP, chain inference creates the missing DO and the
    # DI → DO → VP path, so the protocol is reached through the proper intermediate.
    di = _make_extracted_node(suggested_id="DI-003", node_type=NodeType.DESIGN_INPUT,
                              title="Sensitivity spec")
    vp = _make_extracted_node(suggested_id="VP-003", node_type=NodeType.VV_PROTOCOL,
                              title="Verification protocol")
    edges = [ExtractedEdge("DI-003", "VP-003", EdgeType.LINKED_TO, 0.9, "x")]
    edges = _filter_skip_level_edges([di, vp], edges, None)
    nodes, edges = _infer_missing_chain_links([di, vp], edges)
    do = next(n for n in nodes if n.node_type == NodeType.DESIGN_OUTPUT)
    assert any(e.source_id == "DI-003" and e.target_id == do.suggested_id for e in edges)
    assert any(e.source_id == do.suggested_id and e.target_id == "VP-003" for e in edges)
    # No direct DI → VP edge survives.
    assert not any(e.source_id == "DI-003" and e.target_id == "VP-003" for e in edges)


# Required-title marking: the [Required] prefix is reserved for the inferred ghost
# Test Result (the one artifact the document never describes). LLM-extracted planned
# nodes and the chain-inferred Design Output keep their natural titles.

def test_required_prefix_reserved_for_inferred_test_result():
    di = _make_extracted_node(suggested_id="DI-003", node_type=NodeType.DESIGN_INPUT,
                              title="Minimum Detectable Concentration Specification")
    di.is_required = True  # described-but-planned: no [Required] prefix
    vp = _make_extracted_node(suggested_id="VP-003", node_type=NodeType.VV_PROTOCOL,
                              title="Verification Protocol for Sensitivity")
    vp.is_required = True
    edges = [
        ExtractedEdge("DI-003", "VP-003", EdgeType.LINKED_TO, 0.9, "x"),
    ]

    # DI → DO inference: the chain stub inherits is_required and uses a natural title.
    nodes, edges = _infer_missing_chain_links([di, vp], edges)
    do = next(n for n in nodes if n.node_type == NodeType.DESIGN_OUTPUT)
    assert do.is_required is True
    assert not do.title.startswith("[Inferred] ")

    # VP → TR inference: the ghost Test Result is the only [Required] node.
    nodes, edges = _infer_required_test_results(nodes, edges)
    tr = next(n for n in nodes if n.node_type == NodeType.TEST_RESULT)
    assert tr.title.startswith("[Required] ")

    # Described-but-planned nodes retain their natural titles.
    assert di.title == "Minimum Detectable Concentration Specification"
    assert vp.title == "Verification Protocol for Sensitivity"


# In-review (being revised) detection

def test_is_in_review_artifact_detects_revision_language():
    di = _make_extracted_node(suggested_id="DI-008", node_type=NodeType.DESIGN_INPUT,
                              title="Antibody capture spec")
    di.source_text = "the existing antibody capture specification is to be revised to tighten the LoD"
    assert _is_in_review_artifact(di) is True
    assert _is_planned_artifact(di) is False  # it exists — not planned


def test_in_review_artifact_lands_in_pending_review_not_active(seed):
    di = _make_extracted_node(suggested_id="DI-REV", node_type=NodeType.DESIGN_INPUT,
                              title="Spec under revision", confidence=1.0)
    di.source_text = "DI-REV is under revision pending the new sensitivity data"
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=RTMDocumentExtractor()._build_nodes([{
            "suggested_id": "DI-REV", "node_type": "Design Input", "title": "Spec under revision",
            "description": "", "confidence": 1.0,
            "source_text": "DI-REV is under revision pending the new sensitivity data",
        }]),
        extracted_edges=[],
        raw_llm_response="{}",
    )
    RTMDocumentExtractor().add_to_graph(result, seed, confidence_threshold=0.75)
    # Despite confidence 1.0, an in-review artifact is incomplete → PENDING_REVIEW.
    assert seed.get_node("DI-REV")["status"] == NodeStatus.PENDING_REVIEW.value


def test_planned_takes_precedence_over_in_review():
    # If both could match, planned (does not exist yet) wins → NOT_STARTED path.
    nodes = RTMDocumentExtractor()._build_nodes([{
        "suggested_id": "DO-099", "node_type": "Design Output", "title": "Output",
        "description": "", "confidence": 1.0,
        "source_text": "the team will author a new output spec to be reviewed next quarter",
    }])
    assert nodes[0].is_required is True
    assert nodes[0].is_in_review is False


# VP → TR required-placeholder inference

def test_infer_required_test_result_created_for_unverified_vp():
    vp = _make_extracted_node(suggested_id="VP-003", node_type=NodeType.VV_PROTOCOL,
                              title="Neonatal verification protocol")
    nodes, edges = _infer_required_test_results([vp], [])
    required = [n for n in nodes if n.is_required]
    assert len(required) == 1
    tr = required[0]
    assert tr.suggested_id == "TR-003"
    assert tr.node_type == NodeType.TEST_RESULT
    assert tr.description == "To be defined by the team"
    # a VP → TR linked_to edge wires the required placeholder into the chain
    assert any(e.source_id == "VP-003" and e.target_id == "TR-003" for e in edges)


def test_infer_required_adds_verifies_back_edge_to_design_input():
    di = _make_extracted_node(suggested_id="DI-003", node_type=NodeType.DESIGN_INPUT,
                              title="Sensitivity input neonatal")
    do = _make_extracted_node(suggested_id="DO-003", node_type=NodeType.DESIGN_OUTPUT,
                              title="Capture antibody neonatal")
    vp = _make_extracted_node(suggested_id="VP-003", node_type=NodeType.VV_PROTOCOL,
                              title="Neonatal verification protocol")
    chain_edges = [
        _make_extracted_edge(source_id="DI-003", target_id="DO-003", edge_type=EdgeType.LINKED_TO),
        _make_extracted_edge(source_id="DO-003", target_id="VP-003", edge_type=EdgeType.LINKED_TO),
    ]
    nodes, edges = _infer_required_test_results([di, do, vp], chain_edges)
    # required TR placeholder verifies the upstream Design Input → closes the loop
    assert any(
        e.source_id == "TR-003" and e.target_id == "DI-003"
        and e.edge_type == EdgeType.VERIFIES
        for e in edges
    )


def test_add_to_graph_keeps_verifies_back_edge_despite_cycle(seed):
    # DI-001 already reaches TR via DI-001→DO-001→VP-001→TR-001A, so a fresh
    # TR-NEW→DI-001 verifies edge closes a loop — but verifies is exempt.
    tr = _make_extracted_node(suggested_id="TR-NEW", node_type=NodeType.TEST_RESULT,
                              title="New result", confidence=0.9)
    verifies = _make_extracted_edge(source_id="TR-NEW", target_id="DI-001",
                                    edge_type=EdgeType.VERIFIES)
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[tr],
        extracted_edges=[verifies],
        raw_llm_response="{}",
    )
    _, edges_added = RTMDocumentExtractor().add_to_graph(result, seed)
    assert edges_added == 1
    assert seed.has_path("TR-NEW", "DI-001") is True


def test_infer_required_skips_vp_that_already_has_test_result():
    vp = _make_extracted_node(suggested_id="VP-009", node_type=NodeType.VV_PROTOCOL,
                              title="Protocol with result")
    tr = _make_extracted_node(suggested_id="TR-009", node_type=NodeType.TEST_RESULT,
                              title="Real result report")
    edge = _make_extracted_edge(source_id="VP-009", target_id="TR-009",
                                edge_type=EdgeType.LINKED_TO)
    nodes, edges = _infer_required_test_results([vp, tr], [edge])
    assert not any(n.is_required for n in nodes)


def test_add_to_graph_required_node_is_not_started(seed):
    required = _make_extracted_node(suggested_id="TR-099", node_type=NodeType.TEST_RESULT,
                                    title="[Required] Test Result", confidence=0.0)
    required.is_required = True
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[required],
        extracted_edges=[],
        raw_llm_response="{}",
    )
    RTMDocumentExtractor().add_to_graph(result, seed, confidence_threshold=0.75)
    added = seed.get_node("TR-099")
    assert added["status"] == NodeStatus.NOT_STARTED.value
    assert added["metadata"].get("required") is True


def test_add_to_graph_cycle_rejection(seed):
    cycle_edge = _make_extracted_edge(source_id="DI-001", target_id="RC-001",
                                      edge_type=EdgeType.LINKED_TO)
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[],
        extracted_edges=[cycle_edge],
        raw_llm_response="{}",
    )
    before = len(seed.all_edges())
    nodes_added, edges_added = RTMDocumentExtractor().add_to_graph(result, seed)
    assert edges_added == 0
    assert len(seed.all_edges()) == before


def test_add_to_graph_missing_node_edge_skipped(seed):
    bad_edge = _make_extracted_edge(source_id="DI-001", target_id="GHOST-999")
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[],
        extracted_edges=[bad_edge],
        raw_llm_response="{}",
    )
    before = len(seed.all_edges())
    _, edges_added = RTMDocumentExtractor().add_to_graph(result, seed)
    assert edges_added == 0
    assert len(seed.all_edges()) == before


def test_add_to_graph_returns_counts(seed):
    node = _make_extracted_node(suggested_id="UN-CNT", node_type=NodeType.USER_NEED,
                                title="Count test node", confidence=0.9)
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[node],
        extracted_edges=[],
        raw_llm_response="{}",
    )
    nodes_added, edges_added = RTMDocumentExtractor().add_to_graph(result, seed)
    assert nodes_added == 1
    assert edges_added == 0


# Audit log

def test_audit_log_populated_after_extract(seed):
    mock_response = MagicMock()
    mock_response.content = '{"nodes": [], "edges": []}'
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("extractor.ChatOpenAI", return_value=mock_llm):
        RTMDocumentExtractor().extract("some document text", document_name="test_doc.txt", graph=seed)

    # The extractor instance is local; verify via the returned result's fields
    ext = RTMDocumentExtractor()
    with patch("extractor.ChatOpenAI", return_value=mock_llm):
        ext.extract("some document text", document_name="test_doc.txt", graph=seed)
    log = ext.audit_log()
    assert len(log) == 1
    assert log[0]["document_name"] == "test_doc.txt"
    assert "nodes_extracted" in log[0]
    assert "edges_extracted" in log[0]


def test_extract_llm_error_returns_empty_result(seed):
    with patch("extractor.ChatOpenAI", side_effect=Exception("API error")):
        result = RTMDocumentExtractor().extract("text", document_name="fail.txt", graph=seed)

    assert result.extracted_nodes == []
    assert result.extracted_edges == []
    assert "Extraction error" in result.raw_llm_response


# ===========================================================================
# 5. SME Router — sme_agent.py
# ===========================================================================

def _sme_node(node_id: str, node_type: str, title: str = "title") -> dict:
    return {
        "node_id": node_id,
        "node_type": node_type,
        "title": title,
        "current_status": "active",
        "edge_path": [],
        "edge_types_on_path": [],
        "required_action": "review",
        "direction": "downstream",
    }


def _run_map_teams(nodes: list[dict]) -> SMEState:
    state: SMEState = {
        "impacted_nodes": nodes,
        "sme_notifications": [],
        "team_briefings": {},
    }
    return map_teams_node(state)


# map_teams_node — routing per node type

def test_vv_protocol_routes_to_bioinformatics_and_rd():
    result = _run_map_teams([_sme_node("VP-001", NodeType.VV_PROTOCOL.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Bioinformatics" in teams
    assert "R&D" in teams


def test_capa_routes_to_quality_ra():
    result = _run_map_teams([_sme_node("CAPA-018", NodeType.CAPA.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Quality/RA" in teams
    assert len(teams) == 1


def test_hazard_routes_to_quality_ra():
    result = _run_map_teams([_sme_node("H-001", NodeType.HAZARD.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Quality/RA" in teams


def test_risk_control_routes_to_pathology():
    result = _run_map_teams([_sme_node("RC-001", NodeType.RISK_CONTROL.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Pathology" in teams


def test_test_result_routes_to_bioinformatics():
    result = _run_map_teams([_sme_node("TR-001A", NodeType.TEST_RESULT.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Bioinformatics" in teams


def test_design_input_routes_to_rd():
    result = _run_map_teams([_sme_node("DI-001", NodeType.DESIGN_INPUT.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "R&D" in teams


def test_design_output_routes_to_rd():
    result = _run_map_teams([_sme_node("DO-001", NodeType.DESIGN_OUTPUT.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "R&D" in teams


def test_unmapped_type_produces_no_notification():
    result = _run_map_teams([_sme_node("UN-001", NodeType.USER_NEED.value)])
    assert result["sme_notifications"] == []


def test_empty_input_produces_no_notifications():
    result = _run_map_teams([])
    assert result["sme_notifications"] == []


def test_multiple_nodes_accumulate_all_notifications():
    nodes = [
        _sme_node("VP-001", NodeType.VV_PROTOCOL.value),
        _sme_node("CAPA-018", NodeType.CAPA.value),
        _sme_node("DI-001", NodeType.DESIGN_INPUT.value),
    ]
    result = _run_map_teams(nodes)
    # VP-001 → 2 teams; CAPA-018 → 1; DI-001 → 1 = 4 total
    assert len(result["sme_notifications"]) == 4


def test_notification_carries_trigger_node_id():
    result = _run_map_teams([_sme_node("VP-001", NodeType.VV_PROTOCOL.value, title="VP title")])
    notifs = result["sme_notifications"]
    assert all(n["trigger_node_id"] == "VP-001" for n in notifs)
    assert all(n["trigger_node_title"] == "VP title" for n in notifs)


def test_notification_review_obligation_is_nonempty():
    result = _run_map_teams([_sme_node("CAPA-018", NodeType.CAPA.value)])
    for notif in result["sme_notifications"]:
        assert len(notif["review_obligation"]) > 10


# map_to_teams — Send fan-out

def test_map_to_teams_returns_one_send_per_unique_team():
    state = _run_map_teams([_sme_node("VP-001", NodeType.VV_PROTOCOL.value)])
    sends = map_to_teams(state)
    assert len(sends) == 2
    assert all(s.node == "brief_team" for s in sends)
    send_teams = {s.arg["team"] for s in sends}
    assert "Bioinformatics" in send_teams
    assert "R&D" in send_teams


def test_map_to_teams_deduplicates_teams_across_multiple_nodes():
    state = _run_map_teams([
        _sme_node("VP-001", NodeType.VV_PROTOCOL.value),
        _sme_node("VP-002", NodeType.VV_PROTOCOL.value),
    ])
    sends = map_to_teams(state)
    send_teams = [s.arg["team"] for s in sends]
    assert send_teams.count("Bioinformatics") == 1
    assert send_teams.count("R&D") == 1


def test_map_to_teams_empty_notifications_returns_empty():
    state = _run_map_teams([])
    assert map_to_teams(state) == []


def test_map_to_teams_send_carries_correct_notifications():
    state = _run_map_teams([_sme_node("CAPA-018", NodeType.CAPA.value)])
    sends = map_to_teams(state)
    assert len(sends) == 1
    payload = sends[0].arg
    assert payload["team"] == "Quality/RA"
    assert len(payload["team_notifications"]) == 1
    assert payload["team_notifications"][0]["trigger_node_id"] == "CAPA-018"


# brief_team_node — mocked LLM

def _brief_team_state(team: str, node_id: str, node_type: str) -> SMEState:
    notif = {
        "team": team,
        "trigger_node_id": node_id,
        "trigger_node_type": node_type,
        "trigger_node_title": "test node",
        "review_obligation": "Review this.",
        "llm_briefing": "",
    }
    return {
        "impacted_nodes": [],
        "sme_notifications": [],
        "team_briefings": {},
        "team": team,
        "team_notifications": [notif],
    }


def test_brief_team_node_returns_briefing_for_team():
    state = _brief_team_state("Quality/RA", "CAPA-018", NodeType.CAPA.value)
    mock_response = MagicMock()
    mock_response.content = "mocked Quality/RA briefing"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("sme_agent.ChatOpenAI", return_value=mock_llm):
        result = brief_team_node(state)

    assert "Quality/RA" in result["team_briefings"]
    assert result["team_briefings"]["Quality/RA"] == "mocked Quality/RA briefing"


def test_brief_team_node_does_not_write_sme_notifications():
    state = _brief_team_state("R&D", "DI-001", NodeType.DESIGN_INPUT.value)
    mock_response = MagicMock()
    mock_response.content = "rd briefing"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("sme_agent.ChatOpenAI", return_value=mock_llm):
        result = brief_team_node(state)

    assert "sme_notifications" not in result


def test_brief_team_node_llm_error_uses_fallback():
    state = _brief_team_state("R&D", "DI-001", NodeType.DESIGN_INPUT.value)
    with patch("sme_agent.ChatOpenAI", side_effect=Exception("API down")):
        result = brief_team_node(state)

    assert "R&D" in result["team_briefings"]
    assert "LLM unavailable" in result["team_briefings"]["R&D"]


# finalize_notifications_node — pure join logic

def test_finalize_notifications_writes_briefings_into_notifications():
    notifications = [
        {"team": "Quality/RA", "trigger_node_id": "CAPA-018", "trigger_node_type": "CAPA",
         "trigger_node_title": "CAPA-018", "review_obligation": "Check.", "llm_briefing": ""},
        {"team": "R&D", "trigger_node_id": "DI-001", "trigger_node_type": "Design Input",
         "trigger_node_title": "DI-001", "review_obligation": "Study.", "llm_briefing": ""},
    ]
    state: SMEState = {
        "impacted_nodes": [],
        "sme_notifications": notifications,
        "team_briefings": {"Quality/RA": "QA briefing text", "R&D": "RD briefing text"},
        "team": "",
        "team_notifications": [],
    }
    result = finalize_notifications_node(state)
    updated = result["sme_notifications"]
    assert updated[0]["llm_briefing"] == "QA briefing text"
    assert updated[1]["llm_briefing"] == "RD briefing text"


def test_finalize_notifications_missing_team_briefing_leaves_empty_string():
    notifications = [
        {"team": "Pathology", "trigger_node_id": "RC-001", "trigger_node_type": "Risk Control",
         "trigger_node_title": "RC-001", "review_obligation": "Assess.", "llm_briefing": ""},
    ]
    state: SMEState = {
        "impacted_nodes": [],
        "sme_notifications": notifications,
        "team_briefings": {},
        "team": "",
        "team_notifications": [],
    }
    result = finalize_notifications_node(state)
    assert result["sme_notifications"][0]["llm_briefing"] == ""


# notifications_from_dicts

def test_notifications_from_dicts_roundtrip():
    raw = [
        {
            "team": "Quality/RA",
            "trigger_node_id": "CAPA-018",
            "trigger_node_type": NodeType.CAPA.value,
            "trigger_node_title": "CAPA-018",
            "review_obligation": "Review CAPA scope per ISO 13485 §8.5.2 (QMSR §820.10)",
            "llm_briefing": "Briefing text here.",
        }
    ]
    result = notifications_from_dicts(raw)
    assert len(result) == 1
    assert isinstance(result[0], SMENotification)
    assert result[0].team == "Quality/RA"
    assert result[0].llm_briefing == "Briefing text here."


def test_notifications_from_dicts_missing_llm_briefing_defaults_to_empty():
    raw = [
        {
            "team": "R&D",
            "trigger_node_id": "DI-001",
            "trigger_node_type": NodeType.DESIGN_INPUT.value,
            "trigger_node_title": "DI-001",
            "review_obligation": "Review studies.",
        }
    ]
    result = notifications_from_dicts(raw)
    assert result[0].llm_briefing == ""


def test_notifications_from_dicts_empty_input():
    assert notifications_from_dicts([]) == []


# SME_NOTIFICATION_MAP completeness

def test_sme_map_covers_all_non_root_node_types():
    expected_mapped = {
        NodeType.VV_PROTOCOL.value,
        NodeType.TEST_RESULT.value,
        NodeType.HAZARD.value,
        NodeType.RISK_CONTROL.value,
        NodeType.CAPA.value,
        NodeType.DESIGN_INPUT.value,
        NodeType.DESIGN_OUTPUT.value,
    }
    assert expected_mapped == set(SME_NOTIFICATION_MAP.keys())


# ===========================================================================
# 6. Regulatory grounding — regulations.py
# ===========================================================================

def test_snapshot_loads_without_fallback():
    assert grounding_status()["using_fallback"] is False
    regs = load_regulations()
    assert "ISO 13485" in regs["820.10"]


def test_snapshot_holds_current_qmsr_not_superseded_qs_reg():
    regs = load_regulations()
    # §820.10(c) is the QMSR design-and-development applicability clause.
    assert "Clause 7.3" in regs["820.10"]
    for superseded in ("820.30", "820.40", "820.100", "820.180"):
        assert superseded not in regs


def test_every_grounding_reference_exists_in_snapshot():
    regs = load_regulations()
    referenced = set(CORE_QMSR)
    for ids in TEAM_GROUNDING.values():
        referenced |= set(ids)
    for ids in PATHWAY_GROUNDING.values():
        referenced |= set(ids)
    assert referenced - set(regs) == set()


def test_snapshot_sources_record_provenance():
    for source in grounding_status()["sources"].values():
        assert source["url"].startswith("https://")
        assert source["version_date"]
        assert len(source["content_sha256"]) == 64
        assert source["retrieved_at"]


def test_missing_snapshot_falls_back_to_stubs(tmp_path):
    regs = load_regulations(tmp_path / "missing.json")
    assert "820.10" in regs
    assert grounding_status(tmp_path / "missing.json")["using_fallback"] is True


def test_build_prompt_context_skips_unknown_and_dedupes():
    regs = load_regulations()
    ctx = build_prompt_context(regs, ["820.10", "820.10", "no-such-section"])
    assert ctx.count("[820.10]") == 1
    assert "no-such-section" not in ctx


def test_pathway_grounding_covers_every_device_class():
    assert set(PATHWAY_GROUNDING) == set(DEVICE_CLASSES)
    assert DEFAULT_DEVICE_CLASS in DEVICE_CLASSES


def test_pathway_context_class_ii_uses_510k_change_sources():
    ctx = pathway_context(load_regulations(), "Class II")
    assert "Device classification: Class II" in ctx
    assert "[807.81]" in ctx and "[510k-change:ivd-risk-assessment]" in ctx


@pytest.mark.parametrize("device_class", ["Class I", "Class III"])
def test_pathway_context_unsupported_class_borrows_nothing(device_class):
    ctx = pathway_context(load_regulations(), device_class)
    assert f"Device classification: {device_class}" in ctx
    assert "Class II 510(k) pathway only" in ctx
    assert "[807.81]" not in ctx and "Regulatory Affairs" in ctx


def test_library_is_scoped_to_the_class_ii_device():
    regs = load_regulations()
    assert DEVICE_CLASSES == ["Class II"] and DEFAULT_DEVICE_CLASS == "Class II"
    assert "862.9" not in regs and "814.39" not in regs


def test_quality_ra_prompt_includes_pathway_other_teams_do_not():
    assert "[807.81]" in team_system_prompt("Quality/RA", "Class II")
    assert "[807.81]" not in team_system_prompt("Bioinformatics", "Class II")


def test_map_to_teams_send_carries_device_class():
    state = _run_map_teams([_sme_node("CAPA-018", NodeType.CAPA.value)])
    state["device_class"] = "Class II"
    sends = map_to_teams(state)
    assert sends[0].arg["device_class"] == "Class II"


def test_brief_team_node_grounds_quality_ra_in_device_class_pathway():
    state = _brief_team_state("Quality/RA", "CAPA-018", NodeType.CAPA.value)
    state["device_class"] = "Class II"
    mock_response = MagicMock()
    mock_response.content = "briefing"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    with patch("sme_agent.ChatOpenAI", return_value=mock_llm):
        brief_team_node(state)
    system_message = mock_llm.invoke.call_args[0][0][0]
    assert "[510k-change:documentation]" in system_message.content


@pytest.mark.parametrize("device_class", ["Class II", "Class III"])
def test_device_class_never_changes_risk_level(device_class):
    impact = {"vv_invalidations": ["VP-001"], "capa_triggers": [], "impacted_nodes": []}
    with patch("supervisor.ChatOpenAI", side_effect=Exception("offline")):
        assessment = _assess_risk("Substantive change", "tighten LoD", impact, device_class)
    assert assessment.risk_level == "critical"


# ===========================================================================
# 7. Regulatory update check — regulatory_refresh.py (network mocked)
# ===========================================================================

def _entry(text: str, label: str = "label") -> dict:
    return {"label": label, "source": "ecfr:21-820", "citation": "21 CFR x", "text": text}


@pytest.fixture
def snapshot_copy(tmp_path):
    from regulations import SNAPSHOT_PATH
    path = tmp_path / "snapshot.json"
    path.write_text(SNAPSHOT_PATH.read_text())
    return path


def _live(snapshot_path, prefix: str, overrides: dict[str, str] | None = None, content_changed: bool = False):
    """Fake fetcher returning the pinned sections/sources under a source-ID prefix."""
    snap = json.loads(snapshot_path.read_text())
    overrides = overrides or {}
    sections = {
        sid: {**entry, "text": overrides.get(sid, entry["text"])}
        for sid, entry in snap["sections"].items() if entry["source"].startswith(prefix)
    }
    sources = {
        sid: {**src, "content_sha256": "0" * 64 if content_changed else src["content_sha256"]}
        for sid, src in snap["sources"].items() if sid.startswith(prefix)
    }
    return lambda: (sections, sources)


def _device_ok():
    return {"status": "match"}


def test_diff_sections_reports_only_changed_and_new():
    pinned = {"a": _entry("(a) same."), "b": _entry("(a) old text. (b) kept.")}
    fetched = {
        "a": _entry("(a) same."),
        "b": _entry("(a) new text. (b) kept."),
        "c": _entry("(a) brand new."),
    }
    changes = {c.section_id: c for c in diff_sections(pinned, fetched)}
    assert set(changes) == {"b", "c"}
    assert changes["b"].status == "changed"
    assert "-(a) old text." in changes["b"].diff
    assert "+(a) new text." in changes["b"].diff
    assert " (b) kept." in changes["b"].diff  # unchanged paragraph shown as context
    assert changes["c"].status == "new"


def test_excerpt_missing_anchor_raises():
    with pytest.raises(SourceTextError):
        excerpt("some regulatory text", "absent start", "text", source="test")


def test_check_up_to_date(snapshot_copy):
    fetchers = {"eCFR": _live(snapshot_copy, "ecfr:"), "CP": _live(snapshot_copy, "fda:cp-7382.850")}
    check = check_regulatory_updates(snapshot_copy, fetchers, device_verifier=_device_ok)
    assert check.up_to_date
    assert check.checked == ["eCFR", "CP"]
    assert check.errors == {}


def test_check_detects_section_change_without_writing(snapshot_copy):
    before = snapshot_copy.read_text()
    fetchers = {"eCFR": _live(snapshot_copy, "ecfr:", {"820.35": "§ 820.35 Control of records. (a) Amended."})}
    check = check_regulatory_updates(snapshot_copy, fetchers, device_verifier=_device_ok)
    assert [c.section_id for c in check.changes] == ["820.35"]
    assert snapshot_copy.read_text() == before


def test_check_flags_source_change_when_excerpts_unchanged(snapshot_copy):
    fetchers = {"FAQ": _live(snapshot_copy, "fda:qmsr-faq", content_changed=True)}
    check = check_regulatory_updates(snapshot_copy, fetchers, device_verifier=_device_ok)
    assert check.changes == []
    assert [s.source_id for s in check.source_changes] == ["fda:qmsr-faq"]
    assert not check.up_to_date


def test_failing_source_is_reported_without_blocking_others(snapshot_copy):
    def broken():
        raise SourceTextError("[fda-qmsr:inspections] start anchor not found")
    fetchers = {"Page": broken, "eCFR": _live(snapshot_copy, "ecfr:")}
    check = check_regulatory_updates(snapshot_copy, fetchers, device_verifier=_device_ok)
    assert "anchor not found" in check.errors["Page"]
    assert check.checked == ["eCFR"]
    assert check.up_to_date


def test_apply_requires_reviewer(snapshot_copy):
    fetchers = {"eCFR": _live(snapshot_copy, "ecfr:", {"820.35": "§ 820.35 amended."})}
    check = check_regulatory_updates(snapshot_copy, fetchers, device_verifier=_device_ok)
    with pytest.raises(ValueError):
        apply_regulatory_updates(check, "  ", snapshot_copy)


def test_apply_writes_only_changed_sections_with_provenance(snapshot_copy):
    before = json.loads(snapshot_copy.read_text())
    fetchers = {
        "eCFR": _live(snapshot_copy, "ecfr:", {"820.35": "§ 820.35 amended."}),
        "FAQ": _live(snapshot_copy, "fda:qmsr-faq", content_changed=True),
    }
    check = check_regulatory_updates(snapshot_copy, fetchers, device_verifier=_device_ok)
    applied = apply_regulatory_updates(check, "J. Reviewer", snapshot_copy)

    after = json.loads(snapshot_copy.read_text())
    assert applied == ["820.35"]
    assert after["sections"]["820.35"]["text"] == "§ 820.35 amended."
    assert after["sections"]["820.10"] == before["sections"]["820.10"]
    # Sources that were not fetched (Federal Register) are untouched.
    assert after["sources"]["fr:89-FR-7496"] == before["sources"]["fr:89-FR-7496"]
    assert after["sources"]["fda:qmsr-faq"]["content_sha256"] == "0" * 64
    assert after["last_update"]["reviewer"] == "J. Reviewer"
    assert after["last_update"]["sections"] == ["820.35"]
    assert after["last_update"]["sources"] == ["fda:qmsr-faq"]
    assert load_regulations(snapshot_copy)["820.35"] == "§ 820.35 amended."


# ===========================================================================
# 8. FDA inspection readiness — inspection.py (CP 7382.850)
# ===========================================================================

def _assessment(results, element):
    return next(a for a in results if a.element == element)


def test_inspection_elements_match_cp_7382_850_attachment_a():
    # Element names and clause lists must be copied verbatim from the pinned CP text.
    regs = load_regulations()
    area_excerpt = {
        "Change Control": regs["cp7382850:change-control"],
        "Design and Development": regs["cp7382850:design-development"],
        "Measurement, Analysis, and Improvement": regs["cp7382850:measurement-analysis-improvement"],
    }
    for area, element, requirements, _check, _basis in ELEMENTS:
        assert f"{element} {requirements}" in area_excerpt[area], (area, element)


def test_inspection_seed_graph_flags_open_loops_and_open_capa(g):
    results = assess_inspection_readiness(g)
    verification = _assessment(results, "Design and Development Verification")
    validation = _assessment(results, "Design and Development Validation")
    capa = _assessment(results, "Corrective Action")
    assert verification.status == "gap" and any("DI-001" in f for f in verification.findings)
    assert validation.status == "gap" and any("UN-001" in f for f in validation.findings)
    assert capa.status == "gap" and any("CAPA-018" in f for f in capa.findings)
    assert _assessment(results, "Design and Development Inputs").status == "clear"


def test_inspection_unmodeled_elements_are_not_assessed(g):
    results = assess_inspection_readiness(g)
    assert _assessment(results, "Design and Development Transfer").status == "not_assessed"
    assert _assessment(results, "Purchasing Changes").status == "not_assessed"


def test_inspection_pending_impact_analysis_is_a_change_control_gap(g):
    report = MagicMock(changed_node_id="DI-001", risk_level="critical",
                       vv_invalidations=["VP-001"], capa_triggers=["CAPA-018"])
    results = assess_inspection_readiness(g, [report])
    changes = _assessment(results, "Product and Process Changes")
    assert changes.status == "gap"
    assert any("not been approved" in f for f in changes.findings)
    assert any("VP-001" in f for f in changes.findings)
    assert any("unapproved change" in f for f in _assessment(results, "Corrective Action").findings)


def test_inspection_no_pending_changes_is_clear(g):
    assert _assessment(assess_inspection_readiness(g), "Product and Process Changes").status == "clear"


def test_inspection_areas_cover_all_elements():
    assert {area for area, *_ in ELEMENTS} == set(AREAS)


def test_quality_ra_prompt_grounded_in_change_control_area():
    assert "[cp7382850:change-control]" in team_system_prompt("Quality/RA", "Class II")


def test_quality_ra_user_prompt_requires_pathway_and_inspection_element():
    state = _brief_team_state("Quality/RA", "CAPA-018", NodeType.CAPA.value)
    state["device_class"] = "Class II"
    mock_response = MagicMock(); mock_response.content = "briefing"
    mock_llm = MagicMock(); mock_llm.invoke.return_value = mock_response
    with patch("sme_agent.ChatOpenAI", return_value=mock_llm):
        brief_team_node(state)
    user_message = mock_llm.invoke.call_args[0][0][1].content
    assert "(Class II)" in user_message and "new 510(k)" in user_message
    assert "PMA" not in user_message
    assert "Regulatory Affairs review" not in user_message  # fallback only for unsupported classes
    assert "[cp7382850:change-control]" in user_message


def test_quality_ra_requirements_flag_unsupported_class():
    from sme_agent import quality_ra_requirements
    assert "Regulatory Affairs review" in quality_ra_requirements("Class III")


def test_other_teams_user_prompt_has_no_pathway_requirement():
    state = _brief_team_state("Bioinformatics", "TR-001A", NodeType.TEST_RESULT.value)
    mock_response = MagicMock(); mock_response.content = "briefing"
    mock_llm = MagicMock(); mock_llm.invoke.return_value = mock_response
    with patch("sme_agent.ChatOpenAI", return_value=mock_llm):
        brief_team_node(state)
    assert "Premarket submission" not in mock_llm.invoke.call_args[0][0][1].content


def test_every_element_regulatory_reference_is_pinned(g):
    regs = load_regulations()
    for a in assess_inspection_readiness(g):
        refs = element_references(a)
        assert refs[0] == "820.10"
        assert set(refs) <= set(regs), (a.element, set(refs) - set(regs))


def test_iso_read_only_link_comes_from_fda_faq():
    from regulations import ISO_13485_READ_ONLY_URL
    assert ISO_13485_READ_ONLY_URL in load_regulations()["fda-qmsr-faq:iso-access"]


def test_superseded_qs_regulation_is_pinned_and_labeled(g):
    from regulations import SNAPSHOT_PATH
    sections = json.loads(SNAPSHOT_PATH.read_text())["sections"]
    for a in assess_inspection_readiness(g):
        for sid in superseded_references(a):
            assert sections[sid]["status"] == "superseded"
            assert "superseded" in sections[sid]["label"].lower()
    assert sections["qsreg:820.30(f)"]["text"].startswith("(f) Design verification.")


def test_superseded_text_never_grounds_prompts_or_current_requirements(g):
    from regulations import INSPECTION_GROUNDING
    grounding = set(CORE_QMSR) | set(INSPECTION_GROUNDING)
    for ids in list(TEAM_GROUNDING.values()) + list(PATHWAY_GROUNDING.values()):
        grounding |= set(ids)
    for a in assess_inspection_readiness(g):
        grounding |= set(element_references(a))
    assert not any(sid.startswith("qsreg:") for sid in grounding)


def test_update_check_does_not_refetch_fixed_historical_sources():
    from regulatory_refresh import ALL_FETCHERS, FETCHERS, fetch_superseded_qs_regulation, fetch_fr_preamble
    assert fetch_superseded_qs_regulation not in FETCHERS.values()
    assert fetch_fr_preamble not in FETCHERS.values()
    assert fetch_superseded_qs_regulation in ALL_FETCHERS.values()


def test_every_assessed_element_has_fda_commentary_or_superseded_equivalent(g):
    for a in assess_inspection_readiness(g):
        if a.status != "not_assessed":
            assert superseded_references(a), a.element


# ===========================================================================
# 9. PRD MVP — authority tags, standards metadata, claim check, device definition
# ===========================================================================

from claim_check import check_output
from regulations import GROUNDING_INSTRUCTION, load_device_definition, load_standards
from regulatory_refresh import classify_source, verify_device_definition


def test_every_snapshot_section_has_authority_class_and_status():
    from regulations import SNAPSHOT_PATH
    snap = json.loads(SNAPSHOT_PATH.read_text())
    for sid, entry in snap["sections"].items():
        assert entry["authority_level"] in {"A1", "A4", "A5", "A6"}, sid
        assert entry["content_class"] == "P", sid
        assert entry["status"] in {"current", "superseded"}, sid
    for sid, src in snap["sources"].items():
        assert src["authority_level"] and src["content_class"] == "P", sid


def test_superseded_sections_are_tagged_superseded():
    from regulations import SNAPSHOT_PATH
    sections = json.loads(SNAPSHOT_PATH.read_text())["sections"]
    assert all(e["status"] == "superseded" for k, e in sections.items() if k.startswith("qsreg:"))
    assert all(e["status"] == "current" for k, e in sections.items() if not k.startswith("qsreg:"))


def test_unclassified_source_is_rejected():
    with pytest.raises(ValueError):
        classify_source("blog:someone")


def test_classification_regulation_862_1215_is_pinned():
    text = load_regulations()["862.1215"]
    assert "Class II" in text


def test_standards_registry_holds_metadata_only():
    standards = load_standards()
    assert {s["designation"] for s in standards} == {
        "ISO 13485:2016", "ISO 14971:2019", "CLSI EP17-A2", "CLSI EP05-A3", "CLSI EP07 (3rd Edition)"}
    for s in standards:
        assert s["content_class"] == "L"
        assert not {"text", "requirement_text", "clause_text"} & set(s)
        if s["authority_level"] == "A3":
            assert re.fullmatch(r"\d+-\d+", s["fda_recognition_number"])


def test_iso_13485_clause_labels_come_from_cp_7382_850():
    iso = next(s for s in load_standards() if s["designation"] == "ISO 13485:2016")
    labels = {c["clause"]: c["public_label"] for c in iso["clauses_referenced"]}
    for area, element, requirements, check, _basis in ELEMENTS:
        if check is None or requirements.count(",") or not requirements.startswith("Clause "):
            continue
        clause = requirements.removeprefix("Clause ")
        if clause in labels and labels[clause] == element:
            continue
        # Elements sharing a clause (e.g. 7.3.7) must still map to a CP element name.
        assert clause in labels, (element, clause)


def test_prompt_context_excludes_non_public_domain_text(monkeypatch):
    import regulations
    fake = {"sections": {
        "pd": {"label": "Public", "text": "public text", "content_class": "P", "authority_level": "A1", "status": "current"},
        "lic": {"label": "Licensed", "text": "licensed text", "content_class": "L", "authority_level": "A2", "status": "current"},
    }}
    monkeypatch.setattr(regulations, "_snapshot_cache", fake)
    ctx = build_prompt_context({"pd": "public text", "lic": "licensed text"}, ["pd", "lic"])
    assert "public text" in ctx and "(A1, current)" in ctx
    assert "licensed text" not in ctx
    monkeypatch.setattr(regulations, "_snapshot_cache", None)


def test_grounding_instruction_covers_authority_and_copyright():
    assert "FDA recommends" in GROUNDING_INSTRUCTION
    assert "CLSI" in GROUNDING_INSTRUCTION and "clause number" in GROUNDING_INSTRUCTION


@pytest.mark.parametrize("text, kind", [
    ("Design changes must be documented per 21 CFR 820.30(i) before implementation.", "superseded_citation"),
    ("FDA requires a new 510(k) under the 510(k) change guidance whenever the LoD changes.", "guidance_as_requirement"),
    ("The QMSR requires full revalidation of every lot.", "unsupported_claim"),
    ('ISO 13485 clause 7.3.6 states "the organization shall perform design and development verification in accordance with planned arrangements".', "possible_standard_text"),
    ("Per [820.99] records must be kept.", "unknown_citation"),
    ("Records must follow [qsreg:820.30(j)].", "superseded_citation"),
])
def test_claim_check_flags(text, kind):
    assert kind in {f.kind for f in check_output(text)}


@pytest.mark.parametrize("text", [
    "A new 510(k) is required under 21 CFR 807.81(a)(3) when a change could significantly affect safety or effectiveness.",
    "Design verification is required by ISO 13485 §7.3.6 (QMSR §820.10).",
    "Under the former QS regulation, 21 CFR 820.30(f) covered design verification.",
    "FDA recommends documenting the decision [510k-change:documentation].",
    "Our team must first review CAPA-018 and then re-run VP-001.",
])
def test_claim_check_passes_supported_or_non_regulatory_text(text):
    assert check_output(text) == []


def test_default_device_class_comes_from_verified_definition():
    assert load_device_definition()["product_code"] == "MMI"
    assert DEFAULT_DEVICE_CLASS == load_device_definition()["device_class"] == "Class II"


def test_verify_device_definition_match_mismatch_error():
    pinned = {"product_code": "MMI", "regulation_number": "862.1215", "device_class": "Class II"}
    same = lambda code: {"product_code": code, "regulation_number": "862.1215", "device_class": "Class II", "device_name": "x"}
    other = lambda code: {"product_code": code, "regulation_number": "862.1215", "device_class": "Class III", "device_name": "x"}
    def down(code):
        raise OSError("offline")
    assert verify_device_definition(pinned, same)["status"] == "match"
    mismatch = verify_device_definition(pinned, other)
    assert mismatch["status"] == "mismatch" and "device_class" in mismatch["differences"][0]
    assert verify_device_definition(pinned, down)["status"] == "error"


def test_update_check_reports_device_definition(snapshot_copy):
    check = check_regulatory_updates(snapshot_copy, {}, device_verifier=lambda: {"status": "mismatch", "differences": ["x"]})
    assert check.device_definition["status"] == "mismatch"


def test_every_standard_cited_by_seed_data_is_in_the_registry(g):
    # Guards against the seed citing a standard the library does not know (EP07 was missed once).
    designations = " ".join(s["designation"] for s in load_standards())
    cited = set()
    for n in g.all_nodes():
        text = f"{n['title']} {n.get('description', '')}"
        cited |= set(re.findall(r"CLSI (EP\d+)", text)) | set(re.findall(r"ISO (\d{4,5})", text))
    assert cited, "seed data should cite standards"
    for ref in cited:
        assert ref in designations, ref


# ---------------------------------------------------------------------------
# Section 10: Topic-based query grounding (sections_for_query)
# ---------------------------------------------------------------------------

def test_sections_for_query_always_includes_core_qmsr():
    for question in ["what is the status?", "explain the CAPA", "any 510(k) implications?"]:
        ids = sections_for_query(question)
        for sid in CORE_QMSR:
            assert sid in ids, f"CORE_QMSR id {sid!r} missing for {question!r}"


def test_sections_for_query_keyword_routing():
    # CAPA question → capa topic
    ids = sections_for_query("Is CAPA-018 closed?")
    for sid in TOPIC_GROUNDING["capa"]:
        assert sid in ids

    # 510(k) question → submission_trigger topic
    ids = sections_for_query("Does this change need a new 510(k)?")
    for sid in TOPIC_GROUNDING["submission_trigger"]:
        assert sid in ids

    # LoD question → analytical_performance topic
    ids = sections_for_query("What is the LoD requirement?")
    for sid in TOPIC_GROUNDING["analytical_performance"]:
        assert sid in ids


def test_sections_for_query_no_match_falls_back_to_design_and_inspection():
    # A vague question with no recognized keywords should still get useful grounding.
    ids = sections_for_query("tell me about this device")
    for sid in TOPIC_GROUNDING["design_controls"] + TOPIC_GROUNDING["inspection"]:
        assert sid in ids


def test_sections_for_query_node_type_supplements_keywords():
    # A node_type hint adds its topic even when the question text has no keywords.
    ids = sections_for_query("what should I know?", node_types=["Test Result"])
    for sid in TOPIC_GROUNDING["vv_verification"]:
        assert sid in ids


def test_sections_for_query_all_ids_exist_in_snapshot():
    # Every section ID returned must be present in the pinned snapshot.
    from regulations import _read_snapshot
    snapshot_ids = set(_read_snapshot().get("sections", {}).keys())
    for question in [
        "Is DI-001 verified?",
        "What does CAPA-018 require?",
        "Does this need a 510(k)?",
        "What is the LoD spec?",
        "Show inspection readiness",
        "Explain risk management",
        "Are records current?",
    ]:
        for sid in sections_for_query(question):
            assert sid in snapshot_ids, f"unknown section {sid!r} for {question!r}"
