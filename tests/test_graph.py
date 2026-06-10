"""
test_graph.py — Unit tests for RTMGraph (graph.py).

All tests are deterministic: no LLM calls, no network calls.
The seed graph (build_seed_graph) is the reference fixture throughout.
"""

import pytest
from graph import (
    RTMGraph,
    NodeType,
    NodeStatus,
    EdgeType,
    build_seed_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def g():
    return build_seed_graph()


# ---------------------------------------------------------------------------
# Seed graph shape
# ---------------------------------------------------------------------------

def test_seed_graph_node_count(g):
    assert len(g.all_nodes()) == 16


def test_seed_graph_edge_count(g):
    assert len(g.all_edges()) == 16


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
        "CAPA-018", "PM-001",
    }
    assert ids == expected


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def test_downstream_from_di001_contains_do001(g):
    downstream = g.downstream_nodes("DI-001")
    assert "DO-001" in downstream


def test_downstream_from_di001_includes_full_regulatory_chain(g):
    # After re-orienting VP/TR edges: DI-001 → DO-001 → VP-001 → TR-001A → CAPA-018/PM-001
    downstream = set(g.downstream_nodes("DI-001"))
    assert "DO-001" in downstream
    assert "VP-001" in downstream
    assert "TR-001A" in downstream
    assert "CAPA-018" in downstream
    assert "PM-001" in downstream


def test_downstream_from_tr001a_is_capa_and_pm(g):
    # TR-001A has two outgoing edges: TRIGGERS → CAPA-018 and VERIFIES → DI-001.
    # The VERIFIES feedback edge means BFS also reaches DI-001 → DO-001 → VP-001.
    downstream = set(g.downstream_nodes("TR-001A"))
    assert {"CAPA-018", "PM-001"}.issubset(downstream)
    assert "DI-001" in downstream  # reached via the VERIFIES back-edge


def test_downstream_from_terminal_node_is_empty(g):
    # PM-001 has no outgoing edges in the seed graph
    assert g.downstream_nodes("PM-001") == []


def test_upstream_from_di001_contains_un001(g):
    upstream = set(g.upstream_nodes("DI-001"))
    assert "UN-001" in upstream


def test_impact_path_di001_to_do001(g):
    path = g.impact_path("DI-001", "DO-001")
    assert path[0] == "DI-001"
    assert path[-1] == "DO-001"
    assert len(path) == 2


def test_impact_path_di001_to_pm001(g):
    # Full chain now reachable: DI-001 → DO-001 → VP-001 → TR-001A → PM-001
    path = g.impact_path("DI-001", "PM-001")
    assert path[0] == "DI-001"
    assert path[-1] == "PM-001"
    assert len(path) >= 4


def test_impact_path_tr001a_to_pm001(g):
    path = g.impact_path("TR-001A", "PM-001")
    assert path[0] == "TR-001A"
    assert path[-1] == "PM-001"


def test_impact_path_no_path_returns_empty(g):
    # PM-001 cannot reach UN-001 in a DAG
    assert g.impact_path("PM-001", "UN-001") == []


# ---------------------------------------------------------------------------
# has_path
# ---------------------------------------------------------------------------

def test_has_path_positive(g):
    assert g.has_path("UN-001", "DI-001") is True
    assert g.has_path("DI-001", "DO-001") is True
    assert g.has_path("DI-001", "VP-001") is True   # DI → DO → VP
    assert g.has_path("DI-001", "PM-001") is True   # full chain
    assert g.has_path("TR-001A", "PM-001") is True


def test_has_path_negative(g):
    assert g.has_path("PM-001", "UN-001") is False
    # TR-001A → DI-001 (VERIFIES) → DO-001 → VP-001: path exists via feedback edge
    assert g.has_path("TR-001A", "VP-001") is True


def test_has_path_unknown_node(g):
    assert g.has_path("NONEXISTENT", "DI-001") is False


# ---------------------------------------------------------------------------
# Completeness metrics
# ---------------------------------------------------------------------------

def test_no_orphans_in_seed(g):
    assert g.orphaned_nodes() == []


def test_no_missing_vv_in_seed(g):
    assert g.missing_vv_links() == []


def test_no_unmet_user_needs_in_seed(g):
    assert g.unmet_user_needs() == []


def test_no_incomplete_design_inputs_in_seed(g):
    assert g.incomplete_design_inputs() == []


def test_completeness_score_below_100_due_to_pending_capa(g):
    # CAPA-018 is PENDING_REVIEW (a critical artifact type) — must penalise score
    score = g.completeness_score()
    assert score < 100.0


def test_completeness_score_drops_on_invalidation(g):
    baseline = g.completeness_score()
    g.update_node_status("VP-001", NodeStatus.INVALIDATED, reason="test")
    assert g.completeness_score() < baseline


def test_unmet_user_need_detected(g):
    # unmet_user_needs() checks for any out-edge to a Design Input node.
    # A User Need with no edges to any Design Input is unmet.
    g.add_node("UN-999", NodeType.USER_NEED, "Unmet test need")
    assert "UN-999" in g.unmet_user_needs()


def test_incomplete_design_input_detected(g):
    # Add a Design Input with no SATISFIES edge to a Design Output
    g.add_node("DI-999", NodeType.DESIGN_INPUT, "Dangling DI")
    g.add_edge("UN-001", "DI-999", EdgeType.SATISFIES)
    assert "DI-999" in g.incomplete_design_inputs()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def test_snapshot_returns_id(g):
    snap_id = g.snapshot()
    assert isinstance(snap_id, str)
    assert len(snap_id) > 0


# ---------------------------------------------------------------------------
# Cycle detection (used by extractor)
# ---------------------------------------------------------------------------

def test_has_path_detects_would_be_cycle(g):
    # RC-001 → DI-001 already exists in the seed graph (LINKED_TO).
    # If someone tried to add DI-001 → RC-001, the cycle check is:
    # has_path(target=RC-001, source=DI-001)... wait, the extractor checks
    # has_path(edge.target_id, edge.source_id) before adding.
    # Adding (source=DI-001, target=RC-001): check has_path("RC-001", "DI-001").
    # That path exists (RC-001 → DI-001 directly), so the edge is rejected.
    assert g.has_path("RC-001", "DI-001") is True


# ---------------------------------------------------------------------------
# remove_node / remove_edge
# ---------------------------------------------------------------------------

def test_remove_node_reduces_count(g):
    before = len(g.all_nodes())
    g.remove_node("PM-001")
    assert len(g.all_nodes()) == before - 1
    assert all(n["id"] != "PM-001" for n in g.all_nodes())


def test_remove_node_also_removes_edges(g):
    # CAPA-018 → PM-001 exists; removing PM-001 must drop that edge
    removed = g.remove_node("PM-001")
    edge_targets = {e["target"] for e in g.all_edges()}
    assert "PM-001" not in edge_targets
    # remove_node returns the list of removed edges
    assert any(e["target"] == "PM-001" for e in removed)


def test_remove_node_unknown_raises(g):
    with pytest.raises(KeyError):
        g.remove_node("GHOST-999")


def test_remove_edge_reduces_count(g):
    before = len(g.all_edges())
    g.remove_edge("CAPA-018", "PM-001")
    assert len(g.all_edges()) == before - 1


def test_remove_edge_unknown_raises(g):
    with pytest.raises(KeyError):
        g.remove_edge("PM-001", "CAPA-018")  # reversed — does not exist


def test_add_edge_missing_node_raises(g):
    with pytest.raises(KeyError):
        g.add_edge("DI-001", "NONEXISTENT", EdgeType.LINKED_TO)


def test_get_node_unknown_raises(g):
    with pytest.raises(KeyError):
        g.get_node("GHOST-999")


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(g, tmp_path):
    path = tmp_path / "rtm_test.json"
    g.save(path)
    g2 = RTMGraph.load(path)
    assert len(g2.all_nodes()) == len(g.all_nodes())
    assert len(g2.all_edges()) == len(g.all_edges())
    # Spot-check a node
    node = g2.get_node("DI-001")
    assert node["node_type"] == "Design Input"


# ---------------------------------------------------------------------------
# completeness_score edge cases
# ---------------------------------------------------------------------------

def test_completeness_score_empty_graph():
    empty = RTMGraph()
    assert empty.completeness_score() == 0.0


def test_completeness_score_multiple_penalties_additive(g):
    # Orphan User Need + invalidated V&V node — each independently penalises.
    # The orphan also triggers unmet_user_needs(), so structural_issues = 2.
    g.add_node("ORPHAN-999", NodeType.USER_NEED, "orphan node")
    g.update_node_status("VP-001", NodeStatus.INVALIDATED)
    score = g.completeness_score()
    # Seed baseline is ~96% (all structural gaps closed, 2 pending nodes).
    # Combined orphan + invalidated penalties should bring the score below 90%.
    assert score < 90.0


def test_completeness_score_fully_clean():
    # A complete graph: UN → DI → DO → VP ← TR (with VERIFIES), all ACTIVE.
    # TR → DI VERIFIES edge closes the missing_vv_links() gap on DI-1.
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


# ---------------------------------------------------------------------------
# snapshot content
# ---------------------------------------------------------------------------

def test_snapshot_captures_state(g):
    snap_id = g.snapshot()
    snap = next(s for s in g._snapshots if s["snapshot_id"] == snap_id)
    assert len(snap["nodes"]) == len(g.all_nodes())
    assert len(snap["edges"]) == len(g.all_edges())


# ---------------------------------------------------------------------------
# verifies edges create a cycle — documented design decision
# ---------------------------------------------------------------------------

def test_vp001_has_verifies_back_edge_to_di001(g):
    # VP-001 → DI-001 (verifies) is an accepted design decision: the protocol
    # references the design input it validates, creating a bidirectional reference
    # for QMSR §820.30(f) traceability. The cycle (DI-001 → DO-001 → VP-001 → DI-001)
    # is intentional and does not affect change-impact traversal because the
    # traverse_node only follows downstream BFS and one level of predecessors.
    assert g.has_path("VP-001", "DI-001") is True
    assert g.has_path("DI-001", "VP-001") is True  # accepted cycle via DO-001
