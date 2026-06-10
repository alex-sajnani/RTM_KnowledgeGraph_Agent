"""
test_sme_agent.py — Unit tests for the SME Router sub-agent (sme_agent.py).

map_teams_node is entirely deterministic (no LLM). Tests verify:
  - Correct team routing per node type
  - Edge cases (unmapped types, empty input)
  - notifications_from_dicts deserialization
  - generate_briefings_node with mocked LLM
"""

import pytest
from unittest.mock import patch, MagicMock

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
)
from graph import NodeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(node_id: str, node_type: str, title: str = "title") -> dict:
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


# ---------------------------------------------------------------------------
# map_teams_node — routing per node type
# ---------------------------------------------------------------------------

def test_vv_protocol_routes_to_bioinformatics_and_rd():
    result = _run_map_teams([_node("VP-001", NodeType.VV_PROTOCOL.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Bioinformatics" in teams
    assert "R&D" in teams


def test_pma_supplement_routes_to_quality_ra():
    result = _run_map_teams([_node("PM-001", NodeType.PMA_SUPPLEMENT_TRIGGER.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Quality/RA" in teams
    assert len(teams) == 1


def test_capa_routes_to_quality_ra():
    result = _run_map_teams([_node("CAPA-018", NodeType.CAPA.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Quality/RA" in teams
    assert len(teams) == 1


def test_hazard_routes_to_quality_ra():
    result = _run_map_teams([_node("H-001", NodeType.HAZARD.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Quality/RA" in teams


def test_risk_control_routes_to_pathology():
    result = _run_map_teams([_node("RC-001", NodeType.RISK_CONTROL.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Pathology" in teams


def test_test_result_routes_to_bioinformatics():
    result = _run_map_teams([_node("TR-001A", NodeType.TEST_RESULT.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "Bioinformatics" in teams


def test_design_input_routes_to_rd():
    result = _run_map_teams([_node("DI-001", NodeType.DESIGN_INPUT.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "R&D" in teams


def test_design_output_routes_to_rd():
    result = _run_map_teams([_node("DO-001", NodeType.DESIGN_OUTPUT.value)])
    teams = {n["team"] for n in result["sme_notifications"]}
    assert "R&D" in teams


def test_unmapped_type_produces_no_notification():
    # User Need is not in SME_NOTIFICATION_MAP — should produce no notification
    result = _run_map_teams([_node("UN-001", NodeType.USER_NEED.value)])
    assert result["sme_notifications"] == []


def test_empty_input_produces_no_notifications():
    result = _run_map_teams([])
    assert result["sme_notifications"] == []


def test_multiple_nodes_accumulate_all_notifications():
    nodes = [
        _node("VP-001", NodeType.VV_PROTOCOL.value),
        _node("PM-001", NodeType.PMA_SUPPLEMENT_TRIGGER.value),
        _node("CAPA-018", NodeType.CAPA.value),
    ]
    result = _run_map_teams(nodes)
    # VP-001 → 2 teams; PM-001 → 1; CAPA-018 → 1 = 4 total
    assert len(result["sme_notifications"]) == 4


def test_notification_carries_trigger_node_id():
    result = _run_map_teams([_node("VP-001", NodeType.VV_PROTOCOL.value, title="VP title")])
    notifs = result["sme_notifications"]
    assert all(n["trigger_node_id"] == "VP-001" for n in notifs)
    assert all(n["trigger_node_title"] == "VP title" for n in notifs)


def test_notification_review_obligation_is_nonempty():
    result = _run_map_teams([_node("PM-001", NodeType.PMA_SUPPLEMENT_TRIGGER.value)])
    for notif in result["sme_notifications"]:
        assert len(notif["review_obligation"]) > 10


# ---------------------------------------------------------------------------
# map_to_teams — Send fan-out
# ---------------------------------------------------------------------------

def test_map_to_teams_returns_one_send_per_unique_team():
    # VP-001 maps to 2 teams (Bioinformatics + R&D) → 2 Send objects
    state = _run_map_teams([_node("VP-001", NodeType.VV_PROTOCOL.value)])
    sends = map_to_teams(state)
    teams = {s.node for s in sends}
    assert len(sends) == 2
    assert all(s.node == "brief_team" for s in sends)
    # Each Send carries its team name in the arg payload
    send_teams = {s.arg["team"] for s in sends}
    assert "Bioinformatics" in send_teams
    assert "R&D" in send_teams


def test_map_to_teams_deduplicates_teams_across_multiple_nodes():
    # Two VP nodes → both map to Bioinformatics; should still produce 1 Send for that team
    state = _run_map_teams([
        _node("VP-001", NodeType.VV_PROTOCOL.value),
        _node("VP-002", NodeType.VV_PROTOCOL.value),
    ])
    sends = map_to_teams(state)
    send_teams = [s.arg["team"] for s in sends]
    assert send_teams.count("Bioinformatics") == 1
    assert send_teams.count("R&D") == 1


def test_map_to_teams_empty_notifications_returns_empty():
    state = _run_map_teams([])
    assert map_to_teams(state) == []


def test_map_to_teams_send_carries_correct_notifications():
    state = _run_map_teams([_node("PM-001", NodeType.PMA_SUPPLEMENT_TRIGGER.value)])
    sends = map_to_teams(state)
    assert len(sends) == 1
    payload = sends[0].arg
    assert payload["team"] == "Quality/RA"
    assert len(payload["team_notifications"]) == 1
    assert payload["team_notifications"][0]["trigger_node_id"] == "PM-001"


# ---------------------------------------------------------------------------
# brief_team_node — mocked LLM, one team per invocation
# ---------------------------------------------------------------------------

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
    state = _brief_team_state("Quality/RA", "PM-001", NodeType.PMA_SUPPLEMENT_TRIGGER.value)
    mock_response = MagicMock()
    mock_response.content = "mocked Quality/RA briefing"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("sme_agent.ChatOpenAI", return_value=mock_llm):
        result = brief_team_node(state)

    assert "Quality/RA" in result["team_briefings"]
    assert result["team_briefings"]["Quality/RA"] == "mocked Quality/RA briefing"


def test_brief_team_node_does_not_write_sme_notifications():
    # brief_team_node only writes team_briefings; finalize_notifications handles the join
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


# ---------------------------------------------------------------------------
# finalize_notifications_node — no LLM, pure join logic
# ---------------------------------------------------------------------------

def test_finalize_notifications_writes_briefings_into_notifications():
    notifications = [
        {"team": "Quality/RA", "trigger_node_id": "PM-001", "trigger_node_type": "PMA Supplement Trigger",
         "trigger_node_title": "PM-001", "review_obligation": "Check.", "llm_briefing": ""},
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
        "team_briefings": {},  # no Pathology entry
        "team": "",
        "team_notifications": [],
    }
    result = finalize_notifications_node(state)
    assert result["sme_notifications"][0]["llm_briefing"] == ""


# ---------------------------------------------------------------------------
# notifications_from_dicts
# ---------------------------------------------------------------------------

def test_notifications_from_dicts_roundtrip():
    raw = [
        {
            "team": "Quality/RA",
            "trigger_node_id": "PM-001",
            "trigger_node_type": NodeType.PMA_SUPPLEMENT_TRIGGER.value,
            "trigger_node_title": "PMA Trigger",
            "review_obligation": "Check 21 CFR 814.39",
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


# ---------------------------------------------------------------------------
# SME_NOTIFICATION_MAP completeness
# ---------------------------------------------------------------------------

def test_sme_map_covers_all_non_root_node_types():
    # Every node type that can appear downstream should route to at least one team.
    # User Need is a root artifact and intentionally not in the map.
    expected_mapped = {
        NodeType.VV_PROTOCOL.value,
        NodeType.TEST_RESULT.value,
        NodeType.HAZARD.value,
        NodeType.RISK_CONTROL.value,
        NodeType.CAPA.value,
        NodeType.PMA_SUPPLEMENT_TRIGGER.value,
        NodeType.DESIGN_INPUT.value,
        NodeType.DESIGN_OUTPUT.value,
    }
    assert expected_mapped == set(SME_NOTIFICATION_MAP.keys())
