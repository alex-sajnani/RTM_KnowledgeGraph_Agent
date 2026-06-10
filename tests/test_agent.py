"""
test_agent.py — Tests for the Change Impact sub-agent (agent.py).

The report_node makes an LLM call; it is mocked so tests run without an API key.
The traverse and classify nodes are deterministic and are the focus here.
"""

import pytest
from unittest.mock import patch, MagicMock

from graph import build_seed_graph
from agent import build_impact_agent, AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(node_id: str = "DI-001") -> AgentState:
    return {
        "changed_node_id": node_id,
        "change_description": "unit test change",
        "downstream_ids": [],
        "upstream_ids": [],
        "impacted_nodes": [],
        "vv_invalidations": [],
        "pma_supplement_flags": [],
        "capa_triggers": [],
        "llm_summary": "",
    }


def _run_agent(node_id: str = "DI-001") -> dict:
    """Invoke the agent with LLM mocked out."""
    g = build_seed_graph()
    agent = build_impact_agent(g)
    mock_response = MagicMock()
    mock_response.content = "mocked compliance summary"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("agent.ChatOpenAI", return_value=mock_llm):
        return agent.invoke(_base_state(node_id))


# ---------------------------------------------------------------------------
# Downstream traversal
# ---------------------------------------------------------------------------

def test_downstream_vv_flagged():
    # After re-orienting DO→VP→TR, the full chain is reachable from DI-001.
    result = _run_agent("DI-001")
    assert "VP-001" in result["vv_invalidations"]


def test_downstream_pma_flagged():
    result = _run_agent("DI-001")
    assert "PM-001" in result["pma_supplement_flags"]


def test_downstream_capa_flagged():
    result = _run_agent("DI-001")
    assert "CAPA-018" in result["capa_triggers"]


def test_downstream_nodes_have_direction():
    result = _run_agent("DI-001")
    downstream = [n for n in result["impacted_nodes"] if n.get("direction") == "downstream"]
    assert len(downstream) >= 5  # DO-001, VP-001, TR-001A, CAPA-018, PM-001


def test_di001_downstream_includes_regulatory_chain():
    result = _run_agent("DI-001")
    downstream_ids = {n["node_id"] for n in result["impacted_nodes"] if n.get("direction") == "downstream"}
    assert {"DO-001", "VP-001", "TR-001A", "CAPA-018", "PM-001"}.issubset(downstream_ids)


# ---------------------------------------------------------------------------
# Upstream traversal (gap 5)
# ---------------------------------------------------------------------------

def test_upstream_predecessor_surfaced():
    result = _run_agent("DI-001")
    upstream = [n for n in result["impacted_nodes"] if n.get("direction") == "upstream"]
    assert len(upstream) > 0, "Expected at least one upstream predecessor"


def test_un001_is_upstream_of_di001():
    result = _run_agent("DI-001")
    upstream_ids = [n["node_id"] for n in result["impacted_nodes"] if n.get("direction") == "upstream"]
    assert "UN-001" in upstream_ids


def test_upstream_nodes_not_in_vv_invalidations():
    result = _run_agent("DI-001")
    # UN-001 is upstream; it must never appear in the regulatory flag lists
    assert "UN-001" not in result["vv_invalidations"]
    assert "UN-001" not in result["pma_supplement_flags"]
    assert "UN-001" not in result["capa_triggers"]


def test_upstream_ids_populated():
    result = _run_agent("DI-001")
    assert "UN-001" in result["upstream_ids"]


# ---------------------------------------------------------------------------
# Terminal node (no downstream)
# ---------------------------------------------------------------------------

def test_terminal_node_no_downstream_flags():
    result = _run_agent("PM-001")
    assert result["vv_invalidations"] == []
    assert result["pma_supplement_flags"] == []
    assert result["capa_triggers"] == []
    downstream = [n for n in result["impacted_nodes"] if n.get("direction") == "downstream"]
    assert downstream == []


# ---------------------------------------------------------------------------
# LLM summary is populated (mocked)
# ---------------------------------------------------------------------------

def test_llm_summary_populated():
    result = _run_agent("DI-001")
    assert result["llm_summary"] == "mocked compliance summary"


# ---------------------------------------------------------------------------
# Action strings contain expected regulatory citations
# ---------------------------------------------------------------------------

def test_vv_action_cites_qmsr():
    result = _run_agent("DI-001")
    vv_nodes = [n for n in result["impacted_nodes"] if n["node_id"] == "VP-001"]
    assert vv_nodes, "VP-001 should be in impacted_nodes when DI-001 changes"
    assert "820.30" in vv_nodes[0]["required_action"]


def test_pma_action_cites_cfr_814():
    result = _run_agent("DI-001")
    pma_nodes = [n for n in result["impacted_nodes"] if n["node_id"] == "PM-001"]
    assert pma_nodes, "PM-001 should be in impacted_nodes when DI-001 changes"
    assert "814.39" in pma_nodes[0]["required_action"]


def test_upstream_action_cites_qmsr_820_30b():
    result = _run_agent("DI-001")
    upstream = [n for n in result["impacted_nodes"] if n.get("direction") == "upstream"]
    assert upstream
    assert "820.30" in upstream[0]["required_action"]
