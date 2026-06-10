"""
test_supervisor.py — Tests for the supervisor's deterministic components.

_compute_risk_level is the most safety-critical function in the codebase:
it controls whether the pipeline pauses for human review. These tests
verify every routing branch without an LLM call.
"""

import pytest
from supervisor import _compute_risk_level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _impact(
    vv=None,
    pma=None,
    capa=None,
    nodes=None,
):
    return {
        "vv_invalidations": vv or [],
        "pma_supplement_flags": pma or [],
        "capa_triggers": capa or [],
        "impacted_nodes": nodes or [],
    }


# ---------------------------------------------------------------------------
# "low" branch — nothing flagged
# ---------------------------------------------------------------------------

def test_risk_low_on_empty_impact():
    assert _compute_risk_level(_impact()) == "low"


def test_risk_low_with_only_design_output():
    assert _compute_risk_level(_impact(
        nodes=[{"node_type": "Design Output", "node_id": "DO-001", "title": "spec"}]
    )) == "low"


def test_risk_low_with_only_test_result():
    assert _compute_risk_level(_impact(
        nodes=[{"node_type": "Test Result", "node_id": "TR-001A", "title": "result"}]
    )) == "low"


# ---------------------------------------------------------------------------
# "critical" branch — V&V invalidations
# ---------------------------------------------------------------------------

def test_risk_critical_on_vv_invalidation():
    assert _compute_risk_level(_impact(vv=["VP-001"])) == "critical"


def test_risk_critical_on_multiple_vv():
    assert _compute_risk_level(_impact(vv=["VP-001", "VP-002"])) == "critical"


# ---------------------------------------------------------------------------
# "critical" branch — PMA supplement flags
# ---------------------------------------------------------------------------

def test_risk_critical_on_pma_flag():
    assert _compute_risk_level(_impact(pma=["PM-001"])) == "critical"


def test_risk_critical_on_vv_and_pma():
    assert _compute_risk_level(_impact(vv=["VP-001"], pma=["PM-001"])) == "critical"


# ---------------------------------------------------------------------------
# "high" branch — CAPA triggers
# ---------------------------------------------------------------------------

def test_risk_high_on_capa_trigger():
    assert _compute_risk_level(_impact(capa=["CAPA-018"])) == "high"


# ---------------------------------------------------------------------------
# "high" branch — Hazard node in impact chain
# ---------------------------------------------------------------------------

def test_risk_high_on_hazard_node():
    assert _compute_risk_level(_impact(
        nodes=[{"node_type": "Hazard", "node_id": "H-001", "title": "missed AMI"}]
    )) == "high"


def test_risk_high_on_risk_control_node():
    assert _compute_risk_level(_impact(
        nodes=[{"node_type": "Risk Control", "node_id": "RC-001", "title": "RC"}]
    )) == "high"


# ---------------------------------------------------------------------------
# Priority: critical beats high
# ---------------------------------------------------------------------------

def test_critical_beats_high_when_both_present():
    result = _compute_risk_level(_impact(
        vv=["VP-001"],
        capa=["CAPA-018"],
        nodes=[{"node_type": "Hazard", "node_id": "H-001", "title": "h"}],
    ))
    assert result == "critical"


def test_pma_flag_beats_capa():
    result = _compute_risk_level(_impact(
        pma=["PM-001"],
        capa=["CAPA-018"],
    ))
    assert result == "critical"
