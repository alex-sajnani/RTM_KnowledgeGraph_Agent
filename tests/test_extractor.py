"""
test_extractor.py — Unit tests for RTMDocumentExtractor (extractor.py).

All LLM calls are mocked. Tests focus on the deterministic post-processing
pipeline: JSON parsing, node/edge building, deduplication, cycle rejection,
and the add_to_graph confidence threshold logic.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from graph import RTMGraph, NodeType, NodeStatus, EdgeType, build_seed_graph
from extractor import (
    RTMDocumentExtractor,
    ExtractionResult,
    ExtractedNode,
    ExtractedEdge,
    _find_existing_match,
    _deduplicate_against_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    return RTMDocumentExtractor()


@pytest.fixture
def empty_graph():
    return RTMGraph()


@pytest.fixture
def seed():
    return build_seed_graph()


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


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _build_nodes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# _build_edges
# ---------------------------------------------------------------------------

def test_build_edges_valid(extractor):
    # "satisfies" is in _BLOCKED_EDGE_TYPES and gets coerced to linked_to.
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


# ---------------------------------------------------------------------------
# _find_existing_match
# ---------------------------------------------------------------------------

def test_find_existing_match_exact_id(seed):
    existing = {n["id"]: n for n in seed.all_nodes()}
    node = _make_extracted_node(suggested_id="DI-001", node_type=NodeType.DESIGN_INPUT,
                                title="Some other title")
    result = _find_existing_match(node, existing)
    assert result == "DI-001"


def test_find_existing_match_exact_id_wrong_type_not_matched(seed):
    # Same ID but different type — should NOT match
    existing = {n["id"]: n for n in seed.all_nodes()}
    node = _make_extracted_node(suggested_id="DI-001", node_type=NodeType.VV_PROTOCOL,
                                title="Some other title")
    result = _find_existing_match(node, existing)
    # Should not match on ID alone when type differs
    assert result != "DI-001" or result is None


def test_find_existing_match_title_word_overlap(seed):
    existing = {n["id"]: n for n in seed.all_nodes()}
    # "Analytical Sensitivity" overlaps with DI-001 title which has "Analytical Sensitivity"
    node = _make_extracted_node(suggested_id="DI-NEW",
                                node_type=NodeType.DESIGN_INPUT,
                                title="Analytical Sensitivity Specification")
    result = _find_existing_match(node, existing)
    assert result == "DI-001"


def test_find_existing_match_numeric_token_overlap(seed):
    existing = {n["id"]: n for n in seed.all_nodes()}
    # "18 minutes" is in DI-002 description
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


# ---------------------------------------------------------------------------
# _deduplicate_against_graph
# ---------------------------------------------------------------------------

def test_deduplicate_removes_duplicate_node(seed):
    node = _make_extracted_node(suggested_id="DI-001",
                                node_type=NodeType.DESIGN_INPUT,
                                title="Analytical Sensitivity — LoD ≤ 2.0 pg/mL")
    nodes, edges = _deduplicate_against_graph([node], [], seed)
    assert len(nodes) == 0  # duplicate removed


def test_deduplicate_keeps_novel_node(seed):
    node = _make_extracted_node(suggested_id="DI-999",
                                node_type=NodeType.DESIGN_INPUT,
                                title="Completely novel spec with no overlap",
                                description="no match possible")
    nodes, edges = _deduplicate_against_graph([node], [], seed)
    assert len(nodes) == 1


def test_deduplicate_remaps_edge_source_to_existing_id(seed):
    # Extracted node DI-001 is a duplicate — its edge source should be rewritten
    dup_node = _make_extracted_node(suggested_id="DI-001",
                                    node_type=NodeType.DESIGN_INPUT,
                                    title="Analytical Sensitivity — LoD ≤ 2.0 pg/mL")
    edge = _make_extracted_edge(source_id="DI-001", target_id="DO-001")
    nodes, edges = _deduplicate_against_graph([dup_node], [edge], seed)
    assert len(nodes) == 0
    # Edge must still exist with source remapped to the canonical "DI-001"
    assert len(edges) == 1
    assert edges[0].source_id == "DI-001"


def test_deduplicate_drops_self_loop_edge(seed):
    # If source and target collapse to the same existing node, the edge is a self-loop
    dup_node = _make_extracted_node(suggested_id="DI-NEW",
                                    node_type=NodeType.DESIGN_INPUT,
                                    title="Analytical Sensitivity Specification")  # matches DI-001
    edge = _make_extracted_edge(source_id="DI-NEW", target_id="DI-001")  # self-loop after remap
    nodes, edges = _deduplicate_against_graph([dup_node], [edge], seed)
    assert len(edges) == 0  # self-loop dropped


# ---------------------------------------------------------------------------
# add_to_graph — confidence threshold
# ---------------------------------------------------------------------------

def test_add_to_graph_high_confidence_node_is_active(seed):
    extractor = RTMDocumentExtractor()
    node = _make_extracted_node(suggested_id="UN-NEW", node_type=NodeType.USER_NEED,
                                title="New clinical need", confidence=0.9)
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[node],
        extracted_edges=[],
        raw_llm_response="{}",
    )
    extractor.add_to_graph(result, seed, confidence_threshold=0.75)
    added = seed.get_node("UN-NEW")
    assert added["status"] == NodeStatus.ACTIVE.value


def test_add_to_graph_low_confidence_node_is_pending(seed):
    extractor = RTMDocumentExtractor()
    node = _make_extracted_node(suggested_id="UN-LOW", node_type=NodeType.USER_NEED,
                                title="Low confidence need", confidence=0.5)
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[node],
        extracted_edges=[],
        raw_llm_response="{}",
    )
    extractor.add_to_graph(result, seed, confidence_threshold=0.75)
    added = seed.get_node("UN-LOW")
    assert added["status"] == NodeStatus.PENDING_REVIEW.value


def test_add_to_graph_cycle_rejection(seed):
    # Edge DI-001 → RC-001 would close a cycle (RC-001 → DI-001 already exists)
    extractor = RTMDocumentExtractor()
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
    nodes_added, edges_added = extractor.add_to_graph(result, seed)
    assert edges_added == 0
    assert len(seed.all_edges()) == before


def test_add_to_graph_missing_node_edge_skipped(seed):
    # Edge where target doesn't exist — should be silently skipped
    extractor = RTMDocumentExtractor()
    bad_edge = _make_extracted_edge(source_id="DI-001", target_id="GHOST-999")
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[],
        extracted_edges=[bad_edge],
        raw_llm_response="{}",
    )
    before = len(seed.all_edges())
    _, edges_added = extractor.add_to_graph(result, seed)
    assert edges_added == 0
    assert len(seed.all_edges()) == before


def test_add_to_graph_returns_counts(seed):
    extractor = RTMDocumentExtractor()
    node = _make_extracted_node(suggested_id="UN-CNT", node_type=NodeType.USER_NEED,
                                title="Count test node", confidence=0.9)
    result = ExtractionResult(
        document_name="test",
        timestamp="2026-01-01T00:00:00Z",
        extracted_nodes=[node],
        extracted_edges=[],
        raw_llm_response="{}",
    )
    nodes_added, edges_added = extractor.add_to_graph(result, seed)
    assert nodes_added == 1
    assert edges_added == 0


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def test_audit_log_populated_after_extract(seed):
    extractor = RTMDocumentExtractor()
    mock_response = MagicMock()
    mock_response.content = '{"nodes": [], "edges": []}'
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("extractor.ChatOpenAI", return_value=mock_llm):
        extractor.extract("some document text", document_name="test_doc.txt", graph=seed)

    log = extractor.audit_log()
    assert len(log) == 1
    assert log[0]["document_name"] == "test_doc.txt"
    assert "nodes_extracted" in log[0]
    assert "edges_extracted" in log[0]


def test_extract_llm_error_returns_empty_result(seed):
    extractor = RTMDocumentExtractor()

    with patch("extractor.ChatOpenAI", side_effect=Exception("API error")):
        result = extractor.extract("text", document_name="fail.txt", graph=seed)

    assert result.extracted_nodes == []
    assert result.extracted_edges == []
    assert "Extraction error" in result.raw_llm_response
