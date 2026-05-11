from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault("LANGSMITH_TRACING", "false")

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from GraphRAG.Connection.connection_agents import (
    build_evidence_prompt,
    build_rerank_prompt,
    normalize_evidence_response,
    normalize_rerank_response,
)
from GraphRAG.Connection.connection_workflow import ConnectionWorkflow
from GraphRAG.Connection.validators import validate_evidence_output, validate_evidence_span


def build_payload() -> dict:
    return {
        "analysis_id": "connection-example-1",
        "query_type": "function_mode",
        "query": {
            "Function": "artifact transfer",
            "Failure mode": "incorrect artifact state",
        },
        "candidate_chunks": [
            {
                "retrieval rank": 1,
                "name": "CH_SUPPORT",
                "section_tag": "diagnostic response",
                "text": (
                    "The controller detects incorrect artifact state during transfer. "
                    "The fallback routine resets the affected interface."
                ),
            },
            {
                "retrieval rank": 2,
                "name": "CH_SUSPECT",
                "section_tag": "operating sequence",
                "text": "When the artifact changes state, the receiving module waits for a valid confirmation.",
            },
            {
                "retrieval rank": 3,
                "name": "CH_IRRELEVANT",
                "section_tag": "document metadata",
                "text": "The document index lists artifact owners and revision identifiers.",
            },
        ],
    }


def test_auto_workflow_support_suspect_irrelevant_and_multiple_units() -> None:
    workflow = ConnectionWorkflow(use_placeholder=True)
    result = workflow.run(build_payload(), stage="auto", batch_size=2)

    assert result["stage"] == "auto"
    assert [chunk["rerank_tag"] for chunk in result["reranked_chunks"]] == [
        "support",
        "suspect",
        "irrelevant",
    ]
    assert result["summary"]["num_support"] == 1
    assert result["summary"]["num_suspect"] == 1
    assert result["summary"]["num_irrelevant"] == 1

    support_units = [
        unit for unit in result["evidence_units"] if unit["rank"] == 1
    ]
    assert len(support_units) == 2
    assert {unit["relation_type"] for unit in support_units} == {
        "detection_or_reporting",
        "control_or_mitigation",
    }
    assert all(validate_evidence_span(unit["raw_text"], unit["evidence_span"]) for unit in support_units)


def test_rerank_stage_only() -> None:
    workflow = ConnectionWorkflow(use_placeholder=True)
    result = workflow.run(build_payload(), stage="rerank")

    assert result["stage"] == "rerank"
    assert len(result["reranked_chunks"]) == 3
    assert result["reranked_chunks"][0]["name"] == "CH_SUPPORT"
    assert result["reranked_chunks"][0]["section_tag"] == "diagnostic response"
    assert result["reranked_chunks"][0]["raw_text"] == build_payload()["candidate_chunks"][0]["text"]
    assert "evidence_units" not in result


def test_rerank_prompt_omits_connected_chunk_groups() -> None:
    payload = build_payload()
    payload["connected_chunk_groups"] = {
        "1": {
            "chunks": [{"rank": 1, "name": "CH_SUPPORT"}],
            "relationships": [],
        }
    }

    prompt = build_rerank_prompt(payload)

    assert '"connected_chunk_groups"' not in prompt
    assert '"candidate_chunks"' in prompt


def test_lightweight_rerank_output_is_enriched_from_candidates() -> None:
    response = {
        "analysis_id": "connection-example-1",
        "query_type": "function_mode",
        "stage": "rerank",
        "reranked_chunks": [
            {
                "rank": 1,
                "rerank_tag": "support",
                "reason": "The chunk detects the target condition.",
            }
        ],
    }

    normalized = normalize_rerank_response(response, build_payload())
    chunk = normalized["reranked_chunks"][0]

    assert chunk["rank"] == 1
    assert chunk["name"] == "CH_SUPPORT"
    assert chunk["section_tag"] == "diagnostic response"
    assert chunk["raw_text"] == build_payload()["candidate_chunks"][0]["text"]


def test_extract_stage_from_existing_reranked_chunks() -> None:
    payload = build_payload()
    payload["reranked_chunks"] = [
        {
            "rank": 1,
            "name": "CH_SUPPORT",
            "section_tag": "diagnostic response",
            "raw_text": payload["candidate_chunks"][0]["text"],
            "rerank_tag": "support",
            "reason": "Contains detection and reset text.",
        },
        {
            "rank": 2,
            "name": "CH_SUSPECT",
            "section_tag": "operating sequence",
            "raw_text": payload["candidate_chunks"][1]["text"],
            "rerank_tag": "suspect",
            "reason": "Contains contextual state-change text.",
        },
    ]
    workflow = ConnectionWorkflow(use_placeholder=True)
    result = workflow.run(payload, stage="extract")

    assert result["stage"] == "extract"
    assert {aggregate["rank"] for aggregate in result["chunk_aggregates"]} == {1, 2}
    assert any(aggregate["selected"] for aggregate in result["chunk_aggregates"])


def test_evidence_prompt_omits_connected_groups_and_stage1_reason() -> None:
    payload = {
        "analysis_id": "connection-example-1",
        "query_type": "function_mode",
        "query": {"Failure mode": "incorrect artifact state"},
        "chunks": [
            {
                "rank": 1,
                "name": "CH_SUPPORT",
                "section_tag": "diagnostic response",
                "raw_text": "The monitor reports an invalid artifact state.",
                "rerank_tag": "support",
                "reason": "Stage 1 reason should not be sent to Stage 2.",
            }
        ],
        "connected_chunk_groups": {"1": {"chunks": [], "relationships": []}},
    }

    prompt = build_evidence_prompt(payload)

    assert '"connected_chunk_groups"' not in prompt
    assert "Stage 1 reason should not be sent" not in prompt
    assert '"raw_text"' in prompt


def test_lightweight_evidence_output_is_enriched_from_source_chunks() -> None:
    payload = {
        "analysis_id": "connection-example-1",
        "query_type": "function_mode",
        "query": {"Failure mode": "incorrect artifact state"},
        "chunks": [
            {
                "rank": 1,
                "name": "CH_SUPPORT",
                "section_tag": "diagnostic response",
                "raw_text": "The monitor reports an invalid artifact state.",
                "rerank_tag": "support",
            },
            {
                "rank": 2,
                "name": "CH_REJECTED",
                "section_tag": "ordinary context",
                "raw_text": "The module provides a general interface.",
                "rerank_tag": "suspect",
            },
        ],
    }
    response = {
        "analysis_id": "connection-example-1",
        "query_type": "function_mode",
        "stage": "extract",
        "evidence_units": [
            {
                "rank": 1,
                "evidence_span": "reports an invalid artifact state",
                "relation_type": "detection_or_reporting",
                "support_capability": "strong",
                "directionality": "evidence_to_target",
                "affected_objects": ["artifact state"],
                "causal_subjects": [],
                "justification": "The span reports the target abnormal state.",
            },
            {
                "rank": 2,
                "evidence_span": "provides a general interface",
                "relation_type": "nominal_context_only",
                "support_capability": "weak",
                "directionality": "not_applicable",
                "affected_objects": [],
                "causal_subjects": [],
                "justification": "This should be filtered from evidence_units.",
            },
        ],
        "chunk_aggregates": [
            {
                "rank": 1,
                "rerank_tag": "support",
                "selected": True,
                "primary_relation": "detection_or_reporting",
                "relations": ["detection_or_reporting"],
                "evidence_spans": ["reports an invalid artifact state"],
                "support_capability": "strong",
                "selection_reason": "Detection evidence was extracted.",
            },
            {
                "rank": 2,
                "rerank_tag": "suspect",
                "selected": False,
                "primary_relation": "nominal_context_only",
                "relations": ["nominal_context_only"],
                "evidence_spans": [],
                "support_capability": "weak",
                "selection_reason": "Only ordinary context was present.",
            },
        ],
    }

    normalized = normalize_evidence_response(response, payload)

    assert len(normalized["evidence_units"]) == 1
    unit = normalized["evidence_units"][0]
    assert unit["name"] == "CH_SUPPORT"
    assert unit["section_tag"] == "diagnostic response"
    assert unit["raw_text"] == "The monitor reports an invalid artifact state."
    assert unit["relation_type"] == "detection_or_reporting"
    assert normalized["chunk_aggregates"][1]["name"] == "CH_REJECTED"
    assert normalized["chunk_aggregates"][1]["selected"] is False
    assert normalized["chunk_aggregates"][1]["primary_relation"] == "nominal_context_only"


def test_validation_fails_when_evidence_span_is_not_exact_substring() -> None:
    source_chunks = [
        {
            "rank": 1,
            "name": "CH_SUPPORT",
            "section_tag": "diagnostic response",
            "raw_text": "The monitor reports an invalid artifact state.",
            "rerank_tag": "support",
        }
    ]
    result = {
        "evidence_units": [
            {
                "rank": 1,
                "name": "CH_SUPPORT",
                "section_tag": "diagnostic response",
                "raw_text": "The monitor reports an invalid artifact state.",
                "evidence_span": "The monitor reports a rewritten state.",
                "relation_type": "detection_or_reporting",
                "support_capability": "strong",
                "directionality": "evidence_to_target",
            }
        ]
    }

    errors = validate_evidence_output(result, source_chunks)

    assert any("evidence_span" in error for error in errors)


def test_validation_fails_for_nominal_or_unrelated_evidence_units() -> None:
    source_chunks = [
        {
            "rank": 1,
            "name": "CH_CONTEXT",
            "section_tag": "ordinary context",
            "raw_text": "The module provides a general interface.",
            "rerank_tag": "suspect",
        }
    ]
    result = {
        "evidence_units": [
            {
                "rank": 1,
                "name": "CH_CONTEXT",
                "section_tag": "ordinary context",
                "raw_text": "The module provides a general interface.",
                "evidence_span": "provides a general interface",
                "relation_type": "nominal_context_only",
                "support_capability": "weak",
                "directionality": "not_applicable",
            }
        ]
    }

    errors = validate_evidence_output(result, source_chunks)

    assert any("must not be used in evidence_units" in error for error in errors)


if __name__ == "__main__":
    test_auto_workflow_support_suspect_irrelevant_and_multiple_units()
    test_rerank_stage_only()
    test_rerank_prompt_omits_connected_chunk_groups()
    test_lightweight_rerank_output_is_enriched_from_candidates()
    test_extract_stage_from_existing_reranked_chunks()
    test_evidence_prompt_omits_connected_groups_and_stage1_reason()
    test_lightweight_evidence_output_is_enriched_from_source_chunks()
    test_validation_fails_when_evidence_span_is_not_exact_substring()
    test_validation_fails_for_nominal_or_unrelated_evidence_units()
