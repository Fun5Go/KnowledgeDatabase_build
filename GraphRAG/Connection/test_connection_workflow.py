from __future__ import annotations

import sys
import os
from pathlib import Path

os.environ.setdefault("LANGSMITH_TRACING", "false")

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

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
                "label": "SPEC",
                "name": "CH_SUPPORT",
                "text": (
                    "The controller detects incorrect artifact state during transfer. "
                    "The fallback routine resets the affected interface."
                ),
            },
            {
                "retrieval rank": 2,
                "label": "SPEC",
                "name": "CH_SUSPECT",
                "text": "When the artifact changes state, the receiving module waits for a valid confirmation.",
            },
            {
                "retrieval rank": 3,
                "label": "INFO",
                "name": "CH_IRRELEVANT",
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
    assert "evidence_units" not in result


def test_extract_stage_from_existing_reranked_chunks() -> None:
    payload = build_payload()
    payload["reranked_chunks"] = [
        {
            "rank": 1,
            "label": "SPEC",
            "name": "CH_SUPPORT",
            "raw_text": payload["candidate_chunks"][0]["text"],
            "rerank_tag": "support",
            "reason": "Contains detection and reset text.",
        },
        {
            "rank": 2,
            "label": "SPEC",
            "name": "CH_SUSPECT",
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


def test_validation_fails_when_evidence_span_is_not_exact_substring() -> None:
    source_chunks = [
        {
            "rank": 1,
            "label": "SPEC",
            "name": "CH_SUPPORT",
            "raw_text": "The monitor reports an invalid artifact state.",
            "rerank_tag": "support",
        }
    ]
    result = {
        "evidence_units": [
            {
                "rank": 1,
                "label": "SPEC",
                "name": "CH_SUPPORT",
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


if __name__ == "__main__":
    test_auto_workflow_support_suspect_irrelevant_and_multiple_units()
    test_rerank_stage_only()
    test_extract_stage_from_existing_reranked_chunks()
    test_validation_fails_when_evidence_span_is_not_exact_substring()
