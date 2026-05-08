from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from GraphRAG.Extraction.chunk_selection_agent import normalize_selection_response


def build_payload() -> dict:
    return {
        "analysis_id": "analysis-1",
        "query_type": "cause",
        "candidate_chunks": [
            {
                "retrieval rank": 1,
                "label": "ESW",
                "name": "CH_1",
                "text": "The watchdog detects missing task activity and resets the controller.",
            },
            {
                "retrieval rank": 2,
                "label": "HW",
                "name": "CH_2",
                "text": "Delayed switching may cause unstable current behavior and device damage.",
            },
            {
                "retrieval rank": 3,
                "label": "SYS",
                "name": "CH_3",
                "text": "The relay shall close within 20 ms when the enable signal is valid.",
            },
            {
                "retrieval rank": 4,
                "label": "SYS",
                "name": "REJECTED",
                "text": "The rejected pump module shall report its nominal status.",
            },
        ],
    }


def test_normalize_selection_response_supports_all_evidence_labels() -> None:
    response = {
        "top_chunks": [
            {
                "rank": 1,
                "label": "ESW",
                "name": "CH_1",
                "evidence_span": "detects missing task activity",
                "evidence_label": "control",
                "support_capability": "strong",
                "justification": "Watchdog detection and reset are controls.",
            },
            {
                "rank": 2,
                "label": "HW",
                "name": "CH_2",
                "evidence_span": "unstable current behavior and device damage",
                "evidence_label": "reason",
                "support_capability": "moderate",
                "justification": "The chunk states damage and instability risk.",
            },
            {
                "rank": 3,
                "label": "SYS",
                "name": "CH_3",
                "evidence_span": "shall close within 20 ms",
                "evidence_label": "specification",
                "support_capability": "weak",
                "justification": "The chunk states a timing requirement.",
            },
        ],
        "affected_objects": ["controller", "device", "relay"],
        "causal_subjects": ["missing task activity", "Delayed switching", "enable signal"],
    }

    normalized = normalize_selection_response(response, build_payload())

    assert [chunk["evidence_label"] for chunk in normalized["top_chunks"]] == [
        "control",
        "reason",
        "specification",
    ]
    assert all("relation" + "ship" not in chunk for chunk in normalized["top_chunks"])
    assert all("reason" not in chunk for chunk in normalized["top_chunks"])
    assert normalized["affected_objects"] == ["controller", "device", "relay"]
    assert normalized["causal_subjects"] == [
        "missing task activity",
        "Delayed switching",
        "enable signal",
    ]


def test_evidence_span_must_be_copied_from_raw_text() -> None:
    response = {
        "top_chunks": [
            {
                "rank": 1,
                "label": "MODEL_DRIFT_LABEL",
                "name": "MODEL_DRIFT_NAME",
                "evidence_span": "watchdog detects ... resets the controller",
                "evidence_label": "control",
                "support_capability": "strong",
                "justification": "The watchdog detects and resets.",
            }
        ],
    }

    normalized = normalize_selection_response(response, build_payload())
    chunk = normalized["top_chunks"][0]

    assert chunk["raw_text"] == build_payload()["candidate_chunks"][0]["text"]
    assert chunk["label"] == "ESW"
    assert chunk["name"] == "CH_1"
    assert chunk["evidence_span"] == "watchdog detects ... resets the controller"


def test_invalid_evidence_span_falls_back_to_exact_raw_text_prefix() -> None:
    response = {
        "top_chunks": [
            {
                "rank": 3,
                "label": "SYS",
                "name": "CH_3",
                "evidence_span": "paraphrased timing requirement",
                "evidence_label": "specification",
                "support_capability": "moderate",
                "justification": "Invalid span should be replaced.",
            }
        ],
    }

    normalized = normalize_selection_response(response, build_payload())

    assert normalized["top_chunks"][0]["evidence_span"] == (
        "The relay shall close within 20 ms when the enable signal is valid."
    )


def test_rejected_chunks_do_not_populate_top_level_terms() -> None:
    response = {
        "top_chunks": [
            {
                "rank": 1,
                "label": "ESW",
                "name": "CH_1",
                "evidence_span": "resets the controller",
                "evidence_label": "control",
                "support_capability": "strong",
                "justification": "Selected control chunk.",
            }
        ],
        "affected_objects": ["controller", "rejected pump module"],
        "causal_subjects": ["missing task activity", "nominal status"],
    }

    normalized = normalize_selection_response(response, build_payload())

    assert normalized["affected_objects"] == ["controller"]
    assert normalized["causal_subjects"] == ["missing task activity"]


def test_empty_selection_always_has_top_level_lists() -> None:
    normalized = normalize_selection_response({"top_chunks": []}, build_payload())

    assert normalized["top_chunks"] == []
    assert normalized["affected_objects"] == []
    assert normalized["causal_subjects"] == []


if __name__ == "__main__":
    test_normalize_selection_response_supports_all_evidence_labels()
    test_evidence_span_must_be_copied_from_raw_text()
    test_invalid_evidence_span_falls_back_to_exact_raw_text_prefix()
    test_rejected_chunks_do_not_populate_top_level_terms()
    test_empty_selection_always_has_top_level_lists()
