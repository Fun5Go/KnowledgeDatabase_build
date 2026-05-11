from __future__ import annotations

from typing import Any

ALLOWED_RERANK_TAGS = {"support", "suspect", "irrelevant"}

ALLOWED_RELATION_TYPES = {
    "condition_match",
    "causal_mechanism",
    "trigger_or_context",
    "control_or_mitigation",
    "detection_or_reporting",
    "design_specification",
    "consequence_or_effect",
    "nominal_context_only",
    "unrelated",
}

ALLOWED_SUPPORT_CAPABILITIES = {"weak", "moderate", "strong"}

ALLOWED_DIRECTIONALITIES = {
    "evidence_to_target",
    "target_to_evidence",
    "bidirectional",
    "contextual",
    "not_applicable",
}

PRIMARY_RELATION_PRIORITY = [
    "control_or_mitigation",
    "detection_or_reporting",
    "causal_mechanism",
    "condition_match",
    "consequence_or_effect",
    "design_specification",
    "trigger_or_context",
    "nominal_context_only",
    "unrelated",
]

SUPPORT_CAPABILITY_PRIORITY = ["weak", "moderate", "strong"]


def build_rerank_output_schema() -> dict[str, Any]:
    return {
        "analysis_id": "string",
        "query_type": "function_mode | cause | effect",
        "stage": "rerank",
        "reranked_chunks": [
            {
                "rank": "integer retrieval rank copied from candidate_chunks[].retrieval rank",
                "rerank_tag": "support | suspect | irrelevant",
                "reason": "concise grounded reason",
            }
        ],
    }


def build_evidence_output_schema() -> dict[str, Any]:
    return {
        "analysis_id": "string",
        "query_type": "function_mode | cause | effect",
        "stage": "extract",
        "evidence_units": [
            {
                "rank": "integer retrieval rank copied from the source chunk",
                "evidence_span": "exact substring from raw_text",
                "relation_type": "condition_match | causal_mechanism | trigger_or_context | control_or_mitigation | detection_or_reporting | design_specification | consequence_or_effect",
                "support_capability": "weak | moderate | strong",
                "directionality": "evidence_to_target | target_to_evidence | bidirectional | contextual | not_applicable",
                "affected_objects": ["string"],
                "causal_subjects": ["string"],
                "justification": "short explanation grounded only in evidence_span and raw_text",
            }
        ],
        "chunk_aggregates": [
            {
                "rank": "integer retrieval rank copied from the source chunk",
                "rerank_tag": "support | suspect | irrelevant | unknown",
                "selected": "boolean",
                "primary_relation": "one allowed relation_type",
                "relations": ["string"],
                "evidence_spans": ["string"],
                "support_capability": "weak | moderate | strong",
                "selection_reason": "short explanation grounded only in selected evidence spans",
            }
        ],
    }
