"""Parse and normalize FMEA text before graph-assisted interpretation."""

from __future__ import annotations

import re
from typing import Any

from .schemas import FMEAReviewItem


FIELD_ALIASES = {
    "item_id": ("item_id", "id", "row_id", "fmea_id", "failure_id"),
    "function": ("function", "element_function", "under_element_function"),
    "failure_mode": ("failure_mode", "mode", "failure mode"),
    "failure_cause": ("failure_cause", "cause", "failure cause"),
    "failure_effect": ("failure_effect", "effect", "failure effect"),
    "control_measure": ("control_measure", "control", "prevention_control", "control measure"),
    "detection_measure": ("detection_measure", "detection", "detection_control", "detection measure"),
    "raw_text": ("raw_text", "text", "description"),
}

FAILURE_NODE_GRAPH_NEEDS = {
    "failure_node": "Failure node identified by failure_id.",
    "linked_failure_attributes": ["cause", "mode", "effect"],
    "optional_context": ["element_function"],
    "linked_document_context": ["chunk nodes connected to the failure entity"],
}


def parse_fmea_row(row: dict[str, Any]) -> FMEAReviewItem:
    """Convert a raw FMEA table row into a normalized review item."""
    normalized_row = {_normalize_key(key): value for key, value in row.items()}
    values = {
        field_name: _first_present(normalized_row, aliases)
        for field_name, aliases in FIELD_ALIASES.items()
    }
    item = FMEAReviewItem(
        item_id=_clean_text(values["item_id"]) or _build_stable_item_id(row),
        function=_clean_text(values["function"]),
        failure_mode=_clean_text(values["failure_mode"]),
        failure_cause=_clean_text(values["failure_cause"]),
        failure_effect=_clean_text(values["failure_effect"]),
        control_measure=_clean_text(values["control_measure"]),
        detection_measure=_clean_text(values["detection_measure"]),
        raw_text=_clean_text(values["raw_text"]) or _join_row_text(row),
        metadata={
            "input_type": "fmea_row",
            "source_row": dict(row),
            "parser_notes": [],
        },
    )
    return normalize_fmea_fields(item)


def parse_free_text_item(text: str, item_id: str = "") -> FMEAReviewItem:
    """Convert a free-text FMEA concern into a review item."""
    item = FMEAReviewItem(
        item_id=_clean_text(item_id) or _build_text_item_id(text),
        raw_text=_clean_text(text),
        metadata={
            "input_type": "free_text",
            "parser_notes": ["No field-level structure was assumed."],
        },
    )
    return normalize_fmea_fields(item)


def parse_failure_node_input(failure_id: str) -> FMEAReviewItem:
    """Create an assistant item for a Failure node identified by failure_id.

    This supports the first demonstration input style:
    `failure_id = "DFMEA6011160042R01__R125"`.

    The parser does not fetch the graph itself. It records exactly what later
    modules should retrieve: linked cause/mode/effect, optional element
    function, and connected chunk nodes. The LLM task is also prepared around
    the requested why/what/how causal-chain explanation.
    """
    clean_failure_id = _clean_text(failure_id)
    if not clean_failure_id:
        raise ValueError("failure_id must not be empty.")

    return FMEAReviewItem(
        item_id=clean_failure_id,
        raw_text=clean_failure_id,
        metadata={
            "input_type": "failure_node",
            "failure_id": clean_failure_id,
            "graph_retrieval_plan": FAILURE_NODE_GRAPH_NEEDS,
            "llm_task": build_failure_chain_explanation_task(clean_failure_id),
        },
    )


def parse_failure_attribute_input(
    *,
    attribute_text: str = "",
    semantic_id: str = "",
    attribute_type: str = "unknown",
    user_question: str = "",
) -> FMEAReviewItem:
    """Create an assistant item for one failure attribute text or semantic id.

    This supports the second demonstration input style, for example asking:
    "what could cause this?" or "what would this affect?" for a standalone
    failure mode, cause, effect, or semanticID.
    """
    clean_text = _clean_text(attribute_text)
    clean_semantic_id = _clean_text(semantic_id)
    clean_attribute_type = _normalize_attribute_type(attribute_type)
    clean_question = _clean_text(user_question)

    if not clean_text and not clean_semantic_id:
        raise ValueError("Either attribute_text or semantic_id must be provided.")

    item_id = clean_semantic_id or _build_text_item_id(clean_text)
    field_values = _attribute_to_fields(clean_attribute_type, clean_text)
    item = FMEAReviewItem(
        item_id=item_id,
        raw_text=clean_text or clean_semantic_id,
        metadata={
            "input_type": "failure_attribute",
            "semantic_id": clean_semantic_id,
            "attribute_type": clean_attribute_type,
            "user_question": clean_question,
            "graph_retrieval_plan": build_attribute_graph_retrieval_plan(
                attribute_type=clean_attribute_type,
                user_question=clean_question,
            ),
            "llm_task": build_attribute_detail_task(
                attribute_text=clean_text,
                semantic_id=clean_semantic_id,
                attribute_type=clean_attribute_type,
                user_question=clean_question,
            ),
        },
        **field_values,
    )
    return normalize_fmea_fields(item)


def build_failure_chain_explanation_task(failure_id: str) -> dict[str, Any]:
    """Build the LLM task for explaining one Failure entity's causal chain."""
    return {
        "task_type": "failure_chain_explanation",
        "failure_id": failure_id,
        "required_graph_context": [
            "linked cause nodes",
            "linked failure mode nodes",
            "linked effect nodes",
            "element function if available",
            "connected chunk/document evidence",
        ],
        "questions": {
            "why": "Why does this failure happen according to linked causes and evidence?",
            "what": "What functions, elements, requirements, users, or system behaviors are affected?",
            "how": "How does the failure propagate from cause to mode to effect, including missing intermediate logic?",
            "support": "Do the retrieved evidence nodes sufficiently support this failure entity and causal chain?",
        },
        "expected_output_sections": [
            "causal_chain_summary",
            "why_explanation",
            "what_is_affected",
            "how_it_happens",
            "evidence_support_assessment",
            "missing_or_weak_evidence",
        ],
    }


def build_attribute_detail_task(
    *,
    attribute_text: str = "",
    semantic_id: str = "",
    attribute_type: str = "unknown",
    user_question: str = "",
) -> dict[str, Any]:
    """Build the LLM task for investigating a standalone failure attribute."""
    intent = infer_attribute_question_intent(user_question)
    return {
        "task_type": "failure_attribute_detail",
        "attribute_text": attribute_text,
        "semantic_id": semantic_id,
        "attribute_type": attribute_type,
        "question_intent": intent,
        "required_graph_context": [
            "matching failure attributes",
            "neighboring failure nodes",
            "linked causes, modes, and effects",
            "connected supporting chunks",
        ],
        "questions": _attribute_questions_for_intent(intent),
        "expected_output_sections": [
            "identified_failure_detail",
            "possible_causes",
            "possible_effects",
            "related_failure_modes",
            "supporting_evidence",
            "uncertainties",
        ],
    }


def build_attribute_graph_retrieval_plan(attribute_type: str, user_question: str = "") -> dict[str, Any]:
    """Describe what graph context should be fetched for a standalone attribute."""
    intent = infer_attribute_question_intent(user_question)
    plan = {
        "start_from": "semantic_id" if attribute_type == "semantic_id" else "attribute_text",
        "attribute_type": attribute_type,
        "retrieve": ["matching attribute nodes", "neighbor failure nodes", "connected chunk nodes"],
        "expand_to": ["cause", "mode", "effect", "element_function"],
    }
    if intent == "cause":
        plan["priority"] = ["cause", "upstream failure chains", "evidence explaining why it occurs"]
    elif intent == "effect":
        plan["priority"] = ["effect", "downstream impacts", "evidence explaining what is affected"]
    else:
        plan["priority"] = ["cause", "mode", "effect", "supporting evidence"]
    return plan


def detect_abbreviations(item: FMEAReviewItem) -> list[str]:
    """Detect abbreviated or domain-specific terms requiring interpretation."""
    text = " ".join(
        [
            item.function,
            item.failure_mode,
            item.failure_cause,
            item.failure_effect,
            item.control_measure,
            item.detection_measure,
            item.raw_text,
        ]
    )
    tokens = re.findall(r"\b[A-Z][A-Z0-9/_-]{1,}\b", text)
    return sorted(set(tokens))


def normalize_fmea_fields(item: FMEAReviewItem) -> FMEAReviewItem:
    """Normalize whitespace, field names, and missing values for review."""
    item.item_id = _clean_text(item.item_id)
    item.function = _clean_text(item.function)
    item.failure_mode = _clean_text(item.failure_mode)
    item.failure_cause = _clean_text(item.failure_cause)
    item.failure_effect = _clean_text(item.failure_effect)
    item.control_measure = _clean_text(item.control_measure)
    item.detection_measure = _clean_text(item.detection_measure)
    item.raw_text = _clean_text(item.raw_text)
    item.metadata.setdefault("detected_abbreviations", detect_abbreviations(item))
    return item


def infer_attribute_question_intent(user_question: str) -> str:
    """Infer whether the standalone attribute question asks about cause, effect, or detail."""
    question = _clean_text(user_question).lower()
    if any(phrase in question for phrase in ("what could cause", "why", "root cause", "lead to this")):
        return "cause"
    if any(phrase in question for phrase in ("would affect", "impact", "effect", "consequence", "lead to")):
        return "effect"
    return "detail"


def _attribute_questions_for_intent(intent: str) -> dict[str, str]:
    if intent == "cause":
        return {
            "main": "What could cause this failure attribute?",
            "why": "Why are these causes plausible according to linked evidence?",
            "how": "How would the causes propagate into the queried attribute?",
            "support": "Which evidence supports or weakens each possible cause?",
        }
    if intent == "effect":
        return {
            "main": "What would this failure attribute affect?",
            "what": "Which functions, elements, users, or requirements may be affected?",
            "how": "How would the queried attribute propagate into those effects?",
            "support": "Which evidence supports or weakens each possible effect?",
        }
    return {
        "main": "What is the detailed meaning of this failure attribute?",
        "why": "Why is it relevant to nearby failure chains?",
        "what": "What failure nodes, causes, modes, or effects are connected to it?",
        "how": "How is it supported by document chunks and graph paths?",
    }


def _attribute_to_fields(attribute_type: str, text: str) -> dict[str, str]:
    fields = {
        "function": "",
        "failure_mode": "",
        "failure_cause": "",
        "failure_effect": "",
        "control_measure": "",
        "detection_measure": "",
    }
    if attribute_type == "cause":
        fields["failure_cause"] = text
    elif attribute_type == "mode":
        fields["failure_mode"] = text
    elif attribute_type == "effect":
        fields["failure_effect"] = text
    return fields


def _normalize_attribute_type(attribute_type: str) -> str:
    normalized = _clean_text(attribute_type).lower().replace("failure_", "")
    aliases = {
        "failure mode": "mode",
        "failure cause": "cause",
        "failure effect": "effect",
        "semanticid": "semantic_id",
        "semantic id": "semantic_id",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in {"cause", "mode", "effect", "semantic_id", "unknown"} else "unknown"


def _first_present(row: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        value = row.get(_normalize_key(alias))
        if value not in (None, ""):
            return value
    return ""


def _normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _join_row_text(row: dict[str, Any]) -> str:
    parts = []
    for key, value in row.items():
        clean_value = _clean_text(value)
        if clean_value:
            parts.append(f"{key}: {clean_value}")
    return " | ".join(parts)


def _build_stable_item_id(row: dict[str, Any]) -> str:
    text = _join_row_text(row)
    return _build_text_item_id(text)


def _build_text_item_id(text: str) -> str:
    clean = _clean_text(text)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", clean).strip("_")[:48]
    return f"text__{slug or 'empty'}"
