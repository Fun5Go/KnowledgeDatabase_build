from __future__ import annotations

import json
import os
import re
from typing import Any, Sequence

from DocLLM.LLMs.llm_init import get_llm_backend

try:
    from .prompts import CONNECTION_RERANK_PROMPT, EVIDENCE_RELATION_EXTRACTION_PROMPT
    from .schemas import build_evidence_output_schema, build_rerank_output_schema
    from .validators import normalize_int, normalize_text
except ImportError:  # pragma: no cover - supports direct script execution
    from prompts import CONNECTION_RERANK_PROMPT, EVIDENCE_RELATION_EXTRACTION_PROMPT
    from schemas import build_evidence_output_schema, build_rerank_output_schema
    from validators import normalize_int, normalize_text

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


LANGSMITH_PROJECT_NAME = "GraphRAGConnection"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/gpt-5.2")
LLM_TEMPERATURE = 0.0
LLM_JSON_MODE = True
USE_PLACEHOLDER_LLM = os.getenv("GRAPHRAG_CONNECTION_USE_PLACEHOLDER", "0").lower() in {
    "1",
    "true",
    "yes",
}


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("LLM returned an empty response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON object found in LLM response.")
    return json.loads(match.group(0))


@traceable(
    run_type="prompt",
    name="graphrag_connection_rerank_prompt_builder",
    tags=["graphrag", "connection", "rerank", "prompt"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def build_rerank_prompt(payload: dict[str, Any], template_text: str = CONNECTION_RERANK_PROMPT) -> str:
    prompt_payload = build_rerank_prompt_payload(payload)
    return template_text.format(
        output_schema_json=json.dumps(build_rerank_output_schema(), indent=2, ensure_ascii=False),
        payload_json=json.dumps(json_safe(prompt_payload), indent=2, ensure_ascii=False),
    )


@traceable(
    run_type="prompt",
    name="graphrag_connection_evidence_prompt_builder",
    tags=["graphrag", "connection", "evidence", "prompt"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def build_evidence_prompt(payload: dict[str, Any], template_text: str = EVIDENCE_RELATION_EXTRACTION_PROMPT) -> str:
    prompt_payload = build_evidence_prompt_payload(payload)
    return template_text.format(
        output_schema_json=json.dumps(build_evidence_output_schema(), indent=2, ensure_ascii=False),
        payload_json=json.dumps(json_safe(prompt_payload), indent=2, ensure_ascii=False),
    )


class ConnectionRerankAgent:
    """LLM agent that tags candidate chunks as support, suspect, or irrelevant."""

    def __init__(
        self,
        backend: str = LLM_BACKEND,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        json_mode: bool = LLM_JSON_MODE,
        use_placeholder: bool = USE_PLACEHOLDER_LLM,
        prompt_template: str = CONNECTION_RERANK_PROMPT,
    ) -> None:
        self.prompt_template = prompt_template
        self.use_placeholder = use_placeholder
        self.llm = None if use_placeholder else get_llm_backend(
            backend=backend,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
        )

    def rerank(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.use_placeholder:
            return self._placeholder_response(payload)

        prompt = build_rerank_prompt(payload, self.prompt_template)
        content = self._invoke_model(prompt)
        return normalize_rerank_response(extract_json(content), payload)

    @traceable(
        run_type="llm",
        name="graphrag_connection_rerank_llm_call",
        tags=["graphrag", "connection", "rerank", "llm"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def _invoke_model(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("LLM client is not initialized.")
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _placeholder_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        chunks = []
        for candidate in payload.get("candidate_chunks", []):
            if not isinstance(candidate, dict):
                continue
            rank = normalize_int(candidate.get("retrieval rank")) or 0
            text = normalize_text(candidate.get("text"))
            tag = placeholder_rerank_tag(text, rank)
            chunks.append(
                {
                    "rank": rank,
                    "rerank_tag": tag,
                    "reason": f"Placeholder assigned {tag} using the candidate text and retrieval rank.",
                }
            )
        lightweight = {
            "analysis_id": normalize_text(payload.get("analysis_id")),
            "query_type": normalize_text(payload.get("query_type")),
            "stage": "rerank",
            "reranked_chunks": sort_by_rank(chunks),
        }
        return normalize_rerank_response(lightweight, payload)


class EvidenceRelationExtractionAgent:
    """LLM agent that extracts exact evidence spans and relation classifications."""

    def __init__(
        self,
        backend: str = LLM_BACKEND,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        json_mode: bool = LLM_JSON_MODE,
        use_placeholder: bool = USE_PLACEHOLDER_LLM,
        prompt_template: str = EVIDENCE_RELATION_EXTRACTION_PROMPT,
    ) -> None:
        self.prompt_template = prompt_template
        self.use_placeholder = use_placeholder
        self.llm = None if use_placeholder else get_llm_backend(
            backend=backend,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
        )

    def extract(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.use_placeholder:
            return self._placeholder_response(payload)

        prompt = build_evidence_prompt(payload, self.prompt_template)
        content = self._invoke_model(prompt)
        return normalize_evidence_response(extract_json(content), payload)

    @traceable(
        run_type="llm",
        name="graphrag_connection_evidence_llm_call",
        tags=["graphrag", "connection", "evidence", "llm"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def _invoke_model(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("LLM client is not initialized.")
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _placeholder_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_chunks = payload.get("chunks") or []
        evidence_units: list[dict[str, Any]] = []
        for source in source_chunks:
            if not isinstance(source, dict):
                continue
            text = normalize_text(source.get("raw_text"))
            spans = placeholder_evidence_units(source, text)
            evidence_units.extend(spans)

        lightweight = {
            "analysis_id": normalize_text(payload.get("analysis_id")),
            "query_type": normalize_text(payload.get("query_type")),
            "stage": "extract",
            "evidence_units": sort_by_rank(evidence_units),
            "chunk_aggregates": placeholder_chunk_aggregates(source_chunks, evidence_units),
        }
        return normalize_evidence_response(lightweight, payload)


def normalize_rerank_response(response: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    candidates = build_candidate_lookup(payload)
    chunks = response.get("reranked_chunks", [])
    if not isinstance(chunks, list):
        chunks = []

    normalized: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        rank = normalize_int(chunk.get("rank"))
        candidate = candidates.get(rank)
        if candidate is None or rank is None or rank in seen_ranks:
            continue
        seen_ranks.add(rank)
        tag = normalize_text(chunk.get("rerank_tag")).lower()
        if tag not in {"support", "suspect", "irrelevant"}:
            tag = "suspect"
        normalized.append(
            {
                "rank": rank,
                "name": candidate.get("name", ""),
                "section_tag": candidate.get("section_tag", ""),
                "raw_text": candidate.get("text", ""),
                "rerank_tag": tag,
                "reason": normalize_text(chunk.get("reason")) or "No grounded reason provided.",
            }
        )

    for rank, candidate in candidates.items():
        if rank in seen_ranks or rank is None:
            continue
        normalized.append(
            {
                "rank": rank,
                "name": candidate.get("name", ""),
                "section_tag": candidate.get("section_tag", ""),
                "raw_text": candidate.get("text", ""),
                "rerank_tag": "suspect",
                "reason": "Missing from model output; retained as suspect for high recall.",
            }
        )

    return {
        "analysis_id": normalize_text(response.get("analysis_id") or payload.get("analysis_id")),
        "query_type": normalize_text(response.get("query_type") or payload.get("query_type")),
        "stage": "rerank",
        "reranked_chunks": sort_by_rank(normalized),
    }


def build_rerank_prompt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the Stage 1 LLM input without graph-level connected context."""

    prompt_payload = {
        "analysis_id": normalize_text(payload.get("analysis_id")),
        "query_type": normalize_text(payload.get("query_type")),
        "query": payload.get("query") if isinstance(payload.get("query"), dict) else {},
        "candidate_chunks": [],
    }
    for candidate in payload.get("candidate_chunks", []):
        if not isinstance(candidate, dict):
            continue
        prompt_candidate = {
            "retrieval rank": candidate.get("retrieval rank"),
            "name": strip_constructed_reason_from_name(candidate.get("name", "")),
            "section_tag": candidate.get("section_tag", ""),
            "text": candidate.get("text", ""),
        }
        if "connected_group_id" in candidate:
            prompt_candidate["connected_group_id"] = candidate.get("connected_group_id")
        prompt_payload["candidate_chunks"].append(prompt_candidate)
    return prompt_payload


def build_evidence_prompt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build Stage 2 LLM input without graph context or Stage 1 reasons."""

    prompt_payload = {
        "analysis_id": normalize_text(payload.get("analysis_id")),
        "query_type": normalize_text(payload.get("query_type")),
        "query": payload.get("query") if isinstance(payload.get("query"), dict) else {},
        "chunks": [],
    }
    for chunk in payload.get("chunks", []):
        if not isinstance(chunk, dict):
            continue
        prompt_payload["chunks"].append(
            {
                "rank": chunk.get("rank"),
                "name": strip_constructed_reason_from_name(chunk.get("name", "")),
                "section_tag": chunk.get("section_tag", ""),
                "raw_text": chunk.get("raw_text", ""),
                "rerank_tag": chunk.get("rerank_tag", "unknown"),
            }
        )
    return prompt_payload


def strip_constructed_reason_from_name(value: Any) -> str:
    name = normalize_text(value)
    marker = "\nReason:"
    if marker in name:
        return name.split(marker, 1)[0].strip()
    if name.startswith("Reason:"):
        return ""
    return name


def normalize_evidence_response(response: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    source_chunks = payload.get("chunks") or []
    source_by_rank = {
        normalize_int(chunk.get("rank")): chunk
        for chunk in source_chunks
        if isinstance(chunk, dict) and normalize_int(chunk.get("rank")) is not None
    }
    units = response.get("evidence_units", [])
    if not isinstance(units, list):
        units = []

    normalized_units: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        rank = normalize_int(unit.get("rank"))
        source = source_by_rank.get(rank)
        if source is None:
            continue
        relation_type = normalize_relation_type(unit.get("relation_type"))
        if relation_type in {"nominal_context_only", "unrelated"}:
            continue
        normalized_units.append(
            {
                "rank": rank,
                "name": source.get("name", ""),
                "section_tag": source.get("section_tag", ""),
                "raw_text": source.get("raw_text", ""),
                "evidence_span": normalize_text(unit.get("evidence_span")),
                "relation_type": relation_type,
                "support_capability": normalize_support_capability(unit.get("support_capability")),
                "directionality": normalize_directionality(unit.get("directionality")),
                "affected_objects": normalize_string_list(unit.get("affected_objects")),
                "causal_subjects": normalize_string_list(unit.get("causal_subjects")),
                "justification": normalize_text(unit.get("justification")),
            }
        )

    return {
        "analysis_id": normalize_text(response.get("analysis_id") or payload.get("analysis_id")),
        "query_type": normalize_text(response.get("query_type") or payload.get("query_type")),
        "stage": "extract",
        "evidence_units": sort_by_rank(normalized_units),
        "chunk_aggregates": normalize_chunk_aggregates_response(
            response.get("chunk_aggregates"),
            source_chunks,
            normalized_units,
        ),
    }


def normalize_chunk_aggregates_response(
    value: Any,
    source_chunks: list[dict[str, Any]],
    evidence_units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_rank = {
        normalize_int(chunk.get("rank")): chunk
        for chunk in source_chunks
        if isinstance(chunk, dict) and normalize_int(chunk.get("rank")) is not None
    }
    units_by_rank: dict[int, list[dict[str, Any]]] = {}
    for unit in evidence_units:
        rank = normalize_int(unit.get("rank"))
        if rank is not None:
            units_by_rank.setdefault(rank, []).append(unit)

    aggregate_by_rank: dict[int, dict[str, Any]] = {}
    if isinstance(value, list):
        for aggregate in value:
            if not isinstance(aggregate, dict):
                continue
            rank = normalize_int(aggregate.get("rank"))
            source = source_by_rank.get(rank)
            if rank is None or source is None:
                continue
            units = units_by_rank.get(rank, [])
            aggregate_by_rank[rank] = enrich_chunk_aggregate(aggregate, source, units)

    for rank, source in source_by_rank.items():
        if rank is None or rank in aggregate_by_rank:
            continue
        units = units_by_rank.get(rank, [])
        aggregate_by_rank[rank] = enrich_chunk_aggregate({}, source, units)

    return [aggregate_by_rank[rank] for rank in sorted(aggregate_by_rank)]


def enrich_chunk_aggregate(
    aggregate: dict[str, Any],
    source: dict[str, Any],
    units: list[dict[str, Any]],
) -> dict[str, Any]:
    relations_from_units = unique_preserve_order(
        [normalize_text(unit.get("relation_type")) for unit in units if normalize_text(unit.get("relation_type"))]
    )
    spans_from_units = unique_preserve_order(
        [normalize_text(unit.get("evidence_span")) for unit in units if normalize_text(unit.get("evidence_span"))]
    )
    selected = bool(units)
    if selected:
        primary_relation = aggregate_primary_relation_local(relations_from_units)
        relations = relations_from_units
        evidence_spans = spans_from_units
        support_capability = strongest_support_capability_local(
            [normalize_text(unit.get("support_capability")) for unit in units]
        )
        selection_reason = normalize_text(aggregate.get("selection_reason")) or (
            "Selected from exact evidence span(s): " + " | ".join(evidence_spans[:3])
        )
    else:
        primary_relation = normalize_relation_type(aggregate.get("primary_relation"))
        if primary_relation not in {"nominal_context_only", "unrelated"}:
            primary_relation = "unrelated"
        aggregate_relations = [
            normalize_relation_type(relation)
            for relation in aggregate.get("relations", [])
        ] if isinstance(aggregate.get("relations"), list) else []
        relations = [
            relation
            for relation in unique_preserve_order(aggregate_relations)
            if relation in {"nominal_context_only", "unrelated"}
        ]
        evidence_spans = []
        support_capability = "weak"
        selection_reason = normalize_text(aggregate.get("selection_reason")) or (
            "No valid failure-relevant evidence was extracted from this chunk."
        )

    return {
        "rank": normalize_int(source.get("rank")),
        "name": source.get("name", ""),
        "section_tag": source.get("section_tag", ""),
        "raw_text": source.get("raw_text", ""),
        "rerank_tag": source.get("rerank_tag", "unknown"),
        "selected": selected,
        "primary_relation": primary_relation,
        "relations": relations,
        "evidence_spans": evidence_spans,
        "support_capability": support_capability,
        "selection_reason": selection_reason,
    }


def build_candidate_lookup(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for candidate in payload.get("candidate_chunks", []):
        if not isinstance(candidate, dict):
            continue
        rank = normalize_int(candidate.get("retrieval rank"))
        if rank is not None:
            lookup[rank] = candidate
    return lookup


def normalize_relation_type(value: Any) -> str:
    relation = normalize_text(value)
    if relation in {
        "condition_match",
        "causal_mechanism",
        "trigger_or_context",
        "control_or_mitigation",
        "detection_or_reporting",
        "design_specification",
        "consequence_or_effect",
        "nominal_context_only",
        "unrelated",
    }:
        return relation
    return "unrelated"


def normalize_support_capability(value: Any) -> str:
    capability = normalize_text(value).lower()
    if capability in {"weak", "moderate", "strong"}:
        return capability
    return "weak"


def normalize_directionality(value: Any) -> str:
    directionality = normalize_text(value)
    if directionality in {
        "evidence_to_target",
        "target_to_evidence",
        "bidirectional",
        "contextual",
        "not_applicable",
    }:
        return directionality
    return "contextual"


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = normalize_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        items.append(text)
    return items


def aggregate_primary_relation_local(relations: list[str]) -> str:
    priority = [
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
    relation_set = set(relations)
    for relation in priority:
        if relation in relation_set:
            return relation
    return "unrelated"


def strongest_support_capability_local(values: Sequence[str]) -> str:
    priority = ["weak", "moderate", "strong"]
    best = "weak"
    for value in values:
        if value in priority and priority.index(value) > priority.index(best):
            best = value
    return best


def unique_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def placeholder_rerank_tag(text: str, rank: int) -> str:
    lowered = text.lower()
    support_terms = ("detect", "prevent", "limit", "reset", "causes", "results in", "shall")
    suspect_patterns = (
        r"\bwhen\b",
        r"\bduring\b",
        r"\bafter\b",
        r"\bif\b",
        r"\bstate\b",
        r"\bcondition\b",
        r"\binterface\b",
    )
    if any(term in lowered for term in support_terms):
        return "support"
    if rank <= 2 or any(re.search(pattern, lowered) for pattern in suspect_patterns):
        return "suspect"
    return "irrelevant"


def placeholder_evidence_units(source: dict[str, Any], text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    sentences = split_sentences(text)
    units: list[dict[str, Any]] = []
    for sentence in sentences:
        relation = placeholder_relation(sentence)
        if relation in {"nominal_context_only", "unrelated"}:
            continue
        units.append(
            {
                "rank": normalize_int(source.get("rank")),
                "evidence_span": sentence,
                "relation_type": relation,
                "support_capability": "moderate" if relation != "nominal_context_only" else "weak",
                "directionality": placeholder_directionality(relation),
                "affected_objects": [],
                "causal_subjects": [],
                "justification": "Placeholder extracted an exact sentence from raw_text.",
            }
        )
    return units


def placeholder_chunk_aggregates(
    source_chunks: Sequence[dict[str, Any]],
    evidence_units: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    units_by_rank: dict[int, list[dict[str, Any]]] = {}
    for unit in evidence_units:
        rank = normalize_int(unit.get("rank"))
        if rank is not None:
            units_by_rank.setdefault(rank, []).append(unit)

    aggregates: list[dict[str, Any]] = []
    for source in source_chunks:
        rank = normalize_int(source.get("rank"))
        units = units_by_rank.get(rank or -1, [])
        relations = unique_preserve_order(
            [normalize_text(unit.get("relation_type")) for unit in units if normalize_text(unit.get("relation_type"))]
        )
        spans = unique_preserve_order(
            [normalize_text(unit.get("evidence_span")) for unit in units if normalize_text(unit.get("evidence_span"))]
        )
        aggregates.append(
            {
                "rank": rank,
                "rerank_tag": source.get("rerank_tag", "unknown"),
                "selected": bool(units),
                "primary_relation": aggregate_primary_relation_local(relations),
                "relations": relations,
                "evidence_spans": spans,
                "support_capability": strongest_support_capability_local(
                    [normalize_text(unit.get("support_capability")) for unit in units]
                ),
                "selection_reason": (
                    "Selected from exact evidence span(s): " + " | ".join(spans[:3])
                    if units
                    else "No valid failure-relevant evidence was extracted from this chunk."
                ),
            }
        )
    return aggregates


def placeholder_relation(sentence: str) -> str:
    lowered = sentence.lower()
    if any(term in lowered for term in ("prevent", "limit", "reset", "isolate", "fallback")):
        return "control_or_mitigation"
    if any(term in lowered for term in ("detect", "monitor", "report", "alarm", "log")):
        return "detection_or_reporting"
    if any(term in lowered for term in ("cause", "lead to", "results in", "otherwise")):
        return "causal_mechanism"
    if "shall" in lowered or "must" in lowered:
        return "design_specification"
    if any(term in lowered for term in ("during", "when", "after", "before", "in mode")):
        return "trigger_or_context"
    if any(term in lowered for term in ("part of", "connects", "provides")):
        return "nominal_context_only"
    return "unrelated"


def placeholder_directionality(relation: str) -> str:
    if relation in {"control_or_mitigation", "detection_or_reporting", "causal_mechanism"}:
        return "evidence_to_target"
    if relation == "trigger_or_context":
        return "contextual"
    if relation == "nominal_context_only":
        return "not_applicable"
    return "contextual"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def sort_by_rank(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: normalize_int(item.get("rank")) or 0)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {normalize_text(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return normalize_text(value)
