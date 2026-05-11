from __future__ import annotations

from typing import Any, Sequence

try:
    from .schemas import (
        ALLOWED_DIRECTIONALITIES,
        ALLOWED_RELATION_TYPES,
        ALLOWED_RERANK_TAGS,
        ALLOWED_SUPPORT_CAPABILITIES,
        PRIMARY_RELATION_PRIORITY,
        SUPPORT_CAPABILITY_PRIORITY,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from schemas import (
        ALLOWED_DIRECTIONALITIES,
        ALLOWED_RELATION_TYPES,
        ALLOWED_RERANK_TAGS,
        ALLOWED_SUPPORT_CAPABILITIES,
        PRIMARY_RELATION_PRIORITY,
        SUPPORT_CAPABILITY_PRIORITY,
    )


def validate_rerank_output(result: dict, payload: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["Rerank output must be a dict."]

    candidates = {
        normalize_int(chunk.get("retrieval rank")): chunk
        for chunk in payload.get("candidate_chunks", [])
        if isinstance(chunk, dict) and normalize_int(chunk.get("retrieval rank")) is not None
    }
    chunks = result.get("reranked_chunks")
    if not isinstance(chunks, list):
        return ["reranked_chunks must be a list."]

    for index, chunk in enumerate(chunks):
        prefix = f"reranked_chunks[{index}]"
        if not isinstance(chunk, dict):
            errors.append(f"{prefix} must be a dict.")
            continue
        rank = normalize_int(chunk.get("rank"))
        candidate = candidates.get(rank)
        if candidate is None:
            errors.append(f"{prefix}.rank does not exist in candidate chunks: {chunk.get('rank')!r}.")
            continue
        if chunk.get("rerank_tag") not in ALLOWED_RERANK_TAGS:
            errors.append(f"{prefix}.rerank_tag is invalid: {chunk.get('rerank_tag')!r}.")
        # Stage 1 model output is lightweight, but normalized workflow output is
        # enriched deterministically for backward compatibility.
        if chunk.get("name") != candidate.get("name"):
            errors.append(f"{prefix}.name must copy candidate name exactly.")
        if chunk.get("section_tag", "") != candidate.get("section_tag", ""):
            errors.append(f"{prefix}.section_tag must copy candidate section_tag exactly.")
        if chunk.get("raw_text") != candidate.get("text"):
            errors.append(f"{prefix}.raw_text must copy candidate text exactly.")
        if not normalize_text(chunk.get("reason")):
            errors.append(f"{prefix}.reason is required.")
    return errors


def validate_evidence_output(result: dict, source_chunks: list[dict]) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["Evidence output must be a dict."]

    chunks_by_rank = {
        normalize_int(chunk.get("rank")): chunk
        for chunk in source_chunks
        if isinstance(chunk, dict) and normalize_int(chunk.get("rank")) is not None
    }
    evidence_units = result.get("evidence_units")
    if not isinstance(evidence_units, list):
        return ["evidence_units must be a list."]

    for index, unit in enumerate(evidence_units):
        prefix = f"evidence_units[{index}]"
        if not isinstance(unit, dict):
            errors.append(f"{prefix} must be a dict.")
            continue
        rank = normalize_int(unit.get("rank"))
        source = chunks_by_rank.get(rank)
        if source is None:
            errors.append(f"{prefix}.rank does not exist in source chunks: {unit.get('rank')!r}.")
            continue
        if unit.get("relation_type") not in ALLOWED_RELATION_TYPES:
            errors.append(f"{prefix}.relation_type is invalid: {unit.get('relation_type')!r}.")
        if unit.get("relation_type") in {"nominal_context_only", "unrelated"}:
            errors.append(
                f"{prefix}.relation_type must not be used in evidence_units: "
                f"{unit.get('relation_type')!r}."
            )
        if unit.get("support_capability") not in ALLOWED_SUPPORT_CAPABILITIES:
            errors.append(
                f"{prefix}.support_capability is invalid: {unit.get('support_capability')!r}."
            )
        if unit.get("directionality") not in ALLOWED_DIRECTIONALITIES:
            errors.append(f"{prefix}.directionality is invalid: {unit.get('directionality')!r}.")
        if unit.get("name") != source.get("name"):
            errors.append(f"{prefix}.name must copy source name exactly.")
        if unit.get("section_tag", "") != source.get("section_tag", ""):
            errors.append(f"{prefix}.section_tag must copy source section_tag exactly.")
        if unit.get("raw_text") != source.get("raw_text"):
            errors.append(f"{prefix}.raw_text must copy source raw_text exactly.")
        if not validate_evidence_span(source.get("raw_text", ""), unit.get("evidence_span", "")):
            errors.append(f"{prefix}.evidence_span must be exact substring(s) from raw_text.")
    return errors


def validate_evidence_span(raw_text: str, evidence_span: str) -> bool:
    raw_text = normalize_text(raw_text)
    evidence_span = normalize_text(evidence_span)
    if not raw_text or not evidence_span:
        return False
    parts = [part.strip() for part in evidence_span.split(" ... ")]
    if not parts or any(not part for part in parts):
        return False
    return all(part in raw_text for part in parts)


def aggregate_primary_relation(relations: list[str]) -> str:
    relation_set = set(relations)
    for relation in PRIMARY_RELATION_PRIORITY:
        if relation in relation_set:
            return relation
    return "unrelated"


def build_summary(
    reranked_chunks: Sequence[dict[str, Any]] | None = None,
    evidence_units: Sequence[dict[str, Any]] | None = None,
    chunk_aggregates: Sequence[dict[str, Any]] | None = None,
    candidate_chunks: Sequence[dict[str, Any]] | None = None,
) -> dict[str, int]:
    reranked_chunks = list(reranked_chunks or [])
    evidence_units = list(evidence_units or [])
    chunk_aggregates = list(chunk_aggregates or [])
    candidate_chunks = list(candidate_chunks or [])
    return {
        "num_candidates": len(candidate_chunks) if candidate_chunks else len(reranked_chunks),
        "num_support": count_tag(reranked_chunks, "support"),
        "num_suspect": count_tag(reranked_chunks, "suspect"),
        "num_irrelevant": count_tag(reranked_chunks, "irrelevant"),
        "num_evidence_units": len(evidence_units),
        "num_selected_chunks": sum(1 for chunk in chunk_aggregates if chunk.get("selected") is True),
    }


def build_chunk_aggregates(source_chunks: Sequence[dict[str, Any]], evidence_units: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    units_by_rank: dict[int, list[dict[str, Any]]] = {}
    for unit in evidence_units:
        rank = normalize_int(unit.get("rank"))
        if rank is None:
            continue
        units_by_rank.setdefault(rank, []).append(unit)

    aggregates: list[dict[str, Any]] = []
    for source in sorted(source_chunks, key=lambda item: normalize_int(item.get("rank")) or 0):
        rank = normalize_int(source.get("rank"))
        units = units_by_rank.get(rank or -1, [])
        relations = unique_preserve_order(
            [normalize_text(unit.get("relation_type")) for unit in units if normalize_text(unit.get("relation_type"))]
        )
        selected_units = [
            unit
            for unit in units
            if unit.get("relation_type") not in {"nominal_context_only", "unrelated"}
        ]
        selected = bool(selected_units)
        spans = unique_preserve_order(
            [normalize_text(unit.get("evidence_span")) for unit in units if normalize_text(unit.get("evidence_span"))]
        )
        selected_spans = [
            normalize_text(unit.get("evidence_span"))
            for unit in selected_units
            if normalize_text(unit.get("evidence_span"))
        ]
        aggregates.append(
            {
                "rank": rank,
                "name": source.get("name", ""),
                "section_tag": source.get("section_tag", ""),
                "raw_text": source.get("raw_text", ""),
                "rerank_tag": source.get("rerank_tag", "unknown"),
                "selected": selected,
                "primary_relation": aggregate_primary_relation(relations),
                "relations": relations,
                "evidence_spans": spans,
                "support_capability": strongest_support_capability(
                    [normalize_text(unit.get("support_capability")) for unit in units]
                ),
                "selection_reason": build_selection_reason(selected_spans),
            }
        )
    return aggregates


def count_tag(chunks: Sequence[dict[str, Any]], tag: str) -> int:
    return sum(1 for chunk in chunks if chunk.get("rerank_tag") == tag)


def strongest_support_capability(values: Sequence[str]) -> str:
    best = "weak"
    for value in values:
        if value in SUPPORT_CAPABILITY_PRIORITY and SUPPORT_CAPABILITY_PRIORITY.index(value) > SUPPORT_CAPABILITY_PRIORITY.index(best):
            best = value
    return best


def build_selection_reason(selected_spans: Sequence[str]) -> str:
    if not selected_spans:
        return "No selected evidence span with a technical relation was extracted."
    return "Selected from exact evidence span(s): " + " | ".join(selected_spans[:3])


def normalize_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def unique_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
