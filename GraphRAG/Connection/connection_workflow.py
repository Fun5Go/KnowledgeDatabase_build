from __future__ import annotations

from typing import Any, Literal

try:
    from .connection_agents import ConnectionRerankAgent, EvidenceRelationExtractionAgent, json_safe
    from .validators import (
        build_summary,
        normalize_int,
        normalize_text,
        validate_evidence_output,
        validate_rerank_output,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from connection_agents import ConnectionRerankAgent, EvidenceRelationExtractionAgent, json_safe
    from validators import (
        build_summary,
        normalize_int,
        normalize_text,
        validate_evidence_output,
        validate_rerank_output,
    )

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


LANGSMITH_PROJECT_NAME = "GraphRAGConnection"
DEFAULT_RERANK_BATCH_SIZE = 30
DEFAULT_EXTRACT_BATCH_SIZE = 12


class ConnectionWorkflow:
    """Two-stage query-conditioned Connection workflow."""

    def __init__(
        self,
        rerank_agent: ConnectionRerankAgent | None = None,
        extraction_agent: EvidenceRelationExtractionAgent | None = None,
        raise_on_validation_error: bool = False,
        use_placeholder: bool | None = None,
    ) -> None:
        agent_kwargs: dict[str, Any] = {}
        if use_placeholder is not None:
            agent_kwargs["use_placeholder"] = use_placeholder
        self.rerank_agent = rerank_agent or ConnectionRerankAgent(**agent_kwargs)
        self.extraction_agent = extraction_agent or EvidenceRelationExtractionAgent(**agent_kwargs)
        self.raise_on_validation_error = raise_on_validation_error

    @traceable(
        run_type="chain",
        name="graphrag_connection_workflow",
        tags=["graphrag", "connection", "workflow"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def run(
        self,
        payload: dict,
        stage: Literal["rerank", "extract", "auto"] = "auto",
        batch_size: int | None = None,
    ) -> dict:
        if stage not in {"rerank", "extract", "auto"}:
            raise ValueError(f"stage must be one of rerank, extract, auto; got {stage!r}.")

        normalized_payload = normalize_connection_payload(payload)
        if stage == "rerank":
            return self.run_rerank(normalized_payload, batch_size=batch_size)
        if stage == "extract":
            return self.run_extract(normalized_payload, batch_size=batch_size)
        return self.run_auto(normalized_payload, batch_size=batch_size)

    def run_rerank(self, payload: dict[str, Any], batch_size: int | None = None) -> dict[str, Any]:
        size = batch_size or DEFAULT_RERANK_BATCH_SIZE
        results: list[dict[str, Any]] = []
        validation_errors: list[str] = []
        for batch in batched(payload.get("candidate_chunks", []), size):
            batch_payload = dict(payload)
            batch_payload["candidate_chunks"] = batch
            batch_payload.pop("connected_chunk_groups", None)
            result = self.rerank_agent.rerank(batch_payload)
            validation_errors.extend(validate_rerank_output(result, batch_payload))
            results.extend(result.get("reranked_chunks", []))

        final = {
            "analysis_id": payload.get("analysis_id", ""),
            "query_type": payload.get("query_type", ""),
            "stage": "rerank",
            "reranked_chunks": sort_reranked_chunks(results),
        }
        return self._with_validation(final, validation_errors)

    def run_extract(self, payload: dict[str, Any], batch_size: int | None = None) -> dict[str, Any]:
        size = batch_size or DEFAULT_EXTRACT_BATCH_SIZE
        source_chunks = build_source_chunks_for_extraction(payload)
        evidence_units: list[dict[str, Any]] = []
        aggregates: list[dict[str, Any]] = []
        validation_errors: list[str] = []

        for batch in batched(source_chunks, size):
            batch_payload = build_extraction_payload(payload, batch)
            result = self.extraction_agent.extract(batch_payload)
            validation_errors.extend(validate_evidence_output(result, batch))
            evidence_units.extend(result.get("evidence_units", []))
            aggregates.extend(result.get("chunk_aggregates", []))

        final = {
            "analysis_id": payload.get("analysis_id", ""),
            "query_type": payload.get("query_type", ""),
            "stage": "extract",
            "evidence_units": sort_unique_evidence_units(evidence_units),
            "chunk_aggregates": sort_unique_by_rank(aggregates),
        }
        return self._with_validation(final, validation_errors)

    def run_auto(self, payload: dict[str, Any], batch_size: int | None = None) -> dict[str, Any]:
        rerank_result = self.run_rerank(payload, batch_size=batch_size or DEFAULT_RERANK_BATCH_SIZE)
        extract_payload = dict(payload)
        extract_payload["reranked_chunks"] = [
            chunk
            for chunk in rerank_result.get("reranked_chunks", [])
            if chunk.get("rerank_tag") in {"support", "suspect"}
        ]
        extract_result = self.run_extract(extract_payload, batch_size=batch_size or DEFAULT_EXTRACT_BATCH_SIZE)

        validation_errors = []
        validation_errors.extend(rerank_result.get("validation_errors", []))
        validation_errors.extend(extract_result.get("validation_errors", []))
        final = {
            "analysis_id": payload.get("analysis_id", ""),
            "query_type": payload.get("query_type", ""),
            "stage": "auto",
            "reranked_chunks": rerank_result.get("reranked_chunks", []),
            "evidence_units": extract_result.get("evidence_units", []),
            "chunk_aggregates": extract_result.get("chunk_aggregates", []),
            "summary": build_summary(
                reranked_chunks=rerank_result.get("reranked_chunks", []),
                evidence_units=extract_result.get("evidence_units", []),
                chunk_aggregates=extract_result.get("chunk_aggregates", []),
                candidate_chunks=payload.get("candidate_chunks", []),
            ),
        }
        return self._with_validation(final, validation_errors)

    def _with_validation(self, result: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        if errors and self.raise_on_validation_error:
            raise ValueError("Connection workflow validation failed: " + "; ".join(errors))
        if errors:
            result["validation_errors"] = errors
        return json_safe(result)


def normalize_connection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict.")
    normalized = dict(payload)
    normalized["analysis_id"] = normalize_text(payload.get("analysis_id"))
    normalized["query_type"] = normalize_text(payload.get("query_type"))
    normalized["query"] = payload.get("query") if isinstance(payload.get("query"), dict) else {}
    normalized["candidate_chunks"] = [
        normalize_candidate_chunk(chunk, index)
        for index, chunk in enumerate(payload.get("candidate_chunks", []), start=1)
        if isinstance(chunk, dict)
    ]
    groups = payload.get("connected_chunk_groups")
    normalized["connected_chunk_groups"] = groups if isinstance(groups, dict) else {}
    if isinstance(payload.get("reranked_chunks"), list):
        normalized["reranked_chunks"] = [
            normalize_reranked_chunk(chunk)
            for chunk in payload.get("reranked_chunks", [])
            if isinstance(chunk, dict)
        ]
    return normalized


def normalize_candidate_chunk(chunk: dict[str, Any], fallback_rank: int) -> dict[str, Any]:
    rank = normalize_int(chunk.get("retrieval rank")) or normalize_int(chunk.get("rank")) or fallback_rank
    normalized = {
        "retrieval rank": rank,
        "name": normalize_text(chunk.get("name")),
        "section_tag": normalize_text(chunk.get("section_tag")),
        "text": normalize_text(chunk.get("text") or chunk.get("raw_text")),
    }
    if "connected_group_id" in chunk:
        normalized["connected_group_id"] = chunk.get("connected_group_id")
    return normalized


def normalize_reranked_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": normalize_int(chunk.get("rank")),
        "name": normalize_text(chunk.get("name")),
        "section_tag": normalize_text(chunk.get("section_tag")),
        "raw_text": normalize_text(chunk.get("raw_text") or chunk.get("text")),
        "rerank_tag": normalize_text(chunk.get("rerank_tag") or "unknown"),
        "reason": normalize_text(chunk.get("reason")),
    }


def build_source_chunks_for_extraction(payload: dict[str, Any]) -> list[dict[str, Any]]:
    reranked_chunks = payload.get("reranked_chunks")
    if isinstance(reranked_chunks, list) and reranked_chunks:
        return [
            normalize_reranked_chunk(chunk)
            for chunk in reranked_chunks
            if isinstance(chunk, dict) and normalize_text(chunk.get("rerank_tag") or "unknown") != "irrelevant"
        ]

    chunks: list[dict[str, Any]] = []
    for candidate in payload.get("candidate_chunks", []):
        if not isinstance(candidate, dict):
            continue
        chunks.append(
            {
                "rank": normalize_int(candidate.get("retrieval rank")),
                "name": candidate.get("name", ""),
                "section_tag": candidate.get("section_tag", ""),
                "raw_text": candidate.get("text", ""),
                "rerank_tag": "unknown",
                "reason": "",
            }
        )
    return chunks


def build_extraction_payload(payload: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis_id": payload.get("analysis_id", ""),
        "query_type": payload.get("query_type", ""),
        "query": payload.get("query", {}),
        "chunks": [build_stage2_input_chunk(chunk) for chunk in chunks],
    }


def build_stage2_input_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": normalize_int(chunk.get("rank")),
        "name": normalize_text(chunk.get("name")),
        "section_tag": normalize_text(chunk.get("section_tag")),
        "raw_text": normalize_text(chunk.get("raw_text")),
        "rerank_tag": normalize_text(chunk.get("rerank_tag") or "unknown"),
    }


def filter_connected_chunk_groups(
    connected_groups: Any,
    candidate_chunks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ranks = {
        normalize_int(chunk.get("retrieval rank"))
        for chunk in candidate_chunks
        if normalize_int(chunk.get("retrieval rank")) is not None
    }
    return filter_connected_groups_by_ranks(connected_groups, ranks)


def filter_connected_chunk_groups_by_source(
    connected_groups: Any,
    source_chunks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ranks = {
        normalize_int(chunk.get("rank"))
        for chunk in source_chunks
        if normalize_int(chunk.get("rank")) is not None
    }
    return filter_connected_groups_by_ranks(connected_groups, ranks)


def filter_connected_groups_by_ranks(connected_groups: Any, ranks: set[int | None]) -> dict[str, dict[str, Any]]:
    if not isinstance(connected_groups, dict):
        return {}
    filtered: dict[str, dict[str, Any]] = {}
    for group_id, group in connected_groups.items():
        if not isinstance(group, dict):
            continue
        chunks = [
            chunk
            for chunk in group.get("chunks", [])
            if isinstance(chunk, dict) and normalize_int(chunk.get("rank")) in ranks
        ]
        relationships = [
            relationship
            for relationship in group.get("relationships", [])
            if isinstance(relationship, dict)
            and normalize_int(relationship.get("source_rank")) in ranks
            and normalize_int(relationship.get("target_rank")) in ranks
        ]
        if chunks or relationships:
            filtered[str(group_id)] = {"chunks": chunks, "relationships": relationships}
    return filtered


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def sort_unique_by_rank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_rank: dict[int, dict[str, Any]] = {}
    for item in items:
        rank = normalize_int(item.get("rank"))
        if rank is None or rank in by_rank:
            continue
        by_rank[rank] = item
    return [by_rank[rank] for rank in sorted(by_rank)]


def sort_reranked_chunks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tag_order = {"support": 0, "suspect": 1, "irrelevant": 2}
    unique_items = sort_unique_by_rank(items)
    return sorted(
        unique_items,
        key=lambda item: (
            tag_order.get(normalize_text(item.get("rerank_tag")).lower(), 3),
            normalize_int(item.get("rank")) or 0,
        ),
    )


def sort_unique_evidence_units(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    ordered: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda unit: normalize_int(unit.get("rank")) or 0):
        key = (
            normalize_int(item.get("rank")),
            item.get("evidence_span"),
            item.get("relation_type"),
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered
