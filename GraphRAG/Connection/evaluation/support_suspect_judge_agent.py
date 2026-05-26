from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from GraphRAG.llm_init import configure_langsmith, get_llm_backend

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_PATH = SCRIPT_DIR / "support_suspect_judge_prompt.txt"

LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/gpt-5.2")
LLM_TEMPERATURE = 0.0
LLM_JSON_MODE = True
DEFAULT_MAX_RETRIES = 3
LANGSMITH_PROJECT_NAME = configure_langsmith("Verification")

SUPPORT_SUSPECT_TAGS = {"support", "suspect"}
ISSUE_TYPES = {
    "selected_error",
    "relationship_error",
    "selected_and_relationship_error",
}


def load_prompt(path: Path = PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse strict JSON first, then recover a single JSON object from noisy model text."""
    cleaned = str(text or "").strip()
    if not cleaned:
        raise ValueError("LLM returned an empty response.")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object.")
    return parsed


def compact_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None).strip()
    return str(value or "").strip()


def compact_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    aggregate = chunk.get("chunk_aggregate")
    if not isinstance(aggregate, dict):
        aggregate = {}

    return {
        "rank": chunk.get("rank"),
        "rerank_tag": chunk.get("rerank_tag"),
        "reason": chunk.get("reason"),
        "raw_text": compact_text(chunk.get("raw_text")),
        "extraction": {
            "selected": aggregate.get("selected"),
            "primary_relation": aggregate.get("primary_relation"),
            "relations": aggregate.get("relations") or [],
            "evidence_spans": aggregate.get("evidence_spans") or [],
            "support_capability": aggregate.get("support_capability"),
            "selection_reason": aggregate.get("selection_reason"),
        },
    }


def group_payload(
    query_text: str,
    chunks: list[dict[str, Any]],
    query_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "query_text": query_text,
        "query": query_fields or {"query_type": "unknown", "query_text": query_text},
        "chunks": [compact_chunk(chunk) for chunk in chunks],
    }


def chunk_payload(
    query_text: str,
    chunk: dict[str, Any],
    query_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "query_text": query_text,
        "query": query_fields or {"query_type": "unknown", "query_text": query_text},
        "chunks": [compact_chunk(chunk)],
    }


def build_prompt(template: str, payload: dict[str, Any]) -> str:
    # The prompt contains literal JSON schema braces, so avoid str.format().
    return template.replace("{payload_json}", json.dumps(payload, indent=2, ensure_ascii=False))


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def original_selected(chunk: dict[str, Any]) -> bool:
    aggregate = chunk.get("chunk_aggregate")
    if not isinstance(aggregate, dict):
        return False
    return normalize_bool(aggregate.get("selected"))


def original_primary_relation(chunk: dict[str, Any]) -> str | None:
    aggregate = chunk.get("chunk_aggregate")
    if not isinstance(aggregate, dict):
        return None
    primary = aggregate.get("primary_relation")
    if primary is None:
        return None
    return str(primary).strip() or None


def original_relations(chunk: dict[str, Any]) -> list[str]:
    aggregate = chunk.get("chunk_aggregate")
    if not isinstance(aggregate, dict):
        return []
    relations = aggregate.get("relations")
    if not isinstance(relations, list):
        relations = [] if relations in (None, "") else [relations]
    return [str(relation).strip() for relation in relations if str(relation).strip()]


def issue_type_from_flags(
    selected_correct: bool,
    relationship_correct: bool,
    suggested_selected: bool,
) -> str:
    if not selected_correct and suggested_selected is False:
        return "selected_error"
    if not selected_correct and not relationship_correct:
        return "selected_and_relationship_error"
    if not selected_correct:
        return "selected_error"
    return "relationship_error"


def relation_set(relations: list[str]) -> set[str]:
    return {relation.strip().lower() for relation in relations if relation.strip()}


def normalize_conflict_item(item: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any] | None:
    """Keep only the requested conflict fields and anchor rank/tag to the source chunk."""
    selected_correct = normalize_bool(item.get("selected_correct"))
    relationship_correct = normalize_bool(item.get("relationship_correct"))
    source_selected = original_selected(chunk)
    suggested_selected = normalize_bool(item.get("suggested_selected"), default=source_selected)

    # If selected=false is correct, relation fields are irrelevant to the KG
    # connection result and should not create a conflict.
    if source_selected is False and selected_correct is True:
        return None

    # Keep the judge schema internally consistent even when the model emits
    # contradictory booleans and suggestions.
    if selected_correct:
        suggested_selected = source_selected
    elif suggested_selected == source_selected:
        suggested_selected = not source_selected

    issue_type = issue_type_from_flags(selected_correct, relationship_correct, suggested_selected)

    primary = item.get("suggested_primary_relation")
    if primary is not None:
        primary = str(primary).strip() or None

    relations = item.get("suggested_relations")
    if not isinstance(relations, list):
        relations = [] if relations in (None, "") else [relations]
    suggested_relations = [str(relation).strip() for relation in relations if str(relation).strip()]

    # Primary relation is informational here. A primary-only disagreement is not
    # a conflict when the relation set itself is unchanged.
    if (
        source_selected is True
        and selected_correct is True
        and relationship_correct is False
        and relation_set(suggested_relations) == relation_set(original_relations(chunk))
    ):
        return None

    return {
        "rank": chunk.get("rank"),
        "issue_type": issue_type,
        "rerank_tag": chunk.get("rerank_tag"),
        "original_selected": source_selected,
        "original_primary_relation": original_primary_relation(chunk),
        "original_relations": original_relations(chunk),
        "selected_correct": selected_correct,
        "relationship_correct": relationship_correct,
        "suggested_selected": suggested_selected,
        "suggested_primary_relation": primary,
        "suggested_relations": suggested_relations,
        "comments": str(item.get("comments") or "").strip(),
    }


def validate_response(response: dict[str, Any], chunks_by_rank: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = response.get("conflicts")
    if not isinstance(conflicts, list):
        raise ValueError("LLM response field 'conflicts' must be a list.")

    normalized_conflicts: list[dict[str, Any]] = []
    for item in conflicts:
        if not isinstance(item, dict):
            raise ValueError("Each conflict item must be an object.")
        rank = item.get("rank")
        chunk = chunks_by_rank.get(rank) or chunks_by_rank.get(str(rank))
        if chunk is None:
            raise ValueError(f"Conflict references unknown rank: {rank}")
        normalized = normalize_conflict_item(item, chunk)
        if normalized is None:
            continue
        if normalized["issue_type"] not in ISSUE_TYPES:
            raise ValueError("Invalid support/suspect issue_type.")
        normalized_conflicts.append(normalized)
    return normalized_conflicts


class SupportSuspectJudgeAgent:
    """Judge support/suspect chunks for selected and relationship extraction errors."""

    def __init__(
        self,
        backend: str = LLM_BACKEND,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        json_mode: bool = LLM_JSON_MODE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        prompt_template: str | None = None,
        llm: Any | None = None,
    ) -> None:
        self.prompt_template = prompt_template if prompt_template is not None else load_prompt()
        self.max_retries = max(1, max_retries)
        self.llm = llm or get_llm_backend(
            backend=backend,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
        )

    def judge_chunk(
        self,
        query_text: str,
        chunk: dict[str, Any],
        query_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        conflicts = self.judge_chunks(query_text, [chunk], query_fields=query_fields)
        return conflicts[0] if conflicts else None

    def judge_chunks(
        self,
        query_text: str,
        chunks: list[dict[str, Any]],
        query_fields: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        chunks = [
            chunk
            for chunk in chunks
            if str(chunk.get("rerank_tag") or "").strip().lower() in SUPPORT_SUSPECT_TAGS
        ]
        if not chunks:
            return []

        payload = group_payload(query_text, chunks, query_fields=query_fields)
        chunks_by_rank: dict[Any, dict[str, Any]] = {}
        for chunk in chunks:
            chunks_by_rank[chunk.get("rank")] = chunk
            chunks_by_rank[str(chunk.get("rank"))] = chunk

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                prompt = build_prompt(self.prompt_template, payload)
                content = self._invoke_model(prompt)
                return validate_response(extract_json_object(content), chunks_by_rank)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)

        raise RuntimeError(f"Support/suspect judge failed after {self.max_retries} attempts: {last_error}")

    def judge_chunk_legacy(
        self,
        query_text: str,
        chunk: dict[str, Any],
        query_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        tag = str(chunk.get("rerank_tag") or "").strip().lower()
        if tag not in SUPPORT_SUSPECT_TAGS:
            return None

        payload = chunk_payload(query_text, chunk, query_fields=query_fields)
        chunks_by_rank = {chunk.get("rank"): chunk, str(chunk.get("rank")): chunk}
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                prompt = build_prompt(self.prompt_template, payload)
                content = self._invoke_model(prompt)
                conflicts = validate_response(extract_json_object(content), chunks_by_rank)
                return conflicts[0] if conflicts else None
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)

        raise RuntimeError(f"Support/suspect judge failed after {self.max_retries} attempts: {last_error}")

    @traceable(
        run_type="llm",
        name="support_suspect_judge_llm_call",
        tags=["verification", "support_suspect", "judge"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def _invoke_model(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
