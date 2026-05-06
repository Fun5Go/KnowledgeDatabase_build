from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from DocLLM.LLMs.llm_init import get_llm_backend

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

try:
    from .chunk_selection_agent import (
        MAX_PROMPT_INPUT_TOKENS,
        build_candidate_lookup,
        enforce_prompt_input_token_limit,
        estimate_json_tokens,
        extract_json,
        json_safe,
        normalize_int,
        normalize_text,
        trim_text,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from chunk_selection_agent import (
        MAX_PROMPT_INPUT_TOKENS,
        build_candidate_lookup,
        enforce_prompt_input_token_limit,
        estimate_json_tokens,
        extract_json,
        json_safe,
        normalize_int,
        normalize_text,
        trim_text,
    )


LANGSMITH_PROJECT_NAME = "QdDetectionControlSelection"

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "qd_detection_control_prompt.txt"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/gpt-5.2")
LLM_TEMPERATURE = 0.0
LLM_JSON_MODE = True
USE_PLACEHOLDER_LLM = os.getenv("GRAPHRAG_EXTRACTION_USE_PLACEHOLDER", "0").lower() in {
    "1",
    "true",
    "yes",
}


def load_prompt_template(path: Path = PROMPT_TEMPLATE_PATH) -> str:
    """Load the QD detection-control prompt template from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_output_schema() -> dict[str, Any]:
    """Return the expected QD agent response schema."""

    return {
        "analysis_id": "string",
        "query_type": "function_mode | cause",
        "selected_qd_chunks": [
            {
                "rank": "integer retrieval rank copied from the selected candidate chunk",
                "label": "QDChunk",
                "name": "string",
                "qd_id": "string",
                "qd_title": "string",
                "objectives": "string copied from the selected candidate chunk objectives",
                "support_capability": "weak | moderate | strong",
                "reason": "short explanation of why this is a detection control",
            }
        ],
    }


def build_agent_payload(
    query_result: dict[str, Any],
    analysis_item: dict[str, Any] | None = None,
    max_chunk_text_length: int = 1800,
    max_prompt_input_tokens: int = MAX_PROMPT_INPUT_TOKENS,
) -> dict[str, Any]:
    """Build the LLM input from the QD-only GraphRAG query result."""

    analysis_item = analysis_item or {}
    payload = {
        "analysis_id": normalize_text(analysis_item.get("analysis_id")),
        "query_type": normalize_text(analysis_item.get("query_type")),
        "query": build_llm_query_payload(analysis_item),
        "candidate_chunks": [
            normalize_candidate_chunk(item, index, max_chunk_text_length)
            for index, item in enumerate(query_result.get("evidence", []), start=1)
        ],
    }
    return enforce_prompt_input_token_limit(payload, max_prompt_input_tokens)


def build_llm_query_payload(analysis_item: dict[str, Any]) -> dict[str, str]:
    """Build the compact target payload shown to the LLM."""

    query_type = normalize_text(analysis_item.get("query_type"))
    if query_type == "cause":
        return {
            "Failure cause": normalize_text(analysis_item.get("query_cause")),
            "Discipline": normalize_text(analysis_item.get("cause_discipline")),
            "Failure element": normalize_text(analysis_item.get("failure_element")),
        }

    return {
        "Function": normalize_text(analysis_item.get("function_text")),
        "Failure mode": normalize_text(analysis_item.get("query_mode")),
        "Failure element": normalize_text(analysis_item.get("failure_element")),
    }


def normalize_candidate_chunk(
    item: dict[str, Any],
    retrieval_rank: int,
    max_text_length: int,
) -> dict[str, Any]:
    """Keep only stable QD candidate fields for the prompt."""

    return {
        "retrieval rank": retrieval_rank,
        "label": normalize_text(item.get("label")),
        "name": normalize_text(item.get("name")),
        "section_tag": normalize_text(item.get("section_tag")),
        "qd_id": normalize_text(item.get("qd_id")),
        "qd_title": normalize_text(item.get("qd_title")),
        "objectives": trim_text(item.get("objectives") or item.get("text"), max_text_length),
        "text": trim_text(item.get("text"), max_text_length),
    }


@traceable(
    run_type="prompt",
    name="graphrag_qd_detection_control_prompt_builder",
    tags=["graphrag", "extraction", "qd", "detection-control", "prompt"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def build_prompt(payload: dict[str, Any], template_text: str) -> str:
    """Render the QD detection-control prompt."""

    return template_text.format(
        output_schema_json=json.dumps(build_output_schema(), indent=2, ensure_ascii=False),
        payload_json=json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
    )


class QDDetectionControlAgent:
    """LLM agent that selects one QD chunk as the detection control."""

    def __init__(
        self,
        backend: str = LLM_BACKEND,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        json_mode: bool = LLM_JSON_MODE,
        use_placeholder: bool = USE_PLACEHOLDER_LLM,
        prompt_template_path: Path = PROMPT_TEMPLATE_PATH,
    ) -> None:
        self.prompt_template_text = load_prompt_template(prompt_template_path)
        self.use_placeholder = use_placeholder
        self.llm = None if use_placeholder else get_llm_backend(
            backend=backend,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
        )

    def select_detection_control(
        self,
        query_result: dict[str, Any],
        analysis_item: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select up to three detection-control QD chunks from a QD-only query result."""

        payload = build_agent_payload(query_result=query_result, analysis_item=analysis_item)
        if self.use_placeholder:
            return self._placeholder_response(payload)

        prompt = build_prompt(payload, self.prompt_template_text)
        content = self._invoke_model(prompt)
        return normalize_selection_response(extract_json(content), payload)

    @traceable(
        run_type="llm",
        name="graphrag_qd_detection_control_llm_call",
        tags=["graphrag", "extraction", "qd", "detection-control", "llm"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def _invoke_model(self, prompt: str) -> str:
        """Execute the real LLM call."""

        if self.llm is None:
            raise RuntimeError("LLM client is not initialized.")

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _placeholder_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Offline fallback that selects the highest-ranked QD candidate."""

        candidates = payload.get("candidate_chunks", [])
        selected = candidates[0] if candidates else None
        return {
            "analysis_id": normalize_text(payload.get("analysis_id")),
            "query_type": normalize_text(payload.get("query_type")),
            "selected_qd_chunks": [
                normalize_selected_candidate(
                    item,
                    support_capability="strong" if index == 1 else "moderate",
                    reason="Placeholder selected this QD retrieval result by rank.",
                )
                for index, item in enumerate(candidates[:3], start=1)
            ],
        }


def normalize_selection_response(response: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize an LLM response to up to three selected QD chunks."""

    candidate_by_key = build_candidate_lookup(payload)
    selected_items = response.get("selected_qd_chunks")
    if selected_items is None:
        selected_items = response.get("top_chunks") or response.get("selected_chunks")
    if selected_items is None and isinstance(response.get("selected_qd_chunk"), dict):
        selected_items = [response.get("selected_qd_chunk")]
    if not isinstance(selected_items, list):
        selected_items = []

    normalized_selected: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    for selected in selected_items:
        if not isinstance(selected, dict):
            continue
        rank = normalize_int(selected.get("rank"))
        if rank is None:
            rank = normalize_int(selected.get("retrieval rank"))
        candidate = candidate_by_key.get(("rank", normalize_text(rank))) if rank is not None else None
        if candidate is None:
            candidate = candidate_by_key.get(
                (
                    "label_name",
                    normalize_text(selected.get("label")),
                    normalize_text(selected.get("name")),
                )
            )
        if candidate is None:
            continue
        candidate_rank = normalize_int(candidate.get("retrieval rank"))
        if candidate_rank is None or candidate_rank in seen_ranks:
            continue
        seen_ranks.add(candidate_rank)
        normalized_selected.append(
            normalize_selected_candidate(
                candidate,
                support_capability=normalize_support_capability(selected.get("support_capability")),
                reason=normalize_text(selected.get("reason")),
            )
        )
        if len(normalized_selected) >= 3:
            break

    return {
        "analysis_id": normalize_text(response.get("analysis_id") or payload.get("analysis_id")),
        "query_type": normalize_text(response.get("query_type") or payload.get("query_type")),
        "selected_qd_chunks": normalized_selected,
    }


def normalize_selected_candidate(
    candidate: dict[str, Any],
    support_capability: str,
    reason: str,
) -> dict[str, Any]:
    """Build the selected QD output object from a candidate."""

    return {
        "rank": normalize_int(candidate.get("retrieval rank")) or 0,
        "label": normalize_text(candidate.get("label")),
        "name": normalize_text(candidate.get("name")),
        "qd_id": normalize_text(candidate.get("qd_id")),
        "qd_title": normalize_text(candidate.get("qd_title")),
        "objectives": normalize_text(candidate.get("objectives") or candidate.get("text")),
        "support_capability": normalize_support_capability(support_capability),
        "reason": normalize_text(reason),
    }


def normalize_support_capability(value: Any) -> str:
    """Normalize support labels."""

    capability = normalize_text(value).lower()
    if capability in {"weak", "moderate", "strong"}:
        return capability
    return "weak"


def estimate_payload_tokens(payload: dict[str, Any]) -> int:
    """Expose the shared lightweight token estimate for diagnostics."""

    return estimate_json_tokens(payload)
