from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from DocLLM.LLMs.llm_init import configure_langsmith, get_llm_backend

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


LANGSMITH_PROJECT_NAME = configure_langsmith()

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "chunk_selection_prompt.txt"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/gpt-4.1")
LLM_TEMPERATURE = 0.0
LLM_JSON_MODE = True
MAX_PROMPT_INPUT_TOKENS = int(os.getenv("GRAPHRAG_EXTRACTION_MAX_INPUT_TOKENS", "4000"))
USE_PLACEHOLDER_LLM = os.getenv("GRAPHRAG_EXTRACTION_USE_PLACEHOLDER", "0").lower() in {
    "1",
    "true",
    "yes",
}


def load_prompt_template(path: Path = PROMPT_TEMPLATE_PATH) -> str:
    """Load the chunk-selection prompt template from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from an LLM response."""

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


def build_output_schema() -> dict[str, Any]:
    """Return the expected agent response schema."""

    return {
        "analysis_id": "string",
        "query_type": "function_mode | cause",
        "top_chunks": [
            {
                "rank": "integer rank among selected chunks",
                "label": "string",
                "name": "string",
                "support_capability": "weak | moderate | strong",
                "reason": "short explanation",
            }
        ],
    }


def build_agent_payload(
    query_result: dict[str, Any],
    analysis_item: dict[str, Any] | None = None,
    max_chunk_text_length: int = 1800,
    max_prompt_input_tokens: int = MAX_PROMPT_INPUT_TOKENS,
) -> dict[str, Any]:
    """Build the LLM input from the GraphRAG query result."""

    analysis_item = analysis_item or {}
    evidence = query_result.get("evidence", [])
    query_type = normalize_text(analysis_item.get("query_type"))

    payload = {
        "analysis_id": normalize_text(analysis_item.get("analysis_id")),
        "query_type": query_type,
        "query": build_llm_query_payload(analysis_item),
        "candidate_chunks": [
            normalize_candidate_chunk(item, index, max_chunk_text_length)
            for index, item in enumerate(evidence, start=1)
        ],
    }
    return enforce_prompt_input_token_limit(payload, max_prompt_input_tokens)


def build_llm_query_payload(analysis_item: dict[str, Any]) -> dict[str, str]:
    """Build the compact query payload shown to the LLM."""

    query_type = normalize_text(analysis_item.get("query_type"))
    if query_type == "cause":
        return {
            "Failure cause": normalize_text(analysis_item.get("query_cause")),
            "Discipline": normalize_text(analysis_item.get("cause_discipline")),
        }

    return {
        "Function": normalize_text(analysis_item.get("function_text")),
        "Failure mode": normalize_text(analysis_item.get("query_mode")),
    }


def enforce_prompt_input_token_limit(
    payload: dict[str, Any],
    max_prompt_input_tokens: int,
) -> dict[str, Any]:
    """Limit the query payload placed into the LLM prompt."""

    if max_prompt_input_tokens <= 0:
        return payload

    candidate_chunks = list(payload.get("candidate_chunks", []))
    limited_payload = dict(payload)
    limited_payload["candidate_chunks"] = []

    for chunk in candidate_chunks:
        trial_chunk = dict(chunk)
        trial_payload = dict(limited_payload)
        trial_payload["candidate_chunks"] = limited_payload["candidate_chunks"] + [trial_chunk]
        trial_tokens = estimate_json_tokens(trial_payload)
        if trial_tokens <= max_prompt_input_tokens:
            limited_payload["candidate_chunks"].append(trial_chunk)
            continue

        remaining_tokens = max_prompt_input_tokens - estimate_json_tokens(limited_payload)
        if remaining_tokens <= 80:
            break

        trial_chunk["text"] = trim_text_by_tokens(trial_chunk.get("text", ""), remaining_tokens)
        trial_payload["candidate_chunks"] = limited_payload["candidate_chunks"] + [trial_chunk]
        if estimate_json_tokens(trial_payload) <= max_prompt_input_tokens:
            limited_payload["candidate_chunks"].append(trial_chunk)
        break

    limited_payload["prompt_input_token_threshold"] = max_prompt_input_tokens
    limited_payload["prompt_input_token_estimate"] = estimate_json_tokens(limited_payload)
    return limited_payload


def normalize_candidate_chunk(
    item: dict[str, Any],
    rank: int,
    max_text_length: int,
) -> dict[str, Any]:
    """Keep only stable, prompt-friendly candidate fields."""

    return {
        "rank": rank,
        "label": normalize_text(item.get("label")),
        "name": normalize_text(item.get("name")),
        "section_tag": normalize_text(item.get("section_tag")),
        "text": trim_text(item.get("text"), max_text_length),
    }


@traceable(
    run_type="prompt",
    name="graphrag_chunk_selection_prompt_builder",
    tags=["graphrag", "extraction", "chunk-selection", "prompt"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def build_prompt(payload: dict[str, Any], template_text: str) -> str:
    """Render the chunk-selection prompt."""

    return template_text.format(
        output_schema_json=json.dumps(build_output_schema(), indent=2, ensure_ascii=False),
        payload_json=json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
    )


class ChunkSelectionAgent:
    """LLM agent that selects retrieved chunks supporting a query mode/cause."""

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

    def select_chunks(
        self,
        query_result: dict[str, Any],
        analysis_item: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select chunks from a GraphRAG query result."""

        payload = build_agent_payload(query_result=query_result, analysis_item=analysis_item)
        if self.use_placeholder:
            return self._placeholder_response(payload)

        prompt = build_prompt(payload, self.prompt_template_text)
        content = self._invoke_model(prompt)
        return normalize_selection_response(extract_json(content), payload)

    @traceable(
        run_type="llm",
        name="graphrag_chunk_selection_llm_call",
        tags=["graphrag", "extraction", "chunk-selection", "llm"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def _invoke_model(self, prompt: str) -> str:
        """Execute the real LLM call."""

        if self.llm is None:
            raise RuntimeError("LLM client is not initialized.")

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _placeholder_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Offline fallback that selects the highest ranked chunks."""

        candidates = payload.get("candidate_chunks", [])

        return {
            "analysis_id": normalize_text(payload.get("analysis_id")),
            "query_type": payload.get("query_type", ""),
            "top_chunks": [
                {
                    "rank": index,
                    "label": item.get("label", ""),
                    "name": item.get("name", ""),
                    "support_capability": placeholder_support_capability(index),
                    "reason": "Placeholder selected this chunk from the highest-ranked retrieval results.",
                }
                for index, item in enumerate(candidates, start=1)
            ],
        }


def normalize_selection_response(response: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the LLM response to the required selected-chunk output contract."""

    chunks = response.get("top_chunks") or response.get("selected_chunks") or []
    if not isinstance(chunks, list):
        chunks = []

    normalized_chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        capability = normalize_text(chunk.get("support_capability")).lower()
        if capability not in {"weak", "moderate", "strong"}:
            capability = "weak"

        normalized_chunks.append(
            {
                "rank": index,
                "label": normalize_text(chunk.get("label")),
                "name": normalize_text(chunk.get("name")),
                "support_capability": capability,
                "reason": normalize_text(chunk.get("reason")),
            }
        )

    return {
        "analysis_id": normalize_text(
            response.get("analysis_id") or payload.get("analysis_id")
        ),
        "query_type": normalize_text(response.get("query_type") or payload.get("query_type")),
        "top_chunks": normalized_chunks,
    }


def placeholder_support_capability(rank: int) -> str:
    """Assign simple support labels for offline placeholder output."""

    if rank == 1:
        return "strong"
    if rank == 2:
        return "moderate"
    return "weak"


def normalize_text(value: Any) -> str:
    """Normalize values to safe strings."""

    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def estimate_json_tokens(value: Any) -> int:
    """Estimate tokens for a JSON payload without requiring a tokenizer package."""

    text = json.dumps(json_safe(value), ensure_ascii=False)
    return estimate_text_tokens(text)


def json_safe(value: Any) -> Any:
    """Convert values such as NumPy scalars into JSON-serializable objects."""

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


def estimate_text_tokens(text: Any) -> int:
    """Approximate token count for prompt budgeting."""

    normalized = normalize_text(text)
    if not normalized:
        return 0
    return max(1, len(re.findall(r"\w+|[^\s\w]", normalized, flags=re.UNICODE)))


def trim_text_by_tokens(text: Any, max_tokens: int) -> str:
    """Trim text using the same lightweight token estimate."""

    normalized = normalize_text(text)
    if estimate_text_tokens(normalized) <= max_tokens:
        return normalized

    approx_chars = max(0, max_tokens * 4)
    if approx_chars <= 3:
        return ""
    return trim_text(normalized, approx_chars)


def trim_text(text: Any, max_length: int) -> str:
    """Trim long text for LLM payloads."""

    normalized = normalize_text(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def unique_preserve_order(items: Sequence[str]) -> list[str]:
    """Deduplicate while preserving order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
