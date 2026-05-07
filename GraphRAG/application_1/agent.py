from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from DocLLM.LLMs.llm_init import get_llm_backend

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

try:
    from GraphRAG.Extraction.chunk_selection_agent import (
        MAX_PROMPT_INPUT_TOKENS,
        estimate_json_tokens,
        extract_json,
        json_safe,
        normalize_text,
        trim_text,
        trim_text_by_tokens,
    )
except ImportError:  # pragma: no cover - supports unusual direct execution
    from ..Extraction.chunk_selection_agent import (
        MAX_PROMPT_INPUT_TOKENS,
        estimate_json_tokens,
        extract_json,
        json_safe,
        normalize_text,
        trim_text,
        trim_text_by_tokens,
    )


LANGSMITH_PROJECT_NAME = "GraphRAGApplication1CausalInference"

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "causal_inference_prompt.txt"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/gpt-5.2")
LLM_TEMPERATURE = 0.0
LLM_JSON_MODE = True
MAX_CHUNK_TEXT_LENGTH = int(os.getenv("GRAPHRAG_APP1_MAX_CHUNK_TEXT_LENGTH", "1800"))
USE_PLACEHOLDER_LLM = os.getenv("GRAPHRAG_APP1_USE_PLACEHOLDER", "0").lower() in {
    "1",
    "true",
    "yes",
}


def load_prompt_template(path: Path = PROMPT_TEMPLATE_PATH) -> str:
    """Load the causal-inference prompt template from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_output_schema() -> dict[str, Any]:
    """Return the expected possible-cause response schema."""

    return {
        "analysis_id": "string",
        "failure_element": "string",
        "function_text": "string",
        "mode_text": "string",
        "possible_causes": [
            {
                "cause": "string copied from one cause candidate",
                "supporting_chunks": [
                    {
                        "name": "string",
                        "reason": "short text-grounded reason",
                    }
                ],
                "reason": "short explanation why this cause is possible",
            }
        ],
    }


def build_agent_payload(
    inference_item: dict[str, Any],
    max_chunk_text_length: int = MAX_CHUNK_TEXT_LENGTH,
    max_prompt_input_tokens: int = MAX_PROMPT_INPUT_TOKENS,
) -> dict[str, Any]:
    """Build the prompt payload for one possible-cause inference item."""

    payload = {
        "analysis_id": normalize_text(inference_item.get("analysis_id")),
        "failure_element": normalize_text(inference_item.get("failure_element")),
        "function_text": normalize_text(inference_item.get("function_text")),
        "mode_text": normalize_text(inference_item.get("mode_text")),
        "cause_candidates": [
            normalize_text(item)
            for item in inference_item.get("cause_candidates", [])
            if normalize_text(item)
        ],
        "chunks": [
            normalize_llm_chunk(chunk, max_chunk_text_length)
            for chunk in inference_item.get("selection_chunks", [])
            if isinstance(chunk, dict)
        ],
    }
    return enforce_selection_prompt_input_token_limit(payload, max_prompt_input_tokens)


def normalize_llm_chunk(chunk: dict[str, Any], max_text_length: int) -> dict[str, str]:
    """Keep only chunk fields sent to the LLM."""

    return {
        "name": normalize_text(chunk.get("name")),
        "text": trim_text(chunk.get("raw text") or chunk.get("text"), max_text_length),
    }


def enforce_selection_prompt_input_token_limit(
    payload: dict[str, Any],
    max_prompt_input_tokens: int,
) -> dict[str, Any]:
    """Limit selected chunks placed into the LLM prompt."""

    if max_prompt_input_tokens <= 0:
        return payload

    selection_chunks = list(payload.get("chunks", []))
    limited_payload = dict(payload)
    limited_payload["chunks"] = []

    for chunk in selection_chunks:
        trial_chunk = dict(chunk)
        trial_payload = dict(limited_payload)
        trial_payload["chunks"] = limited_payload["chunks"] + [trial_chunk]
        if estimate_json_tokens(trial_payload) <= max_prompt_input_tokens:
            limited_payload["chunks"].append(trial_chunk)
            continue

        remaining_tokens = max_prompt_input_tokens - estimate_json_tokens(limited_payload)
        if remaining_tokens <= 80:
            break

        trial_chunk["text"] = trim_text_by_tokens(trial_chunk.get("text", ""), remaining_tokens)
        trial_payload["chunks"] = limited_payload["chunks"] + [trial_chunk]
        if estimate_json_tokens(trial_payload) <= max_prompt_input_tokens:
            limited_payload["chunks"].append(trial_chunk)
        break

    limited_payload["prompt_input_token_threshold"] = max_prompt_input_tokens
    limited_payload["prompt_input_token_estimate"] = estimate_json_tokens(limited_payload)
    return limited_payload


@traceable(
    run_type="prompt",
    name="graphrag_application_1_causal_inference_prompt_builder",
    tags=["graphrag", "application_1", "causal-inference", "prompt"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def build_prompt(payload: dict[str, Any], template_text: str) -> str:
    """Render the causal-inference prompt."""

    return template_text.format(
        output_schema_json=json.dumps(build_output_schema(), indent=2, ensure_ascii=False),
        payload_json=json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
    )


class CausalInferenceAgent:
    """LLM agent that selects possible causes for one failure mode."""

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

    def infer_causal_chain(self, inference_item: dict[str, Any]) -> dict[str, Any]:
        """Infer possible causes for one mode from selected chunks."""

        payload = build_agent_payload(inference_item)
        if self.use_placeholder:
            return self._placeholder_response(payload)

        prompt = build_prompt(payload, self.prompt_template_text)
        content = self._invoke_model(prompt)
        return normalize_inference_response(extract_json(content), payload)

    @traceable(
        run_type="llm",
        name="graphrag_application_1_causal_inference_llm_call",
        tags=["graphrag", "application_1", "causal-inference", "llm"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def _invoke_model(self, prompt: str) -> str:
        """Execute the real LLM call."""

        if self.llm is None:
            raise RuntimeError("LLM client is not initialized.")

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _placeholder_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Offline fallback using lexical overlap with selected evidence chunks."""

        mode_text = normalize_text(payload.get("mode_text"))
        chunks = payload.get("chunks", [])
        possible_causes: list[dict[str, Any]] = []
        for cause_text in payload.get("cause_candidates", []):
            cause_text = normalize_text(cause_text)
            strength = lexical_link_strength(cause_text, mode_text, chunks)
            if strength not in {"moderate", "strong"}:
                continue
            possible_causes.append(
                {
                    "cause": cause_text,
                    "supporting_chunks": [
                        {
                            "name": chunk.get("name", ""),
                            "reason": "Placeholder lexical match with this selected chunk.",
                        }
                        for chunk in chunks[:3]
                    ],
                    "reason": "Placeholder lexical check found moderate overlap; use the LLM for real cause inference.",
                }
            )
        return normalize_inference_response(
            {
                "analysis_id": payload.get("analysis_id", ""),
                "failure_element": payload.get("failure_element", ""),
                "function_text": payload.get("function_text", ""),
                "mode_text": mode_text,
                "possible_causes": possible_causes,
            },
            payload,
        )


def normalize_inference_response(response: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize an LLM response to the causal-inference output contract."""

    return {
        "analysis_id": normalize_text(response.get("analysis_id") or payload.get("analysis_id")),
        "failure_element": normalize_text(response.get("failure_element") or payload.get("failure_element")),
        "function_text": normalize_text(response.get("function_text") or payload.get("function_text")),
        "mode_text": normalize_text(response.get("mode_text") or payload.get("mode_text")),
        "possible_causes": normalize_possible_causes(response.get("possible_causes"), payload),
    }


def normalize_possible_causes(value: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize possible-cause rows."""

    rows = value if isinstance(value, list) else []
    known_causes = {normalize_text(item) for item in payload.get("cause_candidates", [])}

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        cause = normalize_text(row.get("cause") or row.get("cause_text"))
        if cause not in known_causes or cause in seen:
            continue
        seen.add(cause)
        normalized.append(
            {
                "cause": cause,
                "supporting_chunks": normalize_supporting_chunks(row.get("supporting_chunks"), payload),
                "reason": normalize_text(row.get("reason")),
            }
        )
    return normalized


def normalize_supporting_chunks(value: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize supporting chunk references and keep only known chunks."""

    chunks = value if isinstance(value, list) else []
    known_chunks = build_selected_chunk_lookup(payload.get("chunks", []))
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in chunks:
        if not isinstance(item, dict):
            continue
        name = normalize_text(item.get("name"))
        if name not in known_chunks or name in seen:
            continue
        seen.add(name)
        normalized.append(
            {
                "name": name,
                "reason": normalize_text(item.get("reason")),
            }
        )
    return normalized


def build_selected_chunk_lookup(chunks: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build lookup keys for selected chunks."""

    lookup: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        name = normalize_text(chunk.get("name"))
        if name:
            lookup[name] = chunk
    return lookup


def lexical_link_strength(left_text: str, right_text: str, chunks: Sequence[dict[str, Any]]) -> str:
    """Simple placeholder signal for offline runs."""

    left_terms = content_terms(left_text)
    right_terms = content_terms(right_text)
    if not left_terms or not right_terms:
        return "none"
    evidence_text = " ".join(normalize_text(chunk.get("text")).lower() for chunk in chunks)
    left_hits = sum(1 for term in left_terms if term in evidence_text)
    right_hits = sum(1 for term in right_terms if term in evidence_text)
    if left_hits >= 2 and right_hits >= 2:
        return "moderate"
    if left_hits >= 1 and right_hits >= 1:
        return "weak"
    return "none"


def content_terms(text: str) -> set[str]:
    """Extract non-trivial lexical terms."""

    stopwords = {"the", "and", "for", "with", "due", "too", "has", "not", "can", "into"}
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", normalize_text(text).lower())
        if len(term) > 2 and term not in stopwords
    }


def estimate_payload_tokens(payload: dict[str, Any]) -> int:
    """Expose the shared lightweight token estimate for diagnostics."""

    return estimate_json_tokens(payload)
