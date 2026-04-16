from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from .llm_init import configure_langsmith, get_llm_backend

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


LANGSMITH_PROJECT_NAME = configure_langsmith()

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "step1_function_mode_prompt.txt"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/gpt-4.1")
LLM_TEMPERATURE = 0.0
LLM_JSON_MODE = True
USE_PLACEHOLDER_LLM = os.getenv("DOC_LLM_USE_PLACEHOLDER", "0").lower() in {"1", "true", "yes"}


def load_prompt_template(path: Path = PROMPT_TEMPLATE_PATH) -> str:
    """Load the Step 1 prompt template from disk."""

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
    """Return the Step 1 response schema."""

    return {
        "failure_element": "string",
        "query_function": "string",
        "mode_assessments": [
            {
                "mode_text": "string",
                "selected": True,
                "supporting_sentence_ids": ["sentence_id"],
                "support_level": "explicit | implicit | mixed",
                "confidence": "high | medium | low",
                "reason": "short engineering explanation",
                "evidence_spans": ["short verbatim span"],
            }
        ],
        "global_reasoning": "short summary",
    }


@traceable(
    run_type="prompt",
    name="docllm_step1_prompt_builder",
    tags=["docllm", "step1", "prompt"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def build_prompt(payload: dict[str, Any], template_text: str) -> str:
    """Render the Step 1 prompt for a grouped function payload."""

    return template_text.format(
        output_schema_json=json.dumps(build_output_schema(), indent=2, ensure_ascii=False),
        payload_json=json.dumps(payload, indent=2, ensure_ascii=False),
    )


class StepOneLLMRunner:
    """Encapsulates Step 1 prompt construction and LLM invocation."""

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

    def infer_modes_for_function(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Build prompt, call the model, and parse its output."""

        prompt = build_prompt(payload, self.prompt_template_text)

        if self.use_placeholder:
            return self._placeholder_response(payload)

        content = self._invoke_model(prompt)
        return extract_json(content)

    @traceable(
        run_type="llm",
        name="docllm_step1_llm_call",
        tags=["docllm", "step1", "llm"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def _invoke_model(self, prompt: str) -> str:
        """Execute the real Step 1 LLM call."""

        if self.llm is None:
            raise RuntimeError("LLM client is not initialized.")

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _placeholder_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Heuristic local fallback for offline execution."""

        candidate_modes = payload.get("candidate_modes", [])
        sentences = payload.get("sentences", [])
        mode_assessments: list[dict[str, Any]] = []

        mode_keywords = {
            "Component break-down": ["failure", "stress", "protect", "surge", "overvoltage", "overload"],
            "Unbalanced motor currents": ["unbalanced", "balanced currents", "current", "currents"],
            "Incorrect interpretation zero-crossing": ["zero crossing", "interrupt", "false triggering", "edge"],
            "Soft start too long": ["soft start", "not finished in time", "duration", "10 seconds"],
            "No detection": ["no detection", "not detect", "miss", "interrupt"],
            "Welded relay": ["relay", "weld", "live switching"],
            "Relay cannot close": ["relay", "close", "release time", "open the relays"],
            "False turn-on / turn-off": ["jitter", "unstable transitions", "switching", "false", "noise"],
        }

        for mode_text in candidate_modes:
            keywords = mode_keywords.get(mode_text, [])
            supporting_sentence_ids: list[str] = []
            evidence_spans: list[str] = []
            support_kinds: set[str] = set()

            for sentence in sentences:
                match_text = normalize_text(sentence.get("match_text", "")).lower()
                usefulness = normalize_text(sentence.get("usefulness", "")).lower()
                if any(keyword in match_text for keyword in keywords):
                    supporting_sentence_ids.append(str(sentence.get("sentence_id", "")))
                    if usefulness in {"explicit", "implicit"}:
                        support_kinds.add(usefulness)
                    evidence_spans.extend(normalize_string_list(sentence.get("key_evidence_spans", []))[:2])
                    if not evidence_spans:
                        evidence_spans.append(trim_text(sentence.get("match_text", ""), 160))

            if supporting_sentence_ids:
                support_level = "mixed" if len(support_kinds) > 1 else (next(iter(support_kinds)) if support_kinds else "implicit")
                mode_assessments.append(
                    {
                        "mode_text": mode_text,
                        "selected": True,
                        "supporting_sentence_ids": unique_preserve_order(supporting_sentence_ids),
                        "support_level": support_level,
                        "confidence": "medium" if len(supporting_sentence_ids) > 1 else "low",
                        "reason": "Heuristic placeholder selected this mode from sentence-level keyword evidence.",
                        "evidence_spans": unique_preserve_order([span for span in evidence_spans if span])[:4],
                    }
                )
            else:
                mode_assessments.append(
                    {
                        "mode_text": mode_text,
                        "selected": False,
                        "supporting_sentence_ids": [],
                        "support_level": "implicit",
                        "confidence": "low",
                        "reason": "No sufficient sentence-level support was found for this candidate mode.",
                        "evidence_spans": [],
                    }
                )

        return {
            "failure_element": payload.get("failure_element", ""),
            "query_function": payload.get("query_function", ""),
            "mode_assessments": mode_assessments,
            "global_reasoning": "Placeholder inference based on candidate-mode keyword overlap in Step 0 evidence.",
        }


def normalize_text(value: Any) -> str:
    """Normalize values to safe strings."""

    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def normalize_string_list(value: Any) -> list[str]:
    """Normalize a value to a list of non-empty strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    if isinstance(value, str):
        normalized = normalize_text(value)
        return [normalized] if normalized else []
    return []


def trim_text(text: Any, max_length: int) -> str:
    """Trim long text for placeholder evidence."""

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
