from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from .llm_init import DOC_LLM_LANGSMITH_PROJECT, configure_langsmith, get_llm_backend

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


LANGSMITH_PROJECT_NAME = configure_langsmith()


PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "step0_sentence_value_prompt.txt"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "azure/gpt-4.1")
LLM_TEMPERATURE = 0.0
LLM_JSON_MODE = True
USE_PLACEHOLDER_LLM = os.getenv("DOC_LLM_USE_PLACEHOLDER", "0").lower() in {"1", "true", "yes"}


def load_prompt_template(path: Path = PROMPT_TEMPLATE_PATH) -> str:
    """Load the Step 0 prompt template from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def build_output_schema() -> dict[str, Any]:
    """Return the required Step 0 JSON schema."""

    return {
        "results": [
            {
                "sentence_id": "string",
                "function_text": "string",
                "usefulness": "explicit | implicit | not_useful",
                "reason": "short explanation",
                "key_evidence_spans": ["short supporting quote"],
            }
        ]
    }


@traceable(
    run_type="prompt",
    name="docllm_step0_prompt_builder",
    tags=["docllm", "step0", "prompt"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def build_batch_prompt(batch_payload: Sequence[dict[str, Any]], template_text: str) -> str:
    """Render the Step 0 prompt for one batch."""

    return template_text.format(
        output_schema_json=json.dumps(build_output_schema(), indent=2, ensure_ascii=False),
        batch_payload_json=json.dumps(list(batch_payload), indent=2, ensure_ascii=False),
    )


def parse_llm_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, tolerating wrapper text when necessary."""

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("LLM returned an empty response.")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the LLM response.")

    return json.loads(match.group(0))


class StepZeroLLMRunner:
    """Encapsulates Step 0 prompt construction and model invocation."""

    def __init__(
        self,
        backend: str = LLM_BACKEND,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        json_mode: bool = LLM_JSON_MODE,
        use_placeholder: bool = USE_PLACEHOLDER_LLM,
        prompt_template_path: Path = PROMPT_TEMPLATE_PATH,
    ) -> None:
        self.prompt_template_path = prompt_template_path
        self.prompt_template_text = load_prompt_template(prompt_template_path)
        self.use_placeholder = use_placeholder
        self.llm = None if use_placeholder else get_llm_backend(
            backend=backend,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
        )

    def invoke_batch(self, batch_payload: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Build the prompt, call the LLM, and parse its JSON output."""

        prompt = build_batch_prompt(batch_payload, self.prompt_template_text)

        if self.use_placeholder:
            return self._placeholder_response(batch_payload)

        content = self._invoke_model(prompt)
        return parse_llm_json(content)

    @traceable(
        run_type="llm",
        name="docllm_step0_llm_call",
        tags=["docllm", "step0", "llm"],
        project_name=LANGSMITH_PROJECT_NAME,
    )
    def _invoke_model(self, prompt: str) -> str:
        """Execute the real LLM call and return raw text content."""

        if self.llm is None:
            raise RuntimeError("LLM client is not initialized.")

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def _placeholder_response(self, batch_payload: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Deterministic fallback response for local dry runs without API access."""

        results: list[dict[str, Any]] = []
        for item in batch_payload:
            sentence_id = normalize_text(item.get("sentence_id", ""))
            function_text = normalize_text(item.get("function_text", ""))
            choice_text = normalize_text(item.get("choice_text", ""))
            rationale_text = normalize_text(item.get("rationale_text", ""))
            combined = f"{choice_text} {rationale_text}".lower()

            explicit_terms = [
                "failure",
                "fault",
                "protect",
                "protection",
                "overvoltage",
                "overload",
                "stress",
                "trip",
                "stop the motor",
                "shutdown",
                "emi",
                "surge",
                "reset",
                "headroom",
                "noise",
            ]
            implicit_terms = [
                function_text.lower(),
                "current",
                "voltage",
                "algorithm",
                "measurement",
                "relay",
                "triac",
                "thyristor",
                "timing",
                "soft start",
            ]

            if any(term for term in explicit_terms if term in combined):
                usefulness = "explicit"
                reason = "Contains direct failure, protection, stress, or abnormal-behavior evidence."
            elif any(term for term in implicit_terms if term and term in combined):
                usefulness = "implicit"
                reason = "Provides contextual technical information that may support later FMEA reasoning."
            else:
                usefulness = "not_useful"
                reason = "Provides little function-specific value for downstream FMEA reasoning."

            results.append(
                {
                    "sentence_id": sentence_id,
                    "function_text": function_text,
                    "usefulness": usefulness,
                    "reason": reason,
                    "key_evidence_spans": extract_evidence_spans(
                        source_texts=[choice_text, rationale_text],
                        terms=[function_text, "failure", "fault", "protect", "protection"],
                        max_spans=3,
                    ),
                }
            )

        return {"results": results}


def normalize_text(value: Any) -> str:
    """Normalize values to a clean string."""

    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def extract_evidence_spans(
    source_texts: Sequence[str],
    terms: Sequence[str],
    max_spans: int = 3,
) -> list[str]:
    """Extract short evidence snippets around matched terms for placeholder mode."""

    spans: list[str] = []
    lowered_terms = [term.lower() for term in terms if normalize_text(term)]

    for text in source_texts:
        normalized = normalize_text(text)
        if not normalized:
            continue

        lower_text = normalized.lower()
        for term in lowered_terms:
            position = lower_text.find(term)
            if position < 0:
                continue

            start = max(0, position - 35)
            end = min(len(normalized), position + len(term) + 35)
            snippet = normalized[start:end].strip()
            if snippet and snippet not in spans:
                spans.append(snippet)
            if len(spans) >= max_spans:
                return spans

    return spans
