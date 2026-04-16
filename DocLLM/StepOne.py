from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from LLMs.llm_init import configure_langsmith
from LLMs.one_LLM import StepOneLLMRunner

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


load_dotenv()
LANGSMITH_PROJECT_NAME = configure_langsmith()


# =========================================================
# Configuration
# =========================================================
BASE_DIR = CURRENT_DIR
INPUT_JSON_PATH = BASE_DIR / "step0_results.json"
OUTPUT_JSON_PATH = BASE_DIR / "step1_results.json"

USEFULNESS_EXCLUSION = {"not_useful"}
QUERY_FUNCTIONS: list[str] = []


MOTOR_CONTROL_STRUCTURE = {
    "product_domain": "motor_drives",
    "nodes": [
        {
            "element_id": "E1",
            "failure_element": "Motor control",
            "modes": {
                "Soft starter": [
                    "Component break-down",
                    "Unbalanced motor currents",
                ],
                "Zero-crossing detection": [
                    "Incorrect interpretation zero-crossing",
                    "Soft start too long",
                    "No detection",
                ],
                "Relay switching": [
                    "Welded relay",
                    "Relay cannot close",
                    "False turn-on / turn-off",
                ],
            },
        }
    ],
}


# =========================================================
# Data models
# =========================================================
@dataclass(slots=True)
class StepZeroFilteredItem:
    """Minimal Step 1 input item coming from Step 0 output."""

    sentence_id: str
    failure_element: str
    function_text: str
    usefulness: str
    key_evidence_spans: list[str]
    match_text: str
    section_tag: str = ""
    chunk_name: str = ""
    discipline: str = ""
    level_identification: str = ""

    def to_llm_sentence(self) -> dict[str, Any]:
        """Return the compact sentence payload for Step 1 LLM input."""

        return {
            "sentence_id": self.sentence_id,
            "usefulness": self.usefulness,
            "key_evidence_spans": self.key_evidence_spans,
            "match_text": self.match_text,
        }


@dataclass(slots=True)
class StepOneModeSelection:
    """Normalized selected failure mode result."""

    mode_text: str
    selected: bool
    supporting_sentence_ids: list[str]
    support_level: str
    confidence: str
    reason: str
    evidence_spans: list[str]


@dataclass(slots=True)
class StepOneResult:
    """Final grouped Step 1 result."""

    failure_element: str
    query_function: str
    candidate_modes: list[str]
    sentence_count: int
    mode_assessments: list[StepOneModeSelection]
    global_reasoning: str
    sentences: list[dict[str, Any]] = field(default_factory=list)


# =========================================================
# IO helpers
# =========================================================
def load_json(path: Path) -> Any:
    """Load a JSON file from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: Path) -> None:
    """Write JSON output to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


# =========================================================
# Filtering and grouping
# =========================================================
def normalize_stepzero_items(raw_data: Any) -> list[StepZeroFilteredItem]:
    """Normalize Step 0 output into the minimal Step 1 input structure."""

    if not isinstance(raw_data, list):
        raise ValueError("Step 1 expects Step 0 output JSON to be a list of records.")

    items: list[StepZeroFilteredItem] = []
    for raw_item in raw_data:
        if not isinstance(raw_item, dict):
            continue
        items.append(
            StepZeroFilteredItem(
                sentence_id=normalize_text(raw_item.get("sentence_id", "")),
                failure_element=normalize_text(raw_item.get("failure_element", "")),
                function_text=normalize_text(raw_item.get("function_text", "")),
                usefulness=normalize_text(raw_item.get("usefulness", "")),
                key_evidence_spans=normalize_string_list(raw_item.get("key_evidence_spans", [])),
                match_text=normalize_text(raw_item.get("match_text", "")),
                section_tag=normalize_text(raw_item.get("section_tag", "")),
                chunk_name=normalize_text(raw_item.get("chunk_name", "")),
                discipline=normalize_text(raw_item.get("discipline", "")),
                level_identification=normalize_text(raw_item.get("level_identification", "")),
            )
        )
    return items


def filter_stepzero_items(
    items: Sequence[StepZeroFilteredItem],
    usefulness_exclusion: set[str],
    query_functions: Sequence[str] | None = None,
) -> list[StepZeroFilteredItem]:
    """Filter Step 0 items by usefulness threshold and optional query functions."""

    excluded = {normalize_text(value).lower() for value in usefulness_exclusion}
    normalized_query = {normalize_text(value).lower() for value in (query_functions or []) if normalize_text(value)}

    filtered: list[StepZeroFilteredItem] = []
    for item in items:
        if item.usefulness.lower() in excluded:
            continue
        if normalized_query and item.function_text.lower() not in normalized_query:
            continue
        filtered.append(item)
    return filtered


def build_function_mode_lookup(structure_input: dict[str, Any]) -> dict[str, list[str]]:
    """Map normalized function text to candidate failure modes."""

    lookup: dict[str, list[str]] = {}
    for node in structure_input.get("nodes", []):
        for function_text, mode_list in node.get("modes", {}).items():
            normalized_function = normalize_text(function_text).lower()
            lookup[normalized_function] = [
                normalize_text(mode_text)
                for mode_text in mode_list
                if normalize_text(mode_text)
            ]
    return lookup


def build_grouped_payloads(
    items: Sequence[StepZeroFilteredItem],
    structure_input: dict[str, Any],
) -> list[dict[str, Any]]:
    """Group filtered Step 0 items by failure element and query function."""

    function_mode_lookup = build_function_mode_lookup(structure_input)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for item in items:
        group_key = (item.failure_element, item.function_text)
        candidate_modes = function_mode_lookup.get(item.function_text.lower(), [])
        if not candidate_modes:
            continue

        if group_key not in grouped:
            grouped[group_key] = {
                "failure_element": item.failure_element,
                "query_function": item.function_text,
                "candidate_modes": candidate_modes,
                "sentences": [],
            }

        grouped[group_key]["sentences"].append(item.to_llm_sentence())

    return list(grouped.values())


# =========================================================
# Result normalization
# =========================================================
def normalize_stepone_result(
    parsed_response: dict[str, Any],
    payload: dict[str, Any],
) -> StepOneResult:
    """Validate and normalize one Step 1 LLM response."""

    valid_modes = set(payload.get("candidate_modes", []))
    sentence_lookup = {
        normalize_text(sentence.get("sentence_id", "")): sentence
        for sentence in payload.get("sentences", [])
    }
    valid_support_levels = {"explicit", "implicit", "mixed"}
    valid_confidence = {"high", "medium", "low"}

    raw_items = parsed_response.get("mode_assessments", parsed_response.get("selected_modes", []))
    if not isinstance(raw_items, list):
        raw_items = []

    normalized_modes_map: dict[str, StepOneModeSelection] = {}

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        mode_text = normalize_text(item.get("mode_text", ""))
        if mode_text not in valid_modes:
            continue

        supporting_ids: list[str] = []
        for sentence_id in item.get("supporting_sentence_ids", []):
            normalized_id = normalize_text(sentence_id)
            if normalized_id in sentence_lookup and normalized_id not in supporting_ids:
                supporting_ids.append(normalized_id)

        support_level = normalize_text(item.get("support_level", "implicit")).lower()
        if support_level not in valid_support_levels:
            support_level = "implicit"

        confidence = normalize_text(item.get("confidence", "low")).lower()
        if confidence not in valid_confidence:
            confidence = "low"

        normalized_modes_map[mode_text] = StepOneModeSelection(
                mode_text=mode_text,
                selected=bool(item.get("selected", bool(supporting_ids))),
                supporting_sentence_ids=supporting_ids,
                support_level=support_level,
                confidence=confidence,
                reason=normalize_text(item.get("reason", "")),
                evidence_spans=normalize_string_list(item.get("evidence_spans", [])),
            )

    completed_mode_assessments: list[StepOneModeSelection] = []
    for mode_text in payload.get("candidate_modes", []):
        assessment = normalized_modes_map.get(mode_text)
        if assessment is None:
            assessment = StepOneModeSelection(
                mode_text=mode_text,
                selected=False,
                supporting_sentence_ids=[],
                support_level="implicit",
                confidence="low",
                reason="Model did not return an assessment for this candidate mode.",
                evidence_spans=[],
            )
        completed_mode_assessments.append(assessment)

    return StepOneResult(
        failure_element=payload.get("failure_element", ""),
        query_function=payload.get("query_function", ""),
        candidate_modes=list(payload.get("candidate_modes", [])),
        sentence_count=len(payload.get("sentences", [])),
        mode_assessments=completed_mode_assessments,
        global_reasoning=normalize_text(parsed_response.get("global_reasoning", "")),
        sentences=list(payload.get("sentences", [])),
    )


# =========================================================
# Pipeline
# =========================================================
@traceable(
    run_type="chain",
    name="docllm_step1_pipeline",
    tags=["docllm", "step1", "pipeline"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def run_step_one() -> list[StepOneResult]:
    """Execute Step 1 using filtered Step 0 outputs."""

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")
    print(f"[INFO] Loading Step 0 results: {INPUT_JSON_PATH}")

    raw_stepzero = load_json(INPUT_JSON_PATH)
    all_items = normalize_stepzero_items(raw_stepzero)
    filtered_items = filter_stepzero_items(
        items=all_items,
        usefulness_exclusion=USEFULNESS_EXCLUSION,
        query_functions=QUERY_FUNCTIONS,
    )
    print(f"[INFO] Loaded {len(all_items)} Step 0 records, kept {len(filtered_items)} after filtering.")

    payloads = build_grouped_payloads(filtered_items, MOTOR_CONTROL_STRUCTURE)
    print(f"[INFO] Built {len(payloads)} grouped function payloads for Step 1.")

    runner = StepOneLLMRunner()
    results: list[StepOneResult] = []

    for index, payload in enumerate(payloads, start=1):
        print(
            f"[INFO] Processing function group {index}/{len(payloads)}: "
            f"{payload.get('query_function', '')} with {len(payload.get('sentences', []))} sentences."
        )
        try:
            parsed = runner.infer_modes_for_function(payload)
            results.append(normalize_stepone_result(parsed, payload))
        except Exception as exc:
            print(f"[WARN] Step 1 group failed for function {payload.get('query_function', '')}: {exc}")
            results.append(
                StepOneResult(
                    failure_element=payload.get("failure_element", ""),
                    query_function=payload.get("query_function", ""),
                    candidate_modes=list(payload.get("candidate_modes", [])),
                    sentence_count=len(payload.get("sentences", [])),
                    mode_assessments=[
                        StepOneModeSelection(
                            mode_text=mode_text,
                            selected=False,
                            supporting_sentence_ids=[],
                            support_level="implicit",
                            confidence="low",
                            reason="Fallback returned no assessment because the LLM result was unavailable.",
                            evidence_spans=[],
                        )
                        for mode_text in payload.get("candidate_modes", [])
                    ],
                    global_reasoning="Fallback returned no selected modes because the LLM result was unavailable.",
                    sentences=list(payload.get("sentences", [])),
                )
            )

    write_json([asdict(item) for item in results], OUTPUT_JSON_PATH)
    print(f"[INFO] Wrote {len(results)} Step 1 results to: {OUTPUT_JSON_PATH}")
    return results


# =========================================================
# Utilities
# =========================================================
def normalize_text(value: Any) -> str:
    """Normalize values to a clean string."""

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


if __name__ == "__main__":
    run_step_one()
