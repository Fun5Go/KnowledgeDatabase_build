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
from LLMs.two_LLM import StepTwoLLMRunner

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
INPUT_JSON_PATH = BASE_DIR / "step1_results.json"
OUTPUT_JSON_PATH = BASE_DIR / "step2_results.json"

ONLY_SELECTED_MODES = True
QUERY_FUNCTIONS: list[str] = []
QUERY_MODES: list[str] = []


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
            "causes": {
                "mechanics": [
                    "Cooling insufficient",
                    "Compressor vibrations",
                ],
                "hardware": [
                    "(Starting) Motor current too high for chosen components",
                    "Overvoltage due to motor disconnect",
                    "Under Voltage due to incorrect triggering",
                    "Live switching of relays",
                ],
                "software": [
                    "Priority zero-crossing interrupt too low",
                    "Open loop control",
                ],
                "other": [
                    "No (correctly designed) snubber design",
                    "Too high dT junction as a result of power cycling of component",
                ],
            },
            "effects": [
                "Motor cannot start",
                "Overcurrent towards motor",
                "Motor starts without soft start",
            ],
        }
    ],
}


# =========================================================
# Data models
# =========================================================
@dataclass(slots=True)
class StepOneModeAssessment:
    """A normalized mode assessment from Step 1."""

    mode_text: str
    selected: bool
    supporting_sentence_ids: list[str]
    support_level: str
    confidence: str
    reason: str
    evidence_spans: list[str]


@dataclass(slots=True)
class StepOneGroup:
    """A grouped Step 1 function result."""

    failure_element: str
    query_function: str
    candidate_modes: list[str]
    sentence_count: int
    mode_assessments: list[StepOneModeAssessment]
    global_reasoning: str
    sentences: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class StepTwoCauseAssessment:
    """Normalized cause assessment for one mode."""

    cause_category: str
    cause_text: str
    selected: bool
    supporting_sentence_ids: list[str]
    support_level: str
    confidence: str
    reason: str
    causal_path: str
    evidence_spans: list[str]


@dataclass(slots=True)
class StepTwoResult:
    """Final grouped Step 2 result."""

    failure_element: str
    query_function: str
    mode_text: str
    selected_in_step1: bool
    support_level_from_step1: str
    candidate_cause_count: int
    cause_assessments: list[StepTwoCauseAssessment]
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
# Normalization helpers
# =========================================================
def normalize_stepone_groups(raw_data: Any) -> list[StepOneGroup]:
    """Normalize Step 1 results into internal group objects."""

    if not isinstance(raw_data, list):
        raise ValueError("Step 2 expects Step 1 output JSON to be a list of records.")

    groups: list[StepOneGroup] = []
    for raw_item in raw_data:
        if not isinstance(raw_item, dict):
            continue

        raw_mode_items = raw_item.get("mode_assessments", raw_item.get("selected_modes", []))
        normalized_mode_assessments: list[StepOneModeAssessment] = []
        for mode_item in raw_mode_items:
            if not isinstance(mode_item, dict):
                continue
            normalized_mode_assessments.append(
                StepOneModeAssessment(
                    mode_text=normalize_text(mode_item.get("mode_text", "")),
                    selected=bool(mode_item.get("selected", True)),
                    supporting_sentence_ids=normalize_string_list(mode_item.get("supporting_sentence_ids", [])),
                    support_level=normalize_text(mode_item.get("support_level", "")),
                    confidence=normalize_text(mode_item.get("confidence", "")),
                    reason=normalize_text(mode_item.get("reason", "")),
                    evidence_spans=normalize_string_list(mode_item.get("evidence_spans", [])),
                )
            )

        groups.append(
            StepOneGroup(
                failure_element=normalize_text(raw_item.get("failure_element", "")),
                query_function=normalize_text(raw_item.get("query_function", "")),
                candidate_modes=normalize_string_list(raw_item.get("candidate_modes", [])),
                sentence_count=int(raw_item.get("sentence_count", 0) or 0),
                mode_assessments=normalized_mode_assessments,
                global_reasoning=normalize_text(raw_item.get("global_reasoning", "")),
                sentences=raw_item.get("sentences", []) if isinstance(raw_item.get("sentences", []), list) else [],
            )
        )

    return groups


def build_cause_candidate_lookup(structure_input: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Map failure element to grouped cause candidates."""

    lookup: dict[str, dict[str, list[str]]] = {}
    for node in structure_input.get("nodes", []):
        failure_element = normalize_text(node.get("failure_element", ""))
        causes = node.get("causes", {})
        grouped_causes: dict[str, list[str]] = {}
        for category, cause_list in causes.items():
            grouped_causes[normalize_text(category)] = [
                normalize_text(cause_text)
                for cause_text in cause_list
                if normalize_text(cause_text)
            ]
        if failure_element:
            lookup[failure_element] = grouped_causes
    return lookup


def build_mode_payloads(
    groups: Sequence[StepOneGroup],
    structure_input: dict[str, Any],
    only_selected_modes: bool = True,
    query_functions: Sequence[str] | None = None,
    query_modes: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build Step 2 payloads for each mode assessment."""

    cause_lookup = build_cause_candidate_lookup(structure_input)
    normalized_query_functions = {
        normalize_text(function_text).lower()
        for function_text in (query_functions or [])
        if normalize_text(function_text)
    }
    normalized_query_modes = {
        normalize_text(mode_text).lower()
        for mode_text in (query_modes or [])
        if normalize_text(mode_text)
    }

    payloads: list[dict[str, Any]] = []

    for group in groups:
        if normalized_query_functions and group.query_function.lower() not in normalized_query_functions:
            continue

        sentence_lookup = {
            normalize_text(sentence.get("sentence_id", "")): sentence
            for sentence in group.sentences
            if isinstance(sentence, dict)
        }

        for mode_assessment in group.mode_assessments:
            if only_selected_modes and not mode_assessment.selected:
                continue
            if normalized_query_modes and mode_assessment.mode_text.lower() not in normalized_query_modes:
                continue

            supporting_sentences = [
                sentence_lookup[sentence_id]
                for sentence_id in mode_assessment.supporting_sentence_ids
                if sentence_id in sentence_lookup
            ]

            payloads.append(
                {
                    "failure_element": group.failure_element,
                    "query_function": group.query_function,
                    "mode_text": mode_assessment.mode_text,
                    "mode_support_level": mode_assessment.support_level,
                    "mode_reason": mode_assessment.reason,
                    "mode_evidence_spans": mode_assessment.evidence_spans,
                    "cause_candidates": cause_lookup.get(group.failure_element, {}),
                    "sentences": supporting_sentences,
                }
            )

    return payloads


# =========================================================
# Result normalization
# =========================================================
def normalize_steptwo_result(
    parsed_response: dict[str, Any],
    payload: dict[str, Any],
) -> StepTwoResult:
    """Validate and normalize one Step 2 LLM response."""

    cause_candidates = payload.get("cause_candidates", {})
    valid_causes: dict[tuple[str, str], None] = {}
    for category, cause_list in cause_candidates.items():
        for cause_text in cause_list:
            valid_causes[(normalize_text(category), normalize_text(cause_text))] = None

    sentence_lookup = {
        normalize_text(sentence.get("sentence_id", "")): sentence
        for sentence in payload.get("sentences", [])
        if isinstance(sentence, dict)
    }
    valid_support_levels = {"explicit", "implicit", "mixed"}
    valid_confidence = {"high", "medium", "low"}

    raw_items = parsed_response.get("cause_assessments", [])
    if not isinstance(raw_items, list):
        raw_items = []

    normalized_map: dict[tuple[str, str], StepTwoCauseAssessment] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue

        cause_category = normalize_text(item.get("cause_category", ""))
        cause_text = normalize_text(item.get("cause_text", ""))
        key = (cause_category, cause_text)
        if key not in valid_causes:
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

        normalized_map[key] = StepTwoCauseAssessment(
            cause_category=cause_category,
            cause_text=cause_text,
            selected=bool(item.get("selected", bool(supporting_ids))),
            supporting_sentence_ids=supporting_ids,
            support_level=support_level,
            confidence=confidence,
            reason=normalize_text(item.get("reason", "")),
            causal_path=normalize_text(item.get("causal_path", "")),
            evidence_spans=normalize_string_list(item.get("evidence_spans", [])),
        )

    completed_assessments: list[StepTwoCauseAssessment] = []
    for category, cause_list in cause_candidates.items():
        for cause_text in cause_list:
            key = (normalize_text(category), normalize_text(cause_text))
            assessment = normalized_map.get(key)
            if assessment is None:
                assessment = StepTwoCauseAssessment(
                    cause_category=normalize_text(category),
                    cause_text=normalize_text(cause_text),
                    selected=False,
                    supporting_sentence_ids=[],
                    support_level="implicit",
                    confidence="low",
                    reason="Model did not return an assessment for this candidate cause.",
                    causal_path="",
                    evidence_spans=[],
                )
            completed_assessments.append(assessment)

    return StepTwoResult(
        failure_element=payload.get("failure_element", ""),
        query_function=payload.get("query_function", ""),
        mode_text=payload.get("mode_text", ""),
        selected_in_step1=True,
        support_level_from_step1=normalize_text(payload.get("mode_support_level", "")),
        candidate_cause_count=sum(len(cause_list) for cause_list in cause_candidates.values()),
        cause_assessments=completed_assessments,
        global_reasoning=normalize_text(parsed_response.get("global_reasoning", "")),
        sentences=list(payload.get("sentences", [])),
    )


# =========================================================
# Pipeline
# =========================================================
@traceable(
    run_type="chain",
    name="docllm_step2_pipeline",
    tags=["docllm", "step2", "pipeline"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def run_step_two() -> list[StepTwoResult]:
    """Execute Step 2 using Step 1 mode assessments."""

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")
    print(f"[INFO] Loading Step 1 results: {INPUT_JSON_PATH}")

    raw_stepone = load_json(INPUT_JSON_PATH)
    groups = normalize_stepone_groups(raw_stepone)
    payloads = build_mode_payloads(
        groups=groups,
        structure_input=MOTOR_CONTROL_STRUCTURE,
        only_selected_modes=ONLY_SELECTED_MODES,
        query_functions=QUERY_FUNCTIONS,
        query_modes=QUERY_MODES,
    )
    print(f"[INFO] Loaded {len(groups)} Step 1 groups and built {len(payloads)} Step 2 mode payloads.")

    runner = StepTwoLLMRunner()
    results: list[StepTwoResult] = []

    for index, payload in enumerate(payloads, start=1):
        print(
            f"[INFO] Processing mode payload {index}/{len(payloads)}: "
            f"{payload.get('query_function', '')} -> {payload.get('mode_text', '')} "
            f"with {len(payload.get('sentences', []))} supporting sentences."
        )
        try:
            parsed = runner.infer_causes_for_mode(payload)
            results.append(normalize_steptwo_result(parsed, payload))
        except Exception as exc:
            print(
                f"[WARN] Step 2 mode payload failed for "
                f"{payload.get('query_function', '')} -> {payload.get('mode_text', '')}: {exc}"
            )
            fallback_assessments: list[StepTwoCauseAssessment] = []
            for category, cause_list in payload.get("cause_candidates", {}).items():
                for cause_text in cause_list:
                    fallback_assessments.append(
                        StepTwoCauseAssessment(
                            cause_category=normalize_text(category),
                            cause_text=normalize_text(cause_text),
                            selected=False,
                            supporting_sentence_ids=[],
                            support_level="implicit",
                            confidence="low",
                            reason="Fallback returned no assessment because the LLM result was unavailable.",
                            causal_path="",
                            evidence_spans=[],
                        )
                    )

            results.append(
                StepTwoResult(
                    failure_element=payload.get("failure_element", ""),
                    query_function=payload.get("query_function", ""),
                    mode_text=payload.get("mode_text", ""),
                    selected_in_step1=True,
                    support_level_from_step1=normalize_text(payload.get("mode_support_level", "")),
                    candidate_cause_count=len(fallback_assessments),
                    cause_assessments=fallback_assessments,
                    global_reasoning="Fallback returned no cause assessments because the LLM result was unavailable.",
                    sentences=list(payload.get("sentences", [])),
                )
            )

    write_json([asdict(item) for item in results], OUTPUT_JSON_PATH)
    print(f"[INFO] Wrote {len(results)} Step 2 results to: {OUTPUT_JSON_PATH}")
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
    run_step_two()
