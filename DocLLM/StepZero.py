from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

from LLMs.llm_init import DOC_LLM_LANGSMITH_PROJECT, configure_langsmith
from LLMs.zero_LLM import StepZeroLLMRunner

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
BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON_PATH = Path(
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\GraphRAG\prepocess\output\motor_control_ts_chunk_function_matches.json"
)
OUTPUT_JSON_PATH = BASE_DIR / "step0_results.json"

TOKEN_THRESHOLD_PER_BATCH = 3500
MAX_ITEMS_PER_BATCH = 12


# =========================================================
# Data models
# =========================================================
@dataclass(slots=True)
class SentenceRecord:
    """Source record loaded from the preprocessing JSON."""

    sentence_id: str
    failure_element: str
    section_tag: str
    discipline: str
    chunk_name: str
    choice_text: str
    rationale_text: str
    match_text: str
    matched_functions: list[str] = field(default_factory=list)
    function_matches: list[dict[str, Any]] = field(default_factory=list)
    element_match_coverage: float | None = None
    source_index: int = -1
    source_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationUnit:
    """One sentence-function pair to be evaluated by Step 0."""

    sentence_id: str
    failure_element: str
    function_text: str
    choice_text: str
    rationale_text: str
    section_tag: str
    discipline: str
    chunk_name: str
    match_text: str
    source_index: int
    function_match_coverage: float | None = None
    source_record: dict[str, Any] = field(default_factory=dict)

    def to_prompt_item(self) -> dict[str, Any]:
        """Return the compact payload passed into the Step 0 LLM runner."""

        return {
            "sentence_id": self.sentence_id,
            "failure_element": self.failure_element,
            "function_text": self.function_text,
            "choice_text": self.choice_text,
            "rationale_text": self.rationale_text,
            "section_tag": self.section_tag,
            "discipline": self.discipline,
            "chunk_name": self.chunk_name,
        }


@dataclass(slots=True)
class StepZeroResult:
    """Final normalized Step 0 output record."""

    sentence_id: str
    failure_element: str
    function_text: str
    usefulness: str
    reason: str
    key_evidence_spans: list[str]
    section_tag: str
    discipline: str
    chunk_name: str
    choice_text: str
    rationale_text: str
    match_text: str
    source_index: int
    function_match_coverage: float | None = None


# =========================================================
# Input loading and expansion
# =========================================================
def load_input_json(path: Path) -> Any:
    """Load the source JSON file with basic validation."""

    if not path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse input JSON: {path}") from exc


def normalize_input_records(raw_data: Any) -> list[SentenceRecord]:
    """Normalize the input JSON into sentence records."""

    raw_items = list(iter_source_records(raw_data))
    records: list[SentenceRecord] = []

    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            print(f"[WARN] Skipping non-dict record at source index {index}.")
            continue

        text_value = normalize_text(item.get("text", ""))
        rationale_value = normalize_text(item.get("rationale_text", ""))
        sentence_id = str(item.get("sentence_id") or build_sentence_id(index, item))
        matched_functions = normalize_string_list(item.get("matched_functions", []))
        function_matches = item.get("function_matches", [])

        records.append(
            SentenceRecord(
                sentence_id=sentence_id,
                failure_element=normalize_text(item.get("failure_element", "")),
                section_tag=normalize_text(item.get("section_tag", "")),
                discipline=normalize_text(item.get("discipline", "")),
                chunk_name=normalize_text(item.get("chunk_name", "")),
                choice_text=text_value,
                rationale_text=rationale_value,
                match_text=normalize_text(item.get("match_text", text_value)),
                matched_functions=matched_functions,
                function_matches=function_matches if isinstance(function_matches, list) else [],
                element_match_coverage=to_optional_float(item.get("element_match_coverage")),
                source_index=index,
                source_payload=item,
            )
        )

    return records


def iter_source_records(raw_data: Any) -> Iterable[Any]:
    """Yield record-like items from flat or grouped JSON structures."""

    if isinstance(raw_data, list):
        yield from raw_data
        return

    if isinstance(raw_data, dict):
        if all(isinstance(value, list) for value in raw_data.values()):
            for group_key, items in raw_data.items():
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict) and "group_key" not in item:
                        enriched = dict(item)
                        enriched["group_key"] = group_key
                        yield enriched
                    else:
                        yield item
            return

        yield raw_data
        return

    raise ValueError(f"Unsupported input JSON structure: {type(raw_data).__name__}")


def expand_sentence_function_units(records: Sequence[SentenceRecord]) -> list[EvaluationUnit]:
    """Expand each sentence into one evaluation unit per matched function."""

    units: list[EvaluationUnit] = []

    for record in records:
        coverage_lookup = build_function_coverage_lookup(record.function_matches)

        for function_text in record.matched_functions:
            function_name = normalize_text(function_text)
            if not function_name:
                continue

            units.append(
                EvaluationUnit(
                    sentence_id=record.sentence_id,
                    failure_element=record.failure_element,
                    function_text=function_name,
                    choice_text=record.choice_text,
                    rationale_text=record.rationale_text,
                    section_tag=record.section_tag,
                    discipline=record.discipline,
                    chunk_name=record.chunk_name,
                    match_text=record.match_text,
                    source_index=record.source_index,
                    function_match_coverage=coverage_lookup.get(function_name),
                    source_record=record.source_payload,
                )
            )

    return units


def build_function_coverage_lookup(function_matches: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """Map function text to function match coverage when available."""

    lookup: dict[str, float | None] = {}
    for item in function_matches:
        if not isinstance(item, dict):
            continue
        function_name = normalize_text(item.get("function", ""))
        if not function_name:
            continue
        lookup[function_name] = to_optional_float(item.get("function_match_coverage"))
    return lookup


# =========================================================
# Batching
# =========================================================
def estimate_tokens_for_text(text: str) -> int:
    """Estimate token count with a simple character-based heuristic."""

    cleaned = normalize_text(text)
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def estimate_tokens_for_unit(unit: EvaluationUnit) -> int:
    """Estimate the token cost of a single evaluation unit."""

    payload = json.dumps(unit.to_prompt_item(), ensure_ascii=False)
    return estimate_tokens_for_text(payload) + 40


def build_batches(
    units: Sequence[EvaluationUnit],
    token_threshold: int,
    max_items_per_batch: int,
) -> list[list[EvaluationUnit]]:
    """Build batches under a token threshold with a max-items safeguard."""

    if token_threshold <= 0:
        raise ValueError("token_threshold must be > 0")
    if max_items_per_batch <= 0:
        raise ValueError("max_items_per_batch must be > 0")

    batches: list[list[EvaluationUnit]] = []
    current_batch: list[EvaluationUnit] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens_for_unit(unit)
        would_exceed_tokens = bool(current_batch) and current_tokens + unit_tokens > token_threshold
        would_exceed_items = len(current_batch) >= max_items_per_batch

        if would_exceed_tokens or would_exceed_items:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(unit)
        current_tokens += unit_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


# =========================================================
# LLM output normalization
# =========================================================
def normalize_batch_results(
    parsed_response: dict[str, Any],
    batch: Sequence[EvaluationUnit],
) -> list[StepZeroResult]:
    """Validate and normalize one parsed LLM response batch."""

    raw_results = parsed_response.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("LLM response must contain a top-level 'results' list.")

    batch_lookup = {(item.sentence_id, item.function_text): item for item in batch}
    normalized_map: dict[tuple[str, str], StepZeroResult] = {}

    for raw_item in raw_results:
        if not isinstance(raw_item, dict):
            print("[WARN] Skipping non-dict LLM result item.")
            continue

        sentence_id = str(raw_item.get("sentence_id", "")).strip()
        function_text = normalize_text(raw_item.get("function_text", ""))
        key = (sentence_id, function_text)
        source_unit = batch_lookup.get(key)

        if source_unit is None:
            print(f"[WARN] Ignoring unmatched LLM item for sentence_id={sentence_id}, function={function_text!r}.")
            continue

        normalized_map[key] = StepZeroResult(
            sentence_id=source_unit.sentence_id,
            failure_element=source_unit.failure_element,
            function_text=source_unit.function_text,
            usefulness=normalize_usefulness(raw_item.get("usefulness")),
            reason=normalize_text(raw_item.get("reason", "")),
            key_evidence_spans=normalize_string_list(raw_item.get("key_evidence_spans", [])),
            section_tag=source_unit.section_tag,
            discipline=source_unit.discipline,
            chunk_name=source_unit.chunk_name,
            choice_text=source_unit.choice_text,
            rationale_text=source_unit.rationale_text,
            match_text=source_unit.match_text,
            source_index=source_unit.source_index,
            function_match_coverage=source_unit.function_match_coverage,
        )

    completed_results: list[StepZeroResult] = []
    for unit in batch:
        key = (unit.sentence_id, unit.function_text)
        result = normalized_map.get(key)
        if result is None:
            result = StepZeroResult(
                sentence_id=unit.sentence_id,
                failure_element=unit.failure_element,
                function_text=unit.function_text,
                usefulness="not_useful",
                reason="No valid classification returned by the model for this sentence-function pair.",
                key_evidence_spans=[],
                section_tag=unit.section_tag,
                discipline=unit.discipline,
                chunk_name=unit.chunk_name,
                choice_text=unit.choice_text,
                rationale_text=unit.rationale_text,
                match_text=unit.match_text,
                source_index=unit.source_index,
                function_match_coverage=unit.function_match_coverage,
            )
        completed_results.append(result)

    return completed_results


def normalize_usefulness(value: Any) -> str:
    """Normalize usefulness labels and clamp unknown values to `not_useful`."""

    text = normalize_text(value).lower()
    if text in {"explicit", "implicit", "not_useful"}:
        return text
    return "not_useful"


# =========================================================
# Output handling
# =========================================================
def write_output_json(path: Path, results: Sequence[StepZeroResult]) -> None:
    """Write final Step 0 results to a JSON file."""

    serializable = [asdict(item) for item in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)


# =========================================================
# Orchestration
# =========================================================
@traceable(
    run_type="chain",
    name="docllm_step0_pipeline",
    tags=["docllm", "step0", "pipeline"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def run_step_zero() -> list[StepZeroResult]:
    """Execute the full Step 0 pipeline."""

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")
    print(f"[INFO] Loading input JSON: {INPUT_JSON_PATH}")
    raw_data = load_input_json(INPUT_JSON_PATH)

    records = normalize_input_records(raw_data)
    print(f"[INFO] Loaded {len(records)} source records.")

    units = expand_sentence_function_units(records)
    print(f"[INFO] Expanded to {len(units)} sentence-function evaluation units.")

    if not units:
        print("[WARN] No sentence-function units were found. Writing empty output.")
        write_output_json(OUTPUT_JSON_PATH, [])
        return []

    batches = build_batches(
        units=units,
        token_threshold=TOKEN_THRESHOLD_PER_BATCH,
        max_items_per_batch=MAX_ITEMS_PER_BATCH,
    )
    print(
        f"[INFO] Built {len(batches)} batches "
        f"(token threshold={TOKEN_THRESHOLD_PER_BATCH}, max items={MAX_ITEMS_PER_BATCH})."
    )

    runner = StepZeroLLMRunner()
    all_results: list[StepZeroResult] = []

    for batch_index, batch in enumerate(batches, start=1):
        batch_payload = [item.to_prompt_item() for item in batch]
        estimated_tokens = sum(estimate_tokens_for_unit(item) for item in batch)
        print(
            f"[INFO] Processing batch {batch_index}/{len(batches)} "
            f"with {len(batch)} items, estimated tokens={estimated_tokens}."
        )

        try:
            parsed_response = runner.invoke_batch(batch_payload)
            batch_results = normalize_batch_results(parsed_response, batch)
        except Exception as exc:
            print(f"[WARN] Batch {batch_index} failed. Falling back to default results. Error: {exc}")
            batch_results = normalize_batch_results({"results": []}, batch)

        all_results.extend(batch_results)

    write_output_json(OUTPUT_JSON_PATH, all_results)
    print(f"[INFO] Wrote {len(all_results)} Step 0 results to: {OUTPUT_JSON_PATH}")
    return all_results


# =========================================================
# Utility helpers
# =========================================================
def build_sentence_id(index: int, item: dict[str, Any]) -> str:
    """Build a stable synthetic sentence ID when the source file does not provide one."""

    chunk_name = normalize_text(item.get("chunk_name", "chunk"))
    compact_chunk = re.sub(r"[^A-Za-z0-9]+", "_", chunk_name).strip("_") or "chunk"
    return f"{compact_chunk}_{index:04d}"


def normalize_text(value: Any) -> str:
    """Normalize input text into a safe printable string."""

    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    return text.strip()


def normalize_string_list(value: Any) -> list[str]:
    """Normalize a value into a list of non-empty strings."""

    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    if isinstance(value, str):
        normalized = normalize_text(value)
        return [normalized] if normalized else []
    return []


def to_optional_float(value: Any) -> float | None:
    """Convert numeric-like values to float when possible."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    run_step_zero()
