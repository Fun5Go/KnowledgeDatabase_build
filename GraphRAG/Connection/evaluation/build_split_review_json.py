"""Build separate human-review JSON files for loose extraction and rerank runs.

This script intentionally does not merge rerank and extraction outputs. It reads
only loose_extract_*.json and rerank_*.json from GraphRAG/Connection/results,
then writes review-ready files into separate evaluation folders.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CONNECTION_DIR = SCRIPT_DIR.parent
RESULTS_DIR = CONNECTION_DIR / "results"
EXTRACT_OUTPUT_DIR = SCRIPT_DIR / "extact_evaluate"
RERANK_OUTPUT_DIR = SCRIPT_DIR / "rerank_evaluate"

EXTRACT_REVIEW_KEYS = [
    "is_selected_correct",
    "is_relation_correct",
    "suggest_relation_tag",
    "comment",
]
RERANK_REVIEW_KEYS = ["is_rerank_correct", "suggest_rerank_tag", "comment"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def find_first_key(obj: Any, keys: list[str]) -> Any:
    if isinstance(obj, dict):
        value = first_present(obj, keys)
        if value is not None:
            return value
        for child in obj.values():
            found = find_first_key(child, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for child in obj:
            found = find_first_key(child, keys)
            if found is not None:
                return found
    return None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def text_or_empty(value: Any) -> str:
    return "" if value is None else str(value)


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", text_or_empty(value)).strip()


def wrap_text(value: Any, width: int = 100) -> list[str]:
    lines: list[str] = []
    for paragraph in text_or_empty(value).splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                paragraph,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [paragraph]
        )
    return lines


def find_named_list(obj: Any, names: set[str]) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names and isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        for child in obj.values():
            found = find_named_list(child, names)
            if found:
                return found
    elif isinstance(obj, list):
        for child in obj:
            found = find_named_list(child, names)
            if found:
                return found
    return []


def query_parts(payload: dict[str, Any]) -> dict[str, Any]:
    analysis_item = as_dict(payload.get("analysis_item"))
    query_spec = as_dict(as_dict(payload.get("query_result")).get("query_spec"))
    source = {**query_spec, **analysis_item}

    discipline_text = first_present(
        source,
        ["discipline_text", "cause_discipline", "discipline", "disciplines"],
    )
    if isinstance(discipline_text, list):
        discipline_text = ", ".join(str(item) for item in discipline_text if item not in (None, ""))

    return {
        "query_text": text_or_empty(first_present(source, ["query_text", "sentence", "target_text"])),
        "query_type": text_or_empty(first_present(source, ["query_type"])),
        "function_text": text_or_empty(first_present(source, ["function_text"])),
        "query_mode": text_or_empty(first_present(source, ["query_mode"])),
        "query_cause": text_or_empty(first_present(source, ["query_cause"])),
        "query_effect": text_or_empty(first_present(source, ["query_effect"])),
        "discipline_text": text_or_empty(discipline_text),
    }


def metadata(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    analysis_item = as_dict(payload.get("analysis_item"))
    connection = as_dict(payload.get("connection"))
    return {
        "source_file": path.name,
        "analysis_id": text_or_empty(first_present(connection, ["analysis_id"]) or analysis_item.get("analysis_id")),
        "element_id": text_or_empty(analysis_item.get("element_id")),
        "failure_element": text_or_empty(analysis_item.get("failure_element")),
    }


def extract_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = as_dict(as_dict(payload.get("connection")).get("summary"))
    if summary:
        return summary
    found = find_first_key(payload, ["summary"])
    return found if isinstance(found, dict) else {}


def review_extract_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": chunk.get("rank"),
        "name": chunk.get("name"),
        "section_tag": chunk.get("section_tag"),
        "raw_text": wrap_text(chunk.get("raw_text")),
        "rerank_tag": chunk.get("rerank_tag"),
        "selected": chunk.get("selected"),
        "primary_relation": chunk.get("primary_relation"),
        "relations": as_list(chunk.get("relations")),
        "evidence_spans": as_list(chunk.get("evidence_spans")),
        "support_capability": chunk.get("support_capability"),
        "selection_reason": chunk.get("selection_reason"),
        "is_selected_correct": None,
        "is_relation_correct": None,
        "suggest_relation_tag": "",
        "comment": "",
    }


def review_rerank_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": chunk.get("rank"),
        "name": chunk.get("name"),
        "section_tag": chunk.get("section_tag"),
        "raw_text": wrap_text(chunk.get("raw_text")),
        "rerank_tag": chunk.get("rerank_tag"),
        "reason": chunk.get("reason"),
        "is_rerank_correct": None,
        "suggest_rerank_tag": "",
        "comment": "",
    }


def output_name(source_path: Path, prefix: str) -> str:
    stem = source_path.stem
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return f"{stem}.review.json"


def build_extract_review(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    chunk_aggregates = find_named_list(payload, {"chunk_aggregates"})
    return {
        "review_type": "loose_extract",
        "review_keys": EXTRACT_REVIEW_KEYS,
        **metadata(payload, path),
        "query": query_parts(payload),
        "summary": extract_summary(payload),
        "chunk_aggregates": [review_extract_chunk(chunk) for chunk in chunk_aggregates],
    }


def build_rerank_review(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    reranked_chunks = find_named_list(payload, {"reranked_chunks"})
    return {
        "review_type": "rerank",
        "review_keys": RERANK_REVIEW_KEYS,
        **metadata(payload, path),
        "query": query_parts(payload),
        "reranked_chunks": [review_rerank_chunk(chunk) for chunk in reranked_chunks],
    }


def build_all() -> tuple[int, int]:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory does not exist: {RESULTS_DIR}")

    extract_count = 0
    for path in sorted(RESULTS_DIR.glob("loose_extract_*.json")):
        review = build_extract_review(path)
        write_json(EXTRACT_OUTPUT_DIR / output_name(path, "loose_extract_"), review)
        extract_count += 1

    rerank_count = 0
    for path in sorted(RESULTS_DIR.glob("rerank_*.json")):
        review = build_rerank_review(path)
        write_json(RERANK_OUTPUT_DIR / output_name(path, "rerank_"), review)
        rerank_count += 1

    return extract_count, rerank_count


def main() -> None:
    extract_count, rerank_count = build_all()
    print(f"Wrote {extract_count} loose extract review files to {EXTRACT_OUTPUT_DIR}")
    print(f"Wrote {rerank_count} rerank review files to {RERANK_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
