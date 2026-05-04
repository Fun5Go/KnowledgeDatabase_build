from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = ("query.json", "chunk_selection_results_query_*.json")
DEFAULT_OUTPUT_NAME = "mode_cause_chunk_name_overlaps.json"


def normalize_text(value: Any) -> str:
    """Return a trimmed string for optional JSON values."""

    if value is None:
        return ""
    return str(value).strip()


def iter_input_files(input_dir: Path, patterns: list[str]) -> list[Path]:
    """Find query result JSON files under input_dir."""

    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in input_dir.rglob(pattern) if path.is_file())
    return sorted(files)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def iter_result_items(data: Any) -> list[dict[str, Any]]:
    """Normalize supported result-file shapes to a list of result dictionaries."""

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return [item for item in data["results"] if isinstance(item, dict)]
        return [data]
    return []


def query_text_from_item(item: dict[str, Any]) -> str:
    analysis_item = item.get("analysis_item")
    if isinstance(analysis_item, dict):
        query_text = normalize_text(analysis_item.get("query_text"))
        if query_text:
            return query_text

    query_result = item.get("query_result")
    if isinstance(query_result, dict):
        query_spec = query_result.get("query_spec")
        if isinstance(query_spec, dict):
            query_text = normalize_text(query_spec.get("query_text"))
            if query_text:
                return query_text
            return normalize_text(query_spec.get("sentence"))

    return ""


def query_type_from_item(item: dict[str, Any]) -> str:
    analysis_item = item.get("analysis_item")
    if isinstance(analysis_item, dict):
        query_type = normalize_text(analysis_item.get("query_type")).lower()
        if query_type:
            return query_type

    selection = item.get("selection")
    if isinstance(selection, dict):
        query_type = normalize_text(selection.get("query_type")).lower()
        if query_type:
            return query_type

    return ""


def collect_selected_chunks(input_files: list[Path]) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Collect selected chunks by name for function_mode and cause queries."""

    chunks_by_name: dict[str, dict[str, list[dict[str, str]]]] = {}

    for path in input_files:
        data = load_json(path)
        for item in iter_result_items(data):
            query_type = query_type_from_item(item)
            if query_type not in {"function_mode", "cause"}:
                continue

            selection = item.get("selection")
            if not isinstance(selection, dict):
                continue

            top_chunks = selection.get("top_chunks")
            if not isinstance(top_chunks, list):
                continue

            output_key = "mode" if query_type == "function_mode" else "cause"
            query_text = query_text_from_item(item)

            for chunk in top_chunks:
                if not isinstance(chunk, dict):
                    continue
                name = normalize_text(chunk.get("name"))
                if not name:
                    continue

                chunks_by_name.setdefault(name, {"mode": [], "cause": []})[output_key].append(
                    {
                        "query text": query_text,
                        "relationship": normalize_text(chunk.get("relationship")),
                        "reason": normalize_text(chunk.get("reason")),
                    }
                )

    return chunks_by_name


def build_overlap_output(
    chunks_by_name: dict[str, dict[str, list[dict[str, str]]]]
) -> list[dict[str, Any]]:
    overlaps: list[dict[str, Any]] = []

    for name in sorted(chunks_by_name):
        grouped_entries = chunks_by_name[name]
        mode_entries = grouped_entries["mode"]
        cause_entries = grouped_entries["cause"]
        if not mode_entries or not cause_entries:
            continue
        overlaps.append(
            {
                "name": name,
                "mode": mode_entries,
                "cause": cause_entries,
            }
        )

    return overlaps


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find chunks whose names appear in both function-mode and cause "
            "query selections, keeping only name, query text, relationship, and reason."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to scan recursively. Defaults to this script's directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_OUTPUT_NAME,
        help=f"Output JSON path. Defaults to {DEFAULT_OUTPUT_NAME}.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help=(
            "Glob pattern to scan recursively. Can be passed multiple times. "
            "Defaults to query.json and chunk_selection_results_query_*.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = args.patterns or list(DEFAULT_PATTERNS)
    input_files = iter_input_files(args.input_dir, patterns)
    chunks_by_name = collect_selected_chunks(input_files)
    overlaps = build_overlap_output(chunks_by_name)
    write_json(args.output, overlaps)
    print(f"Wrote {len(overlaps)} overlapping chunk names to {args.output}")


if __name__ == "__main__":
    main()
