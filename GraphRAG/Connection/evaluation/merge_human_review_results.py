"""Merge rerank and extraction JSON results into a human-review file.

The script reads all JSON files from GraphRAG/Connection/results, groups records
by query text, merges rerank chunks with extraction aggregates, and writes a
compact review-oriented JSON file next to this script.
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
OUTPUT_PATH = SCRIPT_DIR / "human_review_merged_results.json"
PER_QUERY_DIR = SCRIPT_DIR / "human_review_by_query"

HUMAN_REVIEW_KEYS = [
    "rerank_tag_correct",
    "relationship_correct",
    "selected_correct",
    "comments",
]

AGGREGATE_KEYS = [
    "selected",
    "primary_relation",
    "relations",
    "evidence_spans",
    "support_capability",
    "selection_reason",
]

SUMMARY_KEYS = {
    "num_candidates",
    "num_support",
    "num_suspect",
    "num_irrelevant",
    "num_evidence_units",
    "num_selected_chunks",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
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


def extract_summary(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        connection = payload.get("connection")
        if isinstance(connection, dict) and isinstance(connection.get("summary"), dict):
            return connection["summary"]

    summary = find_first_key(payload, ["summary"])
    if isinstance(summary, dict) and SUMMARY_KEYS.intersection(summary):
        return summary
    return {}


def query_text_from_payload(payload: Any) -> str:
    query = find_first_key(payload, ["query_text", "question"])
    if query is None:
        query = find_first_key(payload, ["query"])

    if isinstance(query, dict):
        query = first_present(query, ["query_text", "question", "text", "content"])
    if isinstance(query, list):
        query = " ".join(str(part) for part in query if part not in (None, ""))

    return str(query or "UNKNOWN_QUERY").strip()


def query_group_key(query_text: str) -> str:
    return re.sub(r"\s+", " ", query_text).strip().casefold()


def filename_from_query(query_text: str, index: int) -> str:
    name = query_text.strip().casefold()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = f"query_{index:03d}"
    return f"{index:03d}_{name[:120]}.json"


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def find_named_lists(obj: Any, names: set[str]) -> list[list[Any]]:
    found: list[list[Any]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in names and isinstance(value, list):
                found.append(value)
            found.extend(find_named_lists(value, names))
    elif isinstance(obj, list):
        for child in obj:
            found.extend(find_named_lists(child, names))
    return found


def extract_rerank_chunks(payload: Any) -> list[dict[str, Any]]:
    lists = find_named_lists(payload, {"reranked_chunks", "chunks", "candidate_chunks"})
    for chunks in lists:
        dict_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
        if any("rerank_tag" in chunk or "reason" in chunk for chunk in dict_chunks):
            return dict_chunks
    return []


def extract_aggregate_chunks(payload: Any) -> list[dict[str, Any]]:
    lists = find_named_lists(
        payload,
        {
            "chunk_aggregates",
            "chunk_aggregate",
            "aggregates",
            "aggregate",
            "extractions",
            "extraction",
            "chunks",
        },
    )
    for chunks in lists:
        dict_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
        if any(has_aggregate_data(chunk) for chunk in dict_chunks):
            return dict_chunks
    return []


def raw_text_from_chunk(chunk: dict[str, Any]) -> str:
    value = first_present(chunk, ["raw_text", "chunk_text", "text", "content"])
    if isinstance(value, list):
        return "\n".join(str(line) for line in value)
    if value is None:
        return ""
    return str(value)


def normalize_text_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def chunk_key(chunk: dict[str, Any]) -> tuple[str, str]:
    name = str(chunk.get("name") or chunk.get("chunk_id") or chunk.get("node_id") or "").strip()
    raw_text = normalize_text_key(raw_text_from_chunk(chunk))
    if name:
        return ("name", name.casefold())
    if raw_text:
        return ("text", raw_text)
    rank = chunk.get("rank", chunk.get("retrieval_rank", chunk.get("retrieval rank", "")))
    return ("rank", str(rank))


def wrap_raw_text(text: str, width: int = 100) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text or "").splitlines():
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


def aggregate_source(chunk: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(chunk, dict):
        return {}

    nested = first_present(chunk, ["chunk_aggregate", "aggregate", "extraction"])
    if isinstance(nested, dict):
        merged = dict(nested)
        for key, value in chunk.items():
            if key not in {"chunk_aggregate", "aggregate", "extraction"} and key not in merged:
                merged[key] = value
        return merged
    return chunk


def has_aggregate_data(chunk: dict[str, Any]) -> bool:
    source = aggregate_source(chunk)
    return any(key in source for key in AGGREGATE_KEYS + ["select", "is_selected"])


def aggregate_from_chunk(chunk: dict[str, Any] | None) -> dict[str, Any]:
    source = aggregate_source(chunk)
    selected = first_present(source, ["selected", "select", "is_selected"])
    return {
        "selected": selected,
        "primary_relation": source.get("primary_relation"),
        "relations": as_list(source.get("relations")),
        "evidence_spans": as_list(source.get("evidence_spans")),
        "support_capability": source.get("support_capability"),
        "selection_reason": source.get("selection_reason"),
    }


def review_chunk(rerank_chunk: dict[str, Any] | None, aggregate_chunk: dict[str, Any] | None) -> dict[str, Any]:
    source = rerank_chunk or aggregate_chunk or {}
    raw_text = raw_text_from_chunk(source) or raw_text_from_chunk(aggregate_chunk or {})
    aggregate = aggregate_from_chunk(aggregate_chunk)

    return {
        "rank": first_present(source, ["rank", "retrieval_rank", "retrieval rank"]),
        "raw_text": wrap_raw_text(raw_text),
        "rerank_tag": first_present(source, ["rerank_tag"]),
        "reason": first_present(source, ["reason"]),
        "chunk_aggregate": aggregate,
        "rerank_tag_correct": None,
        "relationship_correct": None,
        "selected_correct": None,
        "comments": "",
    }


def merge_query_chunks(
    rerank_chunks: list[dict[str, Any]],
    aggregate_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregate_by_key = {chunk_key(chunk): chunk for chunk in aggregate_chunks}
    used_keys: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []

    for rerank_chunk in rerank_chunks:
        key = chunk_key(rerank_chunk)
        aggregate_chunk = aggregate_by_key.get(key)
        if aggregate_chunk is None:
            rerank_text_key = normalize_text_key(raw_text_from_chunk(rerank_chunk))
            aggregate_chunk = next(
                (
                    chunk
                    for chunk in aggregate_chunks
                    if normalize_text_key(raw_text_from_chunk(chunk)) == rerank_text_key
                ),
                None,
            )
        if aggregate_chunk is not None:
            used_keys.add(chunk_key(aggregate_chunk))
        merged.append(review_chunk(rerank_chunk, aggregate_chunk))

    for aggregate_chunk in aggregate_chunks:
        key = chunk_key(aggregate_chunk)
        if key not in used_keys and key not in {chunk_key(chunk) for chunk in rerank_chunks}:
            merged.append(review_chunk(None, aggregate_chunk))

    return merged


def collect_results() -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        lowered_name = path.name.casefold()
        if "rerank" not in lowered_name and "extract" not in lowered_name:
            continue

        payload = load_json(path)
        query_text = query_text_from_payload(payload)
        key = query_group_key(query_text)
        group = grouped.setdefault(
            key,
            {
                "query_text": query_text,
                "summary": {},
                "rerank_chunks": [],
                "aggregate_chunks": [],
            },
        )

        if "rerank" in lowered_name:
            group["rerank_chunks"].extend(extract_rerank_chunks(payload))
        if "extract" in lowered_name:
            group["aggregate_chunks"].extend(extract_aggregate_chunks(payload))
            summary = extract_summary(payload)
            if summary:
                group["summary"] = summary

    return grouped


def build_review_payload() -> dict[str, Any]:
    grouped = collect_results()
    queries = []
    for group in sorted(grouped.values(), key=lambda item: item["query_text"].casefold()):
        queries.append(
            {
                "query_text": group["query_text"],
                "summary": group["summary"],
                "chunks": merge_query_chunks(group["rerank_chunks"], group["aggregate_chunks"]),
            }
        )

    return {
        "human_review_keys": HUMAN_REVIEW_KEYS,
        "queries": queries,
    }


def write_per_query_files(review_payload: dict[str, Any]) -> None:
    PER_QUERY_DIR.mkdir(parents=True, exist_ok=True)
    for index, query in enumerate(review_payload["queries"], start=1):
        query_payload = {
            "human_review_keys": HUMAN_REVIEW_KEYS,
            "query_text": query["query_text"],
            "summary": query["summary"],
            "chunks": query["chunks"],
        }
        write_json(PER_QUERY_DIR / filename_from_query(query["query_text"], index), query_payload)


def main() -> None:
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory does not exist: {RESULTS_DIR}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    review_payload = build_review_payload()
    write_json(OUTPUT_PATH, review_payload)
    write_per_query_files(review_payload)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote per-query files to {PER_QUERY_DIR}")


if __name__ == "__main__":
    main()
