from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from neo4j import GraphDatabase


DEFAULT_PATTERNS = ("query.json", "chunk_selection_results_query_*.json")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

QUERY_TYPE_TO_FAILURE_LABEL = {
    "function_mode": "Mode",
    "mode": "Mode",
    "cause": "Cause",
}
QUERY_TYPE_TO_TEXT_FIELD = {
    "function_mode": "query_mode",
    "mode": "query_mode",
    "cause": "query_cause",
}
ALLOWED_RELATIONSHIPS = {"upstream", "downstream", "self"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_query_type(value: Any) -> str:
    return normalize_text(value).lower()


def normalize_relationship(value: Any) -> str:
    relationship = normalize_text(value).lower()
    if relationship not in ALLOWED_RELATIONSHIPS:
        return ""
    return relationship


def iter_input_files(input_dir: Path, patterns: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in input_dir.rglob(pattern) if path.is_file())
    return sorted(files)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def iter_query_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return [item for item in data["results"] if isinstance(item, dict)]
        return [data]
    return []


def get_analysis_item(item: dict[str, Any]) -> dict[str, Any]:
    analysis_item = item.get("analysis_item")
    return analysis_item if isinstance(analysis_item, dict) else {}


def get_selection(item: dict[str, Any]) -> dict[str, Any]:
    selection = item.get("selection")
    return selection if isinstance(selection, dict) else {}


def get_query_type(item: dict[str, Any]) -> str:
    analysis_item = get_analysis_item(item)
    query_type = normalize_query_type(analysis_item.get("query_type"))
    if query_type:
        return query_type

    selection = get_selection(item)
    return normalize_query_type(selection.get("query_type"))


def get_query_text(item: dict[str, Any]) -> str:
    analysis_item = get_analysis_item(item)
    query_text = normalize_text(analysis_item.get("query_text"))
    if query_text:
        return query_text

    query_result = item.get("query_result")
    if not isinstance(query_result, dict):
        return ""

    query_spec = query_result.get("query_spec")
    if not isinstance(query_spec, dict):
        return ""

    return normalize_text(query_spec.get("query_text") or query_spec.get("sentence"))


def build_link_rows_from_query_item(
    item: dict[str, Any],
    source_file: Path | None = None,
) -> list[dict[str, str]]:
    """Build document-chunk to failure-node link rows from one query result item."""

    analysis_item = get_analysis_item(item)
    selection = get_selection(item)
    query_type = get_query_type(item)
    failure_label = QUERY_TYPE_TO_FAILURE_LABEL.get(query_type)
    target_text_field = QUERY_TYPE_TO_TEXT_FIELD.get(query_type)
    if not failure_label or not target_text_field:
        return []

    target_text = normalize_text(analysis_item.get(target_text_field))
    if not target_text:
        return []

    top_chunks = selection.get("top_chunks")
    if not isinstance(top_chunks, list):
        return []

    rows: list[dict[str, str]] = []
    for chunk in top_chunks:
        if not isinstance(chunk, dict):
            continue

        chunk_name = normalize_text(chunk.get("name"))
        relationship = normalize_relationship(chunk.get("relationship"))
        if not chunk_name or not relationship:
            continue

        rows.append(
            {
                "chunk_name": chunk_name,
                "failure_label": failure_label,
                "failure_text": target_text,
                "relationship": relationship,
                "support_capability": normalize_text(chunk.get("support_capability")),
                "reason": normalize_text(chunk.get("reason")),
                "analysis_id": normalize_text(
                    selection.get("analysis_id") or analysis_item.get("analysis_id")
                ),
                "query_type": query_type,
                "query_text": get_query_text(item),
                "source_file": str(source_file) if source_file else "",
            }
        )

    return rows


def build_link_rows_from_query_files(input_files: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in input_files:
        data = load_json(path)
        for item in iter_query_items(data):
            rows.extend(build_link_rows_from_query_item(item, source_file=path))
    return rows


def group_rows_by_label_and_relationship(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = row["failure_label"]
        relationship = row["relationship"]
        if label not in set(QUERY_TYPE_TO_FAILURE_LABEL.values()):
            continue
        if relationship not in ALLOWED_RELATIONSHIPS:
            continue
        grouped[(label, relationship)].append(row)
    return dict(grouped)


def connect_rows(
    rows: list[dict[str, str]],
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    database: str = NEO4J_DATABASE,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Connect selected document chunks to Mode/Cause failure nodes.

    Failure node label is constrained by query type:
    - function_mode/mode -> (:Mode {text: query_mode})
    - cause -> (:Cause {text: query_cause})

    Relationship type is copied from selection.relationship
    (upstream/downstream/self). Each relationship stores support_capability
    and reason, plus trace fields for repeatable imports.
    """

    stats = {
        "input_rows": len(rows),
        "created_or_matched_relationships": 0,
        "missing_chunks": 0,
        "missing_failure_nodes": 0,
    }

    if dry_run or not rows:
        return stats

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            for (failure_label, relationship), group_rows in group_rows_by_label_and_relationship(rows).items():
                query = f"""
                UNWIND $rows AS row
                OPTIONAL MATCH (chunk {{name: row.chunk_name}})
                OPTIONAL MATCH (failure:{failure_label} {{text: row.failure_text}})
                WITH row, chunk, failure
                CALL {{
                    WITH row, chunk, failure
                    WITH row, chunk, failure
                    WHERE chunk IS NOT NULL AND failure IS NOT NULL
                    MERGE (chunk)-[rel:{relationship} {{
                        analysis_id: row.analysis_id,
                        query_text: row.query_text
                    }}]->(failure)
                    SET rel.support_capability = row.support_capability,
                        rel.reason = row.reason,
                        rel.query_type = row.query_type,
                        rel.source_file = row.source_file
                    RETURN count(rel) AS linked
                }}
                RETURN
                    sum(linked) AS linked_count,
                    count(CASE WHEN chunk IS NULL THEN 1 END) AS missing_chunks,
                    count(CASE WHEN failure IS NULL THEN 1 END) AS missing_failure_nodes
                """
                for start in range(0, len(group_rows), batch_size):
                    batch = group_rows[start : start + batch_size]
                    record = session.run(query, rows=batch).single()
                    if not record:
                        continue
                    stats["created_or_matched_relationships"] += int(record["linked_count"] or 0)
                    stats["missing_chunks"] += int(record["missing_chunks"] or 0)
                    stats["missing_failure_nodes"] += int(record["missing_failure_nodes"] or 0)
    finally:
        driver.close()

    return stats


def connect_document_chunk_graph_to_failure_graph(
    input_dir: Path = Path(__file__).resolve().parent,
    patterns: list[str] | None = None,
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    database: str = NEO4J_DATABASE,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict[str, int]:
    """Scan query JSON files and write chunk-to-failure relationships."""

    input_files = iter_input_files(input_dir, patterns or list(DEFAULT_PATTERNS))
    rows = build_link_rows_from_query_files(input_files)
    stats = connect_rows(
        rows=rows,
        uri=uri,
        user=user,
        password=password,
        database=database,
        batch_size=batch_size,
        dry_run=dry_run,
    )
    stats["input_files"] = len(input_files)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Connect selected document chunk nodes to failure graph Mode/Cause nodes "
            "using chunk selection query JSON files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to scan recursively. Defaults to this script's directory.",
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
    parser.add_argument("--uri", default=NEO4J_URI)
    parser.add_argument("--user", default=NEO4J_USER)
    parser.add_argument("--password", default=NEO4J_PASSWORD)
    parser.add_argument("--database", default=NEO4J_DATABASE)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse query JSON and report counts without writing Neo4j relationships.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = connect_document_chunk_graph_to_failure_graph(
        input_dir=args.input_dir,
        patterns=args.patterns,
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
