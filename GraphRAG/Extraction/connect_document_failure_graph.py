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


DEFAULT_PATTERNS = (
    "query.json",
    "chunk_selection_results_query_*.json",
    "qd_detection_control_results_query_*.json",
)
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
ALLOWED_EVIDENCE_LABELS = {"control", "reason", "specification", "detection"}
LEGACY_RELATIONSHIP_TYPES = ("upstream", "downstream", "self")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_query_type(value: Any) -> str:
    return normalize_text(value).lower()


def normalize_evidence_label(value: Any) -> str:
    evidence_label = normalize_text(value).lower()
    if evidence_label not in ALLOWED_EVIDENCE_LABELS:
        return ""
    return evidence_label


def iter_selected_qd_chunks(selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Return selected QD detection-control chunks from supported JSON shapes."""

    selected = selection.get("selected_qd_chunks")
    if isinstance(selected, list):
        return [item for item in selected if isinstance(item, dict)]

    selected_one = selection.get("selected_qd_chunk")
    if isinstance(selected_one, dict):
        return [selected_one]

    return []


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

    rows: list[dict[str, str]] = []
    top_chunks = selection.get("top_chunks")
    if not isinstance(top_chunks, list):
        top_chunks = []

    for chunk in top_chunks:
        if not isinstance(chunk, dict):
            continue

        chunk_name = normalize_text(chunk.get("name"))
        evidence_label = normalize_evidence_label(chunk.get("evidence_label"))
        if not chunk_name or not evidence_label:
            continue

        rows.append(
            {
                "chunk_name": chunk_name,
                "failure_label": failure_label,
                "failure_text": target_text,
                "evidence_label": evidence_label,
                "evidence_span": normalize_text(chunk.get("evidence_span")),
                "support_capability": normalize_text(chunk.get("support_capability")),
                "justification": normalize_text(chunk.get("justification") or chunk.get("reason")),
                "analysis_id": normalize_text(
                    selection.get("analysis_id") or analysis_item.get("analysis_id")
                ),
                "query_type": query_type,
                "query_text": get_query_text(item),
                "source_file": str(source_file) if source_file else "",
                "qd_id": "",
                "qd_title": "",
            }
        )

    for chunk in iter_selected_qd_chunks(selection):
        chunk_name = normalize_text(chunk.get("name"))
        if not chunk_name:
            continue

        rows.append(
            {
                "chunk_name": chunk_name,
                "failure_label": failure_label,
                "failure_text": target_text,
                "evidence_label": "detection",
                "evidence_span": normalize_text(chunk.get("objectives")),
                "support_capability": normalize_text(chunk.get("support_capability")),
                "justification": normalize_text(chunk.get("justification") or chunk.get("reason")),
                "analysis_id": normalize_text(
                    selection.get("analysis_id") or analysis_item.get("analysis_id")
                ),
                "query_type": query_type,
                "query_text": get_query_text(item),
                "source_file": str(source_file) if source_file else "",
                "qd_id": normalize_text(chunk.get("qd_id")),
                "qd_title": normalize_text(chunk.get("qd_title")),
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


def group_rows_by_failure_and_evidence_label(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = row["failure_label"]
        evidence_label = row["evidence_label"]
        if label not in set(QUERY_TYPE_TO_FAILURE_LABEL.values()):
            continue
        if evidence_label not in ALLOWED_EVIDENCE_LABELS:
            continue
        grouped[(label, evidence_label)].append(row)
    return dict(grouped)


def connect_rows(
    rows: list[dict[str, str]],
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    database: str = NEO4J_DATABASE,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Connect selected document chunks to Mode/Cause failure nodes.

    Failure node label is constrained by query type:
    - function_mode/mode -> (:Mode {text: query_mode})
    - cause -> (:Cause {text: query_cause})

    Evidence edge type is copied from selection.evidence_label.
    QD detection-control selections are connected with a detection edge.
    Each edge stores support_capability, evidence_span, and justification,
    plus trace fields for repeatable imports.
    """

    stats = {
        "input_rows": len(rows),
        "created_or_matched_edges": 0,
        "missing_chunks": 0,
        "missing_failure_nodes": 0,
        "missing_failure_node_details": [],
    }

    if not rows:
        return stats

    if dry_run:
        stats["missing_failure_node_details"] = []
        return stats

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            for (failure_label, evidence_label), group_rows in group_rows_by_failure_and_evidence_label(rows).items():
                query = f"""
                UNWIND $rows AS row
                OPTIONAL MATCH (chunk {{name: row.chunk_name}})
                OPTIONAL MATCH (failure:{failure_label} {{text: row.failure_text}})
                WITH row, chunk, failure
                CALL {{
                    WITH row, chunk, failure
                    WITH row, chunk, failure
                    WHERE chunk IS NOT NULL AND failure IS NOT NULL
                    MERGE (chunk)-[rel:{evidence_label} {{
                        analysis_id: row.analysis_id,
                        query_text: row.query_text
                    }}]->(failure)
                    SET rel.support_capability = row.support_capability,
                        rel.evidence_span = row.evidence_span,
                        rel.justification = row.justification,
                        rel.qd_id = row.qd_id,
                        rel.qd_title = row.qd_title,
                        rel.query_type = row.query_type,
                        rel.source_file = row.source_file
                    RETURN count(rel) AS linked
                }}
                RETURN
                    sum(linked) AS linked_count,
                    count(CASE WHEN chunk IS NULL THEN 1 END) AS missing_chunks,
                    count(CASE WHEN failure IS NULL THEN 1 END) AS missing_failure_nodes,
                    collect(DISTINCT CASE
                        WHEN failure IS NULL THEN {{
                            failure_label: row.failure_label,
                            failure_text: row.failure_text,
                            query_type: row.query_type,
                            analysis_id: row.analysis_id,
                            query_text: row.query_text,
                            source_file: row.source_file
                        }}
                    END) AS missing_failure_node_details
                """
                for start in range(0, len(group_rows), batch_size):
                    batch = group_rows[start : start + batch_size]
                    record = session.run(query, rows=batch).single()
                    if not record:
                        continue
                    stats["created_or_matched_edges"] += int(record["linked_count"] or 0)
                    stats["missing_chunks"] += int(record["missing_chunks"] or 0)
                    stats["missing_failure_nodes"] += int(record["missing_failure_nodes"] or 0)
                    stats["missing_failure_node_details"].extend(
                        item
                        for item in record["missing_failure_node_details"]
                        if isinstance(item, dict)
                    )
    finally:
        driver.close()

    stats["missing_failure_node_details"] = dedupe_missing_failure_details(
        stats["missing_failure_node_details"]
    )
    return stats


def dedupe_missing_failure_details(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Deduplicate missing failure-node diagnostics while preserving order."""

    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        failure_label = normalize_text(item.get("failure_label"))
        failure_text = normalize_text(item.get("failure_text"))
        analysis_id = normalize_text(item.get("analysis_id"))
        key = (failure_label, failure_text, analysis_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "failure_label": failure_label,
                "failure_text": failure_text,
                "query_type": normalize_text(item.get("query_type")),
                "analysis_id": analysis_id,
                "query_text": normalize_text(item.get("query_text")),
                "source_file": normalize_text(item.get("source_file")),
            }
        )
    return deduped


def delete_legacy_relationships(
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    database: str = NEO4J_DATABASE,
    dry_run: bool = False,
) -> int:
    """Delete old upstream/downstream/self chunk-to-failure relationships."""

    if dry_run:
        return 0

    query = """
    MATCH (chunk)-[rel:upstream|downstream|self]->(failure)
    WHERE (failure:Mode OR failure:Cause)
    WITH collect(rel) AS legacy_rels
    WITH legacy_rels, size(legacy_rels) AS deleted_count
    UNWIND legacy_rels AS rel
    DELETE rel
    RETURN deleted_count
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            record = session.run(query).single()
            return int(record["deleted_count"] or 0) if record else 0
    finally:
        driver.close()


def connect_document_chunk_graph_to_failure_graph(
    input_dir: Path = Path(__file__).resolve().parent,
    patterns: list[str] | None = None,
    uri: str = NEO4J_URI,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    database: str = NEO4J_DATABASE,
    batch_size: int = 100,
    dry_run: bool = False,
    delete_legacy_edges: bool = True,
) -> dict[str, Any]:
    """Scan query JSON files and write chunk-to-failure evidence edges."""

    input_files = iter_input_files(input_dir, patterns or list(DEFAULT_PATTERNS))
    rows = build_link_rows_from_query_files(input_files)
    legacy_deleted = (
        delete_legacy_relationships(
            uri=uri,
            user=user,
            password=password,
            database=database,
            dry_run=dry_run,
        )
        if delete_legacy_edges
        else 0
    )
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
    stats["legacy_edges_deleted"] = legacy_deleted
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect selected document chunk nodes to failure graph Mode/Cause nodes."
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
            "Defaults to query.json, chunk_selection_results_query_*.json, "
            "and qd_detection_control_results_query_*.json."
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
        help="Parse query JSON and report counts without writing Neo4j evidence edges.",
    )
    parser.add_argument(
        "--keep-legacy-edges",
        action="store_true",
        help="Do not delete old upstream/downstream/self edges before writing new edges.",
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
        delete_legacy_edges=not args.keep_legacy_edges,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
