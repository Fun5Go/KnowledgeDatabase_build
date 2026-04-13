from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def fetch_chunks(label: str) -> List[Dict[str, Any]]:
    try:
        from neo4j import GraphDatabase
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'neo4j'.\n"
            f"Current Python executable: {sys.executable}\n"
            "Install it into this exact interpreter, for example:\n"
            f'  "{sys.executable}" -m pip install neo4j'
        ) from exc

    query = f"""
    MATCH (n:{label})
    OPTIONAL MATCH (r:RationaleChunk)-[:RATIONALE_FOR]->(n)
    RETURN
        coalesce(n.name, "") AS name,
        coalesce(n.text, "") AS text,
        [
            item IN collect(
                DISTINCT CASE
                    WHEN r IS NULL THEN NULL
                    ELSE {{
                        name: coalesce(r.name, ""),
                        text: coalesce(r.text, "")
                    }}
                END
            )
            WHERE item IS NOT NULL
        ] AS rationales
    ORDER BY name
    """

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )
    try:
        records, _, _ = driver.execute_query(query, database_=NEO4J_DATABASE)
        return [record.data() for record in records]
    finally:
        driver.close()


def write_json(data: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FSChunk and TSChunk nodes from Neo4j into JSON files, including linked rationales."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURRENT_DIR / "output",
        help="Directory where fs_chunks.json and ts_chunks.json will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    fs_chunks = fetch_chunks("FSChunk")
    ts_chunks = fetch_chunks("TSChunk")

    fs_output = output_dir / "fs_chunks.json"
    ts_output = output_dir / "ts_chunks.json"

    write_json(fs_chunks, fs_output)
    write_json(ts_chunks, ts_output)

    print(f"Exported {len(fs_chunks)} FSChunk records to {fs_output}")
    print(f"Exported {len(ts_chunks)} TSChunk records to {ts_output}")


if __name__ == "__main__":
    main()
