import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from tqdm import tqdm


# ============================================
# CONFIG
# ============================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

JSON_ROOT = Path(
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database"
    r"\GraphRAG\Extraction\qd_dense_top20_QDFAT"
)

QUERY_TYPE_TARGETS = {
    "function_mode": ("Mode", "query_mode"),
    "cause": ("Cause", "query_cause"),
    "effect": ("Effect", "query_effect"),
}


# ============================================
# HELPERS
# ============================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_json_files(root: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError(f"JSON_ROOT does not exist: {root}")

    for path in sorted(root.glob("*.json")):
        try:
            yield path, json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file: {path}") from exc


def get_nested(data: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key) or {}
    return current if isinstance(current, dict) else {}


def target_from_query(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    analysis_item = get_nested(data, "analysis_item")
    query_spec = get_nested(data, "query_result", "query_spec")

    query_type = safe_text(analysis_item.get("query_type") or query_spec.get("query_type"))
    target_config = QUERY_TYPE_TARGETS.get(query_type)
    if not target_config:
        return None, None, query_type

    label, field = target_config
    target_text = safe_text(analysis_item.get(field) or query_spec.get(field))
    return label, target_text, query_type


def find_selected_qd_chunks(value: Any) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    if isinstance(value, dict):
        selected = value.get("selected_qd_chunks")
        if isinstance(selected, list):
            chunks.extend(item for item in selected if isinstance(item, dict))

        for child in value.values():
            chunks.extend(find_selected_qd_chunks(child))

    elif isinstance(value, list):
        for item in value:
            chunks.extend(find_selected_qd_chunks(item))

    return chunks


# ============================================
# GRAPH CONNECTOR
# ============================================

class Neo4jGraphTestConnectionBuilder:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def connect_selected_chunk(
        self,
        session,
        target_label: str,
        target_text: str,
        chunk_info: Dict[str, Any],
    ) -> str:
        chunk_label = safe_text(chunk_info.get("label"))
        chunk_name = safe_text(chunk_info.get("name"))
        support_capability = safe_text(chunk_info.get("support_capability"))

        if not target_label or not target_text:
            return "missing_target"
        if not chunk_label or not chunk_name:
            return "missing_chunk"

        result = session.run(
            f"""
            MATCH (target:{target_label})
            WHERE toLower(trim(target.text)) = toLower(trim($target_text))
            MATCH (chunk:{chunk_label} {{name:$chunk_name}})
            MERGE (chunk)-[r:test_for]->(target)
            SET r.support_capability = $support_capability
            RETURN count(r) AS linked
            """,
            target_text=target_text,
            chunk_name=chunk_name,
            support_capability=support_capability,
        )
        linked = result.single()["linked"]
        return "linked" if linked else "not_found"

    def build_connections(self, json_root: Path):
        files = list(read_json_files(json_root))
        stats = Counter()

        with self.driver.session(database=self.database) as session:
            for path, data in tqdm(files, desc="Connecting QD/FAT test evidence"):
                target_label, target_text, query_type = target_from_query(data)
                if not target_label:
                    stats[f"unsupported_query_type:{query_type}"] += 1
                    continue

                selected_chunks = find_selected_qd_chunks(data)
                if not selected_chunks:
                    stats["no_selected_qd_chunks"] += 1
                    continue

                for chunk_info in selected_chunks:
                    status = self.connect_selected_chunk(
                        session=session,
                        target_label=target_label,
                        target_text=target_text,
                        chunk_info=chunk_info,
                    )
                    stats[status] += 1
                    if status == "not_found":
                        stats[f"not_found_file:{path.name}"] += 1

        print("QD/FAT test connection build finished.")
        print("Files:", len(files))
        print("Stats:", dict(stats))


# ============================================
# MAIN
# ============================================

def main():
    builder = Neo4jGraphTestConnectionBuilder(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    try:
        builder.build_connections(JSON_ROOT)
    except ServiceUnavailable as exc:
        print(f"Neo4j connection failed: {exc}")
        print(f"Configured URI: {NEO4J_URI}")
    except AuthError as exc:
        print(f"Neo4j authentication failed: {exc}")
        print(f"Configured URI: {NEO4J_URI}")
        print(f"Configured user: {NEO4J_USER}")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
