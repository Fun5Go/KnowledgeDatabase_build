import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase


# =========================================================
# CONFIG
# =========================================================
FS_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\FS6303220015R11.pdf"
ESW_TS_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\TS6303220021R05.pdf"
HW_TS_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\TS6303220029R09.pdf"
ESW_QD_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\QD6303220037R03.pdf"
HW_QD_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\QD6303220030R08 IPS3 - Electronics.pdf"
FAT_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\FAT6303220032R08.pdf"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

SECTION_LABELS = ["Chapter", "Section", "Subsection", "Subsubsection"]
SECTION_REL_TYPE = "SHARED"


# =========================================================
# BASIC UTILS
# =========================================================
def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_spaces(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def normalize_discipline(discipline: Optional[str]) -> Optional[str]:
    discipline = safe_text(discipline).upper()
    if not discipline:
        return None
    if discipline not in {"ESW", "HW"}:
        raise ValueError(f"Unsupported discipline: {discipline}. Expected ESW or HW.")
    return discipline


def document_id_from_path(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def normalize_key_text(text: Optional[str]) -> str:
    text = safe_text(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_prefix(prefix: Optional[str]) -> str:
    return re.sub(r"\s+", "", safe_text(prefix)).strip().lower()


def section_match_key(section: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        safe_text(section.get("hierarchy_level")),
        normalize_key_text(section.get("name")),
        normalize_prefix(section.get("prefix")),
    )


def section_name_key(section: Dict[str, Any]) -> Tuple[str, str]:
    return (
        safe_text(section.get("hierarchy_level")),
        normalize_key_text(section.get("name")),
    )


def format_section_label(row: Dict[str, Any]) -> str:
    prefix = safe_text(row.get("prefix"))
    name = safe_text(row.get("name"))
    level = safe_text(row.get("hierarchy_level"))
    semantic_id = safe_text(row.get("semantic_id"))
    return f"level={level} | {prefix} {name}".strip() + f" | {semantic_id}"


def print_section_rows(label: str, rows: List[Dict[str, Any]]):
    print("\n" + "=" * 120)
    print(f"[{label} SECTION DETAILS]")
    print("=" * 120)
    if not rows:
        print("[INFO] No section rows")
        return
    for row in rows:
        print(f"- {format_section_label(row)}")


# =========================================================
# LINKER
# =========================================================
class DocumentLinker:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def link_documents(self, source_document_id: str, target_document_id: str, rel_type: str) -> bool:
        query = f"""
        MATCH (src {{semantic_id: $source_document_id}})
        MATCH (dst {{semantic_id: $target_document_id}})
        MERGE (src)-[:{rel_type}]->(dst)
        RETURN src.semantic_id AS source_id, dst.semantic_id AS target_id
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                source_document_id=source_document_id,
                target_document_id=target_document_id,
            )
            record = result.single()
            if not record:
                print(f"[WARN] Missing document node for {source_document_id} -> {target_document_id} ({rel_type})")
                return False
        print(f"[INFO] Linked document {source_document_id} -[{rel_type}]-> {target_document_id}")
        return True

    def fetch_section_rows(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        if not document_ids:
            return []

        query = """
        MATCH (n)
        WHERE any(lbl IN labels(n) WHERE lbl IN $section_labels)
          AND any(doc_id IN $document_ids WHERE n.semantic_id STARTS WITH (doc_id + '::'))
        RETURN
            split(n.semantic_id, '::')[0] AS document_id,
            n.semantic_id AS semantic_id,
            coalesce(n.prefix, '') AS prefix,
            coalesce(n.name, '') AS name,
            coalesce(n.hierarchy_level, '') AS hierarchy_level
        ORDER BY document_id, prefix, name
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                document_ids=document_ids,
                section_labels=SECTION_LABELS,
            )
            rows = [dict(record) for record in result]
            return rows

    def link_section_nodes(
        self,
        source_document_id: str,
        target_document_id: str,
        rel_type: str = SECTION_REL_TYPE,
    ) -> int:
        rows = self.fetch_section_rows([source_document_id, target_document_id])
        source_rows = [row for row in rows if row["document_id"] == source_document_id]
        target_rows = [row for row in rows if row["document_id"] == target_document_id]

        if not source_rows or not target_rows:
            print(
                f"[WARN] Missing section rows for {source_document_id} -> {target_document_id} ({rel_type})"
            )
            print(
                f"[DEBUG] fetch_section_rows used document_ids=[{source_document_id}, {target_document_id}] "
                f"and matched labels={SECTION_LABELS}"
            )
            return 0

        print_section_rows(f"SOURCE {source_document_id}", source_rows)
        print_section_rows(f"TARGET {target_document_id}", target_rows)

        target_exact: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        target_level_name: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in target_rows:
            target_exact[section_match_key(row)].append(row)
            target_level_name[section_name_key(row)].append(row)

        pairs: List[Dict[str, str]] = []
        seen_pairs = set()

        for source_row in source_rows:
            source_key = section_match_key(source_row)
            candidates = target_exact.get(source_key, [])
            if not candidates:
                name_candidates = target_level_name.get(section_name_key(source_row), [])
                if len(name_candidates) == 1:
                    candidates = name_candidates
                elif len(name_candidates) > 1:
                    same_prefix = [
                        row
                        for row in name_candidates
                        if normalize_prefix(row.get("prefix")) == normalize_prefix(source_row.get("prefix"))
                    ]
                    if len(same_prefix) == 1:
                        candidates = same_prefix
                    else:
                        candidates = same_prefix or name_candidates[:1]

            print(
                f"[MATCH] {source_document_id} source={format_section_label(source_row)} "
                f"candidates={len(candidates)}"
            )

            if len(candidates) != 1:
                continue

            target_row = candidates[0]
            pair_key = (source_row["semantic_id"], target_row["semantic_id"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            pairs.append(
                {
                    "source_id": source_row["semantic_id"],
                    "target_id": target_row["semantic_id"],
                }
            )

        if not pairs:
            print(f"[WARN] No matching sections found for {source_document_id} -> {target_document_id} ({rel_type})")
            return 0

        query = f"""
        UNWIND $rows AS row
        MATCH (src {{semantic_id: row.source_id}})
        MATCH (dst {{semantic_id: row.target_id}})
        MERGE (src)-[:{rel_type}]->(dst)
        """

        with self.driver.session(database=self.database) as session:
            session.run(query, rows=pairs)

        print(
            f"[INFO] Linked {len(pairs)} section nodes {source_document_id} -[{rel_type}]-> {target_document_id}"
        )
        for pair in pairs:
            source_row = next((row for row in source_rows if row["semantic_id"] == pair["source_id"]), None)
            target_row = next((row for row in target_rows if row["semantic_id"] == pair["target_id"]), None)
            if source_row and target_row:
                print(
                    f"[SECTION] {source_document_id} -> {target_document_id} [{rel_type}] "
                    f"{format_section_label(source_row)} -> {format_section_label(target_row)}"
                )
        return len(pairs)

    def link_pair(
        self,
        source_path: str,
        target_path: str,
        rel_type: str,
    ) -> Dict[str, Any]:
        source_document_id = document_id_from_path(source_path)
        target_document_id = document_id_from_path(target_path)

        root_linked = self.link_documents(source_document_id, target_document_id, rel_type)
        section_link_count = self.link_section_nodes(source_document_id, target_document_id)

        return {
            "source_document_id": source_document_id,
            "target_document_id": target_document_id,
            "rel_type": SECTION_REL_TYPE,
            "root_rel_type": rel_type,
            "root_linked": root_linked,
            "section_link_count": section_link_count,
        }

    def run(self):
        plan = [
            (FAT_PATH, FS_PATH, "ACCEPTED"),
            (ESW_QD_PATH, ESW_TS_PATH, "VERIFIED"),
            (HW_QD_PATH, HW_TS_PATH, "VERIFIED"),
        ]

        results = []
        for source_path, target_path, rel_type in plan:
            results.append(self.link_pair(source_path, target_path, rel_type))

        return results


# =========================================================
# MAIN
# =========================================================
def main():
    linker = DocumentLinker(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    try:
        linker.run()
        print("[DONE]")
    finally:
        linker.close()


if __name__ == "__main__":
    main()
