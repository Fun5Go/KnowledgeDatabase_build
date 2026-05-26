import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from chromadb.utils import embedding_functions
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

BASE_DIR = Path(__file__).resolve().parents[1]
KB_DATA_ROOT = BASE_DIR.parent / "KB_motor_drives_complete"
SENTENCE_KB_DIR = KB_DATA_ROOT / "sentence_kb"
FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"

JSON_ROOT = Path(
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_MD\failure_identification"
)


# ============================================
# HELPERS
# ============================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def text_key(text: str) -> str:
    return " ".join(safe_text(text).lower().split())


def semantic_id(prefix: str, text: str) -> str:
    return f"{prefix}:{stable_id(text_key(text))}"


def read_json_files(root: Path) -> Iterable[Dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(f"JSON_ROOT does not exist: {root}")

    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON file: {path}") from exc

        if isinstance(data, dict):
            data["_source_path"] = str(path)
            yield data
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item["_source_path"] = str(path)
                    yield item


def sentence_group_text(entities: Any) -> str:
    if not isinstance(entities, list):
        return ""
    sentences = []
    for entity in entities:
        if isinstance(entity, dict):
            text = safe_text(entity.get("text"))
        else:
            text = safe_text(entity)
        if text:
            sentences.append(text)
    return "\n".join(sentences)


def sentence_ids(entities: Any) -> List[str]:
    if not isinstance(entities, list):
        return []
    ids = []
    for entity in entities:
        if isinstance(entity, dict):
            sid = safe_text(entity.get("sentence_id"))
            if sid:
                ids.append(sid)
    return ids


def source_sections(entities: Any) -> List[str]:
    if not isinstance(entities, list):
        return []
    sections = []
    for entity in entities:
        if isinstance(entity, dict):
            section = safe_text(entity.get("source_section"))
            if section and section not in sections:
                sections.append(section)
    return sections


def collect_product_pnids(records: Iterable[Dict[str, Any]]) -> List[Any]:
    pnids = []
    seen = set()
    for item in records:
        for document in item.get("documents") or []:
            pnid = document.get("productPnId", document.get("productPnID"))
            if pnid is None or safe_text(pnid) == "":
                continue
            key = safe_text(pnid)
            if key not in seen:
                seen.add(key)
                pnids.append(pnid)
    return pnids


# ============================================
# EMBEDDING
# ============================================

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_embedding_cache: Dict[str, list] = {}


def embed(text: str):
    text = safe_text(text)
    if not text:
        return None
    if text in _embedding_cache:
        return _embedding_cache[text]
    vector = embedder([text])[0]
    _embedding_cache[text] = vector
    return vector


# ============================================
# GRAPH BUILDER
# ============================================

class EightDKGBuilder:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    # ----------------------------------------
    # SCHEMA
    # ----------------------------------------

    def create_constraints(self):
        queries = [
            """
            CREATE CONSTRAINT product_pnid_8d IF NOT EXISTS
            FOR (n:Product) REQUIRE n.productPnID IS UNIQUE
            """,
            """
            CREATE CONSTRAINT document_file_name_8d IF NOT EXISTS
            FOR (n:Document) REQUIRE n.file_name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT element_semantic_id_8d IF NOT EXISTS
            FOR (n:Element) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT mode_semantic_id_8d IF NOT EXISTS
            FOR (n:Mode) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT effect_semantic_id_8d IF NOT EXISTS
            FOR (n:Effect) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT cause_semantic_id_8d IF NOT EXISTS
            FOR (n:Cause) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT sentence_semantic_id_8d IF NOT EXISTS
            FOR (n:Sentence) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT d5_semantic_id_8d IF NOT EXISTS
            FOR (n:D5) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT d6_semantic_id_8d IF NOT EXISTS
            FOR (n:D6) REQUIRE n.semantic_id IS UNIQUE
            """,
        ]

        with self.driver.session(database=self.database) as session:
            for query in queries:
                session.run(query)

    def create_vector_indexes(self):
        labels = ["Element", "Mode", "Effect", "Cause", "Sentence"]

        with self.driver.session(database=self.database) as session:
            for label in labels:
                index_name = f"{label.lower()}_embedding_8d"
                session.run(
                    f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:{label})
                    ON (n.embedding)
                    OPTIONS {{indexConfig:{{
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }}}}
                    """
                )

    # ----------------------------------------
    # NODE MERGE
    # ----------------------------------------

    def merge_product(self, session, document: Dict[str, Any]) -> Optional[Any]:
        pnid = document.get("productPnId", document.get("productPnID"))
        if pnid is None or safe_text(pnid) == "":
            return None

        product_name = safe_text(document.get("product_name"))
        session.run(
            """
            MERGE (p:Product {productPnID:$productPnID})
            SET p.productName = $productName,
                p.name = $productName,
                p.product_domain = $productDomain,
                p.parent_PN = $parentPN
            """,
            productPnID=pnid,
            productName=product_name,
            productDomain=safe_text(document.get("product_domain")),
            parentPN=safe_text(document.get("parent_PN")),
        )
        return pnid

    def merge_document(
        self,
        session,
        document: Dict[str, Any],
        system_name: str,
    ) -> Optional[str]:
        file_name = safe_text(document.get("file_name"))
        if not file_name:
            return None

        session.run(
            """
            MERGE (d:Document {file_name:$file_name})
            SET d.project_name = $project_name,
                d.system_name = $system_name,
                d.name = $file_name
            REMOVE d.released_date,
                   d.released,
                   d.source_path,
                   d.fmea_type,
                   d.source_type
            """,
            file_name=file_name,
            project_name=safe_text(document.get("project_name")),
            system_name=safe_text(system_name),
        )
        return file_name

    def merge_text_node(
        self,
        session,
        label: str,
        prefix: str,
        text: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        text = safe_text(text)
        if not text:
            return None

        node_id = semantic_id(prefix, text)
        params = {
            "id": node_id,
            "text": text,
            "embedding": embed(f"{label}: {text}"),
            "extra": extra or {},
        }

        query = f"""
        MERGE (n:{label} {{semantic_id:$id}})
        SET n.text = $text,
            n.embedding = $embedding
        """
        if label in {"Element", "Mode", "Effect", "Cause"}:
            query += """
REMOVE n.name,
       n.source_type,
       n.file_name,
       n.fmea_type,
       n.failure_id,
       n.failure_level,
       n.cause_ID,
       n.cause_level,
       n.cause_parent,
       n.discipline,
       n.confidence,
       n.severity,
       n.count
            """
        else:
            query += "\nSET n.name = $text"
        if extra and label not in {"Element", "Mode", "Effect", "Cause"}:
            for key in extra:
                query += f"\nSET n.{key} = $extra.{key}"

        session.run(query, **params)
        return node_id

    def merge_sentence(
        self,
        session,
        group_id: str,
        text: str,
        entities: Any,
        group_type: str,
        ref_id: Optional[str],
        file_name: str,
    ) -> Optional[str]:
        text = safe_text(text)
        if not text:
            return None

        node_id = f"sentence:{stable_id(group_id)}"
        ids = sentence_ids(entities)
        sections = source_sections(entities)

        session.run(
            """
            MERGE (s:Sentence {semantic_id:$id})
            SET s.text = $text,
                s.name = $group_id,
                s.embedding = $embedding,
                s.group_id = $group_id,
                s.group_type = $group_type,
                s.ref_id = $ref_id,
                s.file_name = $file_name,
                s.sentence_ids = $sentence_ids,
                s.source_sections = $source_sections,
                s.size = $size,
                s.source_type = '8D'
            """,
            id=node_id,
            text=text,
            embedding=embed(f"Sentence: {text}"),
            group_id=group_id,
            group_type=group_type,
            ref_id=ref_id,
            file_name=file_name,
            sentence_ids=ids,
            source_sections=sections,
            size=len(ids) if ids else len([line for line in text.splitlines() if line.strip()]),
        )
        return node_id

    def merge_d_node(
        self,
        session,
        label: str,
        prefix: str,
        file_name: str,
        text: str,
    ) -> Optional[str]:
        text = safe_text(text)
        if not text:
            return None

        node_id = f"{prefix}:{stable_id(file_name + ':' + text_key(text))}"
        session.run(
            f"""
            MERGE (n:{label} {{semantic_id:$id}})
            SET n.text = $text,
                n.raw_context = $text,
                n.file_name = $file_name,
                n.source_type = '8D',
                n.name = $name
            """,
            id=node_id,
            text=text,
            file_name=file_name,
            name=f"{file_name}_{label}",
        )
        return node_id

    # ----------------------------------------
    # RELATION MERGE
    # ----------------------------------------

    def link_product_document(self, session, product_pnid: Any, file_name: str):
        if product_pnid is None or not file_name:
            return
        session.run(
            """
            MATCH (p:Product {productPnID:$productPnID})
            MATCH (d:Document {file_name:$file_name})
            MERGE (p)-[:HAS_DOCUMENT]->(d)
            """,
            productPnID=product_pnid,
            file_name=file_name,
        )

    def cleanup_stale_product_entry_edges(self, session, product_pnids: List[Any]):
        if not product_pnids:
            return
        session.run(
            """
            UNWIND $productPnIDs AS productPnID
            MATCH (p:Product {productPnID:productPnID})-[r:HAS_ELEMENT|HAS_Mode|HAS_MODE]->()
            DELETE r
            """,
            productPnIDs=product_pnids,
        )

    def link_document_node(self, session, file_name: str, label: str, node_id: str, rel: str):
        if not file_name or not node_id:
            return
        session.run(
            f"""
            MATCH (d:Document {{file_name:$file_name}})
            MATCH (n:{label} {{semantic_id:$node_id}})
            MERGE (d)-[:{rel}]->(n)
            """,
            file_name=file_name,
            node_id=node_id,
        )

    def create_core_edges(
        self,
        session,
        element_id: Optional[str],
        mode_id: Optional[str],
        effect_id: Optional[str],
    ):
        if element_id and mode_id:
            session.run(
                """
                MATCH (e:Element {semantic_id:$element_id})
                MATCH (m:Mode {semantic_id:$mode_id})
                MERGE (e)-[:HAS_MODE]->(m)
                """,
                element_id=element_id,
                mode_id=mode_id,
            )

        if mode_id and effect_id:
            session.run(
                """
                MATCH (m:Mode {semantic_id:$mode_id})
                MATCH (e:Effect {semantic_id:$effect_id})
                MERGE (m)-[r:LEADS_TO]->(e)
                ON CREATE SET r.count = 1
                ON MATCH SET r.count = r.count + 1
                """,
                mode_id=mode_id,
                effect_id=effect_id,
            )

    def link_cause_mode(self, session, cause_id: str, mode_id: str):
        if not cause_id or not mode_id:
            return
        session.run(
            """
            MATCH (c:Cause {semantic_id:$cause_id})
            MATCH (m:Mode {semantic_id:$mode_id})
            MERGE (c)-[r:CAUSES]->(m)
            ON CREATE SET r.count = 1
            ON MATCH SET r.count = r.count + 1
            """,
            cause_id=cause_id,
            mode_id=mode_id,
        )

    def link_sentence_evidence(self, session, sentence_id: str, label: str, target_id: str):
        if not sentence_id or not target_id:
            return
        session.run(
            f"""
            MATCH (s:Sentence {{semantic_id:$sentence_id}})
            MATCH (n:{label} {{semantic_id:$target_id}})
            MERGE (s)-[:EVIDENCE_FOR]->(n)
            """,
            sentence_id=sentence_id,
            target_id=target_id,
        )

    def link_d5_mode(self, session, d5_id: str, mode_id: str):
        if not d5_id or not mode_id:
            return
        session.run(
            """
            MATCH (d5:D5 {semantic_id:$d5_id})
            MATCH (m:Mode {semantic_id:$mode_id})
            MERGE (d5)-[:SOLUTION_FOR]->(m)
            """,
            d5_id=d5_id,
            mode_id=mode_id,
        )

    def link_d6_d5(self, session, d6_id: str, d5_id: str):
        if not d6_id or not d5_id:
            return
        session.run(
            """
            MATCH (d6:D6 {semantic_id:$d6_id})
            MATCH (d5:D5 {semantic_id:$d5_id})
            MERGE (d6)-[:IMPLEMENTS]->(d5)
            """,
            d6_id=d6_id,
            d5_id=d5_id,
        )

    # ----------------------------------------
    # BUILD
    # ----------------------------------------

    def build_graph(self, json_root: Path):
        records = list(read_json_files(json_root))
        product_pnids = collect_product_pnids(records)
        skipped = Counter()

        with self.driver.session(database=self.database) as session:
            self.cleanup_stale_product_entry_edges(session, product_pnids)

            for item in tqdm(records, desc="Building 8D KG"):
                documents = item.get("documents") or []
                document = documents[0] if documents else {}
                failure = item.get("failure") or {}
                sections = item.get("sections") or {}

                system_name = safe_text(item.get("system_name"))
                source_path = safe_text(item.get("_source_path"))

                product_pnid = self.merge_product(session, document)
                if product_pnid is None:
                    skipped["product"] += 1

                file_name = self.merge_document(session, document, system_name)
                if not file_name:
                    skipped["document"] += 1
                    file_name = safe_text(document.get("file_name")) or stable_id(source_path)

                self.link_product_document(session, product_pnid, file_name)

                element_id = self.merge_text_node(
                    session,
                    "Element",
                    "element",
                    safe_text(failure.get("failure_element")),
                )
                if element_id:
                    self.link_document_node(session, file_name, "Element", element_id, "HAS_ELEMENT")
                else:
                    skipped["element"] += 1

                mode_id = self.merge_text_node(
                    session,
                    "Mode",
                    "mode",
                    safe_text(failure.get("failure_mode")),
                )
                if mode_id:
                    if not element_id:
                        self.link_document_node(session, file_name, "Mode", mode_id, "HAS_MODE")
                else:
                    skipped["mode"] += 1

                effect_id = self.merge_text_node(
                    session,
                    "Effect",
                    "effect",
                    safe_text(failure.get("failure_effect")),
                )
                if effect_id:
                    self.link_document_node(session, file_name, "Effect", effect_id, "HAS_EFFECT")
                else:
                    skipped["effect"] += 1

                self.create_core_edges(session, element_id, mode_id, effect_id)

                failure_sentence_text = sentence_group_text(failure.get("supporting_entities"))
                failure_sentence_id = self.merge_sentence(
                    session=session,
                    group_id=f"{file_name}:failure_evidence",
                    text=failure_sentence_text,
                    entities=failure.get("supporting_entities"),
                    group_type="failure_mode_evidence",
                    ref_id=safe_text(failure.get("failure_ID")),
                    file_name=file_name,
                )
                if failure_sentence_id and mode_id:
                    self.link_sentence_evidence(session, failure_sentence_id, "Mode", mode_id)
                    self.link_document_node(session, file_name, "Sentence", failure_sentence_id, "HAS_SENTENCE")
                elif not failure_sentence_id:
                    skipped["failure_sentence"] += 1

                for index, cause in enumerate(failure.get("root_causes") or [], start=1):
                    cause_text = safe_text(cause.get("failure_cause"))
                    cause_id = self.merge_text_node(
                        session,
                        "Cause",
                        "cause",
                        cause_text,
                    )
                    if cause_id:
                        self.link_cause_mode(session, cause_id, mode_id)
                        self.link_document_node(session, file_name, "Cause", cause_id, "HAS_CAUSE")
                    else:
                        skipped["cause"] += 1

                    cause_sentence_text = sentence_group_text(cause.get("supporting_entities"))
                    cause_sentence_id = self.merge_sentence(
                        session=session,
                        group_id=f"{file_name}:cause_{index}_evidence",
                        text=cause_sentence_text,
                        entities=cause.get("supporting_entities"),
                        group_type="cause_evidence",
                        ref_id=safe_text(cause.get("cause_ID")),
                        file_name=file_name,
                    )
                    if cause_sentence_id and cause_id:
                        self.link_sentence_evidence(session, cause_sentence_id, "Cause", cause_id)
                        self.link_document_node(session, file_name, "Sentence", cause_sentence_id, "HAS_SENTENCE")
                    elif not cause_sentence_id:
                        skipped["cause_sentence"] += 1

                d5_text = safe_text((sections.get("D5") or {}).get("raw_context"))
                d6_text = safe_text((sections.get("D6") or {}).get("raw_context"))

                d5_id = self.merge_d_node(session, "D5", "d5", file_name, d5_text)
                if d5_id:
                    self.link_d5_mode(session, d5_id, mode_id)
                    self.link_document_node(session, file_name, "D5", d5_id, "HAS_D5")
                else:
                    skipped["d5"] += 1

                d6_id = self.merge_d_node(session, "D6", "d6", file_name, d6_text)
                if d6_id:
                    self.link_d6_d5(session, d6_id, d5_id)
                    self.link_document_node(session, file_name, "D6", d6_id, "HAS_D6")
                else:
                    skipped["d6"] += 1

        print("8D KG build finished.")
        print("Records:", len(records))
        print("Skipped counts:", dict(skipped))
        print("Embedding cache size:", len(_embedding_cache))


# ============================================
# MAIN
# ============================================

def main():
    builder = EightDKGBuilder(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    try:
        print("Creating constraints...")
        builder.create_constraints()

        print("Creating vector indexes...")
        builder.create_vector_indexes()

        print("Building 8D KG...")
        builder.build_graph(JSON_ROOT)

        print("8D KG build complete.")
    except ServiceUnavailable as exc:
        print(f"Neo4j connection failed: {exc}")
        print(f"Configured URI: {NEO4J_URI}")
        print("Please start Neo4j or update .env to a reachable Neo4j instance.")
    except AuthError as exc:
        print(f"Neo4j authentication failed: {exc}")
        print(f"Configured URI: {NEO4J_URI}")
        print(f"Configured user: {NEO4J_USER}")
        print("Please verify the Neo4j username and password in .env.")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
