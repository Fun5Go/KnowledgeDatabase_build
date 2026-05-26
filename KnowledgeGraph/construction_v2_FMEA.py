import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from tqdm import tqdm
from chromadb.utils import embedding_functions


# ============================================
# CONFIG
# ============================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

JSONL_FILE = (
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON"
    r"\FMEA_motor_drive_recall.jsonl"
)


# ============================================
# HELPERS
# ============================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = safe_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def text_key(text: str) -> str:
    return " ".join(safe_text(text).lower().split())


def semantic_id(prefix: str, text: str) -> str:
    return f"{prefix}:{stable_id(text_key(text))}"


def get_product_pnid(metadata: Dict[str, Any]) -> Any:
    return metadata.get("productPnID", metadata.get("productPnId"))


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc


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
# GRAPH BUILDER V2
# ============================================

class FMEARecallKGBuilderV2:
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
            CREATE CONSTRAINT product_pnid_v2 IF NOT EXISTS
            FOR (n:Product) REQUIRE n.productPnID IS UNIQUE
            """,
            """
            CREATE CONSTRAINT document_file_name_v2 IF NOT EXISTS
            FOR (n:Document) REQUIRE n.file_name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT element_semantic_id_v2 IF NOT EXISTS
            FOR (n:Element) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT function_semantic_id_v2 IF NOT EXISTS
            FOR (n:Function) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT mode_semantic_id_v2 IF NOT EXISTS
            FOR (n:Mode) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT effect_semantic_id_v2 IF NOT EXISTS
            FOR (n:Effect) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT cause_semantic_id_v2 IF NOT EXISTS
            FOR (n:Cause) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT detection_semantic_id_v2 IF NOT EXISTS
            FOR (n:Detection) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT action_semantic_id_v2 IF NOT EXISTS
            FOR (n:Action) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT prevention_semantic_id_v2 IF NOT EXISTS
            FOR (n:Prevention) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT failure_id_v2 IF NOT EXISTS
            FOR (n:Failure) REQUIRE n.failure_id IS UNIQUE
            """,
        ]

        with self.driver.session(database=self.database) as session:
            for query in queries:
                session.run(query)

    def create_vector_indexes(self):
        labels = [
            "Element",
            "Function",
            "Mode",
            "Effect",
            "Cause",
            "Detection",
            "Action",
            "Prevention",
        ]

        with self.driver.session(database=self.database) as session:
            for label in labels:
                index_name = f"{label.lower()}_embedding_v2"
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

    def merge_product(self, session, metadata: Dict[str, Any]) -> Optional[Any]:
        pnid = get_product_pnid(metadata)
        if pnid is None or safe_text(pnid) == "":
            return None

        session.run(
            """
            MERGE (p:Product {productPnID:$productPnID})
            SET p.productName = $productName,
                p.productID = $productID,
                p.name = $productName
            """,
            productPnID=pnid,
            productName=safe_text(metadata.get("productName")),
            productID=metadata.get("productID", metadata.get("productId")),
        )
        return pnid

    def merge_document(self, session, item: Dict[str, Any]) -> Optional[str]:
        file_name = safe_text(item.get("file_name"))
        if not file_name:
            return None

        content = item.get("content") or {}
        system_name = safe_text(content.get("system_name"))
        metadata = item.get("metadata") or {}

        session.run(
            """
            MERGE (d:Document {file_name:$file_name})
            SET d.source_type = $source_type,
                d.released = $released,
                d.name = $file_name
            WITH d
            FOREACH (_ IN CASE WHEN $system_name <> '' THEN [1] ELSE [] END |
                SET d.system_name = $system_name,
                    d.system_names =
                        CASE
                            WHEN d.system_names IS NULL THEN [$system_name]
                            WHEN NOT $system_name IN d.system_names THEN d.system_names + $system_name
                            ELSE d.system_names
                        END
            )
            """,
            file_name=file_name,
            source_type=safe_text(item.get("source_type")),
            released=safe_text(metadata.get("released")),
            system_name=system_name,
        )
        return file_name

    def merge_element(self, session, content: Dict[str, Any]) -> Optional[str]:
        text = safe_text(content.get("system_element") or content.get("process_step"))
        if not text:
            return None

        node_id = semantic_id("element", text)
        session.run(
            """
            MERGE (e:Element {semantic_id:$id})
            SET e.text = $text,
                e.name = $text,
                e.severity = null,
                e.embedding = $embedding
            """,
            id=node_id,
            text=text,
            embedding=embed(f"Element: {text}"),
        )
        return node_id

    def merge_text_node(
        self,
        session,
        label: str,
        prefix: str,
        text: str,
        count: Optional[int] = None,
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
            "count": count,
            "extra": extra or {},
        }

        query = f"""
        MERGE (n:{label} {{semantic_id:$id}})
        SET n.text = $text,
            n.name = $text,
            n.embedding = $embedding
        """
        if count is not None:
            query += "\nSET n.count = $count"
        if extra:
            for key in extra:
                query += f"\nSET n.{key} = $extra.{key}"

        session.run(query, **params)
        return node_id

    def merge_detection(self, session, content: Dict[str, Any], rpn: Dict[str, Any]) -> Optional[str]:
        text = safe_text(content.get("current_detection"))
        return self.merge_text_node(
            session=session,
            label="Detection",
            prefix="detection",
            text=text,
            extra={"detection": safe_number(rpn.get("detection"))},
        )

    def merge_action(self, session, content: Dict[str, Any]) -> Optional[str]:
        return self.merge_text_node(
            session=session,
            label="Action",
            prefix="action",
            text=safe_text(content.get("recommended_action")),
        )

    def merge_prevention(self, session, content: Dict[str, Any], rpn: Dict[str, Any]) -> Optional[str]:
        text = safe_text(content.get("controls_prevention"))
        return self.merge_text_node(
            session=session,
            label="Prevention",
            prefix="prevention",
            text=text,
            extra={"occurrence": safe_number(rpn.get("occurrence"))},
        )

    def merge_failure(
        self,
        session,
        item: Dict[str, Any],
        file_name: Optional[str],
    ) -> str:
        metadata = item.get("metadata") or {}
        content = item.get("content") or {}
        rpn = item.get("RPN") or {}

        file_part = safe_text(file_name) or safe_text(item.get("file_name")) or "unknown_file"
        row_index = item.get("row_index")
        if row_index is None or safe_text(row_index) == "":
            row_index = item.get("index")
        if row_index is None or safe_text(row_index) == "":
            row_index = stable_id(json.dumps(item, sort_keys=True, default=str))

        failure_id = f"{file_part}__R{row_index}"

        session.run(
            """
            MERGE (f:Failure {failure_id:$failure_id})
            SET f.name = $failure_id,
                f.file_name = $file_name,
                f.row_index = $row_index,
                f.source_type = $source_type,
                f.fmea_type = $fmea_type,
                f.productPnID = $productPnID,
                f.system_name = $system_name,
                f.severity = $severity,
                f.occurrence = $occurrence,
                f.detection = $detection,
                f.rpn = $rpn
            """,
            failure_id=failure_id,
            file_name=safe_text(file_name),
            row_index=row_index,
            source_type=safe_text(item.get("source_type")),
            fmea_type=safe_text(metadata.get("fmea_type", item.get("fmea_type"))),
            productPnID=get_product_pnid(metadata),
            system_name=safe_text(content.get("system_name")),
            severity=safe_number(rpn.get("severity")),
            occurrence=safe_number(rpn.get("occurrence")),
            detection=safe_number(rpn.get("detection")),
            rpn=safe_number(rpn.get("rpn", rpn.get("RPN"))),
        )
        return failure_id

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

    def link_document_element(self, session, file_name: str, element_id: str):
        if not file_name or not element_id:
            return
        session.run(
            """
            MATCH (d:Document {file_name:$file_name})
            MATCH (e:Element {semantic_id:$element_id})
            MERGE (d)-[:HAS_ELEMENT]->(e)
            """,
            file_name=file_name,
            element_id=element_id,
        )

    def link_document_effect(self, session, file_name: str, effect_id: str):
        if not file_name or not effect_id:
            return
        session.run(
            """
            MATCH (d:Document {file_name:$file_name})
            MATCH (e:Effect {semantic_id:$effect_id})
            MERGE (d)-[:HAS_EFFECT]->(e)
            """,
            file_name=file_name,
            effect_id=effect_id,
        )

    def link_document_failure(self, session, file_name: str, failure_id: str):
        if not file_name or not failure_id:
            return
        session.run(
            """
            MATCH (d:Document {file_name:$file_name})
            MATCH (f:Failure {failure_id:$failure_id})
            MERGE (d)-[:HAS_Failure]->(f)
            """,
            file_name=file_name,
            failure_id=failure_id,
        )

    def create_core_edges(
        self,
        session,
        element_id: Optional[str],
        function_id: Optional[str],
        mode_id: Optional[str],
        effect_id: Optional[str],
        cause_id: Optional[str],
    ):
        if element_id and function_id:
            session.run(
                """
                MATCH (e:Element {semantic_id:$element_id})
                MATCH (f:Function {semantic_id:$function_id})
                MERGE (e)-[:HAS_FUNCTION]->(f)
                """,
                element_id=element_id,
                function_id=function_id,
            )

        if function_id and mode_id:
            session.run(
                """
                MATCH (f:Function {semantic_id:$function_id})
                MATCH (m:Mode {semantic_id:$mode_id})
                MERGE (f)-[:HAS_MODE]->(m)
                """,
                function_id=function_id,
                mode_id=mode_id,
            )

        if element_id and mode_id and not function_id:
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

        if cause_id and mode_id:
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

    def create_control_edges(
        self,
        session,
        action_id: Optional[str],
        detection_id: Optional[str],
        prevention_id: Optional[str],
        cause_id: Optional[str],
    ):
        if not cause_id:
            return

        if action_id:
            session.run(
                """
                MATCH (a:Action {semantic_id:$action_id})
                MATCH (c:Cause {semantic_id:$cause_id})
                MERGE (a)-[:RECOMMENDS]->(c)
                """,
                action_id=action_id,
                cause_id=cause_id,
            )

        if detection_id:
            session.run(
                """
                MATCH (d:Detection {semantic_id:$detection_id})
                MATCH (c:Cause {semantic_id:$cause_id})
                MERGE (d)-[:CONTROLS]->(c)
                """,
                detection_id=detection_id,
                cause_id=cause_id,
            )

        if prevention_id:
            session.run(
                """
                MATCH (p:Prevention {semantic_id:$prevention_id})
                MATCH (c:Cause {semantic_id:$cause_id})
                MERGE (p)-[:CONTROLS]->(c)
                """,
                prevention_id=prevention_id,
                cause_id=cause_id,
            )

    def create_failure_edges(
        self,
        session,
        failure_id: Optional[str],
        element_id: Optional[str],
        function_id: Optional[str],
        mode_id: Optional[str],
        cause_id: Optional[str],
        effect_id: Optional[str],
        detection_id: Optional[str],
        prevention_id: Optional[str],
        action_id: Optional[str],
    ):
        if not failure_id:
            return

        targets = [
            ("Element", element_id, "HAS_ELEMENT"),
            ("Function", function_id, "HAS_FUNCTION"),
            ("Mode", mode_id, "HAS_MODE"),
            ("Cause", cause_id, "HAS_CAUSE"),
            ("Effect", effect_id, "HAS_EFFECT"),
            ("Detection", detection_id, "HAS_CONTROLS"),
            ("Prevention", prevention_id, "HAS_CONTROLS"),
            ("Action", action_id, "HAS_ACTION"),
        ]

        for label, node_id, rel in targets:
            if not node_id:
                continue
            session.run(
                f"""
                MATCH (f:Failure {{failure_id:$failure_id}})
                MATCH (n:{label} {{semantic_id:$node_id}})
                MERGE (f)-[:{rel}]->(n)
                """,
                failure_id=failure_id,
                node_id=node_id,
            )

    # ----------------------------------------
    # BUILD
    # ----------------------------------------

    def _count_repeated_texts(self, records):
        mode_counts = Counter()
        effect_counts = Counter()
        cause_counts = Counter()

        for item in records:
            content = item.get("content") or {}
            mode = safe_text(content.get("failure_mode"))
            effect = safe_text(content.get("failure_effect"))
            cause = safe_text(content.get("failure_cause"))

            if mode:
                mode_counts[text_key(mode)] += 1
            if effect:
                effect_counts[text_key(effect)] += 1
            if cause:
                cause_counts[text_key(cause)] += 1

        return mode_counts, effect_counts, cause_counts

    def build_graph(self, jsonl_path: str):
        records = list(read_jsonl(jsonl_path))
        mode_counts, effect_counts, cause_counts = self._count_repeated_texts(records)

        skipped = Counter()

        with self.driver.session(database=self.database) as session:
            for item in tqdm(records, desc="Building recall FMEA KG v2"):
                metadata = item.get("metadata") or {}
                content = item.get("content") or {}
                rpn = item.get("RPN") or {}

                product_pnid = self.merge_product(session, metadata)
                if product_pnid is None:
                    skipped["product"] += 1

                file_name = self.merge_document(session, item)
                if not file_name:
                    skipped["document"] += 1

                if product_pnid is not None and file_name:
                    self.link_product_document(session, product_pnid, file_name)

                failure_id = self.merge_failure(
                    session=session,
                    item=item,
                    file_name=file_name,
                )
                if file_name:
                    self.link_document_failure(session, file_name, failure_id)

                element_id = self.merge_element(session, content)
                if not element_id:
                    skipped["element"] += 1
                elif file_name:
                    self.link_document_element(session, file_name, element_id)

                function_text = safe_text(content.get("function"))
                function_id = self.merge_text_node(
                    session, "Function", "function", function_text
                )
                if not function_id:
                    skipped["function"] += 1

                mode_text = safe_text(content.get("failure_mode"))
                mode_id = self.merge_text_node(
                    session,
                    "Mode",
                    "mode",
                    mode_text,
                    count=mode_counts.get(text_key(mode_text), 0) if mode_text else None,
                )
                if not mode_id:
                    skipped["mode"] += 1

                effect_text = safe_text(content.get("failure_effect"))
                effect_id = self.merge_text_node(
                    session,
                    "Effect",
                    "effect",
                    effect_text,
                    count=effect_counts.get(text_key(effect_text), 0) if effect_text else None,
                    extra={"severity": safe_number(rpn.get("severity"))},
                )
                if not effect_id:
                    skipped["effect"] += 1

                if file_name and effect_id and not element_id:
                    self.link_document_effect(session, file_name, effect_id)

                cause_text = safe_text(content.get("failure_cause"))
                cause_id = self.merge_text_node(
                    session,
                    "Cause",
                    "cause",
                    cause_text,
                    count=cause_counts.get(text_key(cause_text), 0) if cause_text else None,
                    extra={"discipline": safe_text(content.get("cause_discipline"))},
                )
                if not cause_id:
                    skipped["cause"] += 1

                self.create_core_edges(
                    session=session,
                    element_id=element_id,
                    function_id=function_id,
                    mode_id=mode_id,
                    effect_id=effect_id,
                    cause_id=cause_id,
                )

                detection_id = self.merge_detection(session, content, rpn)
                if not detection_id:
                    skipped["detection"] += 1

                action_id = self.merge_action(session, content)
                if not action_id:
                    skipped["action"] += 1

                prevention_id = self.merge_prevention(session, content, rpn)
                if not prevention_id:
                    skipped["prevention"] += 1

                self.create_control_edges(
                    session=session,
                    action_id=action_id,
                    detection_id=detection_id,
                    prevention_id=prevention_id,
                    cause_id=cause_id,
                )

                self.create_failure_edges(
                    session=session,
                    failure_id=failure_id,
                    element_id=element_id,
                    function_id=function_id,
                    mode_id=mode_id,
                    cause_id=cause_id,
                    effect_id=effect_id,
                    detection_id=detection_id,
                    prevention_id=prevention_id,
                    action_id=action_id,
                )

        print("Build finished.")
        print("Records:", len(records))
        print("Skipped counts:", dict(skipped))
        print("Embedding cache size:", len(_embedding_cache))


# ============================================
# MAIN
# ============================================

def main():
    builder = FMEARecallKGBuilderV2(
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

        print("Building recall FMEA KG v2...")
        builder.build_graph(JSONL_FILE)

        print("Recall FMEA KG v2 build complete.")
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
