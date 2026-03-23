import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Any, List
from collections import defaultdict

from dotenv import load_dotenv
from tqdm import tqdm
from neo4j import GraphDatabase
from chromadb.utils import embedding_functions
import numpy as np


# ============================================
# CONFIG
# ============================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# failure/entity json
JSON_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_expand\failure_kb\entity_store.json"

# sentence json
SENTENCE_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_expand\sentence_kb\sentence_store.json"

CAUSE_GROUP = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\process_KB\cause_groups_refined_v2.json"
EFFECT_GROUP = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\process_KB\effect_groups_refined_v2.json"
MODE_GROUP = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\process_KB\mode_groups_refined_v2.json"


# ============================================
# HELPERS
# ============================================

def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


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
    vec = embedder([text])[0]
    _embedding_cache[text] = vec
    return vec


# ============================================
# GRAPH BUILDER
# ============================================

class FMEAVectorKGBuilder:
    def __init__(self, uri, user, password, database="neo4j"):
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
            CREATE CONSTRAINT element_id IF NOT EXISTS
            FOR (n:Element) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT function_id IF NOT EXISTS
            FOR (n:Function) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT mode_id IF NOT EXISTS
            FOR (n:Mode) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT cause_id IF NOT EXISTS
            FOR (n:Cause) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT effect_id IF NOT EXISTS
            FOR (n:Effect) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT failure_id IF NOT EXISTS
            FOR (n:Failure) REQUIRE n.failure_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT product_id IF NOT EXISTS
            FOR (n:Product) REQUIRE n.productPnID IS UNIQUE
            """,
            """
            CREATE CONSTRAINT sentence_group_id IF NOT EXISTS
            FOR (n:SentenceGroup) REQUIRE n.group_id IS UNIQUE
            """
        ]

        with self.driver.session(database=self.database) as session:
            for q in queries:
                session.run(q)

    # ----------------------------------------
    # VECTOR INDEX
    # ----------------------------------------

    def create_vector_indexes(self):
        queries = [
            """
            CREATE VECTOR INDEX element_embedding IF NOT EXISTS
            FOR (n:Element)
            ON (n.embedding)
            OPTIONS {indexConfig:{
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX function_embedding IF NOT EXISTS
            FOR (n:Function)
            ON (n.embedding)
            OPTIONS {indexConfig:{
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX mode_embedding IF NOT EXISTS
            FOR (n:Mode)
            ON (n.embedding)
            OPTIONS {indexConfig:{
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX cause_embedding IF NOT EXISTS
            FOR (n:Cause)
            ON (n.embedding)
            OPTIONS {indexConfig:{
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX effect_embedding IF NOT EXISTS
            FOR (n:Effect)
            ON (n.embedding)
            OPTIONS {indexConfig:{
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX sentence_embedding IF NOT EXISTS
            FOR (n:SentenceGroup)
            ON (n.embedding)
            OPTIONS {indexConfig:{
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """
        ]

        with self.driver.session(database=self.database) as session:
            for q in queries:
                session.run(q)

    # ----------------------------------------
    # NODE CREATION
    # ----------------------------------------

    def merge_element(self, session, item) -> Optional[str]:
        element_id = item.get("element_id")
        if not element_id:
            return None

        text = safe_text(item.get("failure_element_text"))
        if not text:
            return None

        embedding = embed(f"Component: {text}")

        session.run(
            """
            MERGE (e:Element {semantic_id:$id})
            SET e.text=$text,
                e.embedding=$embedding,
                e.source_type=$source_type,
                e.fmea_type=$fmea_type,
                e.name=$text
            """,
            id=element_id,
            text=text,
            embedding=embedding,
            source_type=safe_text(item.get("source_type")),
            fmea_type=safe_text(item.get("fmea_type")),
        )
        return element_id

    def merge_function(self, session, item) -> Optional[str]:
        text = safe_text(item.get("function"))
        if not text:
            return None

        function_id = f"function:{stable_id(text)}"
        embedding = embed(f"Function: {text}")

        session.run(
            """
            MERGE (f:Function {semantic_id:$id})
            SET f.text=$text,
                f.embedding=$embedding,
                f.source_type=$source_type,
                f.fmea_type=$fmea_type,
                f.name=$text
            """,
            id=function_id,
            text=text,
            embedding=embedding,
            source_type=safe_text(item.get("source_type")),
            fmea_type=safe_text(item.get("fmea_type")),
        )
        return function_id

    def merge_mode(self, session, item) -> Optional[str]:
        mode_id = item.get("mode_id")
        if not mode_id:
            return None

        text = safe_text(item.get("failure_mode_text"))
        if not text:
            return None

        embedding = embed(f"Failure mode: {text}")

        session.run(
            """
            MERGE (m:Mode {semantic_id:$id})
            SET m.text=$text,
                m.embedding=$embedding,
                m.source_type=$source_type,
                m.fmea_type=$fmea_type,
                m.name=$text
            """,
            id=mode_id,
            text=text,
            embedding=embedding,
            source_type=safe_text(item.get("source_type")),
            fmea_type=safe_text(item.get("fmea_type")),
        )
        return mode_id

    def merge_cause(self, session, item) -> Optional[str]:
        cause_id = item.get("cause_id")
        if not cause_id:
            return None

        text = safe_text(item.get("failure_cause_text"))
        if not text:
            return None

        embedding = embed(f"Failure cause: {text}")

        session.run(
            """
            MERGE (c:Cause {semantic_id:$id})
            SET c.text=$text,
                c.embedding=$embedding,
                c.source_type=$source_type,
                c.fmea_type=$fmea_type,
                c.discipline=$discipline,
                c.name=$text
            """,
            id=cause_id,
            text=text,
            embedding=embedding,
            source_type=safe_text(item.get("source_type")),
            fmea_type=safe_text(item.get("fmea_type")),
            discipline=safe_text(item.get("discipline"))
        )
        return cause_id

    def merge_effect(self, session, item) -> Optional[str]:
        effect_id = item.get("effect_id")
        if not effect_id:
            return None

        text = safe_text(item.get("failure_effect_text"))
        if not text:
            return None

        embedding = embed(f"Failure effect: {text}")

        session.run(
            """
            MERGE (e:Effect {semantic_id:$id})
            SET e.text=$text,
                e.embedding=$embedding,
                e.source_type=$source_type,
                e.fmea_type=$fmea_type,
                e.name=$text
            """,
            id=effect_id,
            text=text,
            embedding=embedding,
            source_type=safe_text(item.get("source_type")),
            fmea_type=safe_text(item.get("fmea_type")),
        )
        return effect_id

    def merge_failure(self, session, item) -> Optional[str]:
        failure_id = item.get("failure_id")
        if not failure_id:
            return None

        session.run(
            """
            MERGE (f:Failure {failure_id:$fid})
            SET f.name=$name,
                f.file_name=$file_name,
                f.system=$system,
                f.severity=$severity,
                f.occurrence=$occurrence,
                f.detection=$detection,
                f.rpn=$rpn,
                f.source_type=$source_type,
                f.fmea_type=$fmea_type,
                f.product_domain=$product_domain,
                f.productPnID=$productPnID,
                f.released_year=$released_year
            """,
            fid=failure_id,
            name=failure_id,
            file_name=safe_text(item.get("file_name")),
            system=safe_text(item.get("system")),
            severity=item.get("severity"),
            occurrence=item.get("occurrence"),
            detection=item.get("detection"),
            rpn=item.get("rpn"),
            source_type=safe_text(item.get("source_type")),
            fmea_type=safe_text(item.get("fmea_type")),
            product_domain=safe_text(item.get("product_domain")),
            productPnID=item.get("productPnID"),
            released_year=item.get("released_year"),
        )
        return failure_id

    def merge_product(self, session, item) -> Optional[int]:
        pnid = item.get("productPnID")
        if pnid is None:
            return None

        session.run(
            """
            MERGE (p:Product {productPnID:$pnid})
            SET p.name=$name,
                p.product_domain=$domain
            """,
            pnid=pnid,
            name=f"product:{pnid}",
            domain=safe_text(item.get("product_domain"))
        )
        return pnid

    def merge_sentence_group(self, session, group_id: str, text: str, source: str, group_type: str, ref_id: Optional[str] = None):
        text = safe_text(text)
        if not text:
            return None

        embedding = embed(text)
        sentence_count = len([x for x in text.split(".") if safe_text(x)])

        session.run(
            """
            MERGE (s:SentenceGroup {group_id:$gid})
            SET s.text=$text,
                s.embedding=$embedding,
                s.source=$source,
                s.group_type=$group_type,
                s.ref_id=$ref_id,
                s.size=$size,
                s.name=$gid
            """,
            gid=group_id,
            text=text,
            embedding=embedding,
            source=source,
            group_type=group_type,
            ref_id=ref_id,
            size=sentence_count
        )
        return group_id

    # ----------------------------------------
    # RELATIONS: ORIGINAL KG
    # 新结构:
    # Element -> Function -> Mode -> Effect
    # Cause  -> Mode
    # 即: (c:Cause)-[:CAUSES]->(m:Mode)
    # ----------------------------------------

    def create_edges(
        self,
        session,
        element_id: Optional[str],
        function_id: Optional[str],
        mode_id: Optional[str],
        cause_id: Optional[str],
        effect_id: Optional[str],
    ):
        if element_id and function_id:
            session.run(
                """
                MATCH (e:Element {semantic_id:$element})
                MATCH (f:Function {semantic_id:$function})
                MERGE (e)-[:HAS_FUNCTION]->(f)
                """,
                element=element_id,
                function=function_id,
            )

        if function_id and mode_id:
            session.run(
                """
                MATCH (f:Function {semantic_id:$function})
                MATCH (m:Mode {semantic_id:$mode})
                MERGE (f)-[:HAS_MODE]->(m)
                """,
                function=function_id,
                mode=mode_id,
            )

        # Cause -> causes -> Mode
        if cause_id and mode_id:
            session.run(
                """
                MATCH (c:Cause {semantic_id:$cause})
                MATCH (m:Mode {semantic_id:$mode})
                MERGE (c)-[r:CAUSES]->(m)
                ON CREATE SET r.weight = 1
                ON MATCH SET r.weight = r.weight + 1
                """,
                cause=cause_id,
                mode=mode_id,
            )

        if mode_id and effect_id:
            session.run(
                """
                MATCH (m:Mode {semantic_id:$mode})
                MATCH (e:Effect {semantic_id:$effect})
                MERGE (m)-[r:LEADS_TO]->(e)
                ON CREATE SET r.weight = 1
                ON MATCH SET r.weight = r.weight + 1
                """,
                mode=mode_id,
                effect=effect_id,
            )

    # ----------------------------------------
    # FAILURE INSTANCE LAYER
    # ----------------------------------------

    def create_failure_edges(
        self,
        session,
        failure_id: Optional[str],
        element_id: Optional[str],
        function_id: Optional[str],
        mode_id: Optional[str],
        cause_id: Optional[str],
        effect_id: Optional[str],
    ):
        if failure_id and element_id:
            session.run(
                """
                MATCH (f:Failure {failure_id:$fid})
                MATCH (e:Element {semantic_id:$eid})
                MERGE (f)-[:HAS_ELEMENT]->(e)
                """,
                fid=failure_id, eid=element_id
            )

        if failure_id and function_id:
            session.run(
                """
                MATCH (f:Failure {failure_id:$fid})
                MATCH (fn:Function {semantic_id:$fnid})
                MERGE (f)-[:HAS_FUNCTION]->(fn)
                """,
                fid=failure_id, fnid=function_id
            )

        if failure_id and mode_id:
            session.run(
                """
                MATCH (f:Failure {failure_id:$fid})
                MATCH (m:Mode {semantic_id:$mid})
                MERGE (f)-[:HAS_MODE]->(m)
                """,
                fid=failure_id, mid=mode_id
            )

        if failure_id and cause_id:
            session.run(
                """
                MATCH (f:Failure {failure_id:$fid})
                MATCH (c:Cause {semantic_id:$cid})
                MERGE (f)-[:HAS_CAUSE]->(c)
                """,
                fid=failure_id, cid=cause_id
            )

        if failure_id and effect_id:
            session.run(
                """
                MATCH (f:Failure {failure_id:$fid})
                MATCH (e:Effect {semantic_id:$eid})
                MERGE (f)-[:HAS_EFFECT]->(e)
                """,
                fid=failure_id, eid=effect_id
            )

    # ----------------------------------------
    # PRODUCT LAYER
    # ----------------------------------------

    def link_product_failure(self, session, pnid: Optional[int], failure_id: Optional[str]):
        if pnid is None or not failure_id:
            return

        session.run(
            """
            MATCH (p:Product {productPnID:$pnid})
            MATCH (f:Failure {failure_id:$fid})
            MERGE (p)-[:HAS_FAILURE]->(f)
            """,
            pnid=pnid,
            fid=failure_id
        )

    # ----------------------------------------
    # EVIDENCE LAYER
    # ----------------------------------------

    def link_cause_sentence(self, session, cause_id: str, group_id: str):
        session.run(
            """
            MATCH (c:Cause {semantic_id:$cid})
            MATCH (s:SentenceGroup {group_id:$gid})
            MERGE (c)-[:INFERRED_BY]->(s)
            """,
            cid=cause_id,
            gid=group_id
        )

    def link_failure_sentence(self, session, failure_id: str, group_id: str):
        session.run(
            """
            MATCH (f:Failure {failure_id:$fid})
            MATCH (s:SentenceGroup {group_id:$gid})
            MERGE (f)-[:INFERRED_BY]->(s)
            """,
            fid=failure_id,
            gid=group_id
        )

    # ----------------------------------------
    # BUILD MAIN GRAPH
    # ----------------------------------------

    def build_graph(self, json_path):
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))

        skipped = {
            "element": 0,
            "function": 0,
            "mode": 0,
            "cause": 0,
            "effect": 0,
            "failure": 0,
            "product": 0,
        }

        with self.driver.session(database=self.database) as session:
            for item in tqdm(data.values(), total=len(data), desc="Building main KG"):
                element_id = self.merge_element(session, item)
                if not element_id:
                    skipped["element"] += 1

                function_id = self.merge_function(session, item)
                if not function_id:
                    skipped["function"] += 1

                mode_id = self.merge_mode(session, item)
                if not mode_id:
                    skipped["mode"] += 1

                cause_id = self.merge_cause(session, item)
                if not cause_id:
                    skipped["cause"] += 1

                effect_id = self.merge_effect(session, item)
                if not effect_id:
                    skipped["effect"] += 1

                failure_id = self.merge_failure(session, item)
                if not failure_id:
                    skipped["failure"] += 1

                pnid = self.merge_product(session, item)
                if pnid is None:
                    skipped["product"] += 1

                # 原始语义层
                self.create_edges(
                    session=session,
                    element_id=element_id,
                    function_id=function_id,
                    mode_id=mode_id,
                    cause_id=cause_id,
                    effect_id=effect_id,
                )

                # Failure 实例层
                self.create_failure_edges(
                    session=session,
                    failure_id=failure_id,
                    element_id=element_id,
                    function_id=function_id,
                    mode_id=mode_id,
                    cause_id=cause_id,
                    effect_id=effect_id
                )

                # Product 聚合层
                self.link_product_failure(
                    session=session,
                    pnid=pnid,
                    failure_id=failure_id
                )

        print("Build finished.")
        print("Skipped counts:", skipped)
        print("Embedding cache size:", len(_embedding_cache))

    # ----------------------------------------
    # BUILD CAUSE -> SENTENCE GROUP
    # ----------------------------------------

    def build_cause_sentence_groups(self, sentence_json_path):
        sentence_data = json.loads(Path(sentence_json_path).read_text(encoding="utf-8"))

        cause_groups = defaultdict(list)

        for sid, item in sentence_data.items():
            cause_id = item.get("cause_id")
            text = safe_text(item.get("text"))

            if cause_id and text:
                cause_groups[cause_id].append(text)

        with self.driver.session(database=self.database) as session:
            for cause_id, texts in tqdm(cause_groups.items(), desc="Building cause sentence groups"):
                texts = unique_preserve_order(texts)
                combined_text = ". ".join(texts)

                group_id = f"cause_group:{cause_id}"

                self.merge_sentence_group(
                    session=session,
                    group_id=group_id,
                    text=combined_text,
                    source="sentence_store",
                    group_type="cause_group",
                    ref_id=cause_id
                )

                self.link_cause_sentence(
                    session=session,
                    cause_id=cause_id,
                    group_id=group_id
                )

    # ----------------------------------------
    # BUILD 8D FAILURE -> SENTENCE GROUP
    # ----------------------------------------

    def build_failure_sentence_groups(self, entity_json_path, sentence_json_path):
        entity_data = json.loads(Path(entity_json_path).read_text(encoding="utf-8"))
        sentence_data = json.loads(Path(sentence_json_path).read_text(encoding="utf-8"))

        with self.driver.session(database=self.database) as session:
            for item in tqdm(entity_data.values(), desc="Building failure sentence groups"):
                if safe_text(item.get("source_type")) != "8D":
                    continue

                failure_id = item.get("failure_id")
                if not failure_id:
                    continue

                sentence_ids = item.get("supporting_sentence_ids", [])
                if not sentence_ids:
                    continue

                texts = []
                for sid in sentence_ids:
                    s_item = sentence_data.get(sid)
                    if not s_item:
                        continue
                    s_text = safe_text(s_item.get("text"))
                    if s_text:
                        texts.append(s_text)

                texts = unique_preserve_order(texts)
                if not texts:
                    continue

                combined_text = ". ".join(texts)
                group_id = f"failure_group:{failure_id}"

                self.merge_sentence_group(
                    session=session,
                    group_id=group_id,
                    text=combined_text,
                    source="8D",
                    group_type="failure_group",
                    ref_id=failure_id
                )

                self.link_failure_sentence(
                    session=session,
                    failure_id=failure_id,
                    group_id=group_id
                )

    # =========================================================
    # Common helper
    # =========================================================
    def _compute_avg_embedding(self, session, label: str, member_ids: List[str]):
        query = f"""
        MATCH (n:{label})
        WHERE n.semantic_id IN $member_ids
        RETURN n.semantic_id AS sid, n.embedding AS emb
        """
        result = session.run(query, member_ids=member_ids)
        rows = [r for r in result if r["emb"] is not None]

        if len(rows) <= 1:
            return None, rows

        embeddings = [r["emb"] for r in rows]
        avg_embedding = np.mean(np.array(embeddings, dtype=float), axis=0).tolist()
        return avg_embedding, rows

    # =========================================================
    # Cause Group Merge
    # 新结构下:
    # SubCause -[:CAUSES]-> Mode
    # 迁移成:
    # GroupCause -[:CAUSES]-> Mode
    # =========================================================
    def merge_cause_group_v3(self, group_item: Dict[str, Any]):
        member_ids = group_item.get("member_node_ids", [])
        canonical_text = safe_text(group_item.get("canonical_text"))
        group_id = safe_text(group_item.get("group_id"))

        if not member_ids or len(member_ids) <= 1:
            return False

        if not canonical_text or not group_id:
            return False

        group_semantic_id = f"group:{group_id}"

        with self.driver.session(database=self.database) as session:
            # 1) 获取已有 Cause embedding
            avg_embedding, rows = self._compute_avg_embedding(session, "Cause", member_ids)
            if avg_embedding is None:
                return False

            # 2) 创建 group cause 节点
            session.run(
                """
                MERGE (cg:Cause {semantic_id:$group_semantic_id})
                SET cg.text = $canonical_text,
                    cg.name = $canonical_text,
                    cg.embedding = $embedding,
                    cg.is_group = true
                """,
                group_semantic_id=group_semantic_id,
                canonical_text=canonical_text,
                embedding=avg_embedding
            )

            # 3) 标记 SubCause，并建立 BELONGS_TO
            session.run(
                """
                MATCH (cg:Cause {semantic_id:$group_semantic_id})
                MATCH (c:Cause)
                WHERE c.semantic_id IN $member_ids
                  AND c.semantic_id <> $group_semantic_id

                SET c:SubCause,
                    c.is_group = false

                MERGE (c)-[:BELONGS_TO]->(cg)
                """,
                group_semantic_id=group_semantic_id,
                member_ids=member_ids
            )

            # 4) 汇总关系到 group，并删除指向 SubCause 的旧边
            session.run(
                """
                MATCH (cg:Cause {semantic_id:$group_semantic_id})
                MATCH (sc:SubCause)-[:BELONGS_TO]->(cg)
                WHERE sc.semantic_id IN $member_ids

                // ---- SubCause -> Mode 迁移到 GroupCause -> Mode ----
                OPTIONAL MATCH (sc)-[r1:CAUSES]->(m:Mode)
                FOREACH (_ IN CASE WHEN r1 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (cg)-[r2:CAUSES]->(m)
                    ON CREATE SET r2.weight = coalesce(r1.weight, 1)
                    ON MATCH SET r2.weight = coalesce(r2.weight, 0) + coalesce(r1.weight, 1)
                    DELETE r1
                )

                WITH cg, sc

                // ---- Failure -> SubCause 迁移到 Failure -> GroupCause ----
                OPTIONAL MATCH (f:Failure)-[r3:HAS_CAUSE]->(sc)
                FOREACH (_ IN CASE WHEN r3 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (f)-[:HAS_CAUSE]->(cg)
                    DELETE r3
                )

                WITH cg, sc

                // ---- Sentence 复制到 group；原 SubCause->Sentence 保留 ----
                OPTIONAL MATCH (sc)-[r4:INFERRED_BY]->(s:SentenceGroup)
                FOREACH (_ IN CASE WHEN r4 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (cg)-[:INFERRED_BY]->(s)
                )
                """,
                group_semantic_id=group_semantic_id,
                member_ids=member_ids
            )

        return True

    # =========================================================
    # Effect Group Merge
    # Effect 方向不变:
    # Mode -[:LEADS_TO]-> Effect
    # =========================================================
    def merge_effect_group_v3(self, group_item: Dict[str, Any]):
        member_ids = group_item.get("member_node_ids", [])
        canonical_text = safe_text(group_item.get("canonical_text"))
        group_id = safe_text(group_item.get("group_id"))

        if not member_ids or len(member_ids) <= 1:
            return False

        if not canonical_text or not group_id:
            return False

        group_semantic_id = f"group:{group_id}"

        with self.driver.session(database=self.database) as session:
            # 1) 获取已有 Effect embedding
            avg_embedding, rows = self._compute_avg_embedding(session, "Effect", member_ids)
            if avg_embedding is None:
                return False

            # 2) 创建 Effect Group
            session.run(
                """
                MERGE (eg:Effect {semantic_id:$group_semantic_id})
                SET eg.text = $canonical_text,
                    eg.name = $canonical_text,
                    eg.embedding = $embedding,
                    eg.is_group = true
                """,
                group_semantic_id=group_semantic_id,
                canonical_text=canonical_text,
                embedding=avg_embedding
            )

            # 3) 标记 SubEffect + BELONGS_TO
            session.run(
                """
                MATCH (eg:Effect {semantic_id:$group_semantic_id})
                MATCH (e:Effect)
                WHERE e.semantic_id IN $member_ids
                  AND e.semantic_id <> $group_semantic_id

                SET e:SubEffect,
                    e.is_group = false

                MERGE (e)-[:BELONGS_TO]->(eg)
                """,
                group_semantic_id=group_semantic_id,
                member_ids=member_ids
            )

            # 4) 迁移关系 + 删除旧边
            session.run(
                """
                MATCH (eg:Effect {semantic_id:$group_semantic_id})
                MATCH (se:SubEffect)-[:BELONGS_TO]->(eg)
                WHERE se.semantic_id IN $member_ids

                // ---- Mode -> SubEffect → Mode -> EffectGroup ----
                OPTIONAL MATCH (m:Mode)-[r1:LEADS_TO]->(se)
                FOREACH (_ IN CASE WHEN r1 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (m)-[r2:LEADS_TO]->(eg)
                    ON CREATE SET r2.weight = coalesce(r1.weight, 1)
                    ON MATCH SET r2.weight = coalesce(r2.weight, 0) + coalesce(r1.weight, 1)
                    DELETE r1
                )

                WITH eg, se

                // ---- Failure -> SubEffect → Failure -> EffectGroup ----
                OPTIONAL MATCH (f:Failure)-[r3:HAS_EFFECT]->(se)
                FOREACH (_ IN CASE WHEN r3 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (f)-[:HAS_EFFECT]->(eg)
                    DELETE r3
                )

                WITH eg, se

                // ---- Sentence → EffectGroup（保留原 SubEffect->Sentence）----
                OPTIONAL MATCH (se)-[r4:INFERRED_BY]->(s:SentenceGroup)
                FOREACH (_ IN CASE WHEN r4 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (eg)-[:INFERRED_BY]->(s)
                )
                """,
                group_semantic_id=group_semantic_id,
                member_ids=member_ids
            )

        return True

    # =========================================================
    # Mode Group Merge
    # Cause -[:CAUSES]-> GroupMode
    # =========================================================
    def merge_mode_group_v3(self, group_item: Dict[str, Any]):
        member_ids = group_item.get("member_node_ids", [])
        canonical_text = safe_text(group_item.get("canonical_text"))
        group_id = safe_text(group_item.get("group_id"))

        if not member_ids or len(member_ids) <= 1:
            return False

        if not canonical_text or not group_id:
            return False

        group_semantic_id = f"group:{group_id}"

        with self.driver.session(database=self.database) as session:
            # 1) 获取 Mode embedding
            avg_embedding, rows = self._compute_avg_embedding(session, "Mode", member_ids)
            if avg_embedding is None:
                return False

            # 2) 创建 Group Mode
            session.run(
                """
                MERGE (gm:Mode {semantic_id:$gid})
                SET gm.text = $text,
                    gm.name = $text,
                    gm.embedding = $embedding,
                    gm.is_group = true
                """,
                gid=group_semantic_id,
                text=canonical_text,
                embedding=avg_embedding
            )

            # 3) SubMode 标记 + BELONGS_TO
            session.run(
                """
                MATCH (gm:Mode {semantic_id:$gid})
                MATCH (m:Mode)
                WHERE m.semantic_id IN $member_ids
                  AND m.semantic_id <> $gid

                SET m:SubMode,
                    m.is_group = false

                MERGE (m)-[:BELONGS_TO]->(gm)
                """,
                gid=group_semantic_id,
                member_ids=member_ids
            )

            # 4) 迁移所有关系到 GroupMode + 删除旧边
            session.run(
                """
                MATCH (gm:Mode {semantic_id:$gid})
                MATCH (sm:SubMode)-[:BELONGS_TO]->(gm)
                WHERE sm.semantic_id IN $member_ids

                // -------------------------
                // Function -> SubMode → GroupMode
                // -------------------------
                OPTIONAL MATCH (f:Function)-[r1:HAS_MODE]->(sm)
                WITH gm, sm, f, r1
                FOREACH (_ IN CASE WHEN r1 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (f)-[:HAS_MODE]->(gm)
                    DELETE r1
                )

                WITH gm, sm

                // -------------------------
                // Cause -> SubMode → Cause -> GroupMode
                // -------------------------
                OPTIONAL MATCH (c:Cause)-[r2:CAUSES]->(sm)
                WITH gm, sm, c, r2, coalesce(r2.weight, 1) AS w
                FOREACH (_ IN CASE WHEN r2 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (c)-[r:CAUSES]->(gm)
                    ON CREATE SET r.weight = w
                    ON MATCH SET r.weight = coalesce(r.weight, 0) + w
                    DELETE r2
                )

                WITH gm, sm

                // -------------------------
                // GroupMode -> Effect
                // -------------------------
                OPTIONAL MATCH (sm)-[r3:LEADS_TO]->(e)
                WITH gm, sm, e, r3, coalesce(r3.weight, 1) AS w
                FOREACH (_ IN CASE WHEN r3 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (gm)-[r:LEADS_TO]->(e)
                    ON CREATE SET r.weight = w
                    ON MATCH SET r.weight = coalesce(r.weight, 0) + w
                    DELETE r3
                )

                WITH gm, sm

                // -------------------------
                // Failure -> SubMode → GroupMode
                // -------------------------
                OPTIONAL MATCH (f:Failure)-[r4:HAS_MODE]->(sm)
                FOREACH (_ IN CASE WHEN r4 IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (f)-[:HAS_MODE]->(gm)
                    DELETE r4
                )
                """,
                gid=group_semantic_id,
                member_ids=member_ids
            )

        return True

    # =========================================================
    # Dispatch
    # =========================================================
    def merge_group_item(self, group_item: Dict[str, Any]):
        """
        自动识别 field_type:
        - cause
        - effect
        - mode

        若 field_type 缺失，则尝试从 group_id 推断。
        """
        field_type = safe_text(group_item.get("field_type")).lower()
        group_id = safe_text(group_item.get("group_id")).lower()

        if not field_type:
            if "cause" in group_id:
                field_type = "cause"
            elif "effect" in group_id:
                field_type = "effect"
            elif "mode" in group_id:
                field_type = "mode"

        if field_type == "cause":
            return self.merge_cause_group_v3(group_item)
        elif field_type == "effect":
            return self.merge_effect_group_v3(group_item)
        elif field_type == "mode":
            return self.merge_mode_group_v3(group_item)
        else:
            return None  # unknown type

    # =========================================================
    # Batch Merge
    # =========================================================
    def merge_all_groups(self, group_json_path):
        groups = json.loads(Path(group_json_path).read_text(encoding="utf-8"))

        total = len(groups)
        created = 0
        skipped_single = 0
        skipped_not_found = 0
        skipped_unknown_type = 0

        for g in tqdm(groups, desc="Merging canonical groups"):
            member_ids = g.get("member_node_ids", [])

            if not member_ids or len(member_ids) <= 1:
                skipped_single += 1
                continue

            result = self.merge_group_item(g)

            if result is True:
                created += 1
            elif result is None:
                skipped_unknown_type += 1
            else:
                skipped_not_found += 1

        print("\n===== Group Summary =====")
        print(f"Total groups          : {total}")
        print(f"Created groups        : {created}")
        print(f"Skipped single nodes  : {skipped_single}")
        print(f"Skipped not found     : {skipped_not_found}")
        print(f"Skipped unknown type  : {skipped_unknown_type}")


# ============================================
# MAIN
# ============================================

def main():
    builder = FMEAVectorKGBuilder(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )

    try:
        print("Creating constraints...")
        builder.create_constraints()

        print("Creating vector indexes...")
        builder.create_vector_indexes()

        print("Building main graph...")
        builder.build_graph(JSON_FILE)

        print("Building cause sentence groups...")
        builder.build_cause_sentence_groups(SENTENCE_JSON)

        print("Building 8D failure sentence groups...")
        builder.build_failure_sentence_groups(JSON_FILE, SENTENCE_JSON)

        print("Vector KG build complete.")

        print("Merge mode groups:")
        builder.merge_all_groups(MODE_GROUP)

        print("\nMerge cause groups:")
        builder.merge_all_groups(CAUSE_GROUP)

        print("\nMerge effect groups:")
        builder.merge_all_groups(EFFECT_GROUP)

    finally:
        builder.close()


if __name__ == "__main__":
    main()