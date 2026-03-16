import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm
from chromadb.utils import embedding_functions


# ============================================
# CONFIG
# ============================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

JSON_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_discipline\failure_kb\entity_store.json"


# ============================================
# HELPERS
# ============================================

def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


# ============================================
# EMBEDDING
# ============================================

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_embedding_cache: Dict[str, list] = {}


def embed(text: str):
    text = safe_text(text)
    if text in _embedding_cache:
        return _embedding_cache[text]
    vec = embedder([text])[0]
    _embedding_cache[text] = vec
    return vec


# ============================================
# GRAPH BUILDER + ENRICHER
# ============================================

class FMEAVectorKGBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

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
            """
        ]

        with self.driver.session() as session:
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
            """
        ]

        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    # ----------------------------------------
    # OPTIONAL RELATIONSHIP INDEXES
    # ----------------------------------------

    def create_relationship_indexes(self):
        queries = [
            """
            CREATE INDEX possible_cause_confidence IF NOT EXISTS
            FOR ()-[r:POSSIBLE_CAUSE]-() ON (r.confidence)
            """,
            """
            CREATE INDEX possible_effect_confidence IF NOT EXISTS
            FOR ()-[r:POSSIBLE_EFFECT]-() ON (r.confidence)
            """,
            """
            CREATE INDEX similar_mode_score IF NOT EXISTS
            FOR ()-[r:SIMILAR_MODE]-() ON (r.score)
            """,
            """
            CREATE INDEX similar_cause_score IF NOT EXISTS
            FOR ()-[r:SIMILAR_CAUSE]-() ON (r.score)
            """,
            """
            CREATE INDEX similar_effect_score IF NOT EXISTS
            FOR ()-[r:SIMILAR_EFFECT]-() ON (r.score)
            """
        ]

        with self.driver.session() as session:
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
                e.fmea_type=$fmea_type
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
                f.fmea_type=$fmea_type
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
                m.fmea_type=$fmea_type
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
                c.fmea_type=$fmea_type
            """,
            id=cause_id,
            text=text,
            embedding=embedding,
            source_type=safe_text(item.get("source_type")),
            fmea_type=safe_text(item.get("fmea_type")),
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
                e.fmea_type=$fmea_type
            """,
            id=effect_id,
            text=text,
            embedding=embedding,
            source_type=safe_text(item.get("source_type")),
            fmea_type=safe_text(item.get("fmea_type")),
        )
        return effect_id

    # ----------------------------------------
    # RELATIONS
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

        if mode_id and cause_id:
            session.run(
                """
                MATCH (m:Mode {semantic_id:$mode})
                MATCH (c:Cause {semantic_id:$cause})
                MERGE (m)-[:CAUSED_BY]->(c)
                """,
                mode=mode_id,
                cause=cause_id,
            )

        if mode_id and effect_id:
            session.run(
                """
                MATCH (m:Mode {semantic_id:$mode})
                MATCH (e:Effect {semantic_id:$effect})
                MERGE (m)-[:LEADS_TO]->(e)
                """,
                mode=mode_id,
                effect=effect_id,
            )

    # ----------------------------------------
    # BUILD BASE GRAPH
    # ----------------------------------------

    def build_graph(self, json_path):
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))

        skipped = {
            "element": 0,
            "function": 0,
            "mode": 0,
            "cause": 0,
            "effect": 0,
        }

        with self.driver.session() as session:
            for item in tqdm(data.values(), total=len(data), desc="Building base KG"):
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

                self.create_edges(
                    session=session,
                    element_id=element_id,
                    function_id=function_id,
                    mode_id=mode_id,
                    cause_id=cause_id,
                    effect_id=effect_id,
                )

        print("Build finished.")
        print("Skipped counts:", skipped)
        print("Embedding cache size:", len(_embedding_cache))

    # ----------------------------------------
    # UTILS
    # ----------------------------------------

    def clear_enrichment_edges(self):
        queries = [
            "MATCH ()-[r:SIMILAR_MODE]->() DELETE r",
            "MATCH ()-[r:SIMILAR_CAUSE]->() DELETE r",
            "MATCH ()-[r:SIMILAR_EFFECT]->() DELETE r",
            "MATCH ()-[r:POSSIBLE_CAUSE]->() DELETE r",
            "MATCH ()-[r:POSSIBLE_EFFECT]->() DELETE r",
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("Old enrichment edges cleared.")

    def count_relationships(self):
        query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS cnt
        ORDER BY cnt DESC
        """
        with self.driver.session() as session:
            result = session.run(query)
            print("\n=== Relationship Counts ===")
            for r in result:
                print(f"{r['rel_type']:<20} {r['cnt']}")

    def count_nodes(self):
        query = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS cnt
        ORDER BY cnt DESC
        """
        with self.driver.session() as session:
            result = session.run(query)
            print("\n=== Node Counts ===")
            for r in result:
                print(f"{r['label']:<12} {r['cnt']}")

    # ----------------------------------------
    # SIMILARITY EDGE BUILDERS
    # ----------------------------------------

    def build_similarity_edges_for_label(
        self,
        label: str,
        index_name: str,
        rel_type: str,
        threshold: float = 0.88,
        top_k: int = 5,
        bidirectional: bool = False,
    ):
        """
        Generic similarity edge builder.
        label: Mode / Cause / Effect
        index_name: mode_embedding / cause_embedding / effect_embedding
        rel_type: SIMILAR_MODE / SIMILAR_CAUSE / SIMILAR_EFFECT
        """

        fetch_query = f"""
        MATCH (n:{label})
        WHERE n.embedding IS NOT NULL
        RETURN n.semantic_id AS id,
               n.text AS text,
               n.embedding AS embedding
        """

        all_edges = 0

        with self.driver.session() as session:
            nodes = list(session.run(fetch_query))
            print(f"\nBuilding {rel_type} edges for {len(nodes)} {label} nodes...")

            for node in tqdm(nodes, desc=f"Building {rel_type}"):
                result = session.run(
                    f"""
                    CALL db.index.vector.queryNodes(
                        '{index_name}',
                        $top_k,
                        $embedding
                    )
                    YIELD node, score
                    WHERE node.semantic_id <> $id
                      AND score >= $threshold
                    RETURN node.semantic_id AS target_id,
                           node.text AS target_text,
                           score
                    ORDER BY score DESC
                    """,
                    embedding=node["embedding"],
                    id=node["id"],
                    top_k=top_k,
                    threshold=threshold,
                )

                for r in result:
                    source_id = node["id"]
                    target_id = r["target_id"]
                    score = float(r["score"])

                    # 去重策略
                    # 若不是双向图，只保留 semantic_id 较小 -> 较大 的单向边，避免重复
                    if not bidirectional and source_id > target_id:
                        continue

                    session.run(
                        f"""
                        MATCH (a:{label} {{semantic_id:$a}})
                        MATCH (b:{label} {{semantic_id:$b}})
                        MERGE (a)-[rel:{rel_type}]->(b)
                        SET rel.score = $score
                        """,
                        a=source_id,
                        b=target_id,
                        score=score,
                    )
                    all_edges += 1

                    if bidirectional:
                        session.run(
                            f"""
                            MATCH (a:{label} {{semantic_id:$a}})
                            MATCH (b:{label} {{semantic_id:$b}})
                            MERGE (b)-[rel:{rel_type}]->(a)
                            SET rel.score = $score
                            """,
                            a=source_id,
                            b=target_id,
                            score=score,
                        )
                        all_edges += 1

        print(f"{rel_type} build complete. Created edges: {all_edges}")

    def build_mode_similarity_edges(self, threshold=0.88, top_k=5, bidirectional=False):
        self.build_similarity_edges_for_label(
            label="Mode",
            index_name="mode_embedding",
            rel_type="SIMILAR_MODE",
            threshold=threshold,
            top_k=top_k,
            bidirectional=bidirectional,
        )

    def build_cause_similarity_edges(self, threshold=0.90, top_k=5, bidirectional=False):
        self.build_similarity_edges_for_label(
            label="Cause",
            index_name="cause_embedding",
            rel_type="SIMILAR_CAUSE",
            threshold=threshold,
            top_k=top_k,
            bidirectional=bidirectional,
        )

    def build_effect_similarity_edges(self, threshold=0.90, top_k=5, bidirectional=False):
        self.build_similarity_edges_for_label(
            label="Effect",
            index_name="effect_embedding",
            rel_type="SIMILAR_EFFECT",
            threshold=threshold,
            top_k=top_k,
            bidirectional=bidirectional,
        )

    # ----------------------------------------
    # PROPAGATION / KB ENRICHMENT
    # ----------------------------------------

    def enrich_possible_causes(self, similarity_threshold=0.90):
        """
        Rule:
        If m1 -[:SIMILAR_MODE]-> m2
        and m1 -[:CAUSED_BY]-> c
        and m2 has no direct CAUSED_BY c
        then add m2 -[:POSSIBLE_CAUSE]-> c
        """

        query = """
        MATCH (m1:Mode)-[s:SIMILAR_MODE]->(m2:Mode)
        MATCH (m1)-[:CAUSED_BY]->(c:Cause)
        WHERE s.score >= $threshold
          AND NOT (m2)-[:CAUSED_BY]->(c)

        MERGE (m2)-[r:POSSIBLE_CAUSE]->(c)
        SET r.confidence = s.score,
            r.source_mode = m1.semantic_id,
            r.rule = 'similar_mode_cause_propagation'
        RETURN count(r) AS created_or_matched
        """

        with self.driver.session() as session:
            result = session.run(query, threshold=similarity_threshold).single()
            print(f"POSSIBLE_CAUSE enrichment complete: {result['created_or_matched']}")

    def enrich_possible_effects(self, similarity_threshold=0.90):
        """
        Rule:
        If m1 -[:SIMILAR_MODE]-> m2
        and m1 -[:LEADS_TO]-> e
        and m2 has no direct LEADS_TO e
        then add m2 -[:POSSIBLE_EFFECT]-> e
        """

        query = """
        MATCH (m1:Mode)-[s:SIMILAR_MODE]->(m2:Mode)
        MATCH (m1)-[:LEADS_TO]->(e:Effect)
        WHERE s.score >= $threshold
          AND NOT (m2)-[:LEADS_TO]->(e)

        MERGE (m2)-[r:POSSIBLE_EFFECT]->(e)
        SET r.confidence = s.score,
            r.source_mode = m1.semantic_id,
            r.rule = 'similar_mode_effect_propagation'
        RETURN count(r) AS created_or_matched
        """

        with self.driver.session() as session:
            result = session.run(query, threshold=similarity_threshold).single()
            print(f"POSSIBLE_EFFECT enrichment complete: {result['created_or_matched']}")

    # ----------------------------------------
    # ADVANCED ENRICHMENT
    # ----------------------------------------

    def enrich_possible_causes_via_similar_cause(self, similarity_threshold=0.90):
        """
        可选增强：
        如果两个 cause 相似，可以帮助后续聚类，但这里不自动建 mode->cause，
        只作为扩展预留。
        """
        query = """
        MATCH (c1:Cause)-[s:SIMILAR_CAUSE]->(c2:Cause)
        WHERE s.score >= $threshold
        RETURN count(s) AS cnt
        """
        with self.driver.session() as session:
            result = session.run(query, threshold=similarity_threshold).single()
            print(f"SIMILAR_CAUSE edges available for advanced reasoning: {result['cnt']}")

    def enrich_possible_effects_via_similar_effect(self, similarity_threshold=0.90):
        query = """
        MATCH (e1:Effect)-[s:SIMILAR_EFFECT]->(e2:Effect)
        WHERE s.score >= $threshold
        RETURN count(s) AS cnt
        """
        with self.driver.session() as session:
            result = session.run(query, threshold=similarity_threshold).single()
            print(f"SIMILAR_EFFECT edges available for advanced reasoning: {result['cnt']}")

    # ----------------------------------------
    # INSPECTION
    # ----------------------------------------

    def print_possible_cause_samples(self, limit=20):
        query = """
        MATCH (m:Mode)-[r:POSSIBLE_CAUSE]->(c:Cause)
        RETURN
            m.text AS mode,
            c.text AS cause,
            r.confidence AS confidence,
            r.rule AS rule,
            r.source_mode AS source_mode
        ORDER BY confidence DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)

            print("\n=== POSSIBLE CAUSE SAMPLES ===")
            for r in result:
                print(f"Mode       : {r['mode']}")
                print(f"Cause      : {r['cause']}")
                print(f"Confidence : {r['confidence']:.4f}")
                print(f"Rule       : {r['rule']}")
                print(f"SourceMode : {r['source_mode']}")
                print("-" * 60)

    def print_possible_effect_samples(self, limit=20):
        query = """
        MATCH (m:Mode)-[r:POSSIBLE_EFFECT]->(e:Effect)
        RETURN
            m.text AS mode,
            e.text AS effect,
            r.confidence AS confidence,
            r.rule AS rule,
            r.source_mode AS source_mode
        ORDER BY confidence DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)

            print("\n=== POSSIBLE EFFECT SAMPLES ===")
            for r in result:
                print(f"Mode       : {r['mode']}")
                print(f"Effect     : {r['effect']}")
                print(f"Confidence : {r['confidence']:.4f}")
                print(f"Rule       : {r['rule']}")
                print(f"SourceMode : {r['source_mode']}")
                print("-" * 60)

    def print_similar_mode_samples(self, limit=20):
        query = """
        MATCH (m1:Mode)-[r:SIMILAR_MODE]->(m2:Mode)
        RETURN
            m1.text AS mode1,
            m2.text AS mode2,
            r.score AS score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)

            print("\n=== SIMILAR MODE SAMPLES ===")
            for r in result:
                print(f"Mode1 : {r['mode1']}")
                print(f"Mode2 : {r['mode2']}")
                print(f"Score : {r['score']:.4f}")
                print("-" * 60)

    # ----------------------------------------
    # QUERY FUNCTIONS
    # ----------------------------------------

    def semantic_failure_chain(self, query_text: str, top_k: int = 5):
        query_vec = embed("Failure mode: " + query_text)

        cypher = """
        CALL db.index.vector.queryNodes(
            'mode_embedding',
            $top_k,
            $embedding
        )
        YIELD node, score

        MATCH (f:Function)-[:HAS_MODE]->(node)
        MATCH (e:Element)-[:HAS_FUNCTION]->(f)

        OPTIONAL MATCH (node)-[:CAUSED_BY]->(c:Cause)
        OPTIONAL MATCH (node)-[:LEADS_TO]->(ef:Effect)
        OPTIONAL MATCH (node)-[:POSSIBLE_CAUSE]->(pc:Cause)
        OPTIONAL MATCH (node)-[:POSSIBLE_EFFECT]->(pe:Effect)

        RETURN
            e.text AS element,
            f.text AS function,
            node.text AS mode,
            c.text AS cause,
            ef.text AS effect,
            collect(DISTINCT pc.text) AS possible_causes,
            collect(DISTINCT pe.text) AS possible_effects,
            score
        ORDER BY score DESC
        """

        with self.driver.session() as session:
            result = session.run(
                cypher,
                embedding=query_vec,
                top_k=top_k
            )

            print("\n=== Semantic Failure Chain ===\n")
            for r in result:
                print(f"Score            : {r['score']:.4f}")
                print(f"Element          : {r['element']}")
                print(f"Function         : {r['function']}")
                print(f"Mode             : {r['mode']}")
                print(f"Cause            : {r['cause']}")
                print(f"Effect           : {r['effect']}")
                print(f"Possible Causes  : {r['possible_causes']}")
                print(f"Possible Effects : {r['possible_effects']}")
                print("-" * 70)


# ============================================
# MAIN
# ============================================

def main():
    builder = FMEAVectorKGBuilder(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
    )

    try:
        # 1. base KG
        builder.create_constraints()
        builder.create_vector_indexes()
        builder.create_relationship_indexes()
        builder.build_graph(JSON_FILE)

        # 2. clear old enrichment edges before re-running
        builder.clear_enrichment_edges()

        # 3. build similarity edges
        builder.build_mode_similarity_edges(
            threshold=0.88,
            top_k=5,
            bidirectional=False
        )

        builder.build_cause_similarity_edges(
            threshold=0.90,
            top_k=5,
            bidirectional=False
        )

        builder.build_effect_similarity_edges(
            threshold=0.90,
            top_k=5,
            bidirectional=False
        )

        # 4. propagate knowledge
        builder.enrich_possible_causes(similarity_threshold=0.90)
        builder.enrich_possible_effects(similarity_threshold=0.90)

        # 5. inspect
        builder.count_nodes()
        builder.count_relationships()
        builder.print_similar_mode_samples(limit=10)
        builder.print_possible_cause_samples(limit=10)
        builder.print_possible_effect_samples(limit=10)

        # 6. test query
        builder.semantic_failure_chain("motor not rotating", top_k=5)

        print("\nVector KG + semantic soft-match edge enrichment complete.")

    finally:
        builder.close()


if __name__ == "__main__":
    main()