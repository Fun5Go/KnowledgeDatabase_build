import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm
from chromadb.utils import embedding_functions


# ============================================
# ENV
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
# GRAPH BUILDER
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
    # NODE CREATION
    # ----------------------------------------

    def merge_element(self, session, item) -> Optional[str]:
        element_id = item.get("element_id")
        if not element_id:
            return None

        text = safe_text(item.get("failure_element_text"))
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
    # BUILD GRAPH
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
            for item in tqdm(data.values(), total=len(data)):
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
        builder.create_constraints()
        builder.create_vector_indexes()
        builder.build_graph(JSON_FILE)
        print("Vector KG build complete")
    finally:
        builder.close()


if __name__ == "__main__":
    main()