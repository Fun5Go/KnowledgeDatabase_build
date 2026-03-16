from pathlib import Path
import json
from neo4j import GraphDatabase


FIELD_LABEL_MAP = {
    "element": "Element",
    "mode": "Mode",
    "effect": "Effect",
    "cause": "Cause",
}

EDGE_REL_MAP = {
    "element_to_mode": "HAS_MODE",
    "mode_to_effect": "LEADS_TO",
    "mode_to_cause": "CAUSED_BY",
    "cause_to_effect": "RESULTS_IN", # To be decided
}


class Neo4jFMEAExporter:
    def __init__(self, uri: str, user: str, password: str, persist_dir: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.persist_dir = Path(persist_dir)

        self.field_store_path = self.persist_dir / "fmea_field_store.json"
        self.entity_store_path = self.persist_dir / "entity_store.json"
        self.edge_store_path = self.persist_dir / "fmea_edge_store.json"

    def close(self):
        self.driver.close()

    def load_json(self, path: Path):
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def create_constraints(self):
        queries = [
            """
            CREATE CONSTRAINT failure_record_id IF NOT EXISTS
            FOR (n:FailureRecord) REQUIRE n.failure_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT element_semantic_id IF NOT EXISTS
            FOR (n:Element) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT mode_semantic_id IF NOT EXISTS
            FOR (n:Mode) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT effect_semantic_id IF NOT EXISTS
            FOR (n:Effect) REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT cause_semantic_id IF NOT EXISTS
            FOR (n:Cause) REQUIRE n.semantic_id IS UNIQUE
            """,
        ]

        with self.driver.session() as session:
            for q in queries:
                session.run(q)

    def export_semantic_nodes(self):
        field_store = self.load_json(self.field_store_path)

        with self.driver.session() as session:
            for semantic_id, node in field_store.items():
                field_type = node.get("field_type")
                label = FIELD_LABEL_MAP.get(field_type)
                if not label:
                    continue

                query = f"""
                MERGE (n:{label} {{semantic_id: $semantic_id}})
                SET n.text = $text,
                    n.field_type = $field_type,
                    n.source_type = $source_type,
                    n.count = $count,
                    n.failure_ids = $failure_ids,
                    n.discipline = $discipline
                """
                session.run(
                    query,
                    semantic_id=semantic_id,
                    text=node.get("text"),
                    field_type=field_type,
                    source_type=node.get("source_type"),
                    count=node.get("count", 0),
                    failure_ids=node.get("failure_ids", []),
                    discipline=node.get("discipline"),
                )

    def export_failure_records(self):
        entity_store = self.load_json(self.entity_store_path)

        with self.driver.session() as session:
            for failure_id, entity in entity_store.items():
                query = """
                MERGE (fr:FailureRecord {failure_id: $failure_id})
                SET fr.file_name = $file_name,
                    fr.failure_mode_text = $failure_mode_text,
                    fr.failure_element_text = $failure_element_text,
                    fr.failure_effect_text = $failure_effect_text,
                    fr.failure_cause_text = $failure_cause_text,
                    fr.process_step = $process_step,
                    fr.system = $system,
                    fr.function = $function,
                    fr.discipline = $discipline,
                    fr.severity = $severity,
                    fr.occurrence = $occurrence,
                    fr.detection = $detection,
                    fr.rpn = $rpn,
                    fr.prevention = $prevention,
                    fr.detection_method = $detection_method,
                    fr.recommended_action = $recommended_action,
                    fr.source_type = $source_type,
                    fr.fmea_type = $fmea_type,
                    fr.productPnID = $productPnID,
                    fr.product_domain = $product_domain,
                    fr.released_year = $released_year,
                    fr.same_id = $same_id
                """
                session.run(
                    query,
                    failure_id=failure_id,
                    file_name=entity.get("file_name"),
                    failure_mode_text=entity.get("failure_mode_text"),
                    failure_element_text=entity.get("failure_element_text"),
                    failure_effect_text=entity.get("failure_effect_text"),
                    failure_cause_text=entity.get("failure_cause_text"),
                    process_step=entity.get("process_step"),
                    system=entity.get("system"),
                    function=entity.get("function"),
                    discipline=entity.get("discipline"),
                    severity=entity.get("severity"),
                    occurrence=entity.get("occurrence"),
                    detection=entity.get("detection"),
                    rpn=entity.get("rpn"),
                    prevention=entity.get("prevention"),
                    detection_method=entity.get("detection_method"),
                    recommended_action=entity.get("recommended_action"),
                    source_type=entity.get("source_type"),
                    fmea_type=entity.get("fmea_type"),
                    productPnID=entity.get("productPnID"),
                    product_domain=entity.get("product_domain"),
                    released_year=entity.get("released_year"),
                    same_id=entity.get("same_id", []),
                )

                # connect FailureRecord -> semantic nodes
                self._link_failure_record(session, failure_id, entity)

    def _link_failure_record(self, session, failure_id: str, entity: dict):
        link_specs = [
            ("element_id", "Element", "HAS_ELEMENT"),
            ("mode_id", "Mode", "HAS_MODE"),
            ("effect_id", "Effect", "HAS_EFFECT"),
            ("cause_id", "Cause", "HAS_CAUSE"),
        ]

        for key, label, rel in link_specs:
            semantic_id = entity.get(key)
            if not semantic_id:
                continue

            query = f"""
            MATCH (fr:FailureRecord {{failure_id: $failure_id}})
            MATCH (n:{label} {{semantic_id: $semantic_id}})
            MERGE (fr)-[:{rel}]->(n)
            """
            session.run(query, failure_id=failure_id, semantic_id=semantic_id)

    def export_semantic_edges(self):
        edge_store = self.load_json(self.edge_store_path)
        field_store = self.load_json(self.field_store_path)

        with self.driver.session() as session:
            for edge_type, src_map in edge_store.items():
                rel = EDGE_REL_MAP.get(edge_type)
                if not rel:
                    continue

                for src_id, tgt_map in src_map.items():
                    src_node = field_store.get(src_id)
                    if not src_node:
                        continue
                    src_label = FIELD_LABEL_MAP.get(src_node.get("field_type"))
                    if not src_label:
                        continue

                    for tgt_id, count in tgt_map.items():
                        tgt_node = field_store.get(tgt_id)
                        if not tgt_node:
                            continue
                        tgt_label = FIELD_LABEL_MAP.get(tgt_node.get("field_type"))
                        if not tgt_label:
                            continue

                        query = f"""
                        MATCH (a:{src_label} {{semantic_id: $src_id}})
                        MATCH (b:{tgt_label} {{semantic_id: $tgt_id}})
                        MERGE (a)-[r:{rel}]->(b)
                        SET r.count = $count
                        """
                        session.run(query, src_id=src_id, tgt_id=tgt_id, count=count)

    def export_all(self):
        self.create_constraints()
        self.export_semantic_nodes()
        self.export_failure_records()
        self.export_semantic_edges()