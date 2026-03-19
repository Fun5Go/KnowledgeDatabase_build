import os
import csv
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

TRIPLES_FILE = "triples.tsv"
NODES_FILE = "nodes.tsv"


def export_kg():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    RELATIONS = [
        "HAS_FUNCTION",
        "HAS_MODE",
        "CAUSES",
        "LEADS_TO",
        "CAUSES_EFFECT",
        "BELONGS_TO"
    ]

    relation_filter = ",".join([f"'{r}'" for r in RELATIONS])

    query = f"""
    MATCH (h)-[r]->(t)
    WHERE type(r) IN [{relation_filter}]
      AND h.semantic_id IS NOT NULL
      AND t.semantic_id IS NOT NULL

    RETURN
        h.semantic_id AS head,
        type(r) AS relation,
        t.semantic_id AS tail,
        head(labels(h)) AS h_type,
        head(labels(t)) AS t_type,
        h.text AS h_text,
        t.text AS t_text,
        h.embedding AS h_emb,
        t.embedding AS t_emb
    """

    with driver.session() as session:
        result = session.run(query)

        triples = []
        nodes = {}

        for record in result:
            h = record["head"]
            r = record["relation"]
            t = record["tail"]

            triples.append((h, r, t))

            # =========================
            # NODE H
            # =========================
            if h not in nodes:
                emb = record["h_emb"]

                if emb is None:
                    continue  # ❗ 没 embedding 的直接跳过（推荐）

                nodes[h] = (
                    record["h_type"],
                    record["h_text"],
                    emb
                )

            # =========================
            # NODE T
            # =========================
            if t not in nodes:
                emb = record["t_emb"]

                if emb is None:
                    continue

                nodes[t] = (
                    record["t_type"],
                    record["t_text"],
                    emb
                )

    # =========================
    # WRITE TRIPLES
    # =========================
    with open(TRIPLES_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["head", "relation", "tail"])

        for row in triples:
            writer.writerow(row)

    # =========================
    # WRITE NODES (with embedding)
    # =========================
    with open(NODES_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow(["node_id", "node_type", "text", "embedding"])

        for node_id, (node_type, text, emb) in nodes.items():
            writer.writerow([
                node_id,
                node_type,
                text,
                json.dumps(emb)  # ⭐ 关键：安全写入
            ])

    driver.close()

    print(f"Exported {len(triples)} triples")
    print(f"Exported {len(nodes)} nodes")


if __name__ == "__main__":
    export_kg()