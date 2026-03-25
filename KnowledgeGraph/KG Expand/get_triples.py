import os
import csv
import json
from neo4j import GraphDatabase

TRIPLES_FILE = "triples.tsv"
NODES_FILE = "nodes.tsv"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


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
    ]

    relation_filter = ",".join([f"'{r}'" for r in RELATIONS])

    query = f"""
    MATCH (h)-[r]->(t)
    WHERE type(r) IN [{relation_filter}]
      AND h.semantic_id IS NOT NULL
      AND t.semantic_id IS NOT NULL
      AND h.embedding IS NOT NULL
      AND t.embedding IS NOT NULL
    RETURN
        h.semantic_id AS head,
        type(r) AS relation,
        t.semantic_id AS tail,
        coalesce(r.weight, 1.0) AS weight,
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
            w = float(record["weight"])

            h_emb = record["h_emb"]
            t_emb = record["t_emb"]

            if h_emb is None or t_emb is None:
                continue

            triples.append((h, r, t, w))

            if h not in nodes:
                nodes[h] = (
                    record["h_type"],
                    record["h_text"],
                    h_emb
                )

            if t not in nodes:
                nodes[t] = (
                    record["t_type"],
                    record["t_text"],
                    t_emb
                )

    with open(TRIPLES_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["head", "relation", "tail", "weight"])
        for row in triples:
            writer.writerow(row)

    with open(NODES_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["node_id", "node_type", "text", "embedding"])
        for node_id, (node_type, text, emb) in nodes.items():
            writer.writerow([
                node_id,
                node_type,
                text,
                json.dumps(emb)
            ])

    driver.close()

    print(f"Exported {len(triples)} triples")
    print(f"Exported {len(nodes)} nodes")


if __name__ == "__main__":
    export_kg()