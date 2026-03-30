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

    # Only relations you want to infer/train
    RELATIONS = ["CAUSES", "LEADS_TO"]
    relation_filter = ",".join([f"'{r}'" for r in RELATIONS])

    query = f"""
    MATCH (h)-[r]->(t)
    WHERE type(r) IN [{relation_filter}]
      AND h.semantic_id IS NOT NULL
      AND t.semantic_id IS NOT NULL
      AND h.embedding IS NOT NULL
      AND t.embedding IS NOT NULL
      AND NOT (h:SubCause OR h:SubMode OR h:SubEffect)
      AND NOT (t:SubCause OR t:SubMode OR t:SubEffect)
    RETURN
        h.semantic_id AS head,
        type(r) AS relation,
        t.semantic_id AS tail,
        coalesce(r.weight, 1.0) AS weight,

        labels(h) AS h_labels,
        labels(t) AS t_labels,
        coalesce(h.text, h.name, h.semantic_id) AS h_text,
        coalesce(t.text, t.name, t.semantic_id) AS t_text,
        h.embedding AS h_emb,
        t.embedding AS t_emb,
        coalesce(h.is_group, false) AS h_is_group,
        coalesce(t.is_group, false) AS t_is_group
    """

    with driver.session() as session:
        result = session.run(query)

        triples = []
        nodes = {}

        def node_kind(is_group: bool) -> str:
            return "group" if is_group else "single"

        def main_type(labels_list):
            # pick one of Cause/Mode/Effect if present; else fallback first label
            for x in ("Cause", "Mode", "Effect"):
                if x in labels_list:
                    return x
            return labels_list[0] if labels_list else "Unknown"

        for record in result:
            h = record["head"]
            r = record["relation"]
            t = record["tail"]
            w = float(record["weight"])

            triples.append((h, r, t, w))

            # store nodes
            if h not in nodes:
                h_labels = record["h_labels"] or []
                nodes[h] = {
                    "node_type": main_type(h_labels),
                    "node_kind": node_kind(bool(record["h_is_group"])),
                    "text": record["h_text"],
                    "embedding": record["h_emb"],
                    "labels": h_labels,
                }

            if t not in nodes:
                t_labels = record["t_labels"] or []
                nodes[t] = {
                    "node_type": main_type(t_labels),
                    "node_kind": node_kind(bool(record["t_is_group"])),
                    "text": record["t_text"],
                    "embedding": record["t_emb"],
                    "labels": t_labels,
                }

    with open(TRIPLES_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["head", "relation", "tail", "weight"])
        writer.writerows(triples)

    with open(NODES_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["node_id", "node_type", "node_kind", "text", "embedding", "labels"])
        for node_id, info in nodes.items():
            writer.writerow([
                node_id,
                info["node_type"],
                info["node_kind"],           # group or single
                info["text"],
                json.dumps(info["embedding"]),
                json.dumps(info["labels"]),  # full labels for debugging
            ])

    driver.close()
    print(f"Exported {len(triples)} triples")
    print(f"Exported {len(nodes)} nodes")


if __name__ == "__main__":
    export_kg()