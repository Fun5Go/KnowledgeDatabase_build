import os
import csv
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

OUTPUT_FILE = "triples.tsv"


def export_triples_to_tsv():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    query = """
    MATCH (h)-[r]->(t)
    WHERE type(r) IN ['HAS_FUNCTION', 'HAS_MODE', 'CAUSED_BY', 'LEADS_TO']
    RETURN h.semantic_id AS head,
           type(r) AS relation,
           t.semantic_id AS tail
    """

    with driver.session() as session:
        result = session.run(query)
        rows = [(
            record["head"],
            record["relation"],
            record["tail"]
        ) for record in result]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in rows:
            writer.writerow(row)

    driver.close()
    print(f"Exported {len(rows)} triples to {OUTPUT_FILE}")


if __name__ == "__main__":
    export_triples_to_tsv()