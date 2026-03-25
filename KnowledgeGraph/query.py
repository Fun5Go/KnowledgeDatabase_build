import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

############################################
# ENV
############################################

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

############################################
# EMBEDDING
############################################

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def embed(text):
    return embedder([text])[0]

############################################
# DRIVER
############################################

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

############################################
# PRINT
############################################
def structured_print(records):
    for i, r in enumerate(records, 1):
        print(f"[{i}]")
        for k, v in r.items():
            print(f"{k:<10}: {v}")
        print("-" * 50)

############################################
# SEMANTIC SEARCH
############################################

def semantic_search(label, index_name, query, discipline=None, top_k=5):
    if label == "Mode":
        embedding = embed(query)

    elif label == "Function":
        embedding = embed(query)

    elif label == "Cause":
        embedding = embed(query)

    elif label == "Effect":
        embedding = embed(query)

    else:
        raise ValueError(f"Unsupported label: {label}")

    where_clause = ""
    if label == "Cause":
        where_clause = """
        WHERE (
            $discipline IS NULL
            OR size($discipline) = 0
            OR toLower(coalesce(node.discipline, "unknown")) IN
            [x IN $discipline | toLower(x)]
        )
        AND (
            coalesce(node.is_group, false) = true
            OR NOT (node)-[:BELONGS_TO]->(:Cause)
        )
        """
    if label == "Effect":
        where_clause = """
        WHERE (
            $discipline IS NULL
            OR size($discipline) = 0
            OR toLower(coalesce(node.discipline, "unknown")) IN
            [x IN $discipline | toLower(x)]
        )
        AND (
            coalesce(node.is_group, false) = true
            OR NOT (node)-[:BELONGS_TO]->(:Effect)
        )
        """

    cypher = f"""
    CALL db.index.vector.queryNodes(
        '{index_name}',
        $k,
        $embedding
    )
    YIELD node, score

    {where_clause}

    RETURN
        node.semantic_id AS id,
        node.text AS text,
        node.discipline AS discipline,
        score
    ORDER BY score DESC
    """

    with driver.session() as session:
        result = session.run(
            cypher,
            embedding=embedding,
            k=top_k,
            discipline=discipline
        )

        records = [dict(r) for r in result]
        structured_print(records)
        return records


############################################
# MODE → CAUSE / EFFECT
############################################

def mode_reasoning(query):

    embedding = embed("Failure mode: " + query)

    cypher = """
    CALL db.index.vector.queryNodes(
        'mode_embedding',
        5,
        $embedding
    )
    YIELD node, score

    OPTIONAL MATCH (node)-[:CAUSED_BY]->(c:Cause)
    OPTIONAL MATCH (node)-[:LEADS_TO]->(e:Effect)

    RETURN
    node.text AS mode,
    c.text AS cause,
    e.text AS effect,
    score
    """

    with driver.session(database="fmeav2") as session:
        result = session.run(cypher, embedding=embedding)

        records = [dict(r) for r in result]     
        structured_print(records)     
        return records


############################################
# FULL FAILURE CHAIN
############################################

def failure_chain_search(query):

    embedding = embed("Failure mode: " + query)

    cypher = """
    CALL db.index.vector.queryNodes(
        'mode_embedding',
        5,
        $embedding
    )
    YIELD node, score

    OPTIONAL MATCH (f:Function)-[:HAS_MODE]->(node)
    OPTIONAL MATCH (e:Element)-[:HAS_FUNCTION]->(f)

    OPTIONAL MATCH (node)-[:CAUSED_BY]->(c:Cause)
    OPTIONAL MATCH (node)-[:LEADS_TO]->(ef:Effect)

    RETURN
    e.text AS element,
    f.text AS function,
    node.text AS mode,
    c.text AS cause,
    ef.text AS effect,
    score
    """

    with driver.session() as session:
        result = session.run(cypher, embedding=embedding)

        records = [dict(r) for r in result]     
        structured_print(records)     
        return records


############################################
# FIND MODES BY CAUSE
############################################

def find_modes_by_cause(query):

    embedding = embed("Failure cause: " + query)

    cypher = """
    CALL db.index.vector.queryNodes(
        'cause_embedding',
        5,
        $embedding
    )
    YIELD node, score

    MATCH (m:Mode)-[:CAUSED_BY]->(node)

    RETURN
    node.text AS cause,
    m.text AS mode,
    score
    """

    with driver.session() as session:
        result = session.run(cypher, embedding=embedding)

        records = [dict(r) for r in result]     
        structured_print(records)     
        return records


############################################
# FIND FAILURE BY ELEMENT
############################################

def find_failure_by_element(query):

    embedding = embed("Component: " + query)

    cypher = """
    CALL db.index.vector.queryNodes(
        'element_embedding',
        5,
        $embedding
    )
    YIELD node, score

    MATCH (node)-[:HAS_FUNCTION]->(f:Function)
    MATCH (f)-[:HAS_MODE]->(m:Mode)

    OPTIONAL MATCH (m)-[:CAUSED_BY]->(c:Cause)
    OPTIONAL MATCH (m)-[:LEADS_TO]->(e:Effect)

    RETURN
    node.text AS element,
    f.text AS function,
    m.text AS mode,
    c.text AS cause,
    e.text AS effect,
    score
    """

    with driver.session() as session:
        result = session.run(cypher, embedding=embedding)

        records = [dict(r) for r in result]     
        structured_print(records)     
        return records

############################################
# GRAPH STATS
############################################

def graph_stats():

    cypher_nodes = """
    MATCH (n)
    RETURN labels(n) AS label, count(*) AS count
    """

    cypher_edges = """
    MATCH ()-[r]->()
    RETURN type(r) AS relationship, count(*) AS count
    """

    with driver.session() as session:

        nodes = [dict(r) for r in session.run(cypher_nodes)]
        edges = [dict(r) for r in session.run(cypher_edges)]

    return nodes, edges


############################################
# MAIN TEST
############################################

if __name__ == "__main__":

    # print("\n=== Semantic Search Mode ===")
    # print(semantic_search(
    #     "Mode",
    #     "mode_embedding",
    #     "Not enough torque",
    #     top_k=20
    # ))

    print("\n=== Semantic Search Mode ===")
    semantic_search(
        "Cause",
        "cause_embedding",
        "(Starting) Motor current too high for chosen components",
        # discipline=["mechanics", "unknown"],
        top_k=20
    )

    # print("\n=== Mode Reasoning ===")
    # print(mode_reasoning("Not enough torque"))

    # print("\n=== Failure Chain ===")
    # print(failure_chain_search("Not enough torque"))

    # print("\n=== Find Modes by Cause ===")
    # find_modes_by_cause("ADC measurements incorrect (incl. bandwidth")

    # print("\n=== Find Failure by Element ===")
    # find_failure_by_element("Power train")

    # print("\n=== Graph Stats ===")
    # nodes, edges = graph_stats()

    # print("Nodes:")
    # print(nodes)

    # print("Edges:")
    # print(edges)

    driver.close()