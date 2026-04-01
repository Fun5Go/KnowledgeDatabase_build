from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import math


class GraphRetriever:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        embedding_model: str = "sentence-transformers/msmarco-MiniLM-L-6-v3",
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.database = database
        self.model = SentenceTransformer(embedding_model)

    def close(self):
        self.driver.close()

    def encode_query(self, query: str) -> List[float]:
        # asymmetric retrieval: query is short, documents are longer
        vec = self.model.encode(query, normalize_embeddings=True)
        return vec.tolist()

    def get_product_anchor(self, product_name: str) -> Dict[str, Any]:
        cypher = """
        MATCH (p:Product)
        WHERE toLower(p.name) = toLower($product_name)
           OR toLower(p.alias) = toLower($product_name)
        RETURN p { .* } AS product
        LIMIT 1
        """
        records, _, _ = self.driver.execute_query(
            cypher,
            product_name=product_name,
            database_=self.database
        )
        return records[0]["product"] if records else None

    def dense_search_requirements(
        self,
        query: str,
        product_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        query_vec = self.encode_query(query)

        cypher = """
        CALL db.index.vector.queryNodes('requirement_embedding_idx', $top_k, $query_vec)
        YIELD node, score
        WHERE node:Requirement
          AND (
                toLower(node.product_name) = toLower($product_name)
                OR EXISTS {
                    MATCH (node)-[:FOR_PRODUCT]->(p:Product)
                    WHERE toLower(p.name) = toLower($product_name)
                }
              )
        RETURN
            elementId(node) AS node_id,
            labels(node) AS labels,
            node.text AS text,
            score,
            'dense' AS source
        ORDER BY score DESC
        LIMIT $top_k
        """
        records, _, _ = self.driver.execute_query(
            cypher,
            query_vec=query_vec,
            top_k=top_k,
            product_name=product_name,
            database_=self.database
        )
        return [r.data() for r in records]

    def sparse_search_requirements(
        self,
        query: str,
        product_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        # query string can be improved with product filter words
        lucene_query = f'{product_name} AND ({query} OR function OR shall OR support)'

        cypher = """
        CALL db.index.fulltext.queryNodes('requirement_text_idx', $lucene_query)
        YIELD node, score
        WHERE (node:Requirement OR node:Chunk)
          AND (
                toLower(coalesce(node.product_name, '')) = toLower($product_name)
                OR EXISTS {
                    MATCH (node)-[:FOR_PRODUCT]->(p:Product)
                    WHERE toLower(p.name) = toLower($product_name)
                }
                OR toLower(coalesce(node.text, '')) CONTAINS toLower($product_name)
              )
        RETURN
            elementId(node) AS node_id,
            labels(node) AS labels,
            node.text AS text,
            score,
            'sparse' AS source
        ORDER BY score DESC
        LIMIT $top_k
        """
        records, _, _ = self.driver.execute_query(
            cypher,
            lucene_query=lucene_query,
            top_k=top_k,
            product_name=product_name,
            database_=self.database
        )
        return [r.data() for r in records]

    def graph_expand_from_product(
        self,
        product_name: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (p:Product)
        WHERE toLower(p.name) = toLower($product_name)
        OPTIONAL MATCH (r:Requirement)-[:FOR_PRODUCT]->(p)
        WITH p, collect(DISTINCT r) AS reqs
        UNWIND reqs AS r
        WHERE r IS NOT NULL
        RETURN
            elementId(r) AS node_id,
            labels(r) AS labels,
            r.text AS text,
            1.0 AS score,
            'graph' AS source
        LIMIT $top_k
        """
        records, _, _ = self.driver.execute_query(
            cypher,
            product_name=product_name,
            top_k=top_k,
            database_=self.database
        )
        return [r.data() for r in records]

    @staticmethod
    def rrf_fuse(result_sets: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion
        score = sum(1 / (k + rank))
        """
        fused = {}
        for result_set in result_sets:
            for rank, item in enumerate(result_set, start=1):
                node_id = item["node_id"]
                if node_id not in fused:
                    fused[node_id] = {
                        "node_id": node_id,
                        "labels": item.get("labels", []),
                        "text": item.get("text", ""),
                        "rrf_score": 0.0,
                        "sources": set()
                    }
                fused[node_id]["rrf_score"] += 1.0 / (k + rank)
                fused[node_id]["sources"].add(item.get("source", "unknown"))

        fused_list = list(fused.values())
        for x in fused_list:
            x["sources"] = list(x["sources"])
        fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_list

    def retrieve_functions_for_product(
        self,
        product_name: str,
        user_query: str,
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        dense = self.dense_search_requirements(user_query, product_name, top_k=top_k)
        sparse = self.sparse_search_requirements(user_query, product_name, top_k=top_k)
        graph = self.graph_expand_from_product(product_name, top_k=top_k)

        fused = self.rrf_fuse([dense, sparse, graph])
        return fused[:top_k]