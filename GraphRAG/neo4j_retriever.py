# chunk_retriever.py

from typing import List, Dict, Any
from neo4j import GraphDatabase

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    embedder,
    _embedding_cache,
)


def get_query_embedding(text: str) -> List[float]:
    """Generate embedding for a query with simple cache."""
    if text in _embedding_cache:
        return _embedding_cache[text]
    emb = embedder([text])[0]
    _embedding_cache[text] = emb
    return emb


class ChunkRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.driver.verify_connectivity()
        self.database = NEO4J_DATABASE

    def close(self):
        self.driver.close()

    def run_query(self, cypher: str, **params) -> List[Dict[str, Any]]:
        records, _, _ = self.driver.execute_query(
            cypher,
            database_=self.database,
            **params
        )
        return [r.data() for r in records]

    def dense_search_fschunks(
        self,
        query_text: str,
        product_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Dense retrieval on FSChunk only.
        Assumes FSChunk nodes have an `embedding` property and `text` property.
        """
        query_embedding = get_query_embedding(query_text)

        cypher = """
        CALL db.index.vector.queryNodes('fs_embedding_idx', $top_k, $query_embedding)
        YIELD node, score
        WHERE node:FSChunk
          AND (
                toLower(coalesce(node.text, "")) CONTAINS toLower($product_name)
                OR toLower(coalesce(node.product_name, "")) = toLower($product_name)
              )
        RETURN
            elementId(node) AS node_id,
            labels(node) AS labels,
            coalesce(node.text, "") AS text,
            score AS score,
            "dense" AS source
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.run_query(
            cypher,
            query_embedding=query_embedding,
            product_name=product_name,
            top_k=top_k
        )

    def sparse_search_fschunks(
        self,
        query_text: str,
        product_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Full-text retrieval on FSChunk.
        """
        lucene_query = f'{product_name} AND ({query_text} OR shall OR support OR provide OR function OR implement Or may)'

        cypher = """
        CALL db.index.fulltext.queryNodes('fs_text_idx', $lucene_query)
        YIELD node, score
        WHERE node:FSChunk
          AND (
                toLower(coalesce(node.text, "")) CONTAINS toLower($product_name)
                OR toLower(coalesce(node.product_name, "")) = toLower($product_name)
              )
        RETURN
            elementId(node) AS node_id,
            labels(node) AS labels,
            coalesce(node.text, "") AS text,
            score AS score,
            "sparse" AS source
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.run_query(
            cypher,
            lucene_query=lucene_query,
            product_name=product_name,
            top_k=top_k
        )

    @staticmethod
    def rrf_fusion(result_sets: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion."""
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
                        "sources": set(),
                    }
                fused[node_id]["rrf_score"] += 1.0 / (k + rank)
                fused[node_id]["sources"].add(item.get("source", "unknown"))

        fused_list = list(fused.values())
        for item in fused_list:
            item["sources"] = list(item["sources"])

        fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_list

    def retrieve_function_seeds(
        self,
        product_name: str,
        query_text: str,
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Get top FSChunk seeds for a function query.
        """
        dense_results = self.dense_search_fschunks(query_text, product_name, top_k=top_k)
        sparse_results = self.sparse_search_fschunks(query_text, product_name, top_k=top_k)
        fused = self.rrf_fusion([dense_results, sparse_results])
        return fused[:top_k]

    def expand_fs_context(self, fs_node_id: str) -> Dict[str, Any]:
        """
        Expand one FSChunk to technical, rationale, and verification evidence.
        """
        cypher = """
        MATCH (fs:FSChunk)
        WHERE elementId(fs) = $fs_node_id

        OPTIONAL MATCH (ts:TSChunk)-[:IMPLEMENT]->(fs)
        OPTIONAL MATCH (r:RationaleChunk)-[:RATIONALE_FOR]->(fs)
        OPTIONAL MATCH (r2:RationaleChunk)-[:RATIONALE_FOR]->(ts)
        OPTIONAL MATCH (qd:QDChunk)-[:VERIFIED]->(ts)
        OPTIONAL MATCH (tst:TSTChunk)-[:VERIFIED]->(ts)

        RETURN
            elementId(fs) AS fs_node_id,
            fs.text AS fs_text,
            collect(DISTINCT ts.text) AS ts_texts,
            collect(DISTINCT qd.text) AS qd_texts,
            collect(DISTINCT tst.text) AS tst_texts,
            collect(DISTINCT r.text) + collect(DISTINCT r2.text) AS rationale_texts
        """
        results = self.run_query(cypher, fs_node_id=fs_node_id)
        return results[0] if results else {}