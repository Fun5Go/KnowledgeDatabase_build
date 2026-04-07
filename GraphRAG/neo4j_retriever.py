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
        """
        Execute a Cypher query and return records as dictionaries.
        """
        records, _, _ = self.driver.execute_query(
            cypher,
            database_=self.database,
            **params
        )
        return [r.data() for r in records]

    def dense_search_chunks(
        self,
        query_text: str,
        label: str,
        vector_index_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generic dense retrieval for chunk nodes.

        Parameters
        ----------
        query_text : str
            Retrieval query text.
        label : str
            Target node label, e.g. 'FSChunk', 'TSChunk', 'QDChunk'.
        vector_index_name : str
            Neo4j vector index name.
        top_k : int
            Number of candidates to return.

        Returns
        -------
        List[Dict[str, Any]]
            Retrieved nodes with dense score.
        """
        query_embedding = get_query_embedding(query_text)

        cypher = f"""
        CALL db.index.vector.queryNodes('{vector_index_name}', $top_k, $query_embedding)
        YIELD node, score
        WHERE node:{label}
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
            top_k=top_k
        )

    def sparse_search_chunks(
        self,
        lucene_query: str,
        label: str,
        fulltext_index_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generic sparse retrieval for chunk nodes using Neo4j fulltext index.

        Parameters
        ----------
        lucene_query : str
            Lucene-style retrieval query.
        label : str
            Target node label, e.g. 'FSChunk', 'TSChunk', 'QDChunk'.
        fulltext_index_name : str
            Neo4j fulltext index name.
        top_k : int
            Number of candidates to return.

        Returns
        -------
        List[Dict[str, Any]]
            Retrieved nodes with sparse score.
        """
        cypher = f"""
        CALL db.index.fulltext.queryNodes('{fulltext_index_name}', $lucene_query)
        YIELD node, score
        WHERE node:{label}
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
            top_k=top_k
        )

    @staticmethod
    def filter_by_product_name(
        results: List[Dict[str, Any]],
        product_name: str
    ) -> List[Dict[str, Any]]:
        """
        Apply a lightweight product-name constraint after retrieval.

        Current heuristic:
        - keep a result if the chunk text explicitly mentions the product name

        This is a post-retrieval lexical constraint. It is simple and practical,
        although graph-structured product linkage would be more robust.
        """
        product_name_lower = product_name.lower().strip()
        filtered = []

        for item in results:
            text = (item.get("text") or "").lower()
            if product_name_lower in text:
                filtered.append(item)

        return filtered

    @staticmethod
    def rrf_fusion(
        result_sets: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.
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
        Retrieve top FSChunk seeds for a function-oriented query.

        The product name is used here as a retrieval constraint, rather than
        being embedded in the low-level dense/sparse retrieval functions.
        """
        dense_results = self.dense_search_chunks(
            query_text=query_text,
            label="FSChunk",
            vector_index_name="fs_embedding_idx",
            top_k=top_k * 3
        )

        lucene_query = (
            f'({query_text} OR shall OR support OR provide OR function '
            f'OR implement OR may)'
        )
        sparse_results = self.sparse_search_chunks(
            lucene_query=lucene_query,
            label="FSChunk",
            fulltext_index_name="fs_text_idx",
            top_k=top_k * 3
        )

        dense_results = self.filter_by_product_name(dense_results, product_name)
        sparse_results = self.filter_by_product_name(sparse_results, product_name)

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
    

    @staticmethod
    def filter_by_element_name(
        results: List[Dict[str, Any]],
        element_name: str
    ) -> List[Dict[str, Any]]:
        """
        Filter retrieved chunks by element mention.

        Heuristic
        ---------
        1. Keep the chunk if it contains the full element name.
        2. Otherwise, keep it if it contains enough meaningful element tokens.

        This is useful when TS sentences use slightly different surface forms such as:
        - "Motor Control Module"
        - "Motor control module"
        - "Motor control"
        """
        element_name_lower = element_name.lower().strip()
        element_tokens = [tok for tok in element_name_lower.split() if len(tok) >= 3]

        filtered = []

        for item in results:
            text = (item.get("text") or "").lower()

            # Exact phrase match
            if element_name_lower in text:
                filtered.append(item)
                continue

            # Token overlap match
            matched = sum(tok in text for tok in element_tokens)
            if matched >= max(1, len(element_tokens) - 1):
                filtered.append(item)

        return filtered

    def retrieve_element_ts_seeds(
        self,
        element_name: str,
        query_text: str = "",
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top TSChunk seeds for an element-centric query.

        Design rationale
        ----------------
        In the current corpus, TS sentences often explicitly mention the element,
        for example:
        - "The Motor Control Module will measure ..."
        - "The Motor control module will be responsible for ..."

        Therefore, retrieval is centered on the element name itself, while the
        optional query text is only used as a secondary refinement signal.

        Parameters
        ----------
        element_name : str
            Queried element name from structure analysis, e.g. "Motor control".
        query_text : str
            Optional refinement such as "frequency", "soft start", "protection",
            or "verification".
        top_k : int
            Number of final TS seed nodes to return.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked TSChunk seed nodes.
        """
        # Main semantic query: the element name should dominate the retrieval.
        retrieval_query = f"{element_name} {query_text}".strip()

        dense_results = self.dense_search_chunks(
            query_text=retrieval_query,
            label="TSChunk",
            vector_index_name="ts_embedding_idx",
            top_k=top_k * 3
        )

        # Sparse query: explicitly require the element phrase.
        if query_text.strip():
            lucene_query = f'"{element_name}" AND ({query_text} OR will OR shall OR responsible OR measure OR filter)'
        else:
            lucene_query = f'"{element_name}" OR ({element_name} AND (will OR shall OR responsible OR measure OR filter))'

        sparse_results = self.sparse_search_chunks(
            lucene_query=lucene_query,
            label="TSChunk",
            fulltext_index_name="ts_text_idx",
            top_k=top_k * 3
        )

        dense_results = self.filter_by_element_name(dense_results, element_name)
        sparse_results = self.filter_by_element_name(sparse_results, element_name)

        fused = self.rrf_fusion([dense_results, sparse_results])
        return fused[:top_k]
    def expand_ts_context(self, ts_node_id: str) -> Dict[str, Any]:
        """
        Expand one TSChunk to its related functional, rationale, and
        verification evidence.

        Graph expansion
        ---------------
        TSChunk -[:IMPLEMENT]-> FSChunk
        RationaleChunk -[:RATIONALE_FOR]-> TSChunk
        RationaleChunk -[:RATIONALE_FOR]-> FSChunk
        QDChunk -[:VERIFIED]-> TSChunk
        TSTChunk -[:VERIFIED]-> TSChunk

        Parameters
        ----------
        ts_node_id : str
            Neo4j elementId of a TSChunk node.

        Returns
        -------
        Dict[str, Any]
            Expanded multi-view evidence around the TSChunk.
        """
        cypher = """
        MATCH (ts:TSChunk)
        WHERE elementId(ts) = $ts_node_id

        OPTIONAL MATCH (ts)-[:IMPLEMENT]->(fs:FSChunk)
        OPTIONAL MATCH (r_ts:RationaleChunk)-[:RATIONALE_FOR]->(ts)
        OPTIONAL MATCH (r_fs:RationaleChunk)-[:RATIONALE_FOR]->(fs)
        OPTIONAL MATCH (qd:QDChunk)-[:VERIFIED]->(ts)
        OPTIONAL MATCH (tst:TSTChunk)-[:VERIFIED]->(ts)

        RETURN
            elementId(ts) AS ts_node_id,
            ts.text AS ts_text,
            collect(DISTINCT fs.text) AS fs_texts,
            collect(DISTINCT qd.text) AS qd_texts,
            collect(DISTINCT tst.text) AS tst_texts,
            collect(DISTINCT r_ts.text) + collect(DISTINCT r_fs.text) AS rationale_texts
        """
        results = self.run_query(cypher, ts_node_id=ts_node_id)
        return results[0] if results else {}