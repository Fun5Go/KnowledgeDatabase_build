# chunk_retriever.py

from typing import List, Dict, Any
import re
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
    
    #====================================
    #======= Product Function============
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
    def _normalize_text(text: str) -> str:
        """
        Normalize text for robust lexical matching.

        Operations
        ----------
        1. Lowercase
        2. Replace non-alphanumeric characters with spaces
        3. Collapse multiple spaces

        This helps align surface forms such as:
        - "Motor Control Module"
        - "motor-control module"
        - "Motor control"
        """
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _meaningful_tokens(text: str, min_len: int = 3) -> List[str]:
        """
        Extract meaningful tokens from text.

        Parameters
        ----------
        text : str
            Input text.
        min_len : int
            Minimum token length to keep.

        Returns
        -------
        List[str]
            Deduplicated token list while preserving order.
        """
        normalized = ChunkRetriever._normalize_text(text)
        tokens = normalized.split()

        seen = set()
        output = []
        for tok in tokens:
            if len(tok) < min_len:
                continue
            if tok not in seen:
                seen.add(tok)
                output.append(tok)
        return output
    @staticmethod
    def filter_by_element_and_function(
        results: List[Dict[str, Any]],
        element_name: str,
        function_text: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Filter retrieved chunks using an element-first strategy, optionally
        refined by function text.

        Retrieval policy
        ----------------
        Case 1: element only
            Keep all chunks clearly related to the queried element.

        Case 2: element + function
            First require the chunk to be element-related.
            Then require lexical evidence that it is also related to the
            function text.

        Rationale
        ---------
        In requirement corpora, an element name is usually the most reliable
        anchor. Function wording can vary more strongly, so it should refine
        the result set rather than replace the element constraint.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Retrieved chunk candidates.
        element_name : str
            Queried element, e.g. "Motor control".
        function_text : str
            Optional function refinement, e.g. "relay switching".

        Returns
        -------
        List[Dict[str, Any]]
            Filtered candidates.
        """
        element_norm = ChunkRetriever._normalize_text(element_name)
        function_norm = ChunkRetriever._normalize_text(function_text)

        element_tokens = ChunkRetriever._meaningful_tokens(element_name, min_len=3)
        function_tokens = ChunkRetriever._meaningful_tokens(function_text, min_len=3)

        filtered = []

        for item in results:
            text = item.get("text") or ""
            text_norm = ChunkRetriever._normalize_text(text)

            # ----------------------------------------------------------
            # Step 1. Element gating: the sentence must be about element.
            # ----------------------------------------------------------
            element_phrase_match = element_norm in text_norm if element_norm else False
            element_token_hits = sum(tok in text_norm for tok in element_tokens)

            # Require strong enough evidence for element relevance
            element_ok = (
                element_phrase_match or
                element_token_hits >= max(1, len(element_tokens) - 1)
            )

            if not element_ok:
                continue

            # ----------------------------------------------------------
            # Step 2. If no function text is provided, keep all element hits.
            # ----------------------------------------------------------
            if not function_tokens:
                filtered.append(item)
                continue

            # ----------------------------------------------------------
            # Step 3. Function refinement.
            # ----------------------------------------------------------
            function_phrase_match = function_norm in text_norm if function_norm else False
            function_token_hits = sum(tok in text_norm for tok in function_tokens)

            # Require at least partial function evidence.
            # For short function phrases like "relay switching", both words are preferred.
            # For longer phrases, allow one token missing.
            function_ok = (
                function_phrase_match or
                function_token_hits >= max(1, len(function_tokens) - 1)
            )

            if function_ok:
                filtered.append(item)

        return filtered
    
    @staticmethod
    def rerank_by_element_and_function(
        results: List[Dict[str, Any]],
        element_name: str,
        function_text: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Re-rank filtered results with element priority and optional function
        relevance.

        Ranking intuition
        -----------------
        1. Exact element phrase match gets a strong bonus.
        2. Function phrase match gets another strong bonus.
        3. Token overlap adds softer evidence.
        4. Original retrieval score is preserved as a weak signal.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Filtered chunk candidates.
        element_name : str
            Queried element.
        function_text : str
            Optional function refinement.

        Returns
        -------
        List[Dict[str, Any]]
            Re-ranked results.
        """
        element_norm = ChunkRetriever._normalize_text(element_name)
        function_norm = ChunkRetriever._normalize_text(function_text)

        element_tokens = ChunkRetriever._meaningful_tokens(element_name, min_len=3)
        function_tokens = ChunkRetriever._meaningful_tokens(function_text, min_len=3)

        rescored = []

        for item in results:
            text = item.get("text") or ""
            text_norm = ChunkRetriever._normalize_text(text)

            score = 0.0

            # Element evidence
            if element_norm and element_norm in text_norm:
                score += 5.0
            score += sum(1.0 for tok in element_tokens if tok in text_norm)

            # Function evidence
            if function_tokens:
                if function_norm and function_norm in text_norm:
                    score += 5.0
                score += sum(1.5 for tok in function_tokens if tok in text_norm)

            # Preserve original retrieval score as weak prior
            base_score = item.get("score", 0.0)
            try:
                score += 0.2 * float(base_score)
            except Exception:
                pass

            new_item = dict(item)
            new_item["rerank_score"] = score
            rescored.append(new_item)

        rescored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return rescored

    def retrieve_element_ts_seeds(
        self,
        element_name: str,
        function_text: str = "",
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Retrieve TSChunk seeds for:
        - element-only query
        - element + function query

        Retrieval strategy
        ------------------
        1. Dense retrieval on a combined query.
        2. Sparse retrieval with element as a hard lexical anchor.
        3. Fuse dense and sparse candidates.
        4. Filter by element relevance.
        5. If function_text is provided, further filter by function relevance.
        6. Re-rank by lexical evidence and return top_k.

        Parameters
        ----------
        element_name : str
            Queried element from structure analysis, e.g. "Motor control".
        function_text : str
            Optional function refinement, e.g. "relay switching".
            If empty, all sentences about the element are returned.
        top_k : int
            Number of final TSChunk seeds.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked TSChunk seed nodes.
        """
        function_text = (function_text or "").strip()

        # --------------------------------------------------------------
        # Dense query:
        # - element only => retrieve element-related TS sentences
        # - element + function => retrieve semantically related candidates
        # --------------------------------------------------------------
        retrieval_query = (
            f"{element_name} {function_text}".strip()
            if function_text
            else element_name
        )

        dense_results = self.dense_search_chunks(
            query_text=retrieval_query,
            label="TSChunk",
            vector_index_name="ts_embedding_idx",
            top_k=top_k * 5
        )

        # --------------------------------------------------------------
        # Sparse query:
        # - element phrase must dominate
        # - function tokens refine only when provided
        # --------------------------------------------------------------
        if function_text:
            function_terms = " OR ".join(
                ChunkRetriever._meaningful_tokens(function_text, min_len=3)
            )
            if function_terms:
                lucene_query = f'"{element_name}" AND ({function_terms})'
            else:
                lucene_query = f'"{element_name}"'
        else:
            lucene_query = f'"{element_name}"'

        sparse_results = self.sparse_search_chunks(
            lucene_query=lucene_query,
            label="TSChunk",
            fulltext_index_name="ts_text_idx",
            top_k=top_k * 5
        )

        # --------------------------------------------------------------
        # Fusion
        # --------------------------------------------------------------
        fused = self.rrf_fusion([dense_results, sparse_results])

        # --------------------------------------------------------------
        # Element-first filter, optional function refinement
        # --------------------------------------------------------------
        filtered = self.filter_by_element_and_function(
            fused,
            element_name=element_name,
            function_text=function_text
        )

        # --------------------------------------------------------------
        # Re-rank so that true function hits go above generic element hits
        # --------------------------------------------------------------
        reranked = self.rerank_by_element_and_function(
            filtered,
            element_name=element_name,
            function_text=function_text
        )

        return reranked[:top_k]
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