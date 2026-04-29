# chunk_retriever.py

from typing import List, Dict, Any
import re
from neo4j import GraphDatabase

from .config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
    embedder,
    _embedding_cache,
)
from .index_manager import Neo4jIndexManager


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
        self._ensure_document_indexes()

    def close(self):
        self.driver.close()

    def _ensure_document_indexes(self) -> None:
        """
        Ensure the current document-KG indexes exist before retrieval.

        The latest KG builder creates labels/constraints but does not create
        vector/fulltext indexes itself, so retrieval should bootstrap them.
        """
        index_manager = Neo4jIndexManager(
            driver=self.driver,
            database=self.database,
        )
        index_manager.create_document_kg_indexes()

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
        top_k: int = 10,
        text_property: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Generic dense retrieval for chunk nodes.

        Parameters
        ----------
        query_text : str
            Retrieval query text.
        label : str
            Target node label, e.g. 'FSChunk', 'TSChunk', 'RationaleChunk'.
        vector_index_name : str
            Neo4j vector index name.
        top_k : int
            Number of candidates to return.
        text_property : str
            Node property to expose as text in the result.

        Returns
        -------
        List[Dict[str, Any]]
            Retrieved nodes with dense score.
        """
        query_embedding = get_query_embedding(query_text)
        text_property_expr = self._safe_property_name(text_property)

        cypher = f"""
        CALL db.index.vector.queryNodes('{vector_index_name}', $top_k, $query_embedding)
        YIELD node, score
        WHERE node:{label}
        RETURN
            elementId(node) AS node_id,
            labels(node) AS labels,
            coalesce(node.name, "") AS name,
            coalesce(node.section_tag, "") AS section_tag,
            coalesce(node.qd_id, "") AS qd_id,
            coalesce(node.qd_title, "") AS qd_title,
            coalesce(node.{text_property_expr}, "") AS text,
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
        top_k: int = 10,
        text_property: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Generic sparse retrieval for chunk nodes using Neo4j fulltext index.

        Parameters
        ----------
        lucene_query : str
            Lucene-style retrieval query.
        label : str
            Target node label, e.g. 'FSChunk', 'TSChunk', 'RationaleChunk'.
        fulltext_index_name : str
            Neo4j fulltext index name.
        top_k : int
            Number of candidates to return.
        text_property : str
            Node property to expose as text in the result.

        Returns
        -------
        List[Dict[str, Any]]
            Retrieved nodes with sparse score.
        """
        text_property_expr = self._safe_property_name(text_property)
        cypher = f"""
        CALL db.index.fulltext.queryNodes('{fulltext_index_name}', $lucene_query)
        YIELD node, score
        WHERE node:{label}
        RETURN
            elementId(node) AS node_id,
            labels(node) AS labels,
            coalesce(node.name, "") AS name,
            coalesce(node.section_tag, "") AS section_tag,
            coalesce(node.qd_id, "") AS qd_id,
            coalesce(node.qd_title, "") AS qd_title,
            coalesce(node.{text_property_expr}, "") AS text,
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
    def _safe_property_name(property_name: str) -> str:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", property_name or ""):
            raise ValueError(f"Unsafe property name: {property_name}")
        return property_name
    
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
                        "name": item.get("name", ""),
                        "section_tag": item.get("section_tag", ""),
                        "qd_id": item.get("qd_id", ""),
                        "qd_title": item.get("qd_title", ""),
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
    
    @staticmethod
    def _score_direct_product_match(text: str, product_text: str) -> float:
        """
        Score direct lexical evidence that a chunk is about the queried product.

        Parameters
        ----------
        text : str
            Chunk text.
        product_text : str
            Queried product text.

        Returns
        -------
        float
            Direct product relevance score.
        """
        product_text = (product_text or "").strip()
        if not product_text:
            return 0.0

        text_norm = ChunkRetriever._normalize_text(text)
        product_norm = ChunkRetriever._normalize_text(product_text)
        product_tokens = ChunkRetriever._meaningful_tokens(product_text, min_len=3)

        score = 0.0

        if product_norm and product_norm in text_norm:
            score += 6.0

        token_hits = ChunkRetriever._token_hit_count(text_norm, product_tokens)
        score += 1.5 * token_hits

        return score

    @staticmethod
    def _score_direct_function_match(text: str, function_text: str) -> float:
        """
        Score direct lexical relevance of a chunk to the function text.

        Parameters
        ----------
        text : str
            Chunk text.
        function_text : str
            Queried function text.

        Returns
        -------
        float
            Direct function relevance score.
        """
        function_text = (function_text or "").strip()
        if not function_text:
            return 0.0

        text_norm = ChunkRetriever._normalize_text(text)
        function_norm = ChunkRetriever._normalize_text(function_text)
        function_tokens = ChunkRetriever._meaningful_tokens(function_text, min_len=3)

        score = 0.0

        if function_norm and function_norm in text_norm:
            score += 8.0

        token_hits = ChunkRetriever._token_hit_count(text_norm, function_tokens)
        score += 2.0 * token_hits

        if function_tokens:
            coverage = token_hits / len(function_tokens)
            score += 4.0 * coverage

        return score

    @staticmethod
    def _score_context_product_match(
        expanded_context: Dict[str, Any],
        product_text: str
    ) -> float:
        """
        Score indirect product relevance using graph-neighbor texts.

        Context sources
        ---------------
        - ts_texts
        - dfs_texts
        - rationale_texts
        - related_fs_texts

        Parameters
        ----------
        expanded_context : Dict[str, Any]
            Expanded context returned by expand_fs_context().
        product_text : str
            Queried product text.

        Returns
        -------
        float
            Context-based product relevance score.
        """
        product_text = (product_text or "").strip()
        if not product_text:
            return 0.0

        product_norm = ChunkRetriever._normalize_text(product_text)
        product_tokens = ChunkRetriever._meaningful_tokens(product_text, min_len=3)

        candidate_texts = []
        for key in ["ts_texts", "related_fs_texts", "rationale_texts"]:
            values = expanded_context.get(key) or []
            candidate_texts.extend([v for v in values if v])

        merged_context = " ".join(candidate_texts)
        merged_norm = ChunkRetriever._normalize_text(merged_context)

        score = 0.0

        if product_norm and product_norm in merged_norm:
            score += 4.0

        token_hits = ChunkRetriever._token_hit_count(merged_norm, product_tokens)
        score += 1.0 * token_hits

        return score

    @staticmethod
    def _score_context_function_match(
        expanded_context: Dict[str, Any],
        function_text: str
    ) -> float:
        """
        Score indirect function relevance using graph-neighbor texts.

        Context sources
        ---------------
        - ts_texts
        - dfs_texts
        - rationale_texts
        - related_fs_texts

        Parameters
        ----------
        expanded_context : Dict[str, Any]
            Expanded context returned by expand_fs_context().
        function_text : str
            Queried function text.

        Returns
        -------
        float
            Context-based function relevance score.
        """
        function_text = (function_text or "").strip()
        if not function_text:
            return 0.0

        function_norm = ChunkRetriever._normalize_text(function_text)
        function_tokens = ChunkRetriever._meaningful_tokens(function_text, min_len=3)

        candidate_texts = []
        for key in ["ts_texts", "related_fs_texts", "rationale_texts"]:
            values = expanded_context.get(key) or []
            candidate_texts.extend([v for v in values if v])

        merged_context = " ".join(candidate_texts)
        merged_norm = ChunkRetriever._normalize_text(merged_context)

        score = 0.0

        if function_norm and function_norm in merged_norm:
            score += 5.0

        token_hits = ChunkRetriever._token_hit_count(merged_norm, function_tokens)
        score += 1.2 * token_hits

        if function_tokens:
            coverage = token_hits / len(function_tokens)
            score += 2.5 * coverage

        return score

    @staticmethod
    def _build_product_function_sparse_queries(
        product_text: str,
        function_text: str
    ) -> List[str]:
        """
        Build multiple sparse retrieval queries for product + function search.

        Design rationale
        ----------------
        Product and function are treated as parallel retrieval signals.
        We do not force product to be a hard lexical gate, because some useful
        function-related chunks may not explicitly repeat the product text.

        Parameters
        ----------
        product_text : str
            Queried product text.
        function_text : str
            Queried function text.

        Returns
        -------
        List[str]
            Sparse query list.
        """
        product_text = (product_text or "").strip()
        function_text = (function_text or "").strip()

        queries = []

        product_tokens = ChunkRetriever._meaningful_tokens(product_text, min_len=3)
        function_tokens = ChunkRetriever._meaningful_tokens(function_text, min_len=3)

        product_terms = " OR ".join(product_tokens)
        function_terms = " OR ".join(function_tokens)

        if product_text and function_text:
            # Product-only views
            queries.append(f'"{product_text}"')
            if product_terms:
                queries.append(f"({product_terms})")

            # Function-only views
            queries.append(f'"{function_text}"')
            if function_terms:
                queries.append(f"({function_terms})")

            # Combined views
            if product_text and function_terms:
                queries.append(f'"{product_text}" AND ({function_terms})')
            if product_terms and function_terms:
                queries.append(f"({product_terms}) AND ({function_terms})")

        elif product_text:
            queries.append(f'"{product_text}"')
            if product_terms:
                queries.append(f"({product_terms})")

        elif function_text:
            queries.append(f'"{function_text}"')
            if function_terms:
                queries.append(f"({function_terms})")

        deduped = []
        seen = set()
        for q in queries:
            if q and q not in seen:
                deduped.append(q)
                seen.add(q)

        return deduped

    def retrieve_function_seeds(
        self,
        product_text: str,
        function_text: str = "",
        top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Retrieve FSChunk seeds for:
        - product-only query
        - product + function query
        - function-only query

        Retrieval strategy
        ------------------
        1. Perform broad dense recall from multiple semantic views.
        2. Perform broad sparse recall from multiple lexical views.
        3. Fuse recalled candidates with RRF.
        4. Expand each FS candidate into graph context.
        5. Re-rank using:
           - direct product relevance
           - direct function relevance
           - context product relevance
           - context function relevance
           - original retrieval score

        Important design choice
        -----------------------
        Product and function are treated in parallel as soft signals.
        We do NOT hard-filter by product name after retrieval.

        Parameters
        ----------
        product_text : str
            Queried product text, e.g. "iPS3".
        function_text : str
            Optional function refinement, e.g. "soft start".
        top_k : int
            Number of final FSChunk seeds.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked FSChunk seed nodes with detailed score decomposition.
        """
        product_text = (product_text or "").strip()
        function_text = (function_text or "").strip()

        dense_result_sets: List[List[Dict[str, Any]]] = []
        sparse_result_sets: List[List[Dict[str, Any]]] = []

        # --------------------------------------------------------------
        # Step 1. Dense retrieval from multiple semantic query views.
        # --------------------------------------------------------------
        dense_queries = []

        if product_text and function_text:
            dense_queries.append(f"{product_text} {function_text}")
            dense_queries.append(product_text)
            dense_queries.append(function_text)
        elif product_text:
            dense_queries.append(product_text)
        elif function_text:
            dense_queries.append(function_text)

        for query in dense_queries:
            results = self.dense_search_chunks(
                query_text=query,
                label="FSChunk",
                vector_index_name="fs_embedding_idx",
                top_k=top_k * 4
            )
            dense_result_sets.append(results)

        # --------------------------------------------------------------
        # Step 2. Sparse retrieval from multiple lexical query views.
        # --------------------------------------------------------------
        sparse_queries = self._build_product_function_sparse_queries(
            product_text=product_text,
            function_text=function_text
        )

        for lucene_query in sparse_queries:
            results = self.sparse_search_chunks(
                lucene_query=lucene_query,
                label="FSChunk",
                fulltext_index_name="fs_text_idx",
                top_k=top_k * 4
            )
            sparse_result_sets.append(results)

        # --------------------------------------------------------------
        # Step 3. Fusion across all retrieval channels.
        # --------------------------------------------------------------
        all_result_sets = dense_result_sets + sparse_result_sets
        if not all_result_sets:
            return []

        fused = self.rrf_fusion(all_result_sets)

        # --------------------------------------------------------------
        # Step 4. Graph-aware reranking.
        # --------------------------------------------------------------
        reranked = self.rerank_product_function_candidates(
            results=fused,
            product_text=product_text,
            function_text=function_text
        )

        return reranked[:top_k]

    def rerank_product_function_candidates(
        self,
        results: List[Dict[str, Any]],
        product_text: str,
        function_text: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Re-rank FS candidates using local text evidence and graph context.

        Ranking signals
        ---------------
        1. Direct product relevance in FS text
        2. Direct function relevance in FS text
        3. Product relevance in graph-neighbor texts
        4. Function relevance in graph-neighbor texts
        5. Original retrieval score as a weak prior

        Scoring policy
        --------------
        - If function_text is provided, function relevance is dominant.
        - Product relevance is always preserved as a parallel signal.
        - If function_text is empty, product relevance dominates.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Fused FSChunk candidates.
        product_text : str
            Queried product text.
        function_text : str
            Optional function refinement.

        Returns
        -------
        List[Dict[str, Any]]
            Re-ranked candidates with score details and expanded context.
        """
        product_text = (product_text or "").strip()
        function_text = (function_text or "").strip()

        reranked = []

        for item in results:
            text = item.get("text") or ""

            fs_node_id = (
                item.get("node_id")
                or item.get("id")
                or item.get("fs_node_id")
            )

            # Direct lexical evidence
            direct_product_score = self._score_direct_product_match(
                text=text,
                product_text=product_text
            )
            direct_function_score = self._score_direct_function_match(
                text=text,
                function_text=function_text
            )

            # Context evidence
            expanded_context = {}
            context_product_score = 0.0
            context_function_score = 0.0

            if fs_node_id:
                expanded_context = self.expand_fs_context(fs_node_id)

                context_product_score = self._score_context_product_match(
                    expanded_context=expanded_context,
                    product_text=product_text
                )
                context_function_score = self._score_context_function_match(
                    expanded_context=expanded_context,
                    function_text=function_text
                )

            # Retrieval prior from fused recall stage.
            # After RRF fusion, the upstream retrieval signal is stored under
            # rrf_score rather than score.
            base_score = item.get("rrf_score", item.get("score", 0.0))
            try:
                base_score = float(base_score)
            except Exception:
                base_score = 0.0

            # Final score
            if function_text:
                final_score = (
                    0.20 * direct_product_score +
                    0.20 * context_product_score +
                    0.35 * direct_function_score +
                    0.15 * context_function_score +
                    0.10 * base_score
                )
            else:
                final_score = (
                    0.40 * direct_product_score +
                    0.30 * context_product_score +
                    0.10 * direct_function_score +
                    0.10 * context_function_score +
                    0.10 * base_score
                )

            new_item = dict(item)
            new_item["direct_product_score"] = direct_product_score
            new_item["context_product_score"] = context_product_score
            new_item["direct_function_score"] = direct_function_score
            new_item["context_function_score"] = context_function_score
            new_item["final_score"] = final_score
            new_item["expanded_context"] = expanded_context

            reranked.append(new_item)

        reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return reranked

    def expand_fs_context(self, fs_node_id: str) -> Dict[str, Any]:
        """
        Expand one FSChunk to technical, related-FS, and rationale evidence.

        Graph expansion
        ---------------
        (ts:TSChunk)-[:IMPLEMENT]->(fs:FSChunk)
        (related_fs:FSChunk)-[:RELATED]->(fs:FSChunk)
        (r:RationaleChunk)-[:RATIONALE_FOR]->(fs:FSChunk)
        (r2:RationaleChunk)-[:RATIONALE_FOR]->(ts:TSChunk)
        (r3:RationaleChunk)-[:RATIONALE_FOR]->(related_fs:FSChunk)

        Notes
        -----
        - RELATED-linked FSChunk texts are returned as related_fs_texts.

        Parameters
        ----------
        fs_node_id : str
            Neo4j elementId of an FSChunk node.

        Returns
        -------
        Dict[str, Any]
            Expanded multi-view evidence around the FSChunk.
        """
        cypher = """
        MATCH (fs:FSChunk)
        WHERE elementId(fs) = $fs_node_id

        OPTIONAL MATCH (ts:TSChunk)-[:IMPLEMENT]->(fs)
        OPTIONAL MATCH (related_fs:FSChunk)-[:RELATED]->(fs)

        OPTIONAL MATCH (r: RationaleChunk)-[:RATIONALE_FOR]->(fs)
        OPTIONAL MATCH (r2:RationaleChunk)-[:RATIONALE_FOR]->(ts)
        OPTIONAL MATCH (r3:RationaleChunk)-[:RATIONALE_FOR]->(related_fs)

        RETURN
            elementId(fs) AS fs_node_id,
            fs.text AS fs_text,
            collect(DISTINCT ts.text) AS ts_texts,
            collect(DISTINCT related_fs.text) AS related_fs_texts,
            collect(DISTINCT r.text)
                + collect(DISTINCT r2.text)
                + collect(DISTINCT r3.text) AS rationale_texts
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
    def _token_hit_count(text_norm: str, tokens: List[str]) -> int:
        """
        Count how many query tokens appear in normalized text.

        Parameters
        ----------
        text_norm : str
            Already-normalized text.
        tokens : List[str]
            Token list.

        Returns
        -------
        int
            Number of matched tokens.
        """
        return sum(tok in text_norm for tok in tokens)

    @staticmethod
    def _score_function_match(text: str, function_text: str) -> float:
        """
        Score lexical relevance of a chunk to the function text.

        Design rationale
        ----------------
        When function_text is provided, it should be the dominant signal,
        because the user is searching for a more specific functional behavior.

        Parameters
        ----------
        text : str
            Chunk text.
        function_text : str
            Queried function text.

        Returns
        -------
        float
            Function relevance score.
        """
        function_text = (function_text or "").strip()
        if not function_text:
            return 0.0

        text_norm = ChunkRetriever._normalize_text(text)
        function_norm = ChunkRetriever._normalize_text(function_text)
        function_tokens = ChunkRetriever._meaningful_tokens(function_text, min_len=3)

        score = 0.0

        # Exact phrase match is a strong signal.
        if function_norm and function_norm in text_norm:
            score += 8.0

        # Token overlap provides softer lexical evidence.
        token_hits = ChunkRetriever._token_hit_count(text_norm, function_tokens)
        score += 2.0 * token_hits

        # Coverage bonus rewards chunks that cover most function tokens.
        if function_tokens:
            coverage = token_hits / len(function_tokens)
            score += 4.0 * coverage

        return score

    @staticmethod
    def _score_direct_element_match(text: str, element_name: str) -> float:
        """
        Score direct lexical evidence that a chunk is about the queried element.

        Parameters
        ----------
        text : str
            Chunk text.
        element_name : str
            Queried element.

        Returns
        -------
        float
            Direct element relevance score.
        """
        element_name = (element_name or "").strip()
        if not element_name:
            return 0.0

        text_norm = ChunkRetriever._normalize_text(text)
        element_norm = ChunkRetriever._normalize_text(element_name)
        element_tokens = ChunkRetriever._meaningful_tokens(element_name, min_len=3)

        score = 0.0

        # Exact phrase match receives a strong bonus.
        if element_norm and element_norm in text_norm:
            score += 6.0

        # Token overlap provides softer evidence.
        token_hits = ChunkRetriever._token_hit_count(text_norm, element_tokens)
        score += 1.5 * token_hits

        return score

    @staticmethod
    def _score_context_element_match(
        expanded_context: Dict[str, Any],
        element_name: str
    ) -> float:
        """
        Score indirect element relevance using graph-neighbor texts.

        Graph sources
        -------------
        - fs_texts
        - rationale_texts
        - fs_texts
        - rationale_texts

        Rationale
        ---------
        A TS sentence may not explicitly mention the queried element,
        but its graph neighborhood may still strongly indicate that it
        belongs to that element.

        Parameters
        ----------
        expanded_context : Dict[str, Any]
            Expanded context returned by expand_ts_context().
        element_name : str
            Queried element.

        Returns
        -------
        float
            Context-based element relevance score.
        """
        element_name = (element_name or "").strip()
        if not element_name:
            return 0.0

        element_norm = ChunkRetriever._normalize_text(element_name)
        element_tokens = ChunkRetriever._meaningful_tokens(element_name, min_len=3)

        candidate_texts = []
        for key in ["fs_texts", "rationale_texts"]:
            values = expanded_context.get(key) or []
            candidate_texts.extend([v for v in values if v])

        merged_context = " ".join(candidate_texts)
        merged_norm = ChunkRetriever._normalize_text(merged_context)

        score = 0.0

        # Exact phrase match in graph context is useful but slightly weaker
        # than direct chunk-level mention.
        if element_norm and element_norm in merged_norm:
            score += 4.0

        token_hits = ChunkRetriever._token_hit_count(merged_norm, element_tokens)
        score += 1.0 * token_hits

        return score

    @staticmethod
    def _score_context_function_match(
        expanded_context: Dict[str, Any],
        function_text: str
    ) -> float:
        """
        Score indirect function relevance using graph-neighbor texts.

        Rationale
        ---------
        Sometimes the TSChunk itself does not explicitly state the queried
        function text, while its linked FS/rationale evidence does.

        Parameters
        ----------
        expanded_context : Dict[str, Any]
            Expanded context returned by expand_ts_context().
        function_text : str
            Queried function text.

        Returns
        -------
        float
            Context-based function relevance score.
        """
        function_text = (function_text or "").strip()
        if not function_text:
            return 0.0

        function_norm = ChunkRetriever._normalize_text(function_text)
        function_tokens = ChunkRetriever._meaningful_tokens(function_text, min_len=3)

        candidate_texts = []
        for key in ["fs_texts", "rationale_texts"]:
            values = expanded_context.get(key) or []
            candidate_texts.extend([v for v in values if v])

        merged_context = " ".join(candidate_texts)
        merged_norm = ChunkRetriever._normalize_text(merged_context)

        score = 0.0

        if function_norm and function_norm in merged_norm:
            score += 5.0

        token_hits = ChunkRetriever._token_hit_count(merged_norm, function_tokens)
        score += 1.2 * token_hits

        if function_tokens:
            coverage = token_hits / len(function_tokens)
            score += 2.5 * coverage

        return score

    @staticmethod
    def _build_sparse_queries(
        element_name: str,
        function_text: str
    ) -> List[str]:
        """
        Build multiple sparse retrieval queries for robust lexical recall.

        Design rationale
        ----------------
        We avoid using a single hard lexical query because:
        1. Some useful chunks mention the function but not the element
        2. Some useful chunks mention the element but not the exact function wording
        3. Different views improve recall before graph-aware reranking

        Parameters
        ----------
        element_name : str
            Queried element.
        function_text : str
            Queried function text.

        Returns
        -------
        List[str]
            Sparse query list.
        """
        element_name = (element_name or "").strip()
        function_text = (function_text or "").strip()

        queries = []

        element_tokens = ChunkRetriever._meaningful_tokens(element_name, min_len=3)
        function_tokens = ChunkRetriever._meaningful_tokens(function_text, min_len=3)

        element_terms = " OR ".join(element_tokens)
        function_terms = " OR ".join(function_tokens)

        if element_name and function_text:
            # Function-only lexical view
            if function_text:
                queries.append(f'"{function_text}"')
            if function_terms:
                queries.append(f"({function_terms})")

            # Element-only lexical view
            if element_name:
                queries.append(f'"{element_name}"')
            if element_terms:
                queries.append(f"({element_terms})")

            # Combined lexical views
            if element_name and function_terms:
                queries.append(f'"{element_name}" AND ({function_terms})')
            if element_terms and function_terms:
                queries.append(f"({element_terms}) AND ({function_terms})")

        elif element_name:
            queries.append(f'"{element_name}"')
            if element_terms:
                queries.append(f"({element_terms})")

        elif function_text:
            queries.append(f'"{function_text}"')
            if function_terms:
                queries.append(f"({function_terms})")

        # Deduplicate while preserving order
        deduped = []
        seen = set()
        for q in queries:
            if q and q not in seen:
                deduped.append(q)
                seen.add(q)

        return deduped

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
        1. Perform broad multi-view recall using both dense and sparse search.
        2. Fuse all recalled candidates.
        3. Expand each TS candidate into graph context.
        4. Re-rank using:
           - direct function evidence
           - direct element evidence
           - context function evidence
           - context element evidence
           - original retrieval score

        Important design choice
        -----------------------
        We do NOT hard-filter by element at the beginning. This avoids losing
        function-relevant sentences that do not explicitly repeat the element
        name in the local text span.

        Parameters
        ----------
        element_name : str
            Queried element from structure analysis, e.g. "Motor control".
        function_text : str
            Optional function refinement, e.g. "relay switching".
            If empty, all sentences related to the element are returned.
        top_k : int
            Number of final TSChunk seeds to return.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked TSChunk seed nodes with expanded context and scoring details.
        """
        element_name = (element_name or "").strip()
        function_text = (function_text or "").strip()

        dense_result_sets: List[List[Dict[str, Any]]] = []
        sparse_result_sets: List[List[Dict[str, Any]]] = []

        # ------------------------------------------------------------------
        # Step 1. Dense retrieval from multiple semantic views.
        # ------------------------------------------------------------------
        dense_queries = []

        if element_name and function_text:
            dense_queries.append(f"{element_name} {function_text}")
            dense_queries.append(function_text)
            dense_queries.append(element_name)
        elif element_name:
            dense_queries.append(element_name)
        elif function_text:
            dense_queries.append(function_text)

        for query in dense_queries:
            results = self.dense_search_chunks(
                query_text=query,
                label="TSChunk",
                vector_index_name="ts_embedding_idx",
                top_k=top_k * 4
            )
            dense_result_sets.append(results)

        # ------------------------------------------------------------------
        # Step 2. Sparse retrieval from multiple lexical views.
        # ------------------------------------------------------------------
        sparse_queries = self._build_sparse_queries(
            element_name=element_name,
            function_text=function_text
        )

        for lucene_query in sparse_queries:
            results = self.sparse_search_chunks(
                lucene_query=lucene_query,
                label="TSChunk",
                fulltext_index_name="ts_text_idx",
                top_k=top_k * 4
            )
            sparse_result_sets.append(results)

        # ------------------------------------------------------------------
        # Step 3. Reciprocal Rank Fusion across all retrieval channels.
        # ------------------------------------------------------------------
        all_result_sets = dense_result_sets + sparse_result_sets

        if not all_result_sets:
            return []

        fused = self.rrf_fusion(all_result_sets)

        # ------------------------------------------------------------------
        # Step 4. Re-rank using local text + graph context.
        # ------------------------------------------------------------------
        reranked = self.rerank_element_function_candidates(
            results=fused,
            element_name=element_name,
            function_text=function_text
        )

        return reranked[:top_k]

    def rerank_element_function_candidates(
        self,
        results: List[Dict[str, Any]],
        element_name: str,
        function_text: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Re-rank TS candidates using direct text evidence and graph context.

        Ranking signals
        ---------------
        1. Direct function relevance in TS text
        2. Direct element relevance in TS text
        3. Function relevance in graph-neighbor texts
        4. Element relevance in graph-neighbor texts
        5. Original retrieval score as a weak prior

        Scoring policy
        --------------
        - If function_text is provided, function relevance is dominant.
        - If function_text is empty, element relevance is dominant.

        Parameters
        ----------
        results : List[Dict[str, Any]]
            Fused TSChunk candidates.
        element_name : str
            Queried element.
        function_text : str
            Optional function refinement.

        Returns
        -------
        List[Dict[str, Any]]
            Re-ranked candidates with detailed score decomposition.
        """
        element_name = (element_name or "").strip()
        function_text = (function_text or "").strip()

        reranked = []

        for item in results:
            text = item.get("text") or ""

            # Use the TS node identifier for graph expansion.
            # Depending on your existing retrieval functions, the id field may
            # be stored under "node_id", "id", or "ts_node_id".
            ts_node_id = (
                item.get("node_id")
                or item.get("id")
                or item.get("ts_node_id")
            )

            # Direct lexical evidence from the chunk itself
            direct_function_score = self._score_function_match(
                text=text,
                function_text=function_text
            )
            direct_element_score = self._score_direct_element_match(
                text=text,
                element_name=element_name
            )

            # Graph-context evidence from related nodes
            expanded_context = {}
            context_function_score = 0.0
            context_element_score = 0.0

            if ts_node_id:
                expanded_context = self.expand_ts_context(ts_node_id)

                context_function_score = self._score_context_function_match(
                    expanded_context=expanded_context,
                    function_text=function_text
                )
                context_element_score = self._score_context_element_match(
                    expanded_context=expanded_context,
                    element_name=element_name
                )

            # Original retrieval score is preserved as a weak prior.
            # After RRF fusion, the upstream retrieval signal is stored under
            # rrf_score rather than score.
            base_score = item.get("rrf_score", item.get("score", 0.0))
            try:
                base_score = float(base_score)
            except Exception:
                base_score = 0.0

            # Final scoring logic
            if function_text:
                # Function-specific query:
                # prioritize function match while still rewarding element relevance.
                final_score = (
                    0.40 * direct_function_score +
                    0.20 * context_function_score +
                    0.15 * direct_element_score +
                    0.15 * context_element_score +
                    0.10 * base_score
                )
            else:
                # Element-only query:
                # prioritize element-related evidence.
                final_score = (
                    0.35 * direct_element_score +
                    0.30 * context_element_score +
                    0.20 * direct_function_score +
                    0.05 * context_function_score +
                    0.10 * base_score
                )

            new_item = dict(item)
            new_item["direct_function_score"] = direct_function_score
            new_item["context_function_score"] = context_function_score
            new_item["direct_element_score"] = direct_element_score
            new_item["context_element_score"] = context_element_score
            new_item["final_score"] = final_score
            new_item["expanded_context"] = expanded_context

            reranked.append(new_item)

        reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return reranked

    def expand_ts_context(self, ts_node_id: str) -> Dict[str, Any]:
        """
        Expand one TSChunk to its related functional and rationale evidence.

        Graph expansion
        ---------------
        TSChunk -[:IMPLEMENT]-> FSChunk
        RationaleChunk -[:RATIONALE_FOR]-> TSChunk
        RationaleChunk -[:RATIONALE_FOR]-> FSChunk
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

        RETURN
            elementId(ts) AS ts_node_id,
            ts.text AS ts_text,
            collect(DISTINCT fs.text) AS fs_texts,
            collect(DISTINCT r_ts.text) + collect(DISTINCT r_fs.text) AS rationale_texts
        """
        results = self.run_query(cypher, ts_node_id=ts_node_id)
        return results[0] if results else {}
