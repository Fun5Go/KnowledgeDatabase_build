from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from neo4j_retriever import ChunkRetriever


class FMEASentenceRetriever(ChunkRetriever):
    """
    Retriever for linking FMEA texts to supporting document sentences.

    The retriever searches TSChunk, FSChunk, and RationaleChunk equally,
    then groups retrieved nodes if they are connected in the graph.
    """

    SEARCH_SPECS = [
        ("TSChunk", "ts_embedding_idx", "ts_text_idx"),
        ("FSChunk", "fs_embedding_idx", "fs_text_idx"),
        ("RationaleChunk", "rationale_embedding_idx", "rationale_text_idx"),
    ]

    def query_fmea_sentences(
        self,
        product_text: str = "",
        function_text: str = "",
        effect_text: str = "",
        element_text: str = "",
        mode_text: str = "",
        cause_text: str = "",
        top_k: int = 8,
        per_label_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Search for document evidence supporting FMEA texts.

        Supported query shapes include:
        - product + function + effect
        - element + function + mode
        - element + cause

        Product, function, and element may be used for recall, but final
        scoring only considers effect / mode / cause texts.
        """
        fields = self._clean_fields(
            product_text=product_text,
            function_text=function_text,
            effect_text=effect_text,
            element_text=element_text,
            mode_text=mode_text,
            cause_text=cause_text,
        )

        dense_queries = self._build_dense_queries(fields)
        sparse_queries = self._build_sparse_queries(fields)

        per_label_k = per_label_k or max(top_k * 4, 12)

        result_sets: List[List[Dict[str, Any]]] = []
        for label, vector_index_name, fulltext_index_name in self.SEARCH_SPECS:
            # Search each sentence family equally instead of privileging
            # one document type at recall time.
            for query_text in dense_queries:
                results = self.dense_search_chunks(
                    query_text=query_text,
                    label=label,
                    vector_index_name=vector_index_name,
                    top_k=per_label_k,
                )
                for item in results:
                    item["label"] = label
                result_sets.append(results)

            for lucene_query in sparse_queries:
                results = self.sparse_search_chunks(
                    lucene_query=lucene_query,
                    label=label,
                    fulltext_index_name=fulltext_index_name,
                    top_k=per_label_k,
                )
                for item in results:
                    item["label"] = label
                result_sets.append(results)

        fused = self.rrf_fusion(result_sets)
        # Re-rank using only the node's own text.
        reranked = self._rerank_candidates(fused, fields)
        # Group connected hits and promote the groups to the primary output.
        grouped = self._group_connected_candidates(reranked[: max(top_k * 4, 12)])

        return {
            "query_fields": fields,
            "dense_queries": dense_queries,
            "sparse_queries": sparse_queries,
            "groups": grouped[:top_k],
            "ungrouped_candidates": reranked[:top_k],
        }

    def query_structure_input(
        self,
        structure_input: Dict[str, Any],
        top_k_per_query: int = 8,
        aggregate_top_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Run multiple FMEA sentence queries derived from a structure-analysis
        payload and aggregate the returned connected text groups.
        """
        queries = self.flatten_structure_queries(structure_input)
        query_results = []

        for query_spec in queries:
            result = self.query_fmea_sentences(
                product_text=query_spec.get("product_text", ""),
                function_text=query_spec.get("function_text", ""),
                effect_text=query_spec.get("effect_text", ""),
                element_text=query_spec.get("element_text", ""),
                mode_text=query_spec.get("mode_text", ""),
                cause_text=query_spec.get("cause_text", ""),
                top_k=top_k_per_query,
            )
            query_results.append({
                "query_spec": query_spec,
                "result": result,
            })

        aggregated_groups = self.aggregate_groups_from_query_results(
            query_results=query_results,
            top_k=aggregate_top_k,
        )

        return {
            "queries": queries,
            "query_results": query_results,
            "groups": aggregated_groups,
        }

    @staticmethod
    def flatten_structure_queries(structure_input: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert structure-analysis input into FMEA-oriented query specs.
        """
        queries: List[Dict[str, str]] = []
        product_text = (structure_input.get("product_domain") or "").strip()

        for node in structure_input.get("nodes", []):
            element_text = (node.get("failure_element") or "").strip()

            modes = node.get("modes", {})
            if isinstance(modes, dict):
                for function_text, mode_list in modes.items():
                    function_text = (function_text or "").strip()
                    for mode_text in mode_list or []:
                        mode_text = (mode_text or "").strip()
                        if mode_text:
                            queries.append({
                                "product_text": product_text,
                                "element_text": element_text,
                                "function_text": function_text,
                                "mode_text": mode_text,
                                "query_type": "mode",
                                "query_text": mode_text,
                            })
            elif isinstance(modes, list):
                for mode_text in modes:
                    mode_text = (mode_text or "").strip()
                    if mode_text:
                        queries.append({
                            "product_text": product_text,
                            "element_text": element_text,
                            "mode_text": mode_text,
                            "query_type": "mode",
                            "query_text": mode_text,
                        })

            causes = node.get("causes", {})
            if isinstance(causes, dict):
                for _, cause_list in causes.items():
                    for cause_text in cause_list or []:
                        cause_text = (cause_text or "").strip()
                        if cause_text:
                            queries.append({
                                "product_text": product_text,
                                "element_text": element_text,
                                "cause_text": cause_text,
                                "query_type": "cause",
                                "query_text": cause_text,
                            })
            elif isinstance(causes, list):
                for cause_text in causes:
                    cause_text = (cause_text or "").strip()
                    if cause_text:
                        queries.append({
                            "product_text": product_text,
                            "element_text": element_text,
                            "cause_text": cause_text,
                            "query_type": "cause",
                            "query_text": cause_text,
                        })

            for effect_text in node.get("effects", []):
                effect_text = (effect_text or "").strip()
                if effect_text:
                    queries.append({
                        "product_text": product_text,
                        "element_text": element_text,
                        "effect_text": effect_text,
                        "query_type": "effect",
                        "query_text": effect_text,
                    })

        return FMEASentenceRetriever._dedupe_query_specs(queries)

    @staticmethod
    def aggregate_groups_from_query_results(
        query_results: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate connected text groups returned by multiple structure-derived
        queries. Groups are merged when they contain the same retrieved nodes.
        """
        group_map = defaultdict(lambda: {
            "group_id": None,
            "score": 0.0,
            "hit_count": 0,
            "matched_queries": [],
            "relationships": [],
            "nodes": [],
        })

        for item in query_results:
            query_spec = item.get("query_spec", {})
            groups = item.get("result", {}).get("groups", [])

            for group in groups:
                node_ids = sorted(
                    node.get("node_id", "")
                    for node in group.get("nodes", [])
                    if node.get("node_id")
                )
                if not node_ids:
                    continue

                group_id = "|".join(node_ids)
                bucket = group_map[group_id]
                bucket["group_id"] = group_id
                bucket["score"] += float(group.get("score", 0.0))
                bucket["hit_count"] += 1
                bucket["nodes"] = group.get("nodes", [])
                bucket["relationships"] = group.get("relationships", [])
                bucket["matched_queries"].append({
                    "query_type": query_spec.get("query_type", ""),
                    "query_text": query_spec.get("query_text", ""),
                    "fields": {
                        key: value
                        for key, value in query_spec.items()
                        if key.endswith("_text") and value
                    },
                    "group_score": float(group.get("score", 0.0)),
                })

        aggregated = list(group_map.values())
        aggregated.sort(
            key=lambda x: (x.get("score", 0.0), x.get("hit_count", 0)),
            reverse=True,
        )
        return aggregated[:top_k]

    @staticmethod
    def _clean_fields(**kwargs: str) -> Dict[str, str]:
        return {
            key: (value or "").strip()
            for key, value in kwargs.items()
            if (value or "").strip()
        }

    def _build_dense_queries(self, fields: Dict[str, str]) -> List[str]:
        queries: List[str] = []
        ordered_keys = [
            "product_text",
            "element_text",
            "function_text",
            "effect_text",
            "mode_text",
            "cause_text",
        ]

        values = [fields[key] for key in ordered_keys if key in fields]

        # Broad semantic query that contains all available signals.
        if values:
            queries.append(" ".join(values))

        target_text = (
            fields.get("effect_text")
            or fields.get("mode_text")
            or fields.get("cause_text")
            or ""
        )
        context_texts = [
            fields.get("product_text", ""),
            fields.get("element_text", ""),
            fields.get("function_text", ""),
        ]
        context_texts = [value for value in context_texts if value]

        # Target text is the FMEA phrase we want to ground in evidence.
        # Context texts are optional anchors such as product/element/function.
        if target_text:
            queries.append(target_text)
            for context in context_texts:
                queries.append(f"{context} {target_text}")

        if fields.get("function_text"):
            queries.append(fields["function_text"])
        if fields.get("element_text"):
            queries.append(fields["element_text"])
        if fields.get("product_text"):
            queries.append(fields["product_text"])

        return self._dedupe_preserve_order(queries)

    def _build_sparse_queries(self, fields: Dict[str, str]) -> List[str]:
        queries: List[str] = []
        for value in fields.values():
            queries.extend(self._phrase_and_token_queries(value))

        target_text = (
            fields.get("effect_text")
            or fields.get("mode_text")
            or fields.get("cause_text")
            or ""
        )

        for context_key in ["product_text", "element_text", "function_text"]:
            context_text = fields.get(context_key, "")
            if context_text and target_text:
                # Add soft lexical conjunctions so sparse search can recover
                # hits where context and target appear together in one sentence.
                context_tokens = self._meaningful_tokens(context_text)
                target_tokens = self._meaningful_tokens(target_text)
                if context_tokens and target_tokens:
                    queries.append(
                        f"({' OR '.join(context_tokens)}) AND ({' OR '.join(target_tokens)})"
                    )
                queries.append(f'"{context_text}" AND "{target_text}"')

        return self._dedupe_preserve_order(queries)

    def _rerank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        fields: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        reranked: List[Dict[str, Any]] = []

        for item in candidates:
            text = item.get("text", "")

            # Direct scores answer:
            # "How much does this sentence itself mention the query fields?"
            direct_scores = {
                "product": self._score_direct_product_match(
                    text=text,
                    product_text=fields.get("product_text", ""),
                ),
                "element": self._score_direct_element_match(
                    text=text,
                    element_name=fields.get("element_text", ""),
                ),
                "function": self._score_function_match(
                    text=text,
                    function_text=fields.get("function_text", ""),
                ),
                "effect": self._score_phrase_match(
                    text=text,
                    phrase_text=fields.get("effect_text", ""),
                ),
                "mode": self._score_phrase_match(
                    text=text,
                    phrase_text=fields.get("mode_text", ""),
                ),
                "cause": self._score_phrase_match(
                    text=text,
                    phrase_text=fields.get("cause_text", ""),
                ),
            }

            # Final scoring uses only effect / mode / cause. Product / element /
            # function may still help recall upstream, but they do not affect
            # the final node ranking.
            weighted = {
                "effect": 0.35 * direct_scores["effect"],
                "mode": 0.35 * direct_scores["mode"],
                "cause": 0.35 * direct_scores["cause"],
                "retrieval_prior": 0.10 * float(item.get("rrf_score", 0.0)),
            }

            final_score = sum(weighted.values())

            new_item = dict(item)
            new_item["direct_scores"] = direct_scores
            new_item["weighted_scores"] = weighted
            new_item["final_score"] = final_score
            reranked.append(new_item)

        reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return reranked

    def _group_connected_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        candidate_map = {item["node_id"]: item for item in candidates}
        relationships = self._fetch_candidate_relationships(list(candidate_map))
        adjacency = {node_id: set() for node_id in candidate_map}
        edge_map: Dict[Tuple[str, str], List[str]] = {}

        for rel in relationships:
            source_id = rel["source_id"]
            target_id = rel["target_id"]
            if source_id not in candidate_map or target_id not in candidate_map:
                continue

            adjacency[source_id].add(target_id)
            adjacency[target_id].add(source_id)

            pair = tuple(sorted((source_id, target_id)))
            edge_map.setdefault(pair, [])
            if rel["relationship"] not in edge_map[pair]:
                edge_map[pair].append(rel["relationship"])

        # Build connected components over the retrieved candidates. Each
        # component becomes one returned group with a summed score.
        visited = set()
        groups: List[Dict[str, Any]] = []

        for node_id in candidate_map:
            if node_id in visited:
                continue

            stack = [node_id]
            component: List[str] = []
            visited.add(node_id)

            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            component_nodes = [candidate_map[item_id] for item_id in component]
            component_nodes.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

            component_edges = []
            for i, source_id in enumerate(component):
                for target_id in component[i + 1:]:
                    pair = tuple(sorted((source_id, target_id)))
                    if pair in edge_map:
                        component_edges.append({
                            "source_id": pair[0],
                            "target_id": pair[1],
                            "relationships": edge_map[pair],
                        })

            group_score = sum(node.get("final_score", 0.0) for node in component_nodes)
            groups.append({
                "score": group_score,
                "node_count": len(component_nodes),
                "nodes": component_nodes,
                "relationships": component_edges,
            })

        groups.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return groups

    def _fetch_candidate_relationships(self, node_ids: List[str]) -> List[Dict[str, str]]:
        cypher = """
        MATCH (a)-[rel]-(b)
        WHERE elementId(a) IN $node_ids
          AND elementId(b) IN $node_ids
          AND elementId(a) <> elementId(b)
          AND (
            (type(rel) = "IMPLEMENT" AND (
                (a:TSChunk AND b:FSChunk) OR
                (a:FSChunk AND b:TSChunk)
            )) OR
            (type(rel) = "RELATED" AND a:FSChunk AND b:FSChunk) OR
            (type(rel) = "RATIONALE_FOR" AND (
                (a:RationaleChunk AND b:TSChunk) OR
                (a:TSChunk AND b:RationaleChunk) OR
                (a:RationaleChunk AND b:FSChunk) OR
                (a:FSChunk AND b:RationaleChunk)
            ))
          )
        RETURN DISTINCT
            elementId(a) AS source_id,
            elementId(b) AS target_id,
            type(rel) AS relationship
        """
        # Restrict grouping to relationships that are already meaningful for
        # sentence-level evidence packaging in the current document graph.
        return self.run_query(cypher, node_ids=node_ids)

    @staticmethod
    def _dedupe_query_specs(query_specs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        output: List[Dict[str, str]] = []
        seen = set()
        for query_spec in query_specs:
            key = tuple(
                (name, query_spec.get(name, ""))
                for name in [
                    "product_text",
                    "element_text",
                    "function_text",
                    "mode_text",
                    "cause_text",
                    "effect_text",
                    "query_type",
                    "query_text",
                ]
            )
            if key not in seen:
                seen.add(key)
                output.append(query_spec)
        return output

    @staticmethod
    def _dedupe_preserve_order(values: List[str]) -> List[str]:
        output: List[str] = []
        seen = set()
        for value in values:
            value = (value or "").strip()
            if value and value not in seen:
                output.append(value)
                seen.add(value)
        return output

    def _phrase_and_token_queries(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        # Pair an exact phrase query with a softer token query so sparse search
        # can recover both exact and partially-overlapping matches.
        queries = [f'"{text}"']
        tokens = self._meaningful_tokens(text)
        if tokens:
            queries.append(f"({' OR '.join(tokens)})")
        return queries

    @staticmethod
    def _score_phrase_match(text: str, phrase_text: str) -> float:
        phrase_text = (phrase_text or "").strip()
        if not phrase_text:
            return 0.0

        # This score is used for effect/mode/cause phrases, which are often
        # short. Phrase hit, token overlap, and token coverage all matter.
        text_norm = ChunkRetriever._normalize_text(text)
        phrase_norm = ChunkRetriever._normalize_text(phrase_text)
        phrase_tokens = ChunkRetriever._meaningful_tokens(phrase_text, min_len=3)

        score = 0.0
        if phrase_norm and phrase_norm in text_norm:
            score += 8.0

        token_hits = ChunkRetriever._token_hit_count(text_norm, phrase_tokens)
        score += 2.0 * token_hits

        if phrase_tokens:
            coverage = token_hits / len(phrase_tokens)
            score += 3.0 * coverage

        return score
