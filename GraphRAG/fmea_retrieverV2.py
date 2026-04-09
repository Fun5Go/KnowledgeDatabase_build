from typing import Any, Callable, Dict, List, Optional, Tuple

from .neo4j_retriever import ChunkRetriever


CrossEncoderScorer = Callable[[str, str], float]


class LocalCrossEncoderScorer:
    """
    Optional local cross-encoder scorer.

    This wrapper is lazy so the retriever can still be imported even when
    sentence-transformers is not installed.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required to use LocalCrossEncoderScorer."
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def __call__(self, query_text: str, doc_text: str) -> float:
        model = self._get_model()
        score = model.predict([(query_text, doc_text)])[0]
        return float(score)


class FMEASentenceRetrieverV2(ChunkRetriever):
    """
    Graph-aware sentence retriever for structure analysis evidence.

    Design goals:
    - retrieve sentence-level evidence for one effect/mode at a time
    - use hybrid retrieval over FS / TS / QD / Rationale nodes
    - use graph neighbors as context features during reranking
    - bias effect evidence toward FS and mode evidence toward TS
    - keep the original effect/mode wording, while optionally adding
      positive-form query variants for better recall on positive sentences
    """

    SEARCH_SPECS = [
        ("FSChunk", "fs_embedding_idx", "fs_text_idx"),
        ("TSChunk", "ts_embedding_idx", "ts_text_idx"),
        ("QDChunk", "qd_embedding_idx", "qd_text_idx"),
        ("RationaleChunk", "rationale_embedding_idx", "rationale_text_idx"),
    ]

    LABEL_PRIORS = {
        "effect": {
            "FSChunk": 2.5,
            "TSChunk": 1.3,
            "QDChunk": 0.8,
            "RationaleChunk": 1.0,
        },
        "mode": {
            "TSChunk": 2.5,
            "FSChunk": 1.3,
            "QDChunk": 0.8,
            "RationaleChunk": 1.0,
        },
    }

    DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def query_effect_support(
        self,
        product_text: str,
        function_text: str,
        effect_texts: List[str],
        element_text: str = "",
        top_k_per_effect: int = 8,
        per_label_k: Optional[int] = None,
        expand_positive_variants: bool = True,
        cross_encoder_scorer: Optional[CrossEncoderScorer] = None,
        cross_encoder_top_n: int = 20,
    ) -> Dict[str, Any]:
        items = []
        for effect_text in self._iter_text_list(effect_texts):
            query_item = self._build_query_item(
                query_type="effect",
                product_text=product_text,
                element_text="",
                function_text=function_text,
                target_text=effect_text,
                expand_positive_variants=expand_positive_variants,
            )
            items.append(
                self._retrieve_for_query_item(
                    query_item=query_item,
                    top_k=top_k_per_effect,
                    per_label_k=per_label_k,
                    cross_encoder_scorer=cross_encoder_scorer,
                    cross_encoder_top_n=cross_encoder_top_n,
                )
            )

        return {
            "query_type": "effect",
            "product_text": (product_text or "").strip(),
            "element_text": "",
            "function_text": (function_text or "").strip(),
            "effect_support": items,
        }

    def query_mode_support(
        self,
        element_text: str,
        function_text: str,
        mode_texts: List[str],
        product_text: str = "",
        top_k_per_mode: int = 8,
        per_label_k: Optional[int] = None,
        expand_positive_variants: bool = True,
        cross_encoder_scorer: Optional[CrossEncoderScorer] = None,
        cross_encoder_top_n: int = 20,
    ) -> Dict[str, Any]:
        items = []
        for mode_text in self._iter_text_list(mode_texts):
            query_item = self._build_query_item(
                query_type="mode",
                product_text="",
                element_text=element_text,
                function_text=function_text,
                target_text=mode_text,
                expand_positive_variants=expand_positive_variants,
            )
            items.append(
                self._retrieve_for_query_item(
                    query_item=query_item,
                    top_k=top_k_per_mode,
                    per_label_k=per_label_k,
                    cross_encoder_scorer=cross_encoder_scorer,
                    cross_encoder_top_n=cross_encoder_top_n,
                )
            )

        return {
            "query_type": "mode",
            "product_text": "",
            "element_text": (element_text or "").strip(),
            "function_text": (function_text or "").strip(),
            "mode_support": items,
        }

    def _retrieve_for_query_item(
        self,
        query_item: Dict[str, Any],
        top_k: int,
        per_label_k: Optional[int],
        cross_encoder_scorer: Optional[CrossEncoderScorer],
        cross_encoder_top_n: int,
    ) -> Dict[str, Any]:
        per_label_k = per_label_k or max(top_k * 4, 16)

        dense_queries = self._build_dense_queries(query_item)
        sparse_queries = self._build_sparse_queries(query_item)

        result_sets: List[List[Dict[str, Any]]] = []
        for label, vector_index_name, fulltext_index_name in self.SEARCH_SPECS:
            for query_text in dense_queries:
                rows = self.dense_search_chunks(
                    query_text=query_text,
                    label=label,
                    vector_index_name=vector_index_name,
                    top_k=per_label_k,
                )
                for row in rows:
                    row["label"] = label
                result_sets.append(rows)

            for lucene_query in sparse_queries:
                rows = self.sparse_search_chunks(
                    lucene_query=lucene_query,
                    label=label,
                    fulltext_index_name=fulltext_index_name,
                    top_k=per_label_k,
                )
                for row in rows:
                    row["label"] = label
                result_sets.append(rows)

        fused = self.rrf_fusion(result_sets)
        candidate_ids = [item["node_id"] for item in fused[: max(top_k * 8, 40)]]
        neighbor_context = self._fetch_neighbor_context(candidate_ids)
        reranked = self._rerank_candidates(
            candidates=fused,
            query_item=query_item,
            neighbor_context=neighbor_context,
        )

        if cross_encoder_scorer:
            reranked = self._cross_encoder_rerank(
                candidates=reranked,
                query_item=query_item,
                cross_encoder_scorer=cross_encoder_scorer,
                top_n=cross_encoder_top_n,
            )

        return {
            "query_item": query_item,
            "dense_queries": dense_queries,
            "sparse_queries": sparse_queries,
            "evidence": self._package_evidence(
                candidates=reranked,
                neighbor_context=neighbor_context,
                top_k=top_k,
            ),
            "candidates": reranked[:top_k],
        }

    def _build_query_item(
        self,
        query_type: str,
        product_text: str,
        element_text: str,
        function_text: str,
        target_text: str,
        expand_positive_variants: bool,
    ) -> Dict[str, Any]:
        product_text = (product_text or "").strip()
        element_text = (element_text or "").strip()
        function_text = (function_text or "").strip()
        target_text = (target_text or "").strip()

        positive_variants = []
        if expand_positive_variants:
            positive_variants = self._build_positive_target_variants(
                query_type=query_type,
                target_text=target_text,
                function_text=function_text,
            )

        return {
            "query_type": query_type,
            "product_text": product_text,
            "element_text": element_text,
            "function_text": function_text,
            "target_text": target_text,
            "positive_target_variants": positive_variants,
        }

    def _build_dense_queries(self, query_item: Dict[str, Any]) -> List[str]:
        product_text = query_item.get("product_text", "")
        element_text = query_item.get("element_text", "")
        function_text = query_item.get("function_text", "")
        target_text = query_item.get("target_text", "")
        positive_variants = query_item.get("positive_target_variants", [])

        contexts = [value for value in [product_text, element_text, function_text] if value]
        targets = [target_text] + positive_variants
        queries: List[str] = []

        for target in targets:
            queries.append(target)
            if function_text:
                queries.append(f"{function_text}. {target}")
            if contexts:
                queries.append(". ".join(contexts + [target]))

            template = self._build_template_query(
                query_type=query_item.get("query_type", ""),
                product_text=product_text,
                element_text=element_text,
                function_text=function_text,
                target_text=target,
            )
            if template:
                queries.append(template)

        return self._dedupe_preserve_order(queries)

    def _build_sparse_queries(self, query_item: Dict[str, Any]) -> List[str]:
        product_text = query_item.get("product_text", "")
        element_text = query_item.get("element_text", "")
        function_text = query_item.get("function_text", "")
        target_text = query_item.get("target_text", "")
        positive_variants = query_item.get("positive_target_variants", [])

        queries: List[str] = []
        target_variants = [target_text] + positive_variants

        for value in [target_text, function_text, element_text, product_text]:
            queries.extend(self._phrase_and_token_queries(value))

        for variant in target_variants:
            variant_tokens = self._meaningful_tokens(variant)
            function_tokens = self._meaningful_tokens(function_text)
            element_tokens = self._meaningful_tokens(element_text)

            if variant:
                queries.append(f'"{variant}"')
            if function_text and variant:
                queries.append(f'"{function_text}" AND "{variant}"')
            if element_text and variant:
                queries.append(f'"{element_text}" AND "{variant}"')
            if function_tokens and variant_tokens:
                queries.append(
                    f"({' OR '.join(function_tokens)}) AND ({' OR '.join(variant_tokens)})"
                )
            if element_tokens and variant_tokens:
                queries.append(
                    f"({' OR '.join(element_tokens)}) AND ({' OR '.join(variant_tokens)})"
                )

        return self._dedupe_preserve_order(queries)

    def _rerank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query_item: Dict[str, Any],
        neighbor_context: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        query_type = query_item.get("query_type", "")
        product_text = query_item.get("product_text", "")
        element_text = query_item.get("element_text", "")
        function_text = query_item.get("function_text", "")
        target_text = query_item.get("target_text", "")
        positive_variants = query_item.get("positive_target_variants", [])

        if query_type == "effect":
            element_text = ""
        elif query_type == "mode":
            product_text = ""

        reranked: List[Dict[str, Any]] = []
        for item in candidates:
            label = item.get("label") or (item.get("labels") or [""])[0]
            text = item.get("text", "")
            context = neighbor_context.get(item["node_id"], {})
            neighbor_text = context.get("neighbor_text", "")

            direct_target = self._score_target_match(
                text=text,
                target_text=target_text,
                positive_variants=positive_variants,
            )
            direct_function = self._score_function_match(text, function_text)
            direct_element = self._score_direct_element_match(text, element_text)
            direct_product = self._score_direct_product_match(text, product_text)

            graph_target = self._score_target_match(
                text=neighbor_text,
                target_text=target_text,
                positive_variants=positive_variants,
            )
            graph_function = self._score_function_match(neighbor_text, function_text)
            graph_element = self._score_direct_element_match(neighbor_text, element_text)

            label_bonus = self.LABEL_PRIORS.get(query_type, {}).get(label, 0.0)
            stage1_score = (
                0.38 * direct_target
                + 0.20 * direct_function
                + 0.08 * direct_element
                + 0.04 * direct_product
                + 0.14 * graph_target
                + 0.08 * graph_function
                + 0.03 * graph_element
                + 0.05 * label_bonus
                + 0.10 * float(item.get("rrf_score", 0.0))
            )

            new_item = dict(item)
            new_item["label"] = label
            new_item["neighbor_context"] = context
            new_item["score_breakdown"] = {
                "direct_target": direct_target,
                "direct_function": direct_function,
                "direct_element": direct_element,
                "direct_product": direct_product,
                "graph_target": graph_target,
                "graph_function": graph_function,
                "graph_element": graph_element,
                "label_bonus": label_bonus,
                "rrf_score": float(item.get("rrf_score", 0.0)),
            }
            new_item["stage1_score"] = stage1_score
            new_item["final_score"] = stage1_score
            reranked.append(new_item)

        reranked.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return reranked

    def _cross_encoder_rerank(
        self,
        candidates: List[Dict[str, Any]],
        query_item: Dict[str, Any],
        cross_encoder_scorer: CrossEncoderScorer,
        top_n: int,
    ) -> List[Dict[str, Any]]:
        kept = candidates[:top_n]
        tail = candidates[top_n:]

        for item in kept:
            pair_query = self._build_cross_encoder_query(query_item)
            pair_doc = self._build_cross_encoder_document(item)
            ce_score = float(cross_encoder_scorer(pair_query, pair_doc))
            item["cross_encoder_score"] = ce_score
            item["final_score"] = 0.65 * item.get("stage1_score", 0.0) + 0.35 * ce_score

        kept.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return kept + tail

    @classmethod
    def build_local_cross_encoder(
        cls,
        model_name: Optional[str] = None,
    ) -> LocalCrossEncoderScorer:
        return LocalCrossEncoderScorer(
            model_name=model_name or cls.DEFAULT_CROSS_ENCODER_MODEL
        )

    def _package_evidence(
        self,
        candidates: List[Dict[str, Any]],
        neighbor_context: Dict[str, Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        seen = set()

        for item in candidates:
            node_id = item.get("node_id", "")
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)

            context = neighbor_context.get(node_id, {})
            supporting_context = context.get("neighbors", [])[:2]
            evidence.append({
                "primary_sentence": {
                    "node_id": node_id,
                    "label": item.get("label", ""),
                    "text": item.get("text", ""),
                    "score": item.get("final_score", 0.0),
                },
                "supporting_context": supporting_context,
                "score_breakdown": item.get("score_breakdown", {}),
            })

            if len(evidence) >= top_k:
                break

        return evidence

    def _fetch_neighbor_context(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not node_ids:
            return {}

        cypher = """
        MATCH (n)
        WHERE elementId(n) IN $node_ids
        OPTIONAL MATCH (n)-[rel]-(m)
        WHERE
            (type(rel) = "IMPLEMENT" AND (
                (n:FSChunk AND m:TSChunk) OR
                (n:TSChunk AND m:FSChunk)
            )) OR
            (type(rel) = "DETAILED_BY" AND (
                (n:TSChunk AND m:QDChunk) OR
                (n:QDChunk AND m:TSChunk)
            )) OR
            (type(rel) = "RATIONALE_FOR" AND (
                (n:RationaleChunk AND (m:FSChunk OR m:TSChunk OR m:QDChunk)) OR
                (m:RationaleChunk AND (n:FSChunk OR n:TSChunk OR n:QDChunk))
            ))
        RETURN
            elementId(n) AS node_id,
            collect(
                CASE
                    WHEN m IS NULL THEN NULL
                    ELSE {
                        node_id: elementId(m),
                        label: labels(m)[0],
                        text: coalesce(m.text, ""),
                        relationship: type(rel)
                    }
                END
            ) AS neighbors
        """
        rows = self.run_query(cypher, node_ids=node_ids)
        output: Dict[str, Dict[str, Any]] = {}

        for row in rows:
            neighbors = [
                item for item in row.get("neighbors", [])
                if item and (item.get("text", "") or "").strip()
            ]
            output[row["node_id"]] = {
                "neighbors": neighbors,
                "neighbor_text": " ".join(item.get("text", "") for item in neighbors),
            }

        return output

    @staticmethod
    def _build_template_query(
        query_type: str,
        product_text: str,
        element_text: str,
        function_text: str,
        target_text: str,
    ) -> str:
        parts = []
        if product_text:
            parts.append(f"product {product_text}")
        if element_text:
            parts.append(f"element {element_text}")
        if function_text:
            parts.append(f"function {function_text}")

        if query_type == "effect":
            parts.append(f"has effect {target_text}")
        elif query_type == "mode":
            parts.append(f"has failure mode {target_text}")

        return ", ".join(parts)

    def _build_positive_target_variants(
        self,
        query_type: str,
        target_text: str,
        function_text: str,
    ) -> List[str]:
        """
        Build auxiliary positive-form variants.

        We do not replace the original effect/mode text because the negative
        wording is still semantically important. Instead, we add a few softer
        variants to improve recall against positive functional statements such
        as FS/TS sentences.
        """
        normalized = self._normalize_text(target_text)
        variants: List[str] = []

        if query_type == "effect":
            if "cannot " in normalized:
                variants.append(normalized.replace("cannot ", "can ", 1))
            if " can not " in f" {normalized} ":
                variants.append(normalized.replace("can not", "can", 1))
            if " without " in f" {normalized} ":
                variants.append(normalized.replace("without", "with", 1))
            if normalized.startswith("no "):
                variants.append(normalized[3:])

            if function_text and variants:
                variants.extend(
                    f"{function_text} {variant}"
                    for variant in variants[:]
                )

        return self._dedupe_preserve_order(
            [variant.strip() for variant in variants if variant.strip()]
        )

    def _score_target_match(
        self,
        text: str,
        target_text: str,
        positive_variants: List[str],
    ) -> float:
        score = self._score_phrase_match(text, target_text)
        for variant in positive_variants:
            score = max(score, 0.85 * self._score_phrase_match(text, variant))
        return score

    def _build_cross_encoder_query(self, query_item: Dict[str, Any]) -> str:
        query_type = query_item.get("query_type", "")
        if query_type == "effect":
            instruction = (
                "Task: score whether the candidate sentence is direct evidence "
                "for the stated effect under the stated function. Prefer direct "
                "support over broad background context."
            )
        else:
            instruction = (
                "Task: score whether the candidate sentence is direct evidence "
                "for the stated failure mode under the stated function. Prefer "
                "direct mode support over general discussion."
            )

        lines = [instruction, f"Type: {query_type}"]
        if query_item.get("product_text"):
            lines.append(f"Product: {query_item['product_text']}")
        if query_item.get("element_text"):
            lines.append(f"Element: {query_item['element_text']}")
        if query_item.get("function_text"):
            lines.append(f"Function: {query_item['function_text']}")
        lines.append(f"Target: {query_item.get('target_text', '')}")
        positive_variants = query_item.get("positive_target_variants", [])
        if positive_variants:
            lines.append(f"Auxiliary positive variants: {' | '.join(positive_variants)}")
        return "\n".join(lines)

    def _build_cross_encoder_document(self, candidate: Dict[str, Any]) -> str:
        lines = [
            f"Label: {candidate.get('label', '')}",
            f"Sentence: {candidate.get('text', '')}",
        ]
        neighbors = candidate.get("neighbor_context", {}).get("neighbors", [])[:3]
        if neighbors:
            lines.append(
                "Linked context: " + " | ".join(
                    f"{item.get('label', '')}: {item.get('text', '')}"
                    for item in neighbors
                )
            )
        return "\n".join(lines)

    def _phrase_and_token_queries(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        queries = [f'"{text}"']
        tokens = self._meaningful_tokens(text)
        if tokens:
            queries.append(f"({' OR '.join(tokens)})")
        return queries

    @staticmethod
    def _iter_text_list(values: Any) -> List[str]:
        if not isinstance(values, list):
            return []
        return [
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        ]

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

    @staticmethod
    def _score_phrase_match(text: str, phrase_text: str) -> float:
        phrase_text = (phrase_text or "").strip()
        if not phrase_text:
            return 0.0

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
