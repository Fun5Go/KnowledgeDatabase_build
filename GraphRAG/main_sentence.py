import math
import re
from typing import Any, Dict, List, Optional

from .fmea_retrieverV2 import CrossEncoderScorer, FMEASentenceRetrieverV2
from .neo4j_retriever import ChunkRetriever, get_query_embedding


def build_sentence_doc_chunk_query(sentence: str) -> Dict[str, Any]:
    sentence = " ".join((sentence or "").split())

    return {
        "query_type": "requirement_sentence_doc_chunk",
        "sentence": sentence,
        "target_text": sentence,
        "query_text": sentence,
    }


def build_dense_queries(query_spec: Dict[str, Any]) -> List[str]:
    return dedupe_preserve_order([
        query_spec.get("target_text", ""),
        query_spec.get("query_text", ""),
    ])


def build_sparse_queries(
    retriever: FMEASentenceRetrieverV2,
    query_spec: Dict[str, Any],
) -> List[str]:
    return dedupe_preserve_order(
        retriever._phrase_and_token_queries(query_spec.get("target_text", ""))
    )


def query_doc_chunks_for_sentence(
    retriever: FMEASentenceRetrieverV2,
    query_spec: Dict[str, Any],
    top_k: int = 8,
    per_label_k: int = 20,
    retrieval_mode: str = "hybrid",
    disciplines: Any = None,
    use_cross_encoder_rerank: bool = False,
    cross_encoder_scorer: Optional[CrossEncoderScorer] = None,
    cross_encoder_top_n: int = 20,
    use_section_tag_bonus: bool = True,
    section_bonus_mode: str = "hybrid",
    section_bonus_weight: float = 0.05,
    is_QD: bool = False,
) -> Dict[str, Any]:
    retrieval_mode = retriever._normalize_retrieval_mode(retrieval_mode)
    section_bonus_mode = retriever._normalize_retrieval_mode(section_bonus_mode)
    allowed_labels = normalize_discipline_labels(disciplines)
    dense_queries = build_dense_queries(query_spec)
    sparse_queries = build_sparse_queries(retriever, query_spec)

    if retrieval_mode == "dense":
        return query_doc_chunks_for_sentence_by_dense_score(
            retriever=retriever,
            query_spec=query_spec,
            top_k=top_k,
            per_label_k=per_label_k,
            disciplines=disciplines,
            use_cross_encoder_rerank=use_cross_encoder_rerank,
            cross_encoder_scorer=cross_encoder_scorer,
            cross_encoder_top_n=cross_encoder_top_n,
            use_section_tag_bonus=use_section_tag_bonus,
            section_bonus_mode=section_bonus_mode,
            section_bonus_weight=section_bonus_weight,
            is_QD=is_QD,
        )
    if retrieval_mode == "sparse":
        return query_doc_chunks_for_sentence_by_sparse_score(
            retriever=retriever,
            query_spec=query_spec,
            top_k=top_k,
            per_label_k=per_label_k,
            disciplines=disciplines,
            use_cross_encoder_rerank=use_cross_encoder_rerank,
            cross_encoder_scorer=cross_encoder_scorer,
            cross_encoder_top_n=cross_encoder_top_n,
            use_section_tag_bonus=use_section_tag_bonus,
            section_bonus_mode=section_bonus_mode,
            section_bonus_weight=section_bonus_weight,
            is_QD=is_QD,
        )

    result_sets: List[List[Dict[str, Any]]] = []
    for spec in build_search_specs(retriever, is_QD=is_QD):
        label = spec["label"]
        vector_index_name = spec["vector_index_name"]
        fulltext_index_name = spec["fulltext_index_name"]
        text_property = spec["text_property"]
        title_property = spec.get("title_property", "")
        id_property = spec.get("id_property", "")

        if retrieval_mode in {"dense", "hybrid"}:
            for query_text in dense_queries:
                if label in {"QDChunk", "FATChunk"}:
                    rows = dense_search_qualification_chunks(
                        retriever=retriever,
                        query_text=query_text,
                        label=label,
                        vector_index_name=vector_index_name,
                        title_property=title_property,
                        text_property=text_property,
                        id_property=id_property,
                        top_k=per_label_k,
                    )
                else:
                    rows = retriever.dense_search_chunks(
                        query_text=query_text,
                        label=label,
                        vector_index_name=vector_index_name,
                        top_k=per_label_k,
                        text_property=text_property,
                    )
                rows = filter_rows_by_labels(rows, allowed_labels, label=label)
                for row in rows:
                    row["label"] = label
                result_sets.append(rows)

        if retrieval_mode in {"sparse", "hybrid"}:
            for lucene_query in sparse_queries:
                if label in {"QDChunk", "FATChunk"}:
                    rows = sparse_search_qualification_chunks(
                        retriever=retriever,
                        lucene_query=lucene_query,
                        label=label,
                        fulltext_index_name=fulltext_index_name,
                        title_property=title_property,
                        text_property=text_property,
                        id_property=id_property,
                        top_k=per_label_k,
                    )
                else:
                    rows = retriever.sparse_search_chunks(
                        lucene_query=lucene_query,
                        label=label,
                        fulltext_index_name=fulltext_index_name,
                        top_k=per_label_k,
                        text_property=text_property,
                    )
                rows = filter_rows_by_labels(rows, allowed_labels, label=label)
                for row in rows:
                    row["label"] = label
                result_sets.append(rows)

    ranked = retriever._apply_base_rrf_scores(retriever.rrf_fusion(result_sets))
    if use_cross_encoder_rerank:
        if cross_encoder_scorer is None:
            cross_encoder_scorer = retriever.build_local_cross_encoder()
        ranked = cross_encoder_rerank_sentence_candidates(
            query_spec=query_spec,
            candidates=ranked,
            cross_encoder_scorer=cross_encoder_scorer,
            top_n=cross_encoder_top_n,
        )
        ranked = ranked[:cross_encoder_top_n]

    if use_section_tag_bonus:
        ranked = apply_section_tag_bonus(
            retriever=retriever,
            query_spec=query_spec,
            candidates=ranked,
            bonus_mode=section_bonus_mode,
            bonus_weight=section_bonus_weight,
        )

    evidence = package_top_k_candidates(retriever, ranked[:top_k])
    evidence_relationships = fetch_retrieved_chunk_relationships(retriever, evidence)
    evidence = attach_retrieved_rationales_to_evidence(evidence, evidence_relationships)

    return {
        "query_spec": query_spec,
        "dense_queries": dense_queries,
        "sparse_queries": sparse_queries,
        "rerank_modes": {
            "cross_encoder": use_cross_encoder_rerank,
            "cross_encoder_top_n": cross_encoder_top_n if use_cross_encoder_rerank else 0,
            "section_tag_bonus": use_section_tag_bonus,
            "section_bonus_mode": section_bonus_mode if use_section_tag_bonus else "",
            "section_bonus_weight": section_bonus_weight if use_section_tag_bonus else 0.0,
            "QD": is_QD,
        },
        "evidence": evidence,
        "evidence_relationships": evidence_relationships,
        "connected_evidence_groups": build_connected_evidence_groups(
            evidence,
            evidence_relationships,
        ),
    }


def query_doc_chunks_for_sentence_by_dense_score(
    retriever: FMEASentenceRetrieverV2,
    query_spec: Dict[str, Any],
    top_k: int = 8,
    per_label_k: int = 20,
    disciplines: Any = None,
    use_cross_encoder_rerank: bool = False,
    cross_encoder_scorer: Optional[CrossEncoderScorer] = None,
    cross_encoder_top_n: int = 20,
    use_section_tag_bonus: bool = True,
    section_bonus_mode: str = "hybrid",
    section_bonus_weight: float = 0.05,
    is_QD: bool = False,
) -> Dict[str, Any]:
    """Dense retrieval that compares FS/TS/Rationale chunks by raw retrieval score.

    The older dense path retrieves per label and then uses RRF, which makes the
    top item from each label receive the same base score. This interface still
    queries the label-specific indexes, but merges all returned chunks by their
    dense similarity score before taking the final top_k.
    """

    section_bonus_mode = retriever._normalize_retrieval_mode(section_bonus_mode)
    allowed_labels = normalize_discipline_labels(disciplines)
    dense_queries = build_dense_queries(query_spec)

    candidates_by_node_id: Dict[str, Dict[str, Any]] = {}
    for spec in build_search_specs(retriever, is_QD=is_QD):
        label = spec["label"]
        vector_index_name = spec["vector_index_name"]
        text_property = spec["text_property"]
        title_property = spec.get("title_property", "")
        id_property = spec.get("id_property", "")

        for query_text in dense_queries:
            if label in {"QDChunk", "FATChunk"}:
                rows = dense_search_qualification_chunks(
                    retriever=retriever,
                    query_text=query_text,
                    label=label,
                    vector_index_name=vector_index_name,
                    title_property=title_property,
                    text_property=text_property,
                    id_property=id_property,
                    top_k=per_label_k,
                )
            else:
                rows = retriever.dense_search_chunks(
                    query_text=query_text,
                    label=label,
                    vector_index_name=vector_index_name,
                    top_k=per_label_k,
                    text_property=text_property,
                )
            rows = filter_rows_by_labels(rows, allowed_labels, label=label)
            for rank, row in enumerate(rows, start=1):
                row["label"] = label
                row["retrieval_source"] = "dense_score"
                row["retrieval_query"] = query_text
                row["retrieval_rank_within_label"] = rank
                keep_best_dense_candidate(candidates_by_node_id, row)

    ranked = sort_dense_score_candidates(list(candidates_by_node_id.values()))
    if use_cross_encoder_rerank:
        if cross_encoder_scorer is None:
            cross_encoder_scorer = retriever.build_local_cross_encoder()
        ranked = cross_encoder_rerank_sentence_candidates(
            query_spec=query_spec,
            candidates=ranked,
            cross_encoder_scorer=cross_encoder_scorer,
            top_n=cross_encoder_top_n,
        )
        ranked = ranked[:cross_encoder_top_n]

    if use_section_tag_bonus:
        ranked = apply_section_tag_bonus(
            retriever=retriever,
            query_spec=query_spec,
            candidates=ranked,
            bonus_mode=section_bonus_mode,
            bonus_weight=section_bonus_weight,
        )

    evidence = package_top_k_candidates(retriever, ranked[:top_k])
    evidence_relationships = fetch_retrieved_chunk_relationships(retriever, evidence)
    evidence = attach_retrieved_rationales_to_evidence(evidence, evidence_relationships)

    return {
        "query_spec": query_spec,
        "dense_queries": dense_queries,
        "sparse_queries": [],
        "retrieval_scoring": "dense_score",
        "rerank_modes": {
            "cross_encoder": use_cross_encoder_rerank,
            "cross_encoder_top_n": cross_encoder_top_n if use_cross_encoder_rerank else 0,
            "section_tag_bonus": use_section_tag_bonus,
            "section_bonus_mode": section_bonus_mode if use_section_tag_bonus else "",
            "section_bonus_weight": section_bonus_weight if use_section_tag_bonus else 0.0,
            "QD": is_QD,
        },
        "evidence": evidence,
        "evidence_relationships": evidence_relationships,
        "connected_evidence_groups": build_connected_evidence_groups(
            evidence,
            evidence_relationships,
        ),
    }


def query_doc_chunks_for_sentence_by_sparse_score(
    retriever: FMEASentenceRetrieverV2,
    query_spec: Dict[str, Any],
    top_k: int = 8,
    per_label_k: int = 20,
    disciplines: Any = None,
    use_cross_encoder_rerank: bool = False,
    cross_encoder_scorer: Optional[CrossEncoderScorer] = None,
    cross_encoder_top_n: int = 20,
    use_section_tag_bonus: bool = True,
    section_bonus_mode: str = "hybrid",
    section_bonus_weight: float = 0.05,
    is_QD: bool = False,
) -> Dict[str, Any]:
    """Sparse retrieval that compares FS/TS/Rationale chunks by raw sparse score."""

    section_bonus_mode = retriever._normalize_retrieval_mode(section_bonus_mode)
    allowed_labels = normalize_discipline_labels(disciplines)
    sparse_queries = build_sparse_queries(retriever, query_spec)

    candidates_by_node_id: Dict[str, Dict[str, Any]] = {}
    for spec in build_search_specs(retriever, is_QD=is_QD):
        label = spec["label"]
        fulltext_index_name = spec["fulltext_index_name"]
        text_property = spec["text_property"]
        title_property = spec.get("title_property", "")
        id_property = spec.get("id_property", "")

        for lucene_query in sparse_queries:
            if label in {"QDChunk", "FATChunk"}:
                rows = sparse_search_qualification_chunks(
                    retriever=retriever,
                    lucene_query=lucene_query,
                    label=label,
                    fulltext_index_name=fulltext_index_name,
                    title_property=title_property,
                    text_property=text_property,
                    id_property=id_property,
                    top_k=per_label_k,
                )
            else:
                rows = retriever.sparse_search_chunks(
                    lucene_query=lucene_query,
                    label=label,
                    fulltext_index_name=fulltext_index_name,
                    top_k=per_label_k,
                    text_property=text_property,
                )
            rows = filter_rows_by_labels(rows, allowed_labels, label=label)
            for rank, row in enumerate(rows, start=1):
                row["label"] = label
                row["retrieval_source"] = "sparse_score"
                row["retrieval_query"] = lucene_query
                row["retrieval_rank_within_label"] = rank
                keep_best_score_candidate(candidates_by_node_id, row, score_name="sparse_score")

    ranked = sort_score_candidates(list(candidates_by_node_id.values()))
    if use_cross_encoder_rerank:
        if cross_encoder_scorer is None:
            cross_encoder_scorer = retriever.build_local_cross_encoder()
        ranked = cross_encoder_rerank_sentence_candidates(
            query_spec=query_spec,
            candidates=ranked,
            cross_encoder_scorer=cross_encoder_scorer,
            top_n=cross_encoder_top_n,
        )
        ranked = ranked[:cross_encoder_top_n]

    if use_section_tag_bonus:
        ranked = apply_section_tag_bonus(
            retriever=retriever,
            query_spec=query_spec,
            candidates=ranked,
            bonus_mode=section_bonus_mode,
            bonus_weight=section_bonus_weight,
        )

    evidence = package_top_k_candidates(retriever, ranked[:top_k])
    evidence_relationships = fetch_retrieved_chunk_relationships(retriever, evidence)
    evidence = attach_retrieved_rationales_to_evidence(evidence, evidence_relationships)

    return {
        "query_spec": query_spec,
        "dense_queries": [],
        "sparse_queries": sparse_queries,
        "retrieval_scoring": "sparse_score",
        "rerank_modes": {
            "cross_encoder": use_cross_encoder_rerank,
            "cross_encoder_top_n": cross_encoder_top_n if use_cross_encoder_rerank else 0,
            "section_tag_bonus": use_section_tag_bonus,
            "section_bonus_mode": section_bonus_mode if use_section_tag_bonus else "",
            "section_bonus_weight": section_bonus_weight if use_section_tag_bonus else 0.0,
            "QD": is_QD,
        },
        "evidence": evidence,
        "evidence_relationships": evidence_relationships,
        "connected_evidence_groups": build_connected_evidence_groups(
            evidence,
            evidence_relationships,
        ),
    }


def keep_best_dense_candidate(
    candidates_by_node_id: Dict[str, Dict[str, Any]],
    row: Dict[str, Any],
) -> None:
    keep_best_score_candidate(candidates_by_node_id, row, score_name="dense_score")


def keep_best_score_candidate(
    candidates_by_node_id: Dict[str, Dict[str, Any]],
    row: Dict[str, Any],
    score_name: str,
) -> None:
    node_id = str(row.get("node_id", ""))
    if not node_id:
        return

    score = float(row.get("score", 0.0))
    existing = candidates_by_node_id.get(node_id)
    if existing is not None and float(existing.get("score", 0.0)) >= score:
        return

    candidate = dict(row)
    candidate["score_breakdown"] = {
        score_name: score,
        "retrieval_source": score_name,
        "retrieval_query": row.get("retrieval_query", ""),
        "retrieval_rank_within_label": row.get("retrieval_rank_within_label"),
    }
    candidate["base_score"] = score
    candidate["final_score"] = score
    candidates_by_node_id[node_id] = candidate


def sort_dense_score_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sort_score_candidates(candidates)


def sort_score_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("final_score", item.get("score", 0.0))),
            -int(item.get("retrieval_rank_within_label") or 0),
        ),
        reverse=True,
    )


def build_search_specs(
    retriever: FMEASentenceRetrieverV2,
    is_QD: bool = False,
) -> List[Dict[str, str]]:
    if is_QD:
        return [
            {
                "label": "QDChunk",
                "vector_index_name": "qd_embedding_idx",
                "fulltext_index_name": "qd_objectives_idx",
                "text_property": "objectives",
                "title_property": "qd_title",
                "id_property": "qd_id",
            },
            {
                "label": "FATChunk",
                "vector_index_name": "fat_embedding_idx",
                "fulltext_index_name": "fat_objectives_idx",
                "text_property": "objectives",
                "title_property": "fat_title",
                "id_property": "fat_id",
            }
        ]

    return [
        {
            "label": label,
            "vector_index_name": vector_index_name,
            "fulltext_index_name": fulltext_index_name,
            "text_property": "text",
        }
        for label, vector_index_name, fulltext_index_name in retriever.SEARCH_SPECS
    ]


def dense_search_qualification_chunks(
    retriever: FMEASentenceRetrieverV2,
    query_text: str,
    label: str,
    vector_index_name: str,
    title_property: str,
    text_property: str,
    id_property: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    query_embedding = get_query_embedding(query_text)
    title_property_expr = ChunkRetriever._safe_property_name(title_property)
    text_property_expr = ChunkRetriever._safe_property_name(text_property)
    id_property_expr = ChunkRetriever._safe_property_name(id_property)

    cypher = f"""
    CALL db.index.vector.queryNodes('{vector_index_name}', $top_k, $query_embedding)
    YIELD node, score
    WHERE node:{label}
    RETURN
        elementId(node) AS node_id,
        labels(node) AS labels,
        coalesce(node.name, "") AS name,
        coalesce(node.section_tag, "") AS section_tag,
        coalesce(node.{id_property_expr}, "") AS qualification_id,
        coalesce(node.qd_id, "") AS qd_id,
        coalesce(node.qd_title, "") AS qd_title,
        coalesce(node.fat_id, "") AS fat_id,
        coalesce(node.fat_title, "") AS fat_title,
        coalesce(node.fat_title, "") AS fat_titile,
        coalesce(node.{text_property_expr}, "") AS objectives,
        trim(coalesce(node.{title_property_expr}, "") + " " + coalesce(node.{text_property_expr}, "")) AS text,
        score AS score,
        "dense" AS source
    ORDER BY score DESC
    LIMIT $top_k
    """
    return retriever.run_query(
        cypher,
        query_embedding=query_embedding,
        top_k=top_k,
    )


def sparse_search_qualification_chunks(
    retriever: FMEASentenceRetrieverV2,
    lucene_query: str,
    label: str,
    fulltext_index_name: str,
    title_property: str,
    text_property: str,
    id_property: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    title_property_expr = ChunkRetriever._safe_property_name(title_property)
    text_property_expr = ChunkRetriever._safe_property_name(text_property)
    id_property_expr = ChunkRetriever._safe_property_name(id_property)

    cypher = f"""
    CALL db.index.fulltext.queryNodes('{fulltext_index_name}', $lucene_query)
    YIELD node, score
    WHERE node:{label}
    RETURN
        elementId(node) AS node_id,
        labels(node) AS labels,
        coalesce(node.name, "") AS name,
        coalesce(node.section_tag, "") AS section_tag,
        coalesce(node.{id_property_expr}, "") AS qualification_id,
        coalesce(node.qd_id, "") AS qd_id,
        coalesce(node.qd_title, "") AS qd_title,
        coalesce(node.fat_id, "") AS fat_id,
        coalesce(node.fat_title, "") AS fat_title,
        coalesce(node.fat_title, "") AS fat_titile,
        coalesce(node.{text_property_expr}, "") AS objectives,
        trim(coalesce(node.{title_property_expr}, "") + " " + coalesce(node.{text_property_expr}, "")) AS text,
        score AS score,
        "sparse" AS source
    ORDER BY score DESC
    LIMIT $top_k
    """
    return retriever.run_query(
        cypher,
        lucene_query=lucene_query,
        top_k=top_k,
    )


def apply_section_tag_bonus(
    retriever: FMEASentenceRetrieverV2,
    query_spec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    bonus_mode: str = "hybrid",
    bonus_weight: float = 0.05,
) -> List[Dict[str, Any]]:
    query_text = query_spec.get("sentence", "")

    reranked = []
    for item in candidates:
        section_tag = clean_section_tag(item.get("section_tag", ""))
        section_score = score_section_tag_match(
            query_text=query_text,
            section_tag=section_tag,
            mode=bonus_mode,
        )
        section_bonus = bonus_weight * section_score

        new_item = dict(item)
        new_item["section_tag"] = section_tag
        new_item["section_tag_score"] = section_score
        new_item["section_tag_bonus"] = section_bonus
        new_item["score_breakdown"] = {
            **dict(item.get("score_breakdown", {})),
            "section_tag_score": section_score,
            "section_tag_bonus": section_bonus,
        }
        new_item["final_score"] = float(item.get("final_score", item.get("rrf_score", 0.0))) + section_bonus
        reranked.append(new_item)

    reranked.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
    return reranked


def score_section_tag_match(
    query_text: str,
    section_tag: str,
    mode: str,
) -> float:
    if not query_text or not section_tag:
        return 0.0

    if mode == "dense":
        return dense_text_similarity(query_text, section_tag)
    if mode == "sparse":
        return sparse_text_similarity(query_text, section_tag)

    dense_score = dense_text_similarity(query_text, section_tag)
    sparse_score = sparse_text_similarity(query_text, section_tag)
    return 0.5 * dense_score + 0.5 * sparse_score


def clean_section_tag(section_tag: str) -> str:
    section_tag = " ".join((section_tag or "").split())
    return re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", section_tag).strip()


def dense_text_similarity(left_text: str, right_text: str) -> float:
    left_embedding = get_query_embedding(left_text)
    right_embedding = get_query_embedding(right_text)
    return cosine_similarity(left_embedding, right_embedding)


def sparse_text_similarity(left_text: str, right_text: str) -> float:
    left_norm = ChunkRetriever._normalize_text(left_text)
    right_norm = ChunkRetriever._normalize_text(right_text)
    left_tokens = set(ChunkRetriever._meaningful_tokens(left_text, min_len=3))
    right_tokens = set(ChunkRetriever._meaningful_tokens(right_text, min_len=3))

    if not left_tokens or not right_tokens:
        return 0.0

    overlap = left_tokens.intersection(right_tokens)
    score = len(overlap) / math.sqrt(len(left_tokens) * len(right_tokens))

    if right_norm and right_norm in left_norm:
        score += 1.0
    if left_norm and left_norm in right_norm:
        score += 1.0

    return score


def cosine_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0

    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values:
        return 0.0

    pair_values = list(zip(left_values, right_values))
    numerator = sum(a * b for a, b in pair_values)
    left_norm = math.sqrt(sum(a * a for a, _ in pair_values))
    right_norm = math.sqrt(sum(b * b for _, b in pair_values))
    if not left_norm or not right_norm:
        return 0.0

    return numerator / (left_norm * right_norm)


def cross_encoder_rerank_sentence_candidates(
    query_spec: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    cross_encoder_scorer: CrossEncoderScorer,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    query_text = build_cross_encoder_query(query_spec)
    kept = candidates[:top_n]
    tail = candidates[top_n:]

    for item in kept:
        ce_score = float(
            cross_encoder_scorer(
                query_text,
                build_cross_encoder_document(item),
            )
        )
        item["cross_encoder_score"] = ce_score
        item["score_breakdown"] = {
            **dict(item.get("score_breakdown", {})),
            "cross_encoder_score": ce_score,
        }
        item["final_score"] = ce_score

    kept.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
    return kept + tail


def build_cross_encoder_query(query_spec: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "Task: score whether the candidate document chunk is relevant evidence for the requirement sentence.",
            f"Requirement sentence: {query_spec.get('sentence', '')}",
        ]
    )


def build_cross_encoder_document(candidate: Dict[str, Any]) -> str:
    label = candidate.get("label") or (candidate.get("labels") or [""])[0]
    return "\n".join(
        [
            f"Label: {label}",
            f"Name: {candidate.get('name', '')}",
            f"Chunk text: {candidate.get('text', '')}",
        ]
    )


def package_top_k_candidates(
    retriever: FMEASentenceRetrieverV2,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    evidence = []
    for retrieval_rank, item in enumerate(candidates, start=1):
        label = retriever._resolve_candidate_label(item)
        node_id = item.get("node_id", "")
        evidence.append(
            {
                "retrieval_rank": retrieval_rank,
                "node_id": node_id,
                "label": label,
                "name": item.get("name", ""),
                "section_tag": item.get("section_tag", ""),
                "qd_id": item.get("qd_id", ""),
                "qd_title": item.get("qd_title", ""),
                "fat_id": item.get("fat_id", ""),
                "fat_title": item.get("fat_title", ""),
                "fat_titile": item.get("fat_titile", item.get("fat_title", "")),
                "objectives": item.get("objectives", ""),
                "text": item.get("text", ""),
                "score": item.get("final_score", item.get("rrf_score", 0.0)),
                "cross_encoder_score": item.get("cross_encoder_score"),
                "section_tag_score": item.get("section_tag_score"),
                "section_tag_bonus": item.get("section_tag_bonus"),
                "score_breakdown": item.get("score_breakdown", {}),
            }
        )
    return evidence


def fetch_retrieved_chunk_relationships(
    retriever: FMEASentenceRetrieverV2,
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    node_ids = dedupe_preserve_order(
        [str(item.get("node_id", "")) for item in evidence if item.get("node_id")]
    )
    if len(node_ids) < 2:
        return []

    cypher = """
    MATCH (source)-[rel]->(target)
    WHERE elementId(source) IN $node_ids
      AND elementId(target) IN $node_ids
      AND elementId(source) <> elementId(target)
      AND (
        (type(rel) = "IMPLEMENT" AND (
            ((source:TSChunk OR source:ESWTSChunk OR source:HWTSChunk) AND target:FSChunk) OR
            (source:FSChunk AND (target:TSChunk OR target:ESWTSChunk OR target:HWTSChunk))
        )) OR
        (type(rel) = "RELATED" AND source:FSChunk AND target:FSChunk) OR
        (type(rel) = "RATIONALE_FOR" AND (
            ((source:RationaleChunk OR source:ESWRationaleChunk OR source:HWRationaleChunk OR source:TSRationaleChunk OR source:FSRationaleChunk)
                AND (target:TSChunk OR target:ESWTSChunk OR target:HWTSChunk OR target:FSChunk)) OR
            ((source:TSChunk OR source:ESWTSChunk OR source:HWTSChunk OR source:FSChunk)
                AND (target:RationaleChunk OR target:ESWRationaleChunk OR target:HWRationaleChunk OR target:TSRationaleChunk OR target:FSRationaleChunk))
        ))
      )
    RETURN DISTINCT
        elementId(source) AS source_id,
        elementId(target) AS target_id,
        type(rel) AS relationship
    """
    return retriever.run_query(cypher, node_ids=node_ids)


def build_connected_evidence_groups(
    evidence: List[Dict[str, Any]],
    relationships: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    evidence_by_id = {
        str(item.get("node_id", "")): item
        for item in evidence
        if item.get("node_id")
    }
    if not evidence_by_id:
        return []

    adjacency = {node_id: set() for node_id in evidence_by_id}
    relationship_by_pair: Dict[tuple[str, str], List[Dict[str, str]]] = {}
    for rel in relationships:
        source_id = str(rel.get("source_id", ""))
        target_id = str(rel.get("target_id", ""))
        relationship = str(rel.get("relationship", ""))
        if relationship == "RATIONALE_FOR":
            continue
        if source_id not in evidence_by_id or target_id not in evidence_by_id or not relationship:
            continue

        adjacency[source_id].add(target_id)
        adjacency[target_id].add(source_id)
        pair = tuple(sorted((source_id, target_id)))
        relationship_by_pair.setdefault(pair, [])
        edge = {
            "source_id": source_id,
            "target_id": target_id,
            "relationship": relationship,
        }
        if edge not in relationship_by_pair[pair]:
            relationship_by_pair[pair].append(edge)

    visited = set()
    groups: List[Dict[str, Any]] = []
    for node_id in evidence_by_id:
        if node_id in visited:
            continue

        stack = [node_id]
        component = []
        visited.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        component.sort(
            key=lambda item_id: int(evidence_by_id[item_id].get("retrieval_rank") or 0)
        )
        component_relationships = []
        for index, source_id in enumerate(component):
            for target_id in component[index + 1:]:
                pair = tuple(sorted((source_id, target_id)))
                component_relationships.extend(relationship_by_pair.get(pair, []))

        groups.append(
            {
                "group_id": len(groups) + 1,
                "node_ids": component,
                "relationships": component_relationships,
                "has_relationships": bool(component_relationships),
                "best_retrieval_rank": min(
                    int(evidence_by_id[item_id].get("retrieval_rank") or 0)
                    for item_id in component
                ),
            }
        )

    groups.sort(key=lambda group: int(group.get("best_retrieval_rank") or 0))
    for index, group in enumerate(groups, start=1):
        group["group_id"] = index
    return groups


def attach_retrieved_rationales_to_evidence(
    evidence: List[Dict[str, Any]],
    relationships: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    evidence_by_id = {
        str(item.get("node_id", "")): item
        for item in evidence
        if item.get("node_id")
    }
    if not evidence_by_id:
        return evidence

    rationale_by_target_id: Dict[str, List[Dict[str, str]]] = {}
    for rel in relationships:
        if str(rel.get("relationship", "")) != "RATIONALE_FOR":
            continue

        source_id = str(rel.get("source_id", ""))
        target_id = str(rel.get("target_id", ""))
        if source_id not in evidence_by_id or target_id not in evidence_by_id:
            continue

        source = evidence_by_id[source_id]
        target = evidence_by_id[target_id]
        if is_rationale_evidence(source) and not is_rationale_evidence(target):
            rationale = source
            rationale_target_id = target_id
        elif is_rationale_evidence(target) and not is_rationale_evidence(source):
            rationale = target
            rationale_target_id = source_id
        else:
            continue

        rationale_text = str(rationale.get("text", "")).strip()
        if not rationale_text:
            continue

        rationale_by_target_id.setdefault(rationale_target_id, [])
        rationale_item = {
            "name": str(rationale.get("name", "")).strip(),
            "label": str(rationale.get("label", "")).strip(),
            "text": rationale_text,
        }
        if rationale_item not in rationale_by_target_id[rationale_target_id]:
            rationale_by_target_id[rationale_target_id].append(rationale_item)

    if not rationale_by_target_id:
        return evidence

    enriched = []
    for item in evidence:
        node_id = str(item.get("node_id", ""))
        rationales = rationale_by_target_id.get(node_id, [])
        if not rationales:
            enriched.append(item)
            continue

        enriched_item = dict(item)
        enriched_item["rationale_chunks"] = rationales
        enriched_item["rationale_texts"] = [rationale["text"] for rationale in rationales]
        enriched.append(enriched_item)

    return enriched


def is_rationale_evidence(item: Dict[str, Any]) -> bool:
    label = str(item.get("label", ""))
    labels = [str(value) for value in item.get("labels", [])]
    return "RationaleChunk" in label or any("RationaleChunk" in value for value in labels)


def normalize_discipline_labels(disciplines: Any) -> List[str]:
    if disciplines is None:
        return []

    if isinstance(disciplines, str):
        values = [disciplines]
    else:
        values = list(disciplines)

    label_map = {
        "ESW": ["ESWTSChunk", "ESWRationaleChunk"],
        "HW": ["HWTSChunk", "HWRationaleChunk"],
        "FS": ["FSChunk", "FSRationaleChunk"],
    }

    labels: List[str] = []
    for value in values:
        key = str(value or "").strip().upper()
        labels.extend(label_map.get(key, []))

    return dedupe_preserve_order(labels)


def filter_rows_by_labels(
    rows: List[Dict[str, Any]],
    allowed_labels: List[str],
    label: str = "",
) -> List[Dict[str, Any]]:
    if not allowed_labels or label in {"QDChunk", "FATChunk"}:
        return rows

    allowed = set(allowed_labels)
    return [
        row
        for row in rows
        if allowed.intersection(set(row.get("labels") or []))
    ]


def print_doc_chunk_results(result: Dict[str, Any]) -> None:
    query_spec = result.get("query_spec", {})
    print("=" * 100)
    print("Query Type: requirement_sentence_doc_chunk")
    print(f"Rerank Modes: {result.get('rerank_modes', {})}")
    print("=" * 100)
    print("Sentence:")
    print(f"  {query_spec.get('sentence', '')}")

    print("\nTop-K Candidates:")
    evidence = result.get("evidence", [])
    if not evidence:
        print("  (none)")
        return

    for i, item in enumerate(evidence, start=1):
        print(
            f"  [{i}] label={item.get('label', '')} "
            f"score={item.get('score', 0.0):.4f} "
            f"node_id={item.get('node_id', '')} "
            f"name={item.get('name', '')}"
        )
        if item.get("qd_id") or item.get("qd_title"):
            print(
                f"      qd_id={item.get('qd_id', '')} "
                f"qd_title={item.get('qd_title', '')}"
            )
        if item.get("fat_id") or item.get("fat_title") or item.get("fat_titile"):
            print(
                f"      fat_id={item.get('fat_id', '')} "
                f"fat_title={item.get('fat_title', item.get('fat_titile', ''))}"
            )
        if item.get("cross_encoder_score") is not None:
            print(f"      cross_encoder_score={item.get('cross_encoder_score', 0.0):.4f}")
        if item.get("section_tag_score") is not None:
            print(
                f"      section_tag={item.get('section_tag', '')} "
                f"section_score={item.get('section_tag_score', 0.0):.4f} "
                f"section_bonus={item.get('section_tag_bonus', 0.0):.4f}"
            )
        print(f"      text={item.get('text', '')}")


def dedupe_preserve_order(values: List[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        value = (value or "").strip()
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def main():
    requirement_sentence = """
    (Starting) Motor current too high for chosen components
    """

    retriever = FMEASentenceRetrieverV2()
    try:
        query_spec = build_sentence_doc_chunk_query(sentence=requirement_sentence)

        result = query_doc_chunks_for_sentence(
            retriever=retriever,
            query_spec=query_spec,
            top_k=20,
            per_label_k=80,
            retrieval_mode="dense",
            disciplines=None,
            use_cross_encoder_rerank=False,
            cross_encoder_top_n=50,
            use_section_tag_bonus=False,
            section_bonus_mode="hybrid",
            section_bonus_weight=0.00,
            is_QD=True,
        )
        print_doc_chunk_results(result)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
