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

    result_sets: List[List[Dict[str, Any]]] = []
    for spec in build_search_specs(retriever, is_QD=is_QD):
        label = spec["label"]
        vector_index_name = spec["vector_index_name"]
        fulltext_index_name = spec["fulltext_index_name"]
        text_property = spec["text_property"]

        if retrieval_mode in {"dense", "hybrid"}:
            for query_text in dense_queries:
                if label == "QDChunk":
                    rows = dense_search_qd_chunks(
                        retriever=retriever,
                        query_text=query_text,
                        vector_index_name=vector_index_name,
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
                if label == "QDChunk":
                    rows = sparse_search_qd_chunks(
                        retriever=retriever,
                        lucene_query=lucene_query,
                        fulltext_index_name=fulltext_index_name,
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
        "evidence": package_top_k_candidates(retriever, ranked[:top_k]),
    }


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


def dense_search_qd_chunks(
    retriever: FMEASentenceRetrieverV2,
    query_text: str,
    vector_index_name: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    query_embedding = get_query_embedding(query_text)

    cypher = f"""
    CALL db.index.vector.queryNodes('{vector_index_name}', $top_k, $query_embedding)
    YIELD node, score
    WHERE node:QDChunk
    RETURN
        elementId(node) AS node_id,
        labels(node) AS labels,
        coalesce(node.name, "") AS name,
        coalesce(node.section_tag, "") AS section_tag,
        coalesce(node.qd_id, "") AS qd_id,
        coalesce(node.qd_title, "") AS qd_title,
        coalesce(node.objectives, "") AS objectives,
        trim(coalesce(node.qd_title, "") + " " + coalesce(node.objectives, "")) AS text,
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


def sparse_search_qd_chunks(
    retriever: FMEASentenceRetrieverV2,
    lucene_query: str,
    fulltext_index_name: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    cypher = f"""
    CALL db.index.fulltext.queryNodes('{fulltext_index_name}', $lucene_query)
    YIELD node, score
    WHERE node:QDChunk
    RETURN
        elementId(node) AS node_id,
        labels(node) AS labels,
        coalesce(node.name, "") AS name,
        coalesce(node.section_tag, "") AS section_tag,
        coalesce(node.qd_id, "") AS qd_id,
        coalesce(node.qd_title, "") AS qd_title,
        coalesce(node.objectives, "") AS objectives,
        trim(coalesce(node.qd_title, "") + " " + coalesce(node.objectives, "")) AS text,
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
    for item in candidates:
        label = retriever._resolve_candidate_label(item)
        node_id = item.get("node_id", "")
        evidence.append(
            {
                "node_id": node_id,
                "label": label,
                "name": item.get("name", ""),
                "section_tag": item.get("section_tag", ""),
                "qd_id": item.get("qd_id", ""),
                "qd_title": item.get("qd_title", ""),
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
    if not allowed_labels or label == "QDChunk":
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
     The ESP32-S3 has two cores with 32 interrupt each. Each interrupt has a  xed priority. In\ncase an interrupt is required for a peripheral any of these interrupts can be used, i.e. in\nrelation to a ARM Cortex the ESP32 has no dedicated interrupts assigned for each peripheral\nwith a con gurable priority.\nAn interrupt enabled by code running a speci c core shall be associated with this core and\nrun the interrupt on that core.\nSee [7] for more details.
    """

    retriever = FMEASentenceRetrieverV2()
    try:
        query_spec = build_sentence_doc_chunk_query(sentence=requirement_sentence)

        result = query_doc_chunks_for_sentence(
            retriever=retriever,
            query_spec=query_spec,
            top_k=15,
            per_label_k=30,
            retrieval_mode="hybrid",
            disciplines=None,
            use_cross_encoder_rerank=False,
            cross_encoder_top_n=50,
            use_section_tag_bonus=True,
            section_bonus_mode="hybrid",
            section_bonus_weight=0.05,
            is_QD=False,
        )
        print_doc_chunk_results(result)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
