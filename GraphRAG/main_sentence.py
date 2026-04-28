from typing import Any, Dict, List

from .fmea_retrieverV2 import FMEASentenceRetrieverV2


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
) -> Dict[str, Any]:
    retrieval_mode = retriever._normalize_retrieval_mode(retrieval_mode)
    allowed_labels = normalize_discipline_labels(disciplines)
    dense_queries = build_dense_queries(query_spec)
    sparse_queries = build_sparse_queries(retriever, query_spec)

    result_sets: List[List[Dict[str, Any]]] = []
    for label, vector_index_name, fulltext_index_name in retriever.SEARCH_SPECS:
        if retrieval_mode in {"dense", "hybrid"}:
            for query_text in dense_queries:
                rows = retriever.dense_search_chunks(
                    query_text=query_text,
                    label=label,
                    vector_index_name=vector_index_name,
                    top_k=per_label_k,
                )
                rows = filter_rows_by_labels(rows, allowed_labels)
                for row in rows:
                    row["label"] = label
                result_sets.append(rows)

        if retrieval_mode in {"sparse", "hybrid"}:
            for lucene_query in sparse_queries:
                rows = retriever.sparse_search_chunks(
                    lucene_query=lucene_query,
                    label=label,
                    fulltext_index_name=fulltext_index_name,
                    top_k=per_label_k,
                )
                rows = filter_rows_by_labels(rows, allowed_labels)
                for row in rows:
                    row["label"] = label
                result_sets.append(rows)

    ranked = retriever._apply_base_rrf_scores(retriever.rrf_fusion(result_sets))
    return {
        "query_spec": query_spec,
        "dense_queries": dense_queries,
        "sparse_queries": sparse_queries,
        "evidence": package_top_k_candidates(retriever, ranked[:top_k]),
    }


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
                "text": item.get("text", ""),
                "score": item.get("final_score", item.get("rrf_score", 0.0)),
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
) -> List[Dict[str, Any]]:
    if not allowed_labels:
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
            top_k=10,
            per_label_k=30,
            retrieval_mode="hybrid",
            disciplines="HW",
        )
        print_doc_chunk_results(result)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
