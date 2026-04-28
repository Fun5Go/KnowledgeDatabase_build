import math
from typing import Any, Dict, List

from .fmea_retrieverV2 import FMEASentenceRetrieverV2
from .neo4j_retriever import ChunkRetriever, get_query_embedding


SUPPORTED_TEXT_TYPES = {"effect", "mode", "cause", "function"}


def fetch_sentence_nodes_by_name(
    retriever: FMEASentenceRetrieverV2,
    names: Any,
) -> List[Dict[str, Any]]:
    names = normalize_text_list(names)
    if not names:
        return []

    cypher = """
    MATCH (node)
    WHERE node.name IN $names
    RETURN
        elementId(node) AS node_id,
        labels(node) AS labels,
        coalesce(node.name, "") AS name,
        coalesce(node.text, "") AS text
    ORDER BY name
    """
    rows = retriever.run_query(cypher, names=names)
    return [
        {
            "source": "graph_name",
            "node_id": row.get("node_id", ""),
            "labels": row.get("labels", []),
            "name": row.get("name", ""),
            "text": " ".join((row.get("text", "") or "").split()),
        }
        for row in rows
        if (row.get("text", "") or "").strip()
    ]


def build_sentence_inputs(
    retriever: FMEASentenceRetrieverV2,
    node_names: Any = None,
    sentence_texts: Any = None,
) -> List[Dict[str, Any]]:
    inputs = fetch_sentence_nodes_by_name(retriever, node_names)

    for index, sentence in enumerate(normalize_text_list(sentence_texts), start=1):
        inputs.append(
            {
                "source": "code_input",
                "node_id": "",
                "labels": [],
                "name": f"code_sentence_{index}",
                "text": sentence,
            }
        )

    return inputs


def build_fmea_candidates(
    fmea_texts: Dict[str, Any],
    selected_text_types: Any = None,
) -> List[Dict[str, str]]:
    selected = normalize_selected_text_types(selected_text_types)
    candidates: List[Dict[str, str]] = []

    for text_type in selected:
        for index, text in enumerate(normalize_text_list(fmea_texts.get(text_type, [])), start=1):
            candidates.append(
                {
                    "candidate_id": f"{text_type}_{index}",
                    "text_type": text_type,
                    "text": text,
                }
            )

    return candidates


def query_fmea_texts_for_sentence(
    retriever: FMEASentenceRetrieverV2,
    sentence_input: Dict[str, Any],
    fmea_candidates: List[Dict[str, str]],
    top_k: int = 8,
    retrieval_mode: str = "hybrid",
) -> Dict[str, Any]:
    retrieval_mode = retriever._normalize_retrieval_mode(retrieval_mode)
    query_text = sentence_input.get("text", "")

    result_sets: List[List[Dict[str, Any]]] = []
    if retrieval_mode in {"dense", "hybrid"}:
        result_sets.append(dense_rank_fmea_candidates(query_text, fmea_candidates))
    if retrieval_mode in {"sparse", "hybrid"}:
        result_sets.append(sparse_rank_fmea_candidates(query_text, fmea_candidates))

    ranked = rrf_fusion_candidates(result_sets)
    return {
        "sentence_input": sentence_input,
        "retrieval_mode": retrieval_mode,
        "selected_candidate_count": len(fmea_candidates),
        "candidates": ranked[:top_k],
    }


def query_fmea_texts_for_sentences(
    retriever: FMEASentenceRetrieverV2,
    sentence_inputs: List[Dict[str, Any]],
    fmea_texts: Dict[str, Any],
    selected_text_types: Any = None,
    top_k: int = 8,
    retrieval_mode: str = "hybrid",
) -> Dict[str, Any]:
    fmea_candidates = build_fmea_candidates(
        fmea_texts=fmea_texts,
        selected_text_types=selected_text_types,
    )

    return {
        "query_type": "sentence_to_fmea_text",
        "selected_text_types": normalize_selected_text_types(selected_text_types),
        "retrieval_mode": retriever._normalize_retrieval_mode(retrieval_mode),
        "results": [
            query_fmea_texts_for_sentence(
                retriever=retriever,
                sentence_input=sentence_input,
                fmea_candidates=fmea_candidates,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
            )
            for sentence_input in sentence_inputs
        ],
    }


def dense_rank_fmea_candidates(
    query_text: str,
    fmea_candidates: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    query_embedding = get_query_embedding(query_text)
    ranked = []

    for candidate in fmea_candidates:
        candidate_embedding = get_query_embedding(candidate.get("text", ""))
        ranked.append(
            {
                **candidate,
                "score": cosine_similarity(query_embedding, candidate_embedding),
                "source": "dense",
            }
        )

    ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return ranked


def sparse_rank_fmea_candidates(
    query_text: str,
    fmea_candidates: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    ranked = []

    for candidate in fmea_candidates:
        ranked.append(
            {
                **candidate,
                "score": sparse_text_score(query_text, candidate.get("text", "")),
                "source": "sparse",
            }
        )

    ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return ranked


def rrf_fusion_candidates(
    result_sets: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}

    for result_set in result_sets:
        for rank, item in enumerate(result_set, start=1):
            candidate_id = item.get("candidate_id", "")
            if candidate_id not in fused:
                fused[candidate_id] = {
                    "candidate_id": candidate_id,
                    "text_type": item.get("text_type", ""),
                    "text": item.get("text", ""),
                    "score": 0.0,
                    "sources": set(),
                    "source_scores": {},
                }

            source = item.get("source", "unknown")
            fused[candidate_id]["score"] += 1.0 / (k + rank)
            fused[candidate_id]["sources"].add(source)
            fused[candidate_id]["source_scores"][source] = item.get("score", 0.0)

    ranked = list(fused.values())
    for item in ranked:
        item["sources"] = sorted(item["sources"])

    ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return ranked


def sparse_text_score(query_text: str, candidate_text: str) -> float:
    query_norm = ChunkRetriever._normalize_text(query_text)
    candidate_norm = ChunkRetriever._normalize_text(candidate_text)
    query_tokens = set(ChunkRetriever._meaningful_tokens(query_text, min_len=3))
    candidate_tokens = set(ChunkRetriever._meaningful_tokens(candidate_text, min_len=3))

    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = query_tokens.intersection(candidate_tokens)
    score = len(overlap) / math.sqrt(len(query_tokens) * len(candidate_tokens))

    if candidate_norm and candidate_norm in query_norm:
        score += 1.0
    if query_norm and query_norm in candidate_norm:
        score += 1.0

    return score


def cosine_similarity(left: List[float], right: List[float]) -> float:
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


def normalize_selected_text_types(selected_text_types: Any) -> List[str]:
    if selected_text_types is None:
        return ["effect", "mode", "cause", "function"]

    values = normalize_text_list(selected_text_types)
    selected = []
    for value in values:
        text_type = value.lower()
        if text_type not in SUPPORTED_TEXT_TYPES:
            raise ValueError(
                f"Unsupported FMEA text type: {value}. "
                "Expected one of: effect, mode, cause, function."
            )
        selected.append(text_type)

    return dedupe_preserve_order(selected)


def normalize_text_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]

    return [
        " ".join(str(value).split())
        for value in values
        if str(value or "").strip()
    ]


def dedupe_preserve_order(values: List[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        value = (value or "").strip()
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def print_sentence_to_fmea_results(result: Dict[str, Any]) -> None:
    print("=" * 100)
    print("Query Type: sentence_to_fmea_text")
    print(f"Retrieval Mode: {result.get('retrieval_mode', '')}")
    print(f"Selected Text Types: {result.get('selected_text_types', [])}")
    print("=" * 100)

    for index, item in enumerate(result.get("results", []), start=1):
        sentence_input = item.get("sentence_input", {})
        print("\n" + "-" * 100)
        print(
            f"[{index}] source={sentence_input.get('source', '')} "
            f"name={sentence_input.get('name', '')} "
            f"node_id={sentence_input.get('node_id', '')}"
        )
        print(f"Sentence: {sentence_input.get('text', '')}")
        print("Top-K FMEA Texts:")

        candidates = item.get("candidates", [])
        if not candidates:
            print("  (none)")
            continue

        for rank, candidate in enumerate(candidates, start=1):
            print(
                f"  [{rank}] type={candidate.get('text_type', '')} "
                f"score={candidate.get('score', 0.0):.4f} "
                f"sources={candidate.get('sources', [])}"
            )
            print(f"      text={candidate.get('text', '')}")
            source_scores = candidate.get("source_scores", {})
            if source_scores:
                print(
                    "      source_scores="
                    + " ".join(
                        f"{name}:{score:.4f}"
                        for name, score in source_scores.items()
                    )
                )


def main():
    node_names = [
        # Put Graph node names here. Their node.text will be used as queries.
        "TS6303220029R09_CHO_27*",
    ]

    sentence_texts = [
        # "In Motor control, Unbalanced motor currents due to Open loop control leading to Motor cannot start",
    ]

    fmea_texts = {
        "function": [
            "Soft starter",
            "Zero-crossing detection",
            "Relay switching",
        ],
        "mode": [
            "Component break-down",
            "Unbalanced motor currents",
            "Soft start too long",
            "No detection",
            "Relay cannot close",
            "Welded relay",
            "Relay cannot close",
            "False turn-on / turn-off",
        ],
        "cause": [
            "Compressor vibrations",
            "Open loop control",
            "Cooling insufficient",
            "Overvoltage due to motor disconnect",
            "Live switching of relays",
            "(Starting) Motor current too high for chosen components",
            "Under Voltage due to incorrect triggering",
            "Priority zero-crossing interrupt too low",
            "Too high dT junction as a result of power cycling of component",
        ],
        "effect": [
            "Motor cannot start",
            "Overcurrent towards motor",
            "Motor starts without soft start",
        ],
    }

    retriever = FMEASentenceRetrieverV2()
    try:
        sentence_inputs = build_sentence_inputs(
            retriever=retriever,
            node_names=node_names,
            sentence_texts=sentence_texts,
        )

        result = query_fmea_texts_for_sentences(
            retriever=retriever,
            sentence_inputs=sentence_inputs,
            fmea_texts=fmea_texts,
            selected_text_types=["mode", "cause","function"],
            top_k=8,
            retrieval_mode="hybrid",
        )
        print_sentence_to_fmea_results(result)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
