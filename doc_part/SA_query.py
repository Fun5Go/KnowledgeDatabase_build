import re
import json
from collections import defaultdict
from typing import List, Dict, Any

from retrieval import create_embeddings, load_vector_store, create_cross_encoder, retrieve

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PERSIST_DIR = "./DATA/chroma_langchain_db"
COLLECTION_NAME = "technical_specification"


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    stopwords = {
        "the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "as",
        "is", "are", "be", "by", "with", "without", "due", "too"
    }
    tokens = normalize_text(text).split()
    return [t for t in tokens if t not in stopwords]


# =========================================================
# Failure signal patterns
# =========================================================
FAILURE_SIGNAL_PATTERNS = {
    "strong": [
        r"\bdue to\b",
        r"\bbecause of\b",
        r"\bleads? to\b",
        r"\bresult(s|ing)? in\b",
        r"\bcaused by\b",
        r"\bcauses?\b",
        r"\bcausing\b",
        r"\bresults? from\b",
        r"\bfails? due to\b",
        r"\bfailure of\b",
        r"\bas a result of\b",
        r"\bif\b",
        r"\bfailure\b",
    ],
    "medium": [
        r"\binduces?\b",
        r"\btriggers?\b",
        r"\bcreates?\b",
        r"\bproduces?\b",
        r"\bgives rise to\b",
        r"\bresponsible for\b",
        r"\boriginates? from\b",
        r"\bstems? from\b",
    ],
    "weak": [
        r"\btherefore\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bconsequently\b",
        r"\bso that\b",
    ]
}


def failure_signal_bonus(chunk_text: str) -> Dict[str, Any]:
    text = normalize_text(chunk_text)

    matched_signals = []
    bonus = 0.0

    weights = {
        "strong": 0.18,
        "medium": 0.10,
        "weak": 0.05,
    }

    for level, patterns in FAILURE_SIGNAL_PATTERNS.items():
        for p in patterns:
            if re.search(p, text):
                matched_signals.append({
                    "level": level,
                    "pattern": p
                })
                bonus += weights[level]

    # cap to avoid overpowering lexical evidence
    bonus = min(bonus, 0.35)

    return {
        "bonus": round(bonus, 4),
        "matched_signals": matched_signals
    }


def flatten_structure(structure_input: Dict[str, Any]) -> List[Dict[str, str]]:
    queries = []

    for node in structure_input.get("nodes", []):
        failure_element = node.get("failure_element")
        if failure_element:
            queries.append({
                "query_text": failure_element,
                "query_type": "failure_element"
            })

        for mode_group, mode_list in node.get("modes", {}).items():
            queries.append({
                "query_text": mode_group,
                "query_type": "mode_group"
            })
            for mode in mode_list:
                queries.append({
                    "query_text": mode,
                    "query_type": "mode"
                })

        for cause_group, cause_list in node.get("causes", {}).items():
            for cause in cause_list:
                queries.append({
                    "query_text": cause,
                    "query_type": "cause"
                })

        for effect in node.get("effects", []):
            queries.append({
                "query_text": effect,
                "query_type": "effect"
            })

    return queries


def lexical_match_score(query_text: str, chunk_text: str) -> Dict[str, Any]:
    q_norm = normalize_text(query_text)
    c_norm = normalize_text(chunk_text)

    q_tokens = tokenize(query_text)
    c_tokens = set(tokenize(chunk_text))

    matched_terms = [t for t in q_tokens if t in c_tokens]
    base_lexical_score = len(set(matched_terms)) / max(len(set(q_tokens)), 1)

    phrase_bonus = 0.0
    if q_norm in c_norm:
        phrase_bonus += 0.3

    signal_info = failure_signal_bonus(chunk_text)

    # 注意：这里只用于 per-query 检索，不把 signal bonus 算进去
    final_score = base_lexical_score + phrase_bonus

    return {
        "score": round(final_score, 4),   # 不含 signal bonus
        "base_lexical_score": round(base_lexical_score, 4),
        "phrase_bonus": round(phrase_bonus, 4),
        "signal_bonus": round(signal_info["bonus"], 4),   # 只记录，不参与当前分数
        "matched_terms": matched_terms,
        "matched_signals": signal_info["matched_signals"]
    }

def get_chunk_text(doc: Any) -> str:
    """
    兼容 LangChain Document 或 dict
    """
    if hasattr(doc, "page_content"):
        return doc.page_content or ""
    if isinstance(doc, dict):
        return doc.get("page_content", "") or doc.get("content", "")
    return ""


def get_chunk_metadata(doc: Any) -> Dict[str, Any]:
    """
    兼容 LangChain Document 或 dict
    """
    if hasattr(doc, "metadata"):
        return doc.metadata or {}
    if isinstance(doc, dict):
        return doc.get("metadata", {})
    return {}


def build_chunk_id(metadata: Dict[str, Any], fallback_idx: int) -> str:
    source = metadata.get("source", "unknown_source")
    req_ids = metadata.get("requirement_ids") or metadata.get("Requirement IDs")
    pages = metadata.get("pages")

    if req_ids:
        if isinstance(req_ids, list):
            req_part = "_".join(map(str, req_ids))
        else:
            req_part = str(req_ids)
        return f"{source}::{req_part}"

    if pages:
        if isinstance(pages, list):
            page_part = "_".join(map(str, pages))
        else:
            page_part = str(pages)
        return f"{source}::pages_{page_part}"

    return f"{source}::chunk_{fallback_idx}"


def search_top_chunks_per_query(
    queries: List[Dict[str, str]],
    vector_store,
    cross_encoder,
    retrieval_mode: str = "hybrid",
    use_rerank: bool = False,
    doc_type: str = "FS",
    candidate_k: int = 30,
    final_k: int = 5,
    hybrid_alpha: float = 0.5,
    min_lexical_score: float = 0.4,
) -> List[Dict[str, Any]]:

    query_results = []

    for q in queries:
        query_text = q["query_text"]
        query_type = q["query_type"]

        retrieved_docs = retrieve(
            vector_store=vector_store,
            query_text=query_text,
            retrieval_mode=retrieval_mode,
            use_rerank=use_rerank,
            doc_type=doc_type,
            candidate_k=candidate_k,
            final_k=candidate_k,
            hybrid_alpha=hybrid_alpha,
            cross_encoder=cross_encoder
        )

        best_chunks = {}

        for idx, doc in enumerate(retrieved_docs):
            chunk_text = get_chunk_text(doc)
            metadata = get_chunk_metadata(doc)

            lex_info = lexical_match_score(query_text, chunk_text)
            if lex_info["score"] < min_lexical_score:
                continue

            chunk_id = build_chunk_id(metadata, idx)

            item = {
                "chunk_id": chunk_id,
                "source": metadata.get("source"),
                "pages": metadata.get("pages"),
                "requirement_ids": metadata.get("requirement_ids") or metadata.get("Requirement IDs"),
                "chunk_mode": metadata.get("chunk_mode"),
                "doc_type": metadata.get("type") or metadata.get("doc_type"),
                "retrieval_score": metadata.get("score"),

                # per-query score: 不含 signal bonus
                "lexical_score": lex_info["score"],
                "base_lexical_score": lex_info["base_lexical_score"],
                "phrase_bonus": lex_info["phrase_bonus"],

                # chunk-level feature: 只记录
                "signal_bonus": lex_info["signal_bonus"],
                "matched_terms": lex_info["matched_terms"],
                "matched_signals": lex_info["matched_signals"],
                "content": chunk_text
            }

            if chunk_id not in best_chunks:
                best_chunks[chunk_id] = item
            else:
                if item["lexical_score"] > best_chunks[chunk_id]["lexical_score"]:
                    best_chunks[chunk_id] = item

        top_chunks = sorted(
            best_chunks.values(),
            key=lambda x: (
                x["lexical_score"],
                x["base_lexical_score"],
                x["phrase_bonus"]
            ),
            reverse=True
        )[:final_k]

        query_results.append({
            "query_text": query_text,
            "query_type": query_type,
            "top_chunks": top_chunks
        })

    return query_results


def aggregate_chunks_from_query_results(
    query_results: List[Dict[str, Any]],
    query_type_weights: Dict[str, float] = None,
    chunk_signal_weight: float = 1.0
) -> List[Dict[str, Any]]:
    """
    把所有 query 的 top chunks 反向聚合：
    一个 chunk 被多少 query 命中？
    被哪些 text 命中？
    signal bonus 作为 chunk-level bonus，只在聚合时加一次
    """
    if query_type_weights is None:
        query_type_weights = {
            "failure_element": 1.0,
            "mode_group": 1.0,
            "mode": 1.0,
            "effect": 1.0,
            "cause": 1.0,
        }

    chunk_map = defaultdict(lambda: {
        "chunk_id": None,
        "source": None,
        "pages": None,
        "requirement_ids": None,
        "chunk_mode": None,
        "doc_type": None,
        "content": None,
        "matched_queries": [],
        "hit_count": 0,
        "weighted_hit_count": 0.0,
        "max_lexical_score": 0.0,
        "chunk_signal_bonus": 0.0,
        "final_aggregate_score": 0.0,
    })

    for qr in query_results:
        qtext = qr["query_text"]
        qtype = qr["query_type"]
        qweight = query_type_weights.get(qtype, 1.0)

        for ch in qr["top_chunks"]:
            chunk_id = ch["chunk_id"]
            entry = chunk_map[chunk_id]

            entry["chunk_id"] = chunk_id
            entry["source"] = ch.get("source")
            entry["pages"] = ch.get("pages")
            entry["requirement_ids"] = ch.get("requirement_ids")
            entry["chunk_mode"] = ch.get("chunk_mode")
            entry["doc_type"] = ch.get("doc_type")
            entry["content"] = ch.get("content")

            entry["matched_queries"].append({
                "query_text": qtext,
                "query_type": qtype,
                "matched_terms": ch.get("matched_terms", []),
                "matched_signals": ch.get("matched_signals", []),
                "base_lexical_score": ch.get("base_lexical_score", 0.0),
                "phrase_bonus": ch.get("phrase_bonus", 0.0),
                "signal_bonus": ch.get("signal_bonus", 0.0),   # 只是记录
                "lexical_score": ch.get("lexical_score", 0.0)  # 不含 signal bonus
            })

            entry["hit_count"] += 1
            entry["weighted_hit_count"] += qweight
            entry["max_lexical_score"] = max(
                entry["max_lexical_score"],
                ch.get("lexical_score", 0.0)
            )

            # chunk-level bonus: 同一个 chunk 只保留一次
            entry["chunk_signal_bonus"] = max(
                entry["chunk_signal_bonus"],
                ch.get("signal_bonus", 0.0)
            )

    aggregated = list(chunk_map.values())

    for entry in aggregated:
        entry["final_aggregate_score"] = round(
            entry["weighted_hit_count"] + chunk_signal_weight * entry["chunk_signal_bonus"],
            4
        )

    aggregated.sort(
        key=lambda x: (
            x["final_aggregate_score"],
            x["hit_count"],
            x["max_lexical_score"]
        ),
        reverse=True
    )
    return aggregated


def print_query_results(query_results: List[Dict[str, Any]], max_content_len: int = 1000):
    print("\n" + "=" * 100)
    print("PER-QUERY TOP CHUNKS")
    print("=" * 100)

    for qr in query_results:
        print(f"\nQuery Text : {qr['query_text']}")
        print(f"Query Type : {qr['query_type']}")
        print("-" * 100)

        if not qr["top_chunks"]:
            print("  No hits.")
            continue

        for i, ch in enumerate(qr["top_chunks"], 1):
            content_preview = ch["content"][:max_content_len].replace("\n", " ")
            print(f"[{i}] Chunk ID         : {ch['chunk_id']}")
            print(f"    Source          : {ch.get('source')}")
            print(f"    Pages           : {ch.get('pages')}")
            print(f"    Requirement ID  : {ch.get('requirement_ids')}")
            print(f"    Chunk Mode      : {ch.get('chunk_mode')}")
            print(f"    Retrieval Score : {ch.get('retrieval_score')}")
            print(f"    Lexical Score   : {ch.get('lexical_score')}")
            print(f"    Base Lex Score  : {ch.get('base_lexical_score')}")
            print(f"    Phrase Bonus    : {ch.get('phrase_bonus')}")
            print(f"    Signal Bonus    : {ch.get('signal_bonus')}")
            print(f"    Matched Terms   : {ch.get('matched_terms')}")
            print(f"    Matched Signals : {ch.get('matched_signals')}")
            print(f"    Content         : {content_preview}")
            print()


def print_aggregated_chunks(aggregated_chunks: List[Dict[str, Any]], top_n: int = 20, max_content_len: int = 1000):
    print("\n" + "=" * 100)
    print("AGGREGATED TOP CHUNKS")
    print("=" * 100)

    for i, ch in enumerate(aggregated_chunks[:top_n], 1):
        content_preview = ch["content"][:max_content_len].replace("\n", " ")
        print(f"\n[{i}] Chunk ID              : {ch['chunk_id']}")
        print(f"    Source                : {ch.get('source')}")
        print(f"    Pages                 : {ch.get('pages')}")
        print(f"    Requirement IDs       : {ch.get('requirement_ids')}")
        print(f"    Hit Count             : {ch.get('hit_count')}")
        print(f"    Weighted Hit Count    : {round(ch.get('weighted_hit_count', 0.0), 4)}")
        print(f"    Chunk Signal Bonus    : {ch.get('chunk_signal_bonus')}")
        print(f"    Final Aggregate Score : {ch.get('final_aggregate_score')}")
        print(f"    Max Lexical Score     : {ch.get('max_lexical_score')}")

        print("    Matched Queries:")
        for mq in ch["matched_queries"]:
            print(
                f"      - [{mq['query_type']}] {mq['query_text']} "
                f"(matched_terms={mq['matched_terms']}, "
                f"matched_signals={mq.get('matched_signals', [])}, "
                f"base={mq.get('base_lexical_score', 0.0)}, "
                f"phrase_bonus={mq.get('phrase_bonus', 0.0)}, "
                f"signal_bonus={mq.get('signal_bonus', 0.0)}, "
                f"lexical_score={mq['lexical_score']})"
            )

        print(f"    Content               : {content_preview}")


if __name__ == "__main__":

    structure_input_motorcontrol = {
        "product_domain": "motor_drives",
        "nodes": [
            {
                "element_id": "E1",
                "failure_element": "Motor control",
                "modes": {
                    "Soft starter": [
                        "Component break-down",
                        "Unbalanced motor currents",
                    ],
                    "Zero-crossing detection": [
                        "Incorrect interpretation zero-crossing",
                        "Soft start too long",
                        "No detection",
                    ],
                    "Relay switching": [
                        "Welded relay",
                        "Relay cannot close",
                        "False turn-on / turn-off"
                    ],
                },
                "causes": {
                    "mechanics": [
                        "Cooling insufficient",
                        "Compressor vibrations"
                    ],
                    "hardware": [
                        "(Starting) Motor current too high for chosen components",
                        "Overvoltage due to motor disconnect",
                        "Under Voltage due to incorrect triggering",
                        "Live switching of relays"
                    ],
                    "software": [
                        "Priority zero-crossing interrupt too low",
                        "Open loop control"
                    ],
                    "other": [
                        "No (correctly designed) snubber design",
                        "Too high dT junction as a result of power cycling of component"
                    ]
                },
                "effects": [
                    "Motor cannot start",
                    "Overcurrent towards motor",
                    "Motor starts without soft start",
                ]
            }
        ]
    }

    embeddings = create_embeddings()
    vector_store = load_vector_store(
        embeddings=embeddings,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    cross_encoder = create_cross_encoder()

    retrieval_mode = "hybrid"   # dense / bm25 / hybrid
    use_rerank = True
    doc_type = "FS"
    candidate_k = 50
    final_k = 5
    hybrid_alpha = 0.4

    # 1) flatten structure -> query list
    queries = flatten_structure(structure_input_motorcontrol)

    # 2) each text separately search
    query_results = search_top_chunks_per_query(
        queries=queries,
        vector_store=vector_store,
        cross_encoder=cross_encoder,
        retrieval_mode=retrieval_mode,
        use_rerank=use_rerank,
        doc_type=doc_type,
        candidate_k=candidate_k,
        final_k=final_k,
        hybrid_alpha=hybrid_alpha,
        min_lexical_score=0.3,   # 调试时可设 0.0；正式建议 0.1 ~ 0.2
    )

    # 3) aggregate chunks across all query hits
    aggregated_chunks = aggregate_chunks_from_query_results(
    query_results,
    chunk_signal_weight=5
)

    # 4) print results
    print_query_results(query_results)
    print_aggregated_chunks(aggregated_chunks, top_n=20)

    # 5) optional: save json
    output = {
        "query_results": query_results,
        "aggregated_chunks": aggregated_chunks
    }
    with open("fmea_retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)