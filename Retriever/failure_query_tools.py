from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List, Union

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import json
from typing import Optional, Dict, Any, List, Union
from collections import defaultdict
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
from sentence_transformers import CrossEncoder
import numpy as np
from .utils import load_group_maps,print_semantic_results_with_group, print_semantic_results, compute_bm25

# =========================================================
# 1) Load KB
# =========================================================
def _load_kb(persist_dir: Union[str, Path]) -> FMEAFailureKB:
    return FMEAFailureKB(Path(persist_dir))


def _get_collection(persist_dir: Union[str, Path]):
    kb = _load_kb(persist_dir)
    return kb.collection


def _build_where(
    field_type: Optional[Union[str, List[str]]] = None,
    source_type: Optional[Union[str, List[str]]] = None,
    min_count: Optional[int] = None,
):
    def clause(key, value):
        if value is None:
            return None
        if isinstance(value, list):
            return {key: {"$in": value}}
        return {key: value}

    clauses = []

    for k, v in [
        ("field_type", field_type),
        ("source_type", source_type),
    ]:
        c = clause(k, v)
        if c:
            clauses.append(c)

    if min_count is not None:
        clauses.append({"count": {"$gte": int(min_count)}})

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}



# =========================================================
# 1) Semantic search
# =========================================================
# -------------------------------------------------
# Hybrid Retrieval Function
# -------------------------------------------------

def normalize(scores):
    scores = np.array(scores, dtype=float)

    if len(scores) == 0:
        return scores

    min_s = scores.min()
    max_s = scores.max()

    if max_s - min_s < 1e-8:
        return np.zeros_like(scores)

    return (scores - min_s) / (max_s - min_s)
def sort_by_score(documents, metadatas, scores):
    items = []

    for doc, meta, score in zip(documents, metadatas, scores):
        items.append({
            "text": doc,
            "metadata": meta,
            "score": float(score)
        })

    items.sort(key=lambda x: x["score"], reverse=True)
    return items


def query_semantic_kb(
    persist_dir: Union[str, Path],
    query_text: str,
    n_results: int = 10,
    field_type: Optional[Union[str, List[str]]] = None,
    source_type: Optional[Union[str, List[str]]] = None,
    min_count: Optional[int] = None,
    include: Optional[List[str]] = None,
    alpha: float = 0.6,
    hybrid: bool = True,
):
    col = _get_collection(persist_dir)

    where = _build_where(
        field_type=field_type,
        source_type=source_type,
        min_count=min_count,
    )

    if include is None:
        include = ["documents", "metadatas", "distances"]

    results: Dict[str, Any] = col.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        include=include,
    )

    # If not hybrid -> keep original output exactly (do nothing)
    if not hybrid:
        return results

    # Hybrid rerank: add fields only when hybrid=True
    # Keep chroma distances semantics (cosine distance), but reorder them.
    if "documents" not in results or not results["documents"]:
        return results

    # Prepare hybrid fields with same outer shape: List[List[...]]
    results["hybrid_scores"] = []
    results["hybrid_distances"] = []

    num_queries = len(results["documents"])

    for qi in range(num_queries):
        docs = results["documents"][qi] or []
        if not docs:
            results["hybrid_scores"].append([])
            results["hybrid_distances"].append([])
            continue

        ids = results["ids"][qi] if "ids" in results and results["ids"] else []
        metas = results["metadatas"][qi] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][qi] if results.get("distances") else [1.0] * len(docs)

        # ---- semantic similarity (cosine) ----
        sem = 1.0 - np.array(dists, dtype=float)
        sem = np.clip(sem, 0.0, 1.0)

        # ---- bm25 ----
        bm25 = compute_bm25(query_text, docs)

        # ---- normalize ----
        sem_n = normalize(sem)
        bm25_n = normalize(bm25)

        # ---- hybrid score ----
        final = alpha * sem_n + (1.0 - alpha) * bm25_n  # 0..1

        # ---- sort index ----
        order = np.argsort(-final)

        # ---- reorder all aligned fields ----
        if ids:
            results["ids"][qi] = [ids[i] for i in order]

        results["documents"][qi] = [docs[i] for i in order]

        if results.get("metadatas"):
            results["metadatas"][qi] = [metas[i] for i in order]

        if results.get("distances"):
            # IMPORTANT: keep original cosine distances, only reorder
            results["distances"][qi] = [float(dists[i]) for i in order]

        # ---- add hybrid fields (aligned to reranked order) ----
        hyb_scores = [float(final[i]) for i in order]
        results["hybrid_scores"].append(hyb_scores)
        results["hybrid_distances"].append([float(1.0 - s) for s in hyb_scores])

    return results


# =========================================================
# 2) Query by metadata (group results by metadata)
# =========================================================
def get_by_metadata(
    persist_dir: Union[str, Path],
    collection_name: str = "all_failure_kb",
    limit: int = 1000,
    offset: int = 0,
    # filters:
    failure_id: Optional[str] = None,
    cause_id: Optional[str] = None,
    role: Optional[Union[str, List[str]]] = None,
    source_type: Optional[Union[str, List[str]]] = None,
    fmea_type: Optional[Union[str, List[str]]] = None,
    productPnID: Optional[Union[str, List[str]]] = None,
    product_domain: Optional[Union[str, List[str]]] = None,
    system: Optional[Union[str, List[str]]] = None,
    discipline: Optional[Union[str, List[str]]] = None,
    extra_where: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
):
    """
    Meta filter：where + get（返回所有匹配项，带分页 offset/limit）
    """
    col = _get_collection(persist_dir, collection_name=collection_name)

    where = _build_where(
        failure_id=failure_id,
        cause_id=cause_id,
        role=role,
        source_type=source_type,
        fmea_type=fmea_type,
        productPnID=productPnID,
        product_domain=product_domain,
        system=system,
        discipline=discipline,
        extra_where=extra_where,
    )

    if include is None:
        include = ["documents", "metadatas"]

    where_arg = where if where else None

    return col.get(
        where=where_arg,
        limit=limit,
        offset=offset,
        include=include,
    )

def get_semantic_structured(
    persist_dir: Union[str, Path],
    semantic_id: str,
):
    kb = _load_kb(persist_dir)
    return kb.field_store.get(semantic_id)


# =========================================================
# 3) Query by ID + metadata
# =========================================================
def get_semantic_by_ids(
    persist_dir: Union[str, Path],
    ids: List[str],
    include: Optional[List[str]] = None,
):
    col = _get_collection(persist_dir)

    if include is None:
        include = ["documents", "metadatas"]

    return col.get(ids=ids, include=include)


def get_failure_entity(
    persist_dir: Union[str, Path],
    failure_id: str,
):
    kb = _load_kb(persist_dir)
    return kb.entity_store.get(failure_id)

def cosine_distance(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    return 1 - np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

def get_embedding_vector(
    persist_dir,
    semantic_id: str,
) -> Optional[List[float]]:
    col = _get_collection(persist_dir)

    result = col.get(
        ids=[semantic_id],
        include=["embeddings"],
    )

    # result 可能是 dict；embeddings 可能是 None / list / np.ndarray
    if result is None:
        return None

    embeddings = result.get("embeddings", None)
    if embeddings is None:
        return None

    # embeddings 可能是 np.ndarray 或 list，统一用 len 判断
    try:
        if len(embeddings) == 0:
            return None
    except TypeError:
        # 万一 embeddings 不是可 len 的对象
        return None

    vec = embeddings[0]
    if vec is None:
        return None

    # vec 可能是 np.ndarray，转成 python list 方便序列化/存储
    if isinstance(vec, np.ndarray):
        return vec.astype(float).tolist()

    # vec 可能已经是 list[float]
    return list(vec)

def get_distance_between_semantic_nodes(
    persist_dir: Union[str, Path],
    semantic_id_1: str,
    semantic_id_2: str,
) -> float:

    vec1 = get_embedding_vector(persist_dir, semantic_id_1)
    vec2 = get_embedding_vector(persist_dir, semantic_id_2)

    if vec1 is None:
        raise ValueError(f"No embedding for {semantic_id_1}")

    if vec2 is None:
        raise ValueError(f"No embedding for {semantic_id_2}")

    return float(cosine_distance(vec1, vec2))

FIELD_ID_MAP = {
    "element": "element_id",
    "mode": "mode_id",
    "effect": "effect_id",
    "cause": "cause_id",
}

def query_linked_failure_fields(
    persist_dir: Union[str, Path],
    query_text: str,
    field_type: str,
    linked_fields: List[str],
    n_results: int = 5,
    min_count: Optional[int] = None,
):
    """
    Input field type and text
    Output the linked failure field (mode/effect/cause)

    Returns:
        {
            linked_field_type: {
                semantic_id: {
                    "text": ...,
                    "failure_ids": [...],
                    "count": ...
                }
            }
        }
    """

    kb = _load_kb(persist_dir)

    # --------------------------------------------
    # 1) Semantic search on input field
    # --------------------------------------------
    res = query_semantic_kb(
        persist_dir=persist_dir,
        query_text=query_text,
        field_type=field_type,
        n_results=n_results,
        min_count=min_count,
    )

    semantic_ids = res.get("ids", [[]])[0]

    # --------------------------------------------
    # 2) Collect failure_ids
    # --------------------------------------------
    all_failure_ids = set()

    for sid in semantic_ids:
        node = kb.field_store.get(sid, {}) or {}
        failure_ids = node.get("failure_ids", []) or []
        all_failure_ids.update(failure_ids)

    # --------------------------------------------
    # 3) Traverse failure entities
    # --------------------------------------------
    result = {lf: {} for lf in linked_fields}

    for fid in all_failure_ids:

        entity = kb.entity_store.get(fid)
        if not entity:
            continue

        for lf in linked_fields:

            real_key = FIELD_ID_MAP.get(lf, lf)

            semantic_id = entity.get(real_key)
            if not semantic_id:
                continue

            node = kb.field_store.get(semantic_id)
            if not node:
                continue

            if semantic_id not in result[lf]:
                result[lf][semantic_id] = {
                    "text": node.get("text"),
                    "count": len(node.get("failure_ids", [])),
                    "failure_ids": list(node.get("failure_ids", [])),
                }

    # --------------------------------------------
    # 4) Sort by frequency
    # --------------------------------------------
    for lf in result:
        result[lf] = dict(
            sorted(
                result[lf].items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )
        )

    return result


semantic_reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank_semantic_results(query_text, res, top_k=None):

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    pairs = [[query_text, doc] for doc in docs]
    scores = semantic_reranker.predict(pairs)

    items = []
    for sid, doc, meta, dist, score in zip(ids, docs, metas, dists, scores):
        items.append({
            "id": sid,
            "doc": doc,
            "meta": meta,
            "dist": dist,
            "ce_score": float(score),
        })

    items = sorted(items, key=lambda x: x["ce_score"], reverse=True)

    if top_k:
        items = items[:top_k]

    return items


def retrieve_similar_failures_from_entity(
    persist_dir: Union[str, Path],
    failure_entity: Dict[str, Any],

    top_k_per_field: int = 25,
    weight_element: float = 0.5,
    weight_mode: float = 1.5,
    weight_cause: float = 1.5,
    weight_effect: float = 1.5,
    top_n: int = 50,
    min_similarity: float = 0.3,
):
    """
    Simplified failure-to-failure retrieval.

    For each field:
        - Take top_k semantic hits
        - Add similarity * weight to all mapped failure_ids
        - Record match_detail per failure_id per field

    Return top_n failures by accumulated score, with match_detail.
    """

    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)

    element_text = (failure_entity.get("failure_element_text") or "").strip()
    mode_text    = (failure_entity.get("failure_mode_text") or "").strip()
    cause_text   = (failure_entity.get("failure_cause_text") or "").strip()
    effect_text  = (failure_entity.get("failure_effect_text") or "").strip()

    # score + match_detail accumulator
    acc = defaultdict(lambda: {
        "score": 0.0,
        "match_detail": {"element": [], "mode": [], "cause": [], "effect": []},
        "_seen": {"element": set(), "mode": set(), "cause": set(), "effect": set()},  # internal dedupe
    })

    def query_and_score(text: str, field: str, weight: float):
        if not text:
            return

        res = query_semantic_kb(
            persist_dir,
            text,
            field_type=field,
            n_results=top_k_per_field,
        )

        ids = (res.get("ids", [[]]) or [[]])[0] or []
        dists = (res.get("distances", [[]]) or [[]])[0] or []

        for sid, dist in zip(ids, dists):
            try:
                similarity = max(0.0, 1.0 - float(dist))
            except Exception:
                continue

            if similarity < min_similarity:
                continue

            node = kb.field_store.get(sid, {}) or {}
            failure_ids = node.get("failure_ids", []) or []
            if not failure_ids:
                continue

            for fid in failure_ids:
                item = acc[fid]

                # ---- score ----
                item["score"] += similarity * float(weight)

                # ---- match_detail (dedupe by semantic_id per field per failure) ----
                if sid in item["_seen"][field]:
                    continue
                item["_seen"][field].add(sid)

                item["match_detail"][field].append({
                    "semantic_id": sid,
                    "similarity": round(float(similarity), 4),
                    "query_text": text,
                })

    # ---- run 4 fields ----
    query_and_score(element_text, "element", weight_element)
    query_and_score(mode_text,    "mode",    weight_mode)
    query_and_score(cause_text,   "cause",   weight_cause)
    query_and_score(effect_text,  "effect",  weight_effect)

    # ---- build result ----
    results = []
    for fid, info in acc.items():
        entity = kb.entity_store.get(fid)
        if not entity:
            continue

        results.append({
            "failure_id": fid,
            "element": entity.get("failure_element_text"),
            "mode": entity.get("failure_mode_text"),
            "cause": entity.get("failure_cause_text"),
            "effect": entity.get("failure_effect_text"),
            "score": round(float(info["score"]), 4),
            "match_detail": info["match_detail"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[: int(top_n)]



if  __name__ == "__main__":
    KB_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_miniLM\failure_kb")
    kb = FMEAFailureKB(KB_PATH)

    group_files = {
    "mode": r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\process_KB\mode_groups_refined_v2.json",
    "cause": r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\process_KB\cause_groups_refined_v2.json",
    "effect": r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\process_KB\effect_groups_refined_v2.json",
    "element": r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\process_KB\element_groups_refined_v2.json",
}
    group_maps = load_group_maps(group_files)

    query_text = "creates too much noise"  

    res = query_semantic_kb(
        persist_dir=KB_PATH,
        query_text=query_text,
        field_type=["mode"],
        n_results=10,
        min_count=1,
        hybrid=False,
        source_type="8D"
    )
    # for r in res[:30]:
    #     print(r["score"], r["text"])
    print_semantic_results(res,kb,max_failure_ids=5)
    # print_semantic_results_with_group(res,kb=kb, group_maps=group_maps,top_n=10)

    # result = query_linked_failure_fields(
    #     persist_dir=KB_PATH,
    #     query_text=query_text,
    #     field_type="mode",
    #     linked_fields=["element","effect","cause"]
    # )


    # reranked = rerank_semantic_results(query_text, res, top_k=20)

    # print("\n====== Reranked Semantic Results ======\n")

    # for i, item in enumerate(reranked, start=1):

    #     semantic_id = item["id"]
    #     node = kb.field_store.get(semantic_id, {}) or {}
    #     failure_ids = node.get("failure_ids", [])

    #     print("=" * 100)
    #     print(f"[{i:02d}] semantic_id: {semantic_id}")
    #     print(f"  text: {item['doc']}")
    #     print(f"  CE score: {item['ce_score']:.4f}")
    #     print(f"  original similarity: {1 - float(item['dist']):.4f}")
    #     print(f"  failure_ids({len(failure_ids)}): {failure_ids[:10]}")


#--------------Entity retrieval
#     FAILURE_ENTITY = {
#     "failure_mode_text": "Motor stalls during operation",
#     "failure_element_text": "Conveyor mechanism",
#     "failure_effect_text": "Process is delayed",
#     "failure_cause_text": "Incorrect motor or driver specification"
#   }
#     result = retrieve_similar_failures_from_entity(persist_dir=KB_PATH, failure_entity=FAILURE_ENTITY,top_n=20,min_similarity=0.4, top_k_per_field=15)
    def build_ground_truth_input(
            results: List[Dict],
            target_n: Optional[int] = None,
            strict_unique: bool = False,
        ) -> str:
            """
            Build LLM-readable GT examples.

            Args:
                results: retrieved chains
                target_n: desired number of output cases
                strict_unique:
                    False → backfill duplicates if unique not enough
                    True  → do NOT backfill, return fewer and warn
            """

            if not results:
                return "No similar failure chains were retrieved from the knowledge base."

            if target_n is None:
                target_n = len(results)

            def norm(x):
                if x is None:
                    return ""
                return " ".join(str(x).strip().split()).lower()

            def sig(r):
                return (
                    norm(r.get("element")),
                    norm(r.get("function")),
                    norm(r.get("mode")),
                    norm(r.get("effect")),
                    norm(r.get("cause")),
                )

            # -------------------------------------------------
            # 1️⃣ Collect unique chains
            # -------------------------------------------------
            selected = []
            seen = set()
            duplicates = []

            for r in results:
                s = sig(r)
                if s in seen:
                    duplicates.append(r)
                    continue
                seen.add(s)
                selected.append(r)
                if len(selected) >= target_n:
                    break

            # -------------------------------------------------
            # 2️⃣ Backfill only if NOT strict
            # -------------------------------------------------
            if not strict_unique and len(selected) < target_n:
                need = target_n - len(selected)
                selected.extend(duplicates[:need])

            # In strict mode, do nothing (may be fewer)

            # -------------------------------------------------
            # 3️⃣ Format
            # -------------------------------------------------
            lines = []
            lines.append("Retrieved Similar FMEA Failure Chains:\n")

            if strict_unique and len(selected) < target_n:
                lines.append(
                    f"⚠ WARNING: Only {len(selected)} unique chains available "
                    f"(requested {target_n}). No duplicate backfilling applied.\n"
                )

            for idx, r in enumerate(selected, start=1):
                lines.append(f"Rank {idx}")
                lines.append(f"Failure ID: {r.get('failure_id')}")
                lines.append(f"Relevance Score: {r.get('score')}")
                lines.append(f"Matched Fields: {', '.join(r.get('matched_fields', []))}")

                lines.append("Failure Chain:")
                tagged = r.get("tagged") or {}
                def fmt(field_name: str, fallback_key: str):
                    obj = tagged.get(field_name)
                    if isinstance(obj, dict) and "text" in obj:
                        return f"{obj.get('text')}  [{obj.get('tag', 'UNKNOWN')}]"
                    return f"{r.get(fallback_key)}"

                lines.append(f"  Element : {fmt('element', 'element')}")
                lines.append(f"  Mode    : {fmt('mode', 'mode')}")
                lines.append(f"  Effect  : {fmt('effect', 'effect')}")
                lines.append(f"  Cause   : {fmt('cause', 'cause')}")
                match_detail = r.get("match_detail", {}) or {}
                if isinstance(match_detail, dict):
                    for field_type, matches in match_detail.items():
                        if matches:
                            lines.append(f"  {str(field_type).upper()}:")
                            for m in matches:
                                display = dict(m)
                                lines.append(f"    - {display}")
                lines.append("-" * 60)
                lines.append("-" * 60)

            return "\n".join(lines)


    # results = build_ground_truth_input(result,target_n=10,strict_unique=True)
    # print(results)