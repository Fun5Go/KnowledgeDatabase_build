from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List, Union

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

from typing import Optional, Dict, Any, List, Union
from collections import defaultdict
from JSON_FMEA_KB.kb_structure import FMEAFailureKB


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
def query_semantic_kb(
    persist_dir: Union[str, Path],
    query_text: str,
    n_results: int = 10,
    field_type: Optional[Union[str, List[str]]] = None,
    source_type: Optional[Union[str, List[str]]] = None,
    min_count: Optional[int] = None,
    include: Optional[List[str]] = None,
):
    col = _get_collection(persist_dir)

    where = _build_where(
        field_type=field_type,
        source_type=source_type,
        min_count=min_count,
    )

    if include is None:
        include = ["documents", "metadatas", "distances"]

    return col.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        include=include,
    )



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


def print_semantic_results(res, kb, max_failure_ids: int = 15):
    """
    res: chroma query result
    kb: FMEAFailureKB or EightDFailureKB (must have .field_store)
    """
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    n = min(len(ids), len(docs), len(metas), len(dists))
    print(f"Returned semantic nodes: {n}")

    for i in range(n):
        semantic_id = ids[i]
        meta = metas[i] or {}

        # ---- get failure_ids from structured store ----
        node = kb.field_store.get(semantic_id, {}) or {}
        failure_ids = node.get("failure_ids", []) or []

        # ---- optional truncate ----
        shown = failure_ids[:max_failure_ids]
        more = len(failure_ids) - len(shown)

        print("=" * 100)
        print(f"[{i:02d}] semantic_id: {semantic_id}")
        print(f"  text: {docs[i]}")
        print(f"  similarity: {1 - float(dists[i]):.4f}")
        print(f"  field_type: {meta.get('field_type')}")
        print(f"  source_type: {meta.get('source_type')}")
        print(f"  count: {meta.get('count')}")

        print(f"  failure_ids({len(failure_ids)}): {shown}" + (f" ... (+{more})" if more > 0 else ""))



if  __name__ == "__main__":
    KB_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb")
    kb = FMEAFailureKB(KB_PATH)

    res = query_semantic_kb(
        persist_dir=KB_PATH,
        query_text="Incorrect gear shift",
        field_type="effect",
        n_results=5,
        min_count=1,
        source_type=["8D"],
    )
  
    print_semantic_results(res, kb)
    # res =  get_semantic_by_ids(persist_dir=KB_PATH, ids="effect:eb3b72714761")

#     linked = query_linked_failure_fields(
#     persist_dir=KB_PATH,
#     query_text="power train",
#     field_type="element",
#     linked_fields=["cause"],
#     n_results=5,
#     min_count=1,
# )

# for field_type, items in linked.items():
#     print(f"\n==== LINKED FIELD: {field_type} ====")
#     for sid, data in items.items():
#         print(f"{sid}")
#         print(f"  text: {data['text']}")
#         print(f"  count: {data['count']}")