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

def print_semantic_results(res):
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    n = min(len(ids), len(docs), len(metas), len(dists))

    print(f"Returned semantic nodes: {n}")

    for i in range(n):
        print("=" * 100)
        print(f"[{i:02d}] semantic_id: {ids[i]}")
        print(f"  text: {docs[i]}")
        print(f"  similarity: {1 - float(dists[i]):.4f}")
        print(f"  field_type: {metas[i].get('field_type')}")
        print(f"  source_type: {metas[i].get('source_type')}")
        print(f"  count: {metas[i].get('count')}")



if  __name__ == "__main__":
    KB_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb")

    res = query_semantic_kb(
        persist_dir=KB_PATH,
        query_text="power train",
        field_type="element",
        n_results=5,
        min_count=1,
    )

    print_semantic_results(res)

    