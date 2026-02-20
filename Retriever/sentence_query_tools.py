from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List, Union

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

from typing import Optional, Dict, Any, List, Union
from collections import defaultdict
from sentence_transformers import CrossEncoder

def _get_sentence_collection(
    persist_dir: Union[str, Path],
    collection_name: str = "all_failure_kb",
):
    client = chromadb.PersistentClient(path=str(persist_dir))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
    )

def _build_sentence_where(
    failure_id=None,
    cause_id=None,
    sentence_role=None,
    case_id=None,
    productPnID=None,
    product_domain=None,
    source_section=None,
    status=None,
    subject = None,
    min_faithful_score=None,
    min_year: int | None = None,
    max_year: int | None = None,
    extra_where=None,
    source_type=None,
):
    def clause(k, v):
        if v is None:
            return None
        if isinstance(v, list):
            return {k: {"$in": v}}
        return {k: v}

    clauses = []

    for k, v in [
        ("failure_id", failure_id),
        ("cause_id", cause_id),
        ("sentence_role", sentence_role),
        ("case_id", case_id),
        ("productPnID", productPnID),
        ("product_domain", product_domain),
        ("source_section", source_section),
        ("status", status),
        ("subject",subject),
        ("source_type", source_type),
    ]:
        c = clause(k, v)
        if c:
            clauses.append(c)
            # year range filter (numeric metadata)
    if min_year is not None:
        clauses.append({"released_year": {"$gte": int(min_year)}})
    if max_year is not None:
        clauses.append({"released_year": {"$lte": int(max_year)}})        

    if min_faithful_score is not None:
        clauses.append({"faithful_score": {"$gte": int(min_faithful_score)}})

    if extra_where:
        for k, v in extra_where.items():
            c = clause(k, v)
            if c:
                clauses.append(c)

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

def query_sentence_kb(
    persist_dir: Union[str, Path],
    query_text: str,
    n_results: int = 10,
    collection_name: str = "sentences",
    # filters
    failure_id: Optional[str] = None,
    cause_id: Optional[str] = None,
    sentence_role: Optional[Union[str, List[str]]] = None,
    case_id: Optional[str] = None,
    productPnID: Optional[Union[str, List[str]]] = None,
    product_domain: Optional[Union[str, List[str]]] = None,
    source_section: Optional[Union[str, List[str]]] = None,
    status: Optional[Union[str, List[str]]] = None,
    subject: Optional[Union[str, List[str]]] = None,
    min_faithful_score: Optional[int] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    extra_where: Optional[Dict[str, Any]] = None,
    source_type: Optional[Union[str, List[str]]] = None,
    include: Optional[List[str]] = None,
):
    col = _get_sentence_collection(persist_dir, collection_name)

    where = _build_sentence_where(
        failure_id=failure_id,
        cause_id=cause_id,
        sentence_role=sentence_role,
        case_id=case_id,
        productPnID=productPnID,
        product_domain=product_domain,
        source_section=source_section,
        status=status,
        subject=subject,
        min_faithful_score=min_faithful_score,
        min_year=min_year,
        max_year=max_year,
        extra_where=extra_where,
        source_type=source_type
    )

    if include is None:
        include = ["documents", "metadatas", "distances"]

    return col.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where if where else None,
        include=include,
    )

def get_sentences_by_metadata(
    persist_dir: Union[str, Path],
    collection_name: str = "sentences",
    limit: int = 1000,
    offset: int = 0,
    # filters
    failure_id: Optional[str] = None,
    cause_id: Optional[str] = None,
    sentence_role: Optional[Union[str, List[str]]] = None,
    case_id: Optional[str] = None,
    productPnID: Optional[Union[str, List[str]]] = None,
    product_domain: Optional[Union[str, List[str]]] = None,
    source_section: Optional[Union[str, List[str]]] = None,
    status: Optional[Union[str, List[str]]] = None,
    subject: Optional[Union[str, List[str]]] = None,
    min_faithful_score: Optional[int] = None,
    extra_where: Optional[Dict[str, Any]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    include: Optional[List[str]] = None,
):
    col = _get_sentence_collection(persist_dir, collection_name)

    where = _build_sentence_where(
        failure_id=failure_id,
        cause_id=cause_id,
        sentence_role=sentence_role,
        case_id=case_id,
        productPnID=productPnID,
        product_domain=product_domain,
        source_section=source_section,
        status=status,
        subject = subject,
        min_year=min_year,
        max_year=max_year,
        min_faithful_score=min_faithful_score,
        extra_where=extra_where,
    )

    if include is None:
        include = ["documents", "metadatas"]

    return col.get(
        where=where if where else None,
        limit=limit,
        offset=offset,
        include=include,
    )


# =========================================================
# Query by IDs
# =========================================================
def get_sentences_by_ids(
    persist_dir: Union[str, Path],
    ids: List[str],
    collection_name: str = "sentences",
    include: Optional[List[str]] = None,
):
    col = _get_sentence_collection(persist_dir, collection_name=collection_name)
    if include is None:
        include = ["documents", "metadatas"]
    return col.get(ids=ids, include=include)


from collections import defaultdict
from typing import Dict, Optional, Union, Any, List
from pathlib import Path


def query_sentence_kb_by_chunks(
    persist_dir: Union[str, Path],
    entity: Dict[str, Optional[str]],
    n_results_each: int = 5,
    collection_name: str = "sentences",
    # meta filters
    case_id=None,
    failure_id=None,
    cause_id=None,
    source_section=None,
    status=None,
    subject=None,
    faithful_score=None,
    productPnID=None,
    product_domain=None,
    extra_where=None,
) -> Dict[str, Any]:

    QUERY_SECTION_MAP = {
        "failure_element": ["D2", "D3"],
        "failure_mode": ["D2", "D3"],
        "failure_effect": ["D2", "D3"],
        "failure_cause": ["D3", "D4"],
    }

    CHUNK_WEIGHT = {
        "failure_mode": 1.3,
        "failure_cause": 1.5,
        "failure_element": 1.0,
        "failure_effect": 1.0,
    }

    all_hits = []
    seen_sentence_ids = set()

    # ===============================
    # Phase 1: query by role
    # ===============================
    for role, text in entity.items():
        if not text or not str(text).strip():
            continue

        r = query_sentence_kb(
            persist_dir=persist_dir,
            query_text=str(text),
            n_results=n_results_each,
            collection_name=collection_name,
            sentence_role=None,
            case_id=case_id,
            failure_id=failure_id,
            cause_id=cause_id,
            source_section=source_section,
            status=status,
            subject=subject,
            min_faithful_score=faithful_score,
            productPnID=productPnID,
            product_domain=product_domain,
            extra_where=extra_where,
            include=["documents", "metadatas", "distances"],
        )

        ids0 = r.get("ids", [[]])[0]
        docs0 = r.get("documents", [[]])[0]
        metas0 = r.get("metadatas", [[]])[0]
        dists0 = r.get("distances", [[]])[0]

        weight = CHUNK_WEIGHT.get(role, 1.0)

        for rid, doc, meta, dist in zip(ids0, docs0, metas0, dists0):
            if rid in seen_sentence_ids:
                continue
            seen_sentence_ids.add(rid)

            sim = 1.0 / (1.0 + float(dist))
            weighted_score = sim * weight

            all_hits.append({
                "sentence_id": rid,
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "role": role,
                "embedding_score": weighted_score,
            })

    # 先按 embedding score 排序
    all_hits = sorted(all_hits, key=lambda x: x["embedding_score"], reverse=True)

    return {
        "sentences": all_hits
    }


ROLE_ORDER = ["failure_element", "failure_mode", "failure_effect", "failure_cause"]

def build_concat_query(entity: Dict[str, Optional[str]]) -> str:
    parts = []
    for role in ROLE_ORDER:
        v = entity.get(role)
        if v and str(v).strip():
            # Will role prefix
            parts.append(f"{role.replace('failure_', '').title()}: {str(v).strip()}")
    return ". ".join(parts)

def query_sentence_kb_by_concat(
    persist_dir: Union[str, Path],
    entity: Dict[str, Optional[str]],
    *,
    top_k: int = 50,
    collection_name: str = "sentences",
    # meta filters
    case_id=None,
    failure_id=None,
    cause_id=None,
    status=None,
    subject=None,
    faithful_score=None,
    productPnID=None,
    product_domain=None,
    extra_where=None,
    # aggregation
    group_by: str = "case_id",
    # scoring
    use_rank_bonus: bool = True,
    rank_bonus_weight: float = 0.2,
    per_group_cap: int = 20,  # Max number of sentences for each case
) -> Dict[str, Any]:

    query_text = build_concat_query(entity)
    if not query_text.strip():
        return {"query_text": query_text, "raw": {}, "merged": []}

    # 1) Query KB by a whole sentence without senction limitation
    raw = query_sentence_kb(
        persist_dir=persist_dir,
        query_text=query_text,
        n_results=top_k,
        collection_name=collection_name,
        sentence_role=None,
        case_id=case_id,
        failure_id=failure_id,
        cause_id=cause_id,
        source_section=None,  
        status=status,
        subject=subject,
        min_faithful_score=faithful_score,
        productPnID=productPnID,
        product_domain=product_domain,
        extra_where=extra_where,
        include=["documents", "metadatas", "distances"],
    )

    ids0 = raw.get("ids", [[]])[0]
    docs0 = raw.get("documents", [[]])[0]
    metas0 = raw.get("metadatas", [[]])[0]
    dists0 = raw.get("distances", [[]])[0]

    # 2) Aggregation by the case id
    agg = defaultdict(lambda: {
        "group_id": None,
        "score": 0.0,
        "hits": [],
    })

    # 
    scored_hits = []
    for rank, (rid, doc, meta, dist) in enumerate(zip(ids0, docs0, metas0, dists0), start=1):
        gid = meta.get(group_by)
        if not gid:
            continue

        # similarity（main）
        sim = 1.0 / (1.0 + float(dist))  
        bonus = (1.0 / rank) if use_rank_bonus else 0.0
        item_score = sim + rank_bonus_weight * bonus

        scored_hits.append({
            "sentence_id": rid,
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "rank": rank,
            "sim": sim,
            "item_score": item_score,
        })

    # 3) 
    hits_by_gid = defaultdict(list)
    for h in scored_hits:
        hits_by_gid[h["metadata"].get(group_by)].append(h)

    for gid, hs in hits_by_gid.items():
        hs_sorted = sorted(hs, key=lambda x: x["item_score"], reverse=True)
        hs_take = hs_sorted[:per_group_cap]

        a = agg[gid]
        a["group_id"] = gid
        a["hits"] = hs_take
        a["score"] = sum(x["item_score"] for x in hs_take)

    merged = sorted(agg.values(), key=lambda x: x["score"], reverse=True)

    return {
        "query_text": query_text,
        "raw": raw,
        "merged": merged,
    }


def _pick(meta: Optional[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    meta = meta or {}
    return {k: meta.get(k) for k in keys if k in meta}

def print_concat_result_structured(
    result: Dict[str, Any],
    *,
    top_groups: int = 10,
    top_hits_each: int = 5,
    show_meta_keys: List[str] = None,
):
    if show_meta_keys is None:
        show_meta_keys = [
            "case_id", "failure_id", "cause_id",
            "sentence_role", "source_section",
            "productPnID", "product_domain",
        ]

    print("\n" + "=" * 80)
    print("[CONCAT QUERY]")
    print(result.get("query_text", "").strip() or "<EMPTY>")
    print("=" * 80)

    merged = result.get("merged") or []
    if not merged:
        print("\n[NO MERGED RESULTS]\n")
        return

    print(f"\n[MERGED GROUPS TOP {min(top_groups, len(merged))}]")
    for gi, g in enumerate(merged[:top_groups], start=1):
        gid = g.get("group_id")
        gscore = g.get("score", 0.0)
        hits = g.get("hits") or []

        print(f"\n[{gi}] group_id={gid}  score={gscore:.4f}  hits={len(hits)}")
        for hi, h in enumerate(hits[:top_hits_each], start=1):
            sid = h.get("sentence_id")
            rank = h.get("rank")
            dist = h.get("distance")
            sim = h.get("sim")
            item_score = h.get("item_score")
            text = (h.get("text") or "").strip().replace("\n", " ")
            if len(text) > 180:
                text = text[:180] + "..."

            meta = h.get("metadata") or {}
            meta_small = _pick(meta, show_meta_keys)

            print(
                f"   - ({hi}) sentence_id={sid} rank={rank} "
                f"dist={dist:.4f} sim={sim:.4f} item_score={item_score:.4f}"
            )
            if meta_small:
                print(f"       meta={meta_small}")
            print(f"       text={text}")

def print_get_result_items(res, show_all_metadata=False):
    ids = res.get("ids", [])
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])

    n = min(len(ids), len(docs), len(metas))
    print(f"Returned items: {n}")

    for i in range(n):
        m = metas[i] or {}
        print("=" * 110)
        print(f"[{i:02d}] id: {ids[i]}")
        print(f" case id: {m.get('case_id')} | text: {docs[i]}")
        print(f" role: {m.get('sentence_role')}  | Year:{m.get('released_year')} | subject:{m.get('subject')}"   )
        # print(f"  productPnID: {m.get('productPnID')} | domain: {m.get('product_domain')} ")

        if show_all_metadata:
            # print any extra metadata keys that exist
            extra = {k: v for k, v in m.items() if k not in {
                "role","source_type","fmea_type","failure_id","cause_id",
                "productPnID","product_domain","system"
            }}
            if extra:
                print(f"  extra_metadata: {extra}")

ROLE_ORDER = ["failure_element", "failure_mode", "failure_effect", "failure_cause"]

def build_concat_query(entity: Dict[str, str]) -> str:
    parts = []
    for role in ROLE_ORDER:
        value = entity.get(role)
        if value and str(value).strip():
            role_name = role.replace("failure_", "").title()
            parts.append(f"{role_name}: {value.strip()}")
    return ". ".join(parts)

ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def cross_encoder_rerank(query_text: str, hits: List[Dict], top_k: int = 20):

    pairs = [[query_text, h["text"]] for h in hits]
    scores = ce_model.predict(pairs)

    for h, s in zip(hits, scores):
        h["ce_score"] = float(s)

    reranked = sorted(hits, key=lambda x: x["ce_score"], reverse=True)

    return reranked[:top_k]

if __name__ == "__main__":

    KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\sentence_kb"
    )

    query_sentence = "In Safety protection, Motor overtemp false trip due to Trips on short spikes leading to Pump stops."
    result = query_sentence_kb(persist_dir=KB_PATH,query_text=query_sentence,n_results=20, source_type=["new_fmea", "old_fmea"])
    def structured_print(results):
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        print("=" * 140)
        print(f"{'Rank':<5} | {'Failure ID':<35} | {'Distance':<10} | {'Source':<12} | Document")
        print("=" * 140)

        for i, (fid, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), start=1):
            source = meta.get("source_type", "8D_case")
            doc_short = doc[:80] + "..." if len(doc) > 80 else doc

            print(f"{i:<5} | {fid:<35} | {dist:<10.6f} | {source:<12} | {doc_short}")

        print("=" * 140)


# Example usage:
    structured_print(result)
