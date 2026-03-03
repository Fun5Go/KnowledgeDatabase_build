from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Tuple
from pathlib import Path
import json
from typing import Optional, Dict, Any, List, Union
import numpy as np
import re
from rank_bm25 import BM25Okapi
#------------------------
# Group 
#-------------------------
def load_group_maps(
    group_files: Dict[str, Union[str, Path]]
) -> Dict[str, Dict[str, str]]:
    """
    group_files example:
        {
            "mode": ".../mode_groups.json",
            "cause": ".../cause_groups.json",
            "effect": ".../effect_groups.json",
            "element": ".../element_groups.json",
        }

    return:
        {
            "mode": { semantic_id -> group_id },
            "cause": {...},
            ...
        }
    """

    group_maps: Dict[str, Dict[str, str]] = {}

    for field_type, file_path in group_files.items():

        path = Path(file_path)
        if not path.exists():
            print(f"[WARN] Group file not found: {path}")
            group_maps[field_type] = {}
            continue

        data = json.loads(path.read_text(encoding="utf-8"))

        sid_to_group: Dict[str, str] = {}

        for group_obj in data:

            group_id = group_obj["group_id"]
            member_ids = group_obj.get("member_node_ids", [])

            for sid in member_ids:
                sid_to_group[sid] = group_id

        group_maps[field_type] = sid_to_group

    return group_maps

def collapse_query_results_by_group(
    res: Dict[str, Any],
    group_maps: Dict[str, Dict[str, str]],
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    res: chroma query result with keys: ids, documents, metadatas, distances
    Assumption: res["ids"][0][i] is the semantic_id (e.g. 'effect:xxxx', 'mode:xxxx').
    """

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    group_best: Dict[str, Tuple[str, str, dict, float, float]] = {}
    group_hit_count: Dict[str, int] = {}
    non_group_hits: List[Tuple[str, str, dict, float, float]] = []

    for sid, doc, meta, dist in zip(ids, docs, metas, dists):
        field = meta.get("field_type")
        sim = 1.0 - float(dist)

        gid = None
        if field in group_maps:
            gid = group_maps[field].get(sid)

        if not gid:
            non_group_hits.append((sid, doc, meta, float(dist), sim))
            continue

        group_hit_count[gid] = group_hit_count.get(gid, 0) + 1

        cur = group_best.get(gid)
        if (cur is None) or (sim > cur[4]):
            # store best representative for this group
            meta2 = dict(meta)
            meta2["group_id"] = gid
            group_best[gid] = (sid, doc, meta2, float(dist), sim)

    final_hits = list(group_best.values()) + non_group_hits
    final_hits.sort(key=lambda x: x[4], reverse=True)  # by sim desc

    if top_n:
        final_hits = final_hits[:top_n]

    # attach group_hit_count for visibility
    new_ids, new_docs, new_metas, new_dists = [], [], [], []
    for sid, doc, meta, dist, sim in final_hits:
        if "group_id" in meta:
            meta = dict(meta)
            meta["group_hit_count_in_group"] = group_hit_count.get(meta["group_id"], 1)
        new_ids.append(sid)
        new_docs.append(doc)
        new_metas.append(meta)
        new_dists.append(dist)

    return {
        "ids": [new_ids],
        "documents": [new_docs],
        "metadatas": [new_metas],
        "distances": [new_dists],
    }

def print_semantic_results_with_group(
    res: Dict[str, Any],
    kb,
    group_maps: Optional[Dict[str, Dict[str, str]]] = None,
    collapse_groups: bool = True,
    top_n: int = 15,
):
    """
    kb: your KB object (not strictly required here, kept for compatibility)
    group_maps: {field_type: {semantic_id -> group_id}}
    """
    if collapse_groups and group_maps:
        res = collapse_query_results_by_group(res, group_maps, top_n=top_n)

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i, (sid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        sim = 1.0 - float(dist)
        gid = meta.get("group_id")

        print(f"[{i}] sim={sim:.4f}  dist={float(dist):.4f}")
        print(f"  semantic_id : {sid}")
        print(f"  field_type  : {meta.get('field_type')}")
        if gid:
            print(f"  group_id    : {gid}  (hits_in_group={meta.get('group_hit_count_in_group')})")
        print(f"  text        : {doc}")
        print("-" * 60)
#===========================
#===== BM25 Calculation ====
#==========================
def tokenize(text):
    # Simple tokenization
    return re.findall(r"\w+", text.lower())

def compute_bm25(query_text, documents):
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)

    return np.array(scores)



#----------------------
# Print
# --------------------
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

