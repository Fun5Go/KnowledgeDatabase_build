from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

import numpy as np

import numpy as np
from pathlib import Path
from KnowledgeBase.JSON_FMEA_KB.kb_structure import FMEAFailureKB

# ---------------------------
# Config
# ---------------------------
SIM_CONFIG = {
    # role gates: if role is present for both sides but sim < gate => hard reject
    "element_threshold": 0.2,
    "mode_threshold": 0.78,
    "effect_threshold": 0.78,

    # final decision threshold after weighted combination
    "final_threshold": 0.80,

    # weights are normalized dynamically by available roles
    "weights": {
        "element": 0.5,
        "mode": 0.3,
        "effect": 0.2,
    },

    # candidate retrieval
    "top_k_candidates_element": 80,   # more recall
    "top_k_candidates_mode": 50,      # secondary recall
}


# ---------------------------
# Math helpers
# ---------------------------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return None
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return None
    return float(np.dot(a, b) / denom)


# ---------------------------
# Embedding cache to avoid slow per-id fetch
# ---------------------------
class EmbeddingCache:
    """
    Cache embeddings by (failure_id, role) from Chroma.
    role here must match the embedding id suffix you used:
      - f"{failure_id}::failure_mode"
      - f"{failure_id}::failure_element"
      - f"{failure_id}::failure_effect"
    """
    def __init__(self, collection):
        self.collection = collection
        self._cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}

    def get(self, failure_id: str, role: str) -> Optional[np.ndarray]:
        key = (failure_id, role)
        if key in self._cache:
            return self._cache[key]

        emb_id = f"{failure_id}::{role}"
        try:
            res = self.collection.get(ids=[emb_id], include=["embeddings"])
            if res and res.get("embeddings"):
                emb = res["embeddings"][0]
                out = np.array(emb) if emb is not None else None
                self._cache[key] = out
                return out
        except Exception:
            pass

        self._cache[key] = None
        return None

    def has_any_signal(self, failure_id: str) -> bool:
        return (
            self.get(failure_id, "failure_element") is not None
            or self.get(failure_id, "failure_mode") is not None
            or self.get(failure_id, "failure_effect") is not None
        )
    
def element_similarity(f1, f2, emb_cache):
    e1 = emb_cache.get(f1, "failure_element")
    e2 = emb_cache.get(f2, "failure_element")
    if e1 is None or e2 is None:
        return None
    return cosine(e1, e2)

def mode_effect_similarity(f1, f2, emb_cache):
    sims = []

    m1 = emb_cache.get(f1, "failure_mode")
    m2 = emb_cache.get(f2, "failure_mode")
    if m1 is not None and m2 is not None:
        s = cosine(m1, m2)
        if s is not None:
            sims.append(s)

    e1 = emb_cache.get(f1, "failure_effect")
    e2 = emb_cache.get(f2, "failure_effect")
    if e1 is not None and e2 is not None:
        s = cosine(e1, e2)
        if s is not None:
            sims.append(s)

    if not sims:
        return None

    return sum(sims) / len(sims)

def group_by_similarity(failure_ids, sim_fn, threshold):
    groups = []
    visited = set()

    for i, fid in enumerate(failure_ids):
        if fid in visited:
            continue

        g = {fid}
        visited.add(fid)

        for other in failure_ids[i+1:]:
            if other in visited:
                continue

            sim = sim_fn(fid, other)
            if sim is not None and sim >= threshold:
                g.add(other)
                visited.add(other)

        if len(g) > 1:
            groups.append(g)

    return groups
# ---------------------------
# Role-aware similarity
# ---------------------------
def compute_stage1_similarity(
    f1_id: str,
    f2_id: str,
    emb_cache: EmbeddingCache,
    cfg=SIM_CONFIG,
) -> float:
    scores = []
    weights = []

    for role in ["element", "mode", "effect"]:
        w = cfg["weights"][role]

        emb1 = emb_cache.get(f1_id, f"failure_{role}")
        emb2 = emb_cache.get(f2_id, f"failure_{role}")

        # missing role on either side => skip (no penalty)
        if emb1 is None or emb2 is None:
            continue

        sim = cosine(emb1, emb2)
        if sim is None:
            continue

        # hard gate if role is present for both and sim is too low
        if sim < cfg[f"{role}_threshold"]:
            return 0.0

        scores.append(sim * w)
        weights.append(w)

    if not weights:
        return 0.0

    return float(sum(scores) / sum(weights))


# ---------------------------
# Candidate retrieval using Chroma query
# ---------------------------
def _query_candidates_by_role(
    collection,
    query_emb: np.ndarray,
    role: str,
    top_k: int,
) -> List[str]:
    """
    Query Chroma for nearest neighbors among vectors with metadata role=={role}.
    Returns failure_ids for candidates (deduped).
    """
    if query_emb is None:
        return []

    try:
        res = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k,
            where={"role": role},
            include=["metadatas", "distances"],
        )
    except Exception:
        return []

    cands: List[str] = []
    metas = (res.get("metadatas") or [[]])[0]
    for m in metas:
        if not m:
            continue
        fid = m.get("failure_id")
        if fid:
            cands.append(fid)
    # de-dup keep order
    seen = set()
    out = []
    for x in cands:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def get_candidates(
    fid: str,
    emb_cache: EmbeddingCache,
    collection,
    cfg=SIM_CONFIG,
) -> Set[str]:
    """
    Stage-1 candidate set:
      - First from element neighbors (if element exists)
      - Then from mode neighbors (if mode exists)
    """
    cands: Set[str] = set()

    e = emb_cache.get(fid, "failure_element")
    if e is not None:
        cands.update(
            _query_candidates_by_role(
                collection,
                query_emb=e,
                role="failure_element",
                top_k=cfg["top_k_candidates_element"],
            )
        )

    m = emb_cache.get(fid, "failure_mode")
    if m is not None:
        cands.update(
            _query_candidates_by_role(
                collection,
                query_emb=m,
                role="failure_mode",
                top_k=cfg["top_k_candidates_mode"],
            )
        )

    # never include itself
    cands.discard(fid)
    return cands


# ---------------------------
# Union-Find (DSU) for robust grouping
# ---------------------------
class DSU:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def groups(self) -> List[Set[str]]:
        bucket: Dict[str, Set[str]] = defaultdict(set)
        for x in list(self.parent.keys()):
            bucket[self.find(x)].add(x)
        return list(bucket.values())


# ---------------------------
# Stage-1 grouping
# ---------------------------
def stage1_group_failures(
    failure_kb: FMEAFailureKB,
    cfg=SIM_CONFIG,
) -> List[Set[str]]:
    collection = failure_kb.collection
    emb_cache = EmbeddingCache(collection)

    all_ids = list(failure_kb.store.keys())
    # optional: only consider failures that have at least one role embedding
    all_ids = [fid for fid in all_ids if emb_cache.has_any_signal(fid)]

    dsu = DSU()
    for fid in all_ids:
        dsu.find(fid)

    # compare only with candidates, not all pairs
    for fid in all_ids:
        candidates = get_candidates(fid, emb_cache, collection, cfg=cfg)
        for other in candidates:
            # ensure other exists in store
            if other not in failure_kb.store:
                continue

            score = compute_stage1_similarity(fid, other, emb_cache, cfg=cfg)
            if score >= cfg["final_threshold"]:
                dsu.union(fid, other)

    groups = dsu.groups()
    return groups


# ---------------------------
# Apply groups: write failure_id_group into canonical record
# ---------------------------
def apply_stage1_groups(
    failure_kb: FMEAFailureKB,
    groups: List[Set[str]],
):
    """
    Writes back:
      - Only canonical record gets updated with failure_id_group.
      - Non-canonical records remain unchanged (no deletion).
    Canonical selection:
      - If failures have row-based IDs like XXX__R89, we prefer smallest R number.
      - Otherwise fallback to lexicographic.
    """
    def canonical_key(fid: str) -> Tuple[int, str]:
        # try parse __R<number>
        try:
            if "__R" in fid:
                r = fid.split("__R", 1)[1]
                # strip any suffix after digits
                num = ""
                for ch in r:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num:
                    return (int(num), fid)
        except Exception:
            pass
        return (10**12, fid)

    merged_count = 0

    for group in groups:
        if len(group) <= 1:
            continue

        canonical = sorted(group, key=canonical_key)[0]
        merged_ids = sorted(group, key=canonical_key)

        rec = failure_kb.store.get(canonical)
        if rec is None:
            continue

        rec.setdefault("failure_id_group", [])

        for fid in merged_ids:
            if fid not in rec["failure_id_group"]:
                rec["failure_id_group"].append(fid)

        merged_count += 1

    # persist store
    failure_kb.store_path.write_text(
        json.dumps(failure_kb.store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Applied Stage-1 groups to {merged_count} canonical failures")


# ---------------------------
# Optional: Quick audit
# ---------------------------
def audit_groups(groups: List[Set[str]]):
    sizes = [len(g) for g in groups]
    c = Counter(sizes)
    big = sorted([s for s in sizes if s > 1], reverse=True)[:10]
    print("[AUDIT] group size histogram (size -> count):", dict(sorted(c.items())))
    print("[AUDIT] top group sizes:", big)

def element_groups_by_knn(kb, k=10, dist_threshold=0.35):
    col = kb.collection

    res = col.get(
        where={"role": "failure_element"},
        include=["documents", "metadatas"],
    )

    groups = []
    visited = set()

    for doc, meta in zip(res["documents"], res["metadatas"]):
        fid = meta["failure_id"]
        if fid in visited:
            continue

        q = col.query(
            query_texts=[doc],
            n_results=k,
            where={"role": "failure_element"},
            include=["distances", "metadatas"],
        )

        g = {fid}
        for d, m in zip(q["distances"][0], q["metadatas"][0]):
            if d <= dist_threshold:
                g.add(m["failure_id"])

        if len(g) > 1:
            groups.append(g)
            visited |= g

    return groups

def mode_groups_by_knn(kb, k=15, dist_threshold=0.40):
    col = kb.collection

    res = col.get(
        where={"role": "failure_mode"},
        include=["documents", "metadatas"],
    )

    groups = []
    visited = set()

    for doc, meta in zip(res["documents"], res["metadatas"]):
        fid = meta["failure_id"]
        if fid in visited:
            continue

        q = col.query(
            query_texts=[doc],
            n_results=k,
            where={"role": "failure_mode"},
            include=["distances", "metadatas"],
        )

        g = {fid}
        for d, m in zip(q["distances"][0], q["metadatas"][0]):
            if d <= dist_threshold:
                g.add(m["failure_id"])

        if len(g) > 1:
            groups.append(g)
            visited |= g

    return groups
# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    FAILURE_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb"
    )

    kb = FMEAFailureKB(FAILURE_PATH)
    # print(f"[INFO] Loaded failures: {len(kb.store)}")
    # print(f"[INFO] Loaded causes  : {len(kb.cause_store)}")

    # groups = stage1_group_failures(kb, cfg=SIM_CONFIG)
    # print(f"[INFO] Total groups: {len(groups)}")

    # audit_groups(groups)

    # apply_stage1_groups(kb, groups)

    # print("[DONE] Stage-1 failure grouping finished.")
    emb_cache = EmbeddingCache(kb.collection)
    failure_ids = list(kb.store.keys())
    res = kb.collection.get(include=["metadatas"])
    print("[DEBUG] vector count:", len(res["ids"]))
    print(
        "[DEBUG] role counts:",
        Counter(m["role"] for m in res["metadatas"] if m and "role" in m)
    )

    # -------- Pass A: element groups --------
    element_groups = element_groups_by_knn(
            kb,
            k=10,
            dist_threshold=0.35,   # 先用这个
        )

    print(f"[ELEMENT] groups: {len(element_groups)}")
    print("[ELEMENT] top sizes:", sorted([len(g) for g in element_groups], reverse=True)[:10])

    # -------- Pass B: mode + effect groups --------
    mode_groups = mode_groups_by_knn(
        kb,
        k=15,
        dist_threshold=0.40,
    )

    print("[MODE] group count:", len(mode_groups))
    print("[MODE] top sizes:", sorted([len(g) for g in mode_groups], reverse=True)[:10])
