from typing import Dict, List, Union, Optional, Any
from collections import defaultdict
from pathlib import Path
from Retriever.failure_query_tools import _load_kb,query_semantic_kb
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
import math
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_bge\failure_kb"
    )


def probe_kb_distance_range(
    persist_dir: str | Path,
    field_type: str = "mode",
    n_queries: int = 200,
    n_results: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Randomly pick the text for querying, to evaluate the distance range
    """
    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)
    field_store: Dict[str, dict] = getattr(kb, "field_store", {}) or {}

    # randomly choose
    pool = []
    for sid, rec in field_store.items():
        if (rec or {}).get("field_type") == field_type:
            t = (rec or {}).get("text")
            if t and str(t).strip():
                pool.append(str(t).strip())

    if not pool:
        raise ValueError(f"No texts found in field_store for field_type={field_type}")

    random.seed(seed)
    queries = [random.choice(pool) for _ in range(min(n_queries, len(pool)))]

    # query to get distances
    dists: List[float] = []
    sims: List[float] = []
    for q in queries:
        res = query_semantic_kb(
            persist_dir,
            q,
            field_type=field_type,
            n_results=n_results,
        ) or {}
        arr = (res.get("distances") or [[]])[0] or []
        for d in arr:
            try:
                similarity = max(0.0, 1.0 - float(d))
                dists.append(float(d))
                sims.append(float(similarity))
            except Exception:
                pass

    if not dists:
        raise ValueError("No distances returned from queries.")

    # dists_sorted = sorted(dists)
    # def pct(p: float) -> float:
    #     idx = int(round((len(dists_sorted) - 1) * p))
    #     return dists_sorted[idx]
    
    sims_sorted = sorted(sims)
    def pct(p: float) -> float:
        idx = int(round((len(sims_sorted) - 1) * p))
        return sims_sorted[idx]

    return {
        "count": float(len(sims_sorted)),
        "min": sims_sorted[0],
        "p05": pct(0.05),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": sims_sorted[-1],
    }

def compute_full_pairwise_similarity(
    persist_dir: str | Path,
    field_type: str = "mode",
    sample_size: int | None = 1500,
    seed: int = 42,
) -> Dict[str, float]:
    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)
    field_store: Dict[str, dict] = getattr(kb, "field_store", {}) or {}

    # 1) collect texts
    texts = [
        str((rec or {}).get("text")).strip()
        for rec in field_store.values()
        if (rec or {}).get("field_type") == field_type and (rec or {}).get("text")
    ]
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError(f"No texts found for field_type={field_type}")

    # 2) sample
    if sample_size and len(texts) > sample_size:
        random.seed(seed)
        texts = random.sample(texts, sample_size)

    print(f"Using {len(texts)} samples for pairwise similarity...")

    # 3) embed with CURRENT embedder (this is the key)
    E = np.array(kb.embedder(texts), dtype=np.float32)

    # (optional sanity) norms
    norms = np.linalg.norm(E, axis=1)
    print("norm min/mean/max:", float(norms.min()), float(norms.mean()), float(norms.max()))

    # 4) pairwise cosine
    sim_matrix = cosine_similarity(E)
    n = sim_matrix.shape[0]
    sims = sim_matrix[np.triu_indices(n, k=1)]

    return {
        "count": float(len(sims)),
        "min": float(np.min(sims)),
        "p05": float(np.percentile(sims, 5)),
        "p50": float(np.percentile(sims, 50)),
        "p95": float(np.percentile(sims, 95)),
        "max": float(np.max(sims)),
    }

def print_high_similarity_pairs(persist_dir: str, field_type: str,
                                similarity_threshold: float = 0.8,
                                max_print: int = 50):
    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)

    res = kb.collection.get(
        where={"field_type": field_type},
        include=["embeddings", "documents"],
    )

    embeddings = np.array(res["embeddings"], dtype=np.float32)
    ids = res["ids"]
    documents = res["documents"]

    n = len(ids)
    if n == 0:
        print(f"No nodes found for field={field_type}")
        return

    print(f"[{field_type}] Checking {n} nodes...")

    sim_matrix = cosine_similarity(embeddings)

    found = 0
    for i in range(n):
        sid1 = ids[i]
        node1 = kb.field_store.get(sid1, {}) or {}
        fids1 = node1.get("failure_ids", []) or []

        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim < similarity_threshold:
                continue

            sid2 = ids[j]
            node2 = kb.field_store.get(sid2, {}) or {}
            fids2 = node2.get("failure_ids", []) or []

            print("=" * 80)
            print(f"Similarity: {sim:.4f}")
            print(f"ID1: {sid1}")
            print(f"Text1: {documents[i][:200]}")
            print(f"failure_ids_1 (n={len(fids1)}): {fids1[:20]}")
            print("-" * 40)
            print(f"ID2: {sid2}")
            print(f"Text2: {documents[j][:200]}")
            print(f"failure_ids_2 (n={len(fids2)}): {fids2[:20]}")

            found += 1
            if found >= max_print:
                print(f"\nReached max_print={max_print}")
                return

    print(f"\nTotal high-similarity pairs found: {found}")

if __name__ == "__main__":
    field_list = [
        "element",
        "mode",
        "cause",
        "effect"
    ]
    for field in field_list:
        result = compute_full_pairwise_similarity(persist_dir=KB_PATH,field_type=field)
        print(f"Field: {field}|{result}")

    # print_high_similarity_pairs(persist_dir=KB_PATH, field_type = "cause", similarity_threshold=0.9,max_print=70)