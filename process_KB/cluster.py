from typing import Dict, List, Union, Optional, Any
from collections import defaultdict
from pathlib import Path
from Retriever.failure_query_tools import _load_kb,query_semantic_kb
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import re


BASE_DIR = Path(__file__).resolve().parent
KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_discipline\failure_kb"
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

def merge_semantic_nodes_to_groups(
    persist_dir: str,
    field_type: str,
    similarity_threshold: float = 0.8,
):
    """
    Merge highly similar semantic nodes into groups
    and export result to JSON file.

    Does NOT modify KB.
    """

    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)

    BASE_DIR = Path(__file__).resolve().parent
  
    print(f"\n[INFO] Loading nodes for field_type = {field_type}")

    res = kb.collection.get(
        where={"field_type": field_type},
        include=["embeddings", "documents"],
    )

    embeddings = np.array(res["embeddings"], dtype=np.float32)
    ids = res["ids"]
    documents = res["documents"]

    n = len(ids)
    if n == 0:
        print("No nodes found.")
        return

    print(f"[INFO] Total nodes: {n}")
    print("[INFO] Computing cosine similarity...")

    sim_matrix = cosine_similarity(embeddings)

    # -------------------------------------------------
    # 1️⃣ Build similarity graph
    # -------------------------------------------------
    adj = defaultdict(set)

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= similarity_threshold:
                adj[ids[i]].add(ids[j])
                adj[ids[j]].add(ids[i])

    # -------------------------------------------------
    # 2️⃣ Find connected components
    # -------------------------------------------------
    visited = set()
    groups = []

    for sid in ids:
        if sid in visited:
            continue

        stack = [sid]
        comp = []

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue

            visited.add(cur)
            comp.append(cur)
            stack.extend(adj[cur])

        groups.append(comp)

    print(f"[INFO] Total groups formed: {len(groups)}")

    # -------------------------------------------------
    # 3️⃣ Build group JSON structure
    # -------------------------------------------------
    id_to_index = {sid: idx for idx, sid in enumerate(ids)}

    group_results = []

    for gid, comp in enumerate(groups, start=1):

        member_texts = []
        all_failure_ids = []
        node_info = []

        for sid in comp:
            idx = id_to_index[sid]
            text = documents[idx]

            node_data = kb.field_store.get(sid, {}) or {}
            fids = node_data.get("failure_ids", []) or []

            member_texts.append(text)
            all_failure_ids.extend(fids)

            node_info.append(
                {
                    "node_id": sid,
                    "text": text,
                    "failure_count": len(fids),
                }
            )

        # remove duplicate failure_ids
        all_failure_ids = list(set(all_failure_ids))

        # -------------------------------------------------
        # 4️⃣ Choose canonical text
        # rule: most failure_ids → shortest length
        # -------------------------------------------------
        node_info_sorted = sorted(
            node_info,
            key=lambda x: (x["failure_count"], -len(x["text"])),
            reverse=True,
        )

        canonical_node = node_info_sorted[0]
        canonical_text = canonical_node["text"]

        group_results.append(
            {
                "group_id": f"{field_type}_group_{gid:04d}",
                "field_type": field_type,
                "canonical_text": canonical_text,
                "member_node_ids": comp,
                "variant_texts": member_texts,
                "failure_ids": all_failure_ids,
                "count": len(all_failure_ids),
                "group_size": len(comp),
            }
        )

    # sort groups by count descending
    group_results = sorted(
        group_results,
        key=lambda x: x["count"],
        reverse=True,
    )

    # -------------------------------------------------
    # 5️⃣ Save JSON
    # -------------------------------------------------
    output_path = BASE_DIR / f"{field_type}_groups.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(group_results, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Group file saved to:")
    print(output_path)
    print(f"[INFO] Total groups saved: {len(group_results)}")

    return group_results

def _tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(t for t in text.split() if t)


def refine_and_regroup(
    persist_dir: str,
    group_json_path: str,
    field_type: str,
    embed_threshold: float = 0.6,
    lexical_threshold: float = 0.3,
    regroup_threshold: float = 0.8,
):

    persist_dir = Path(persist_dir)
    group_json_path = Path(group_json_path)

    kb = _load_kb(persist_dir)

    with group_json_path.open("r", encoding="utf-8") as f:
        groups = json.load(f)

    print("\n[INFO] Refining and regrouping...")

    outlier_ids = []
    refined_groups = []

    # -------------------------------------------------
    # STEP 1: refine existing groups
    # -------------------------------------------------
    for group in groups:

        canonical_text = group["canonical_text"]
        canonical_tokens = _tokenize(canonical_text)

        canonical_id = None
        for sid in group["member_node_ids"]:
            if kb.field_store[sid]["text"] == canonical_text:
                canonical_id = sid
                break

        if canonical_id is None:
            continue

        canonical_emb = kb.collection.get(
            ids=[canonical_id],
            include=["embeddings"]
        )["embeddings"][0]

        canonical_emb = np.array(canonical_emb).reshape(1, -1)

        core_members = []
        core_texts = []
        core_failure_ids = []

        for sid in group["member_node_ids"]:

            node_text = kb.field_store[sid]["text"]
            node_tokens = _tokenize(node_text)

            node_emb = kb.collection.get(
                ids=[sid],
                include=["embeddings"]
            )["embeddings"][0]

            node_emb = np.array(node_emb).reshape(1, -1)

            emb_sim = cosine_similarity(
                canonical_emb,
                node_emb
            )[0][0]

            intersection = canonical_tokens.intersection(node_tokens)
            overlap_ratio = len(intersection) / max(len(canonical_tokens), 1)

            if emb_sim >= embed_threshold and overlap_ratio >= lexical_threshold:
                core_members.append(sid)
                core_texts.append(node_text)
                core_failure_ids.extend(
                    kb.field_store[sid].get("failure_ids", [])
                )
            else:
                outlier_ids.append(sid)

        if core_members:
            refined_groups.append(
                {
                    "group_id": group["group_id"],
                    "field_type": field_type,
                    "canonical_text": canonical_text,
                    "member_node_ids": core_members,
                    "variant_texts": core_texts,
                    "failure_ids": list(set(core_failure_ids)),
                    "count": len(set(core_failure_ids)),
                    "group_size": len(core_members),
                }
            )

    print(f"[INFO] Outliers collected: {len(outlier_ids)}")

    # -------------------------------------------------
    # STEP 2: regroup outliers strictly
    # -------------------------------------------------
    if outlier_ids:

        print("[INFO] Regrouping outliers...")

        res = kb.collection.get(
            ids=outlier_ids,
            include=["embeddings", "documents"]
        )

        embeddings = np.array(res["embeddings"], dtype=np.float32)
        ids = res["ids"]
        documents = res["documents"]

        sim_matrix = cosine_similarity(embeddings)

        visited = set()
        new_groups = []

        for i in range(len(ids)):

            if ids[i] in visited:
                continue

            center = ids[i]
            group = [center]
            visited.add(center)

            for j in range(len(ids)):
                if i == j:
                    continue
                if ids[j] in visited:
                    continue

                if sim_matrix[i, j] >= regroup_threshold:
                    group.append(ids[j])
                    visited.add(ids[j])

            new_groups.append(group)

        # build new group structures
        for idx, comp in enumerate(new_groups, start=1):

            texts = []
            fids = []

            for sid in comp:
                texts.append(kb.field_store[sid]["text"])
                fids.extend(
                    kb.field_store[sid].get("failure_ids", [])
                )

            refined_groups.append(
                {
                    "group_id": f"{field_type}_regroup_{idx:04d}",
                    "field_type": field_type,
                    "canonical_text": texts[0],
                    "member_node_ids": comp,
                    "variant_texts": texts,
                    "failure_ids": list(set(fids)),
                    "count": len(set(fids)),
                    "group_size": len(comp),
                }
            )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    output_path = group_json_path.parent / f"{field_type}_groups_refined_v2.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(refined_groups, f, indent=2, ensure_ascii=False)

    print("\n[INFO] Refinement + regroup completed.")
    print(output_path)

    return refined_groups

if __name__ == "__main__":
    field_list = [
        "element",
        "mode",
        "cause",
        "effect"
    ]
    # for field in field_list:
    #     result = compute_full_pairwise_similarity(persist_dir=KB_PATH,field_type=field)
    #     print(f"Field: {field}|{result}")

    # print_high_similarity_pairs(persist_dir=KB_PATH, field_type = "cause", similarity_threshold=0.85,max_print=70)

    merge_semantic_nodes_to_groups(
    persist_dir=KB_PATH,
    field_type="cause",
    similarity_threshold=0.7,
    )
    # retriever = GroupBM25Retriever(
    #     Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\process_KB\cause_groups.json")
    # )

    # results = retriever.query(
    #     query_text="short circuit",
    #     top_k=5
    # )

    # for r in results:
    #     print(r)
#     GROUP_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\process_KB\element_groups.json")
#     refine_and_regroup(
#     persist_dir=KB_PATH,
#     group_json_path=GROUP_PATH,
#     field_type="element",
#     embed_threshold=0.6,
#     lexical_threshold=0.0,
#     regroup_threshold=0.7,
# )