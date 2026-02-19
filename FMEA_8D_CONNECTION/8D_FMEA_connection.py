import json
import math
from pathlib import Path
from typing import Dict, Any, List, Union
from collections import defaultdict
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
import chromadb
from chromadb.utils import embedding_functions
import json
from pathlib import Path
from typing import Any, Dict, List, Optional



# =========================================================
# CONFIG
# =========================================================

ENTITY_WEIGHT = {
    "mode": 1.5,
    "cause": 1.5,
    "effect": 1.2,
    "element": 1.5,
}

ROLE_ORDER = [
    "element",
    "mode",
    "effect",
    "cause",
]


# =========================================================
# Load Sentence Collection
# =========================================================

def get_sentence_collection(
    persist_dir: Union[str, Path],
    collection_name: str = "sentences",
):
    client = chromadb.PersistentClient(path=str(persist_dir))

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )


# =========================================================
# MATCH SINGLE FAILURE
# =========================================================

def match_fmea_failure_to_8d(
    *,
    fmea_kb,
    sentence_collection,
    fmea_failure_json: dict,
    n_results_each: int = 20,
    min_similarity: float = 0.65,
    case_score_threshold: float = 2.0,
    group_by: str = "case_id",
    debug: bool = False,
):

    agg = defaultdict(lambda: {
        "group_id": None,
        "raw_score": 0.0,
        "hit_count": 0,
        "matched_roles": set(),
        "hits": [],
    })

    for role in ROLE_ORDER:

        semantic_id = fmea_failure_json.get(role + "_id")
        if not semantic_id:
            continue
        if debug:
            print(f"\n---- Role: {role} | semantic_id: {semantic_id}")

        # --- get embedding from FMEA KB ---
        res = fmea_kb.collection.get(
            ids=[semantic_id],
            include=["embeddings"]
        )

        embs = res.get("embeddings", None)
        if embs is None or len(embs) == 0 or embs[0] is None:
            continue

        entity_vector = embs[0]
        weight = ENTITY_WEIGHT.get(role, 1.0)

        # --- query sentence KB ---
        result = sentence_collection.query(
            query_embeddings=[entity_vector],
            n_results=n_results_each,
            include=["documents", "metadatas", "distances"],
        )

        ids0 = result.get("ids", [[]])[0]
        docs0 = result.get("documents", [[]])[0]
        metas0 = result.get("metadatas", [[]])[0]
        dists0 = result.get("distances", [[]])[0]
        if debug:
            print(f"Returned sentences: {len(ids0)}")

        for sid, doc, meta, dist in zip(ids0, docs0, metas0, dists0):

            gid = meta.get(group_by)
            if not gid:
                continue

            # cosine similarity
            sim = 1 - float(dist)

            if sim < min_similarity:
                continue

            score = weight * sim
            if debug and sim >= min_similarity:
                print(f"  hit case={gid} sim={sim:.3f}")

            a = agg[gid]
            a["group_id"] = gid
            a["raw_score"] += score
            a["hit_count"] += 1
            a["matched_roles"].add(role)

            a["hits"].append({
                "case_id": gid,
                "sentence_id": sid,
                "role": role,
                "similarity": float(sim),
                "distance": float(dist),
                "text": (doc or "").strip(),
                "meta": meta or {},
            })

    # =====================================================
    # Final scoring
    # =====================================================

    final_results = []

    for case_id, a in agg.items():

        coverage = len(a["matched_roles"]) / len(ROLE_ORDER)
        coverage_bonus = 0.3 * coverage
        density_bonus = 0.1 * math.log(1 + a["hit_count"])

        final_score = a["raw_score"] * (1 + coverage_bonus + density_bonus)

        if final_score < case_score_threshold:
            continue

        a["score"] = final_score
        a["coverage"] = coverage

        final_results.append(a)
        if debug:
            print(f"\nCase {case_id}:")
            print(f"  raw_score={a['raw_score']:.3f}")
            print(f"  hit_count={a['hit_count']}")
            print(f"  coverage={coverage:.2f}")
            print(f"  final_score={final_score:.3f}")

    final_results.sort(key=lambda x: x["score"], reverse=True)

    return final_results


# =========================================================
# BATCH MATCH
# =========================================================

def batch_match_fmea_to_8d(
    *,
    fmea_kb,
    sentence_collection,
    fmea_failure_list,
    verbose_progress: bool = True,
    **kwargs,
):
    all_results = {}
    failure_map = {}

    total = len(fmea_failure_list)
    for idx, failure in enumerate(fmea_failure_list, start=1):
        failure_id = failure.get("failure_id")
        if not failure_id:
            continue

        failure_map[failure_id] = failure  # raw failure json

        if verbose_progress:
            print(f"[{idx}/{total}] Matching failure: {failure_id}")

        matches = match_fmea_failure_to_8d(
            fmea_kb=fmea_kb,
            sentence_collection=sentence_collection,
            fmea_failure_json=failure,
            **kwargs,
        )

        all_results[failure_id] = matches

    return all_results, failure_map



def print_connected_results(
    results: Dict[str, List[Dict[str, Any]]],
    *,
    fmea_kb=None,                         # ✅ 新增：用于取 semantic text
    failure_map: Dict[str, dict] = None,  # ✅ 新增：failure_id -> raw failure json
    top_failures: int = 999999,
    top_cases_each: int = 3,
    top_hits_each: int = 5,
    min_hit_sim: float = None,
    show_text_chars: int = 220,
    show_meta_keys: List[str] = None,
):
    """
    results: {failure_id: [case_match_dicts...]}
    在原来打印基础上：每条 hit 额外打印该 role 对应的 semantic_id + semantic_text
    """

    if show_meta_keys is None:
        show_meta_keys = [
            "case_id", "failure_id", "cause_id",
            "sentence_role", "source_section",
            "productPnID", "product_domain",
            "released_year", "status", "subject",
        ]

    # ---- semantic text cache: semantic_id -> text ----
    semantic_cache = {}

    def _get_semantic_text(semantic_id: str) -> str:
        if not semantic_id or fmea_kb is None:
            return ""
        if semantic_id in semantic_cache:
            return semantic_cache[semantic_id]

        text = ""
        try:
            r = fmea_kb.collection.get(ids=[semantic_id], include=["documents", "metadatas"])
            doc = (r.get("documents") or [None])[0]
            if doc:
                text = str(doc).strip()
            else:
                meta = (r.get("metadatas") or [None])[0] or {}
                # 兜底：如果你的 KB 文本在 meta 里
                for k in ("text", "name", "label", "content", "semantic_text"):
                    if meta.get(k):
                        text = str(meta[k]).strip()
                        break
        except Exception:
            text = ""

        semantic_cache[semantic_id] = text
        return text

    # 仅保留有匹配的
    connected = [(fid, ms) for fid, ms in results.items() if ms]
    connected.sort(key=lambda x: x[1][0].get("score", 0.0), reverse=True)

    print("\n" + "=" * 100)
    print(f"Connected failures: {len(connected)} / {len(results)}")
    print("=" * 100)

    for idx, (failure_id, matches) in enumerate(connected[:top_failures], start=1):
        print("\n" + "#" * 100)
        print(f"[{idx}] Failure: {failure_id}  |  matched_cases: {len(matches)}")
        print("#" * 100)

        # ✅ 取出该 failure 的原始 json（用于 role -> semantic_id）
        failure_json = (failure_map or {}).get(failure_id, {}) or {}

        for ci, case in enumerate(matches[:top_cases_each], start=1):
            case_id = case.get("group_id")
            score = case.get("score", 0.0)
            raw = case.get("raw_score", 0.0)
            hits = case.get("hit_count", 0)
            cov = case.get("coverage", 0.0)

            print(f"\n  ({ci}) Case: {case_id} | score={score:.3f} raw={raw:.3f} hits={hits} coverage={cov:.2f}")

            # 命中句子按 similarity 排序打印
            hit_list = case.get("hits", []) or []
            hit_list = sorted(hit_list, key=lambda h: h.get("similarity", 0.0), reverse=True)

            printed = 0
            for hi, h in enumerate(hit_list, start=1):
                sim = h.get("similarity", 0.0)
                if (min_hit_sim is not None) and (sim < min_hit_sim):
                    continue

                text = (h.get("text") or "").replace("\n", " ").strip()
                if len(text) > show_text_chars:
                    text = text[:show_text_chars] + "..."

                meta = h.get("meta") or {}
                meta_small = {k: meta.get(k) for k in show_meta_keys if k in meta and meta.get(k) is not None}

                role = h.get("role")
                sid = h.get("sentence_id")

                # ✅ 新增：role -> semantic_id/text
                semantic_id = failure_json.get(role + "_id") if role else None
                semantic_text = _get_semantic_text(semantic_id)
                semantic_text_show = (semantic_text or "").replace("\n", " ").strip()
                if len(semantic_text_show) > show_text_chars:
                    semantic_text_show = semantic_text_show[:show_text_chars] + "..."

                print(f"      - [{hi:02d}] sim={sim:.3f} role={role} sid={sid}")

                # ✅ 新增 semantic 输出（不影响原打印）
                if semantic_id:
                    print(f"           semantic_id={semantic_id}")
                if semantic_text_show:
                    print(f"           semantic_text={semantic_text_show}")

                if meta_small:
                    print(f"           meta={meta_small}")
                print(f"           text={text}")

                printed += 1
                if printed >= top_hits_each:
                    break

            if printed == 0:
                print("      (no hits printed under current filters)")



def _to_jsonable(x: Any):
    """Convert numpy/scalar/set types to JSON-serializable Python types."""
    if isinstance(x, set):
        return sorted(list(x))
    # numpy scalar
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass
    return x


def sanitize_match_results(
    results: Dict[str, List[Dict[str, Any]]],
    *,
    fmea_kb=None,
    failure_map: Dict[str, dict] = None,
    only_connected: bool = True,
    top_cases_each: Optional[int] = 3,
    top_hits_each: Optional[int] = 8,
    drop_text: bool = False,
    keep_meta_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:

    if keep_meta_keys is None:
        keep_meta_keys = [
            "case_id", "failure_id", "cause_id",
            "sentence_role", "source_section",
            "productPnID", "product_domain",
            "released_year", "status", "subject",
        ]

    semantic_cache = {}

    def _get_semantic_text(semantic_id: str) -> str:
        if not semantic_id or fmea_kb is None:
            return ""

        if semantic_id in semantic_cache:
            return semantic_cache[semantic_id]

        text = ""
        try:
            r = fmea_kb.collection.get(
                ids=[semantic_id],
                include=["documents", "metadatas"]
            )
            doc = (r.get("documents") or [None])[0]
            if doc:
                text = str(doc).strip()
            else:
                meta = (r.get("metadatas") or [None])[0] or {}
                for k in ("text", "name", "label", "content", "semantic_text"):
                    if meta.get(k):
                        text = str(meta[k]).strip()
                        break
        except Exception:
            text = ""

        semantic_cache[semantic_id] = text
        return text

    out = {}

    for failure_id, matches in results.items():

        if only_connected and not matches:
            continue

        failure_json = (failure_map or {}).get(failure_id, {}) or {}

        ms = matches or []
        if top_cases_each is not None:
            ms = ms[:top_cases_each]

        new_ms = []

        for m in ms:
            m2 = dict(m)

            # set -> list
            if "matched_roles" in m2:
                m2["matched_roles"] = sorted(list(m2["matched_roles"]))

            hits = m2.get("hits", []) or []
            hits = sorted(hits, key=lambda h: float(h.get("similarity", 0.0)), reverse=True)

            if top_hits_each is not None:
                hits = hits[:top_hits_each]

            new_hits = []

            for h in hits:
                h2 = dict(h)

                role = h2.get("role")
                semantic_id = failure_json.get(role + "_id") if role else None
                semantic_text = _get_semantic_text(semantic_id)

                # 附加 semantic 信息
                h2["semantic_id"] = semantic_id
                h2["semantic_text"] = semantic_text

                # 限制 meta 字段
                meta = h2.get("meta") or {}
                if keep_meta_keys:
                    meta = {
                        k: meta.get(k)
                        for k in keep_meta_keys
                        if meta.get(k) is not None
                    }
                h2["meta"] = meta

                if drop_text:
                    h2.pop("text", None)

                new_hits.append(h2)

            m2["hits"] = new_hits
            new_ms.append(m2)

        out[failure_id] = new_ms

    return out



def export_results_to_json(
    results,
    output_path: Path,
    *,
    fmea_kb=None,
    failure_map=None,
    only_connected=True,
    top_cases_each=3,
    top_hits_each=8,
    drop_text=False,
):
    payload = sanitize_match_results(
        results,
        fmea_kb=fmea_kb,
        failure_map=failure_map,
        only_connected=only_connected,
        top_cases_each=top_cases_each,
        top_hits_each=top_hits_each,
        drop_text=drop_text,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n✅ Exported JSON: {output_path}  (failures={len(payload)})")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    FAILURE_KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb"
    )

    SENTENCE_KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\sentence_kb"
    )

    # --- Initialize FMEA KB ---
    fmea_kb = FMEAFailureKB(FAILURE_KB_PATH)

    # --- Load sentence collection ---
    sentence_collection = get_sentence_collection(
        persist_dir=SENTENCE_KB_PATH,
        collection_name="sentences",
    )

    # --- Load FMEA failures ---
    fmea_failures = list(fmea_kb.entity_store.values())

    # --- Run matching ---
    results,failure_map = batch_match_fmea_to_8d(
        fmea_kb=fmea_kb,
        sentence_collection=sentence_collection,
        fmea_failure_list=fmea_failures,
        n_results_each=25,
        min_similarity=0.55,
        case_score_threshold=2.0,
        verbose_progress=False,  
    )
    print_connected_results(
        results,
        fmea_kb=fmea_kb,
        failure_map=failure_map,
        top_cases_each=3,
        top_hits_each=6,
        min_hit_sim=0.65,
    )
    export_results_to_json(
        results,
        output_path=Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\fmea_8d_matches_with_semantic_v2.json"),
        fmea_kb=fmea_kb,
        failure_map=failure_map,
        only_connected=True,
        top_cases_each=5,
        top_hits_each=10,
    )