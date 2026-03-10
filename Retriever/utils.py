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

    # NEW: hybrid fields (safe read)
    hybrid_scores = res.get("hybrid_scores", [[]])
    hybrid_scores = hybrid_scores[0] if hybrid_scores else []

    for i, (sid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):

        sim = 1.0 - float(dist)
        gid = meta.get("group_id")

        hybrid_score = hybrid_scores[i-1] if i-1 < len(hybrid_scores) else None

        print(f"[{i}] sim={sim:.4f}  dist={float(dist):.4f}", end="")

        if hybrid_score is not None:
            print(f"  hybrid={hybrid_score:.4f}")
        else:
            print()

        print(f"  semantic_id : {sid}")
        # print(f"  field_type  : {meta.get('field_type')}")
        print(f"  discipline  : {meta.get('discipline')}")

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

def get_strong_semi_chains_from_query_combo_stats(
    results_graph: Dict[str, Any],
    *,
    min_count: int = 2,
    min_best_score: float = 0.5,
    keep_example: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    从 generate_query_unique_chains() 的输出 results_graph 中，
    筛选 query_combo_stats 里满足:
      - count >= min_count
      - best_score >= min_best_score
    的 MC / ME semi chain “统计项”。

    注意：
    - 这里返回的是 query_combo_stats 的条目（含 example），不是具体 unique_mc/unique_me 的链记录。
    - example 字段里会带 matched_mode/matched_cause/matched_effect 等，适合作为“代表性 semi chain”。

    Return:
      {
        "MC": [ {chain_type, query_combo, count, best_score, example}, ... ],
        "ME": [ ... ]
      }
    """
    stats = (results_graph or {}).get("query_combo_stats") or {}
    out: Dict[str, List[Dict[str, Any]]] = {"MC": [], "ME": []}

    for kind in ("MC", "ME"):
        rows = stats.get(kind) or []
        for r in rows:
            try:
                c = int(r.get("count", 0) or 0)
                s = float(r.get("best_score", 0.0) or 0.0)
            except Exception:
                continue

            if c < min_count or s < min_best_score:
                continue

            item = {
                "chain_type": kind,
                "query_combo": r.get("query_combo") or [],
                "count": c,
                "best_score": s,
            }
            if keep_example:
                item["example"] = r.get("example") or {}

            out[kind].append(item)

        # 默认按 (count, best_score) 降序
        out[kind].sort(key=lambda x: (int(x["count"]), float(x["best_score"])), reverse=True)

    return out

def rerank_strong_semi_chains_with_cross_encoder(
    results_graph: Dict[str, Any],
    cross_encoder_model: Any,
    *,
    min_count: int = 2,
    min_best_score: float = 0.5,
    ce_weight: float = 0.7,
    original_weight: float = 0.3,
    use_matched_pair_text: bool = True,
    sigmoid_ce_score: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    对 strong MC / ME semi-chains 做 cross-encoder 打分并 rerank。

    参数
    ----
    results_graph:
        generate_query_unique_chains(...) 的返回结果
    cross_encoder_model:
        需要有 predict(pairs) 方法
        其中 pairs 形如:
            [(text_a, text_b), ...]
    min_count:
        strong semi-chain 的最小 count
    min_best_score:
        strong semi-chain 的最小 best_score
    ce_weight:
        CE 分数权重
    original_weight:
        原始 best_score 权重
    use_matched_pair_text:
        True:
            MC 用 ("query_mode [SEP] query_cause", "matched_mode [SEP] matched_cause")
            ME 用 ("query_mode [SEP] query_effect", "matched_mode [SEP] matched_effect")
        False:
            MC 用 ("query_mode", "matched_mode") 与 ("query_cause", "matched_cause") 分别打分再平均
            ME 用 ("query_mode", "matched_mode") 与 ("query_effect", "matched_effect") 分别打分再平均
    sigmoid_ce_score:
        如果 cross_encoder_model.predict(...) 输出的是 logit，可设为 True 做 sigmoid

    返回
    ----
    {
        "MC": [... reranked rows ...],
        "ME": [... reranked rows ...],
    }
    """

    strong = get_strong_semi_chains_from_query_combo_stats(
        results_graph,
        min_count=min_count,
        min_best_score=min_best_score,
        keep_example=True,
    )

    mc_rows: List[Dict[str, Any]] = strong.get("MC", []) or []
    me_rows: List[Dict[str, Any]] = strong.get("ME", []) or []

    def _safe(x: Any) -> str:
        return "" if x is None else str(x).strip()

    def _sigmoid(x: float) -> float:
        import math
        return 1.0 / (1.0 + math.exp(-float(x)))

    def _get_qc(r: Dict[str, Any]) -> List[str]:
        return list(r.get("query_combo") or [])

    def _get_mc_qmode_qcause(r: Dict[str, Any]) -> Tuple[str, str]:
        qc = _get_qc(r)
        q_mode, q_cause = (qc + ["", ""])[:2]
        return _safe(q_mode), _safe(q_cause)

    def _get_me_qmode_qeffect(r: Dict[str, Any]) -> Tuple[str, str]:
        qc = _get_qc(r)
        q_mode, q_effect = (qc + ["", ""])[:2]
        return _safe(q_mode), _safe(q_effect)

    def _normalize_ce_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        vals = [float(s) for s in scores]
        if sigmoid_ce_score:
            vals = [_sigmoid(v) for v in vals]
            return vals

        # 若不是 sigmoid 模式，做一个稳妥的 min-max 到 [0,1]
        lo = min(vals)
        hi = max(vals)
        if abs(hi - lo) < 1e-12:
            return [0.5 for _ in vals]
        return [(v - lo) / (hi - lo) for v in vals]

    def _rerank_mc(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []

        if use_matched_pair_text:
            pairs: List[Tuple[str, str]] = []
            for r in rows:
                q_mode, q_cause = _get_mc_qmode_qcause(r)
                ex = r.get("example") or {}
                matched_mode = _safe(ex.get("matched_mode"))
                matched_cause = _safe(ex.get("matched_cause"))

                left = f"Failure mode: {q_mode}. Caused by: {q_cause}."
                right = f"Failure mode: {matched_mode}. Caused by: {matched_cause}."
                pairs.append((left, right))

            ce_scores_raw = list(cross_encoder_model.predict(pairs))
            ce_scores = _normalize_ce_scores([float(s) for s in ce_scores_raw])

            out: List[Dict[str, Any]] = []
            for r, ce_score in zip(rows, ce_scores):
                row = dict(r)
                row["ce_score"] = float(ce_score)
                row["rerank_score"] = (
                    float(original_weight) * float(r.get("best_score", 0.0))
                    + float(ce_weight) * float(ce_score)
                )
                out.append(row)

            out.sort(key=lambda x: (float(x["rerank_score"]), float(x.get("best_score", 0.0)), int(x.get("count", 0))), reverse=True)
            return out

        # 分别打 mode / cause 再平均
        pairs: List[Tuple[str, str]] = []
        pair_owner: List[Tuple[int, str]] = []

        for idx, r in enumerate(rows):
            q_mode, q_cause = _get_mc_qmode_qcause(r)
            ex = r.get("example") or {}
            matched_mode = _safe(ex.get("matched_mode"))
            matched_cause = _safe(ex.get("matched_cause"))

            pairs.append((q_mode, matched_mode))
            pair_owner.append((idx, "mode"))

            pairs.append((q_cause, matched_cause))
            pair_owner.append((idx, "cause"))

        ce_scores_raw = list(cross_encoder_model.predict(pairs))
        ce_scores = _normalize_ce_scores([float(s) for s in ce_scores_raw])

        agg: Dict[int, Dict[str, float]] = {}
        for (idx, tag), score in zip(pair_owner, ce_scores):
            agg.setdefault(idx, {})
            agg[idx][tag] = float(score)

        out = []
        for idx, r in enumerate(rows):
            mode_score = float((agg.get(idx) or {}).get("mode", 0.0))
            cause_score = float((agg.get(idx) or {}).get("cause", 0.0))
            ce_score = (mode_score + cause_score) / 2.0

            row = dict(r)
            row["ce_mode_score"] = mode_score
            row["ce_cause_score"] = cause_score
            row["ce_score"] = ce_score
            row["rerank_score"] = (
                float(original_weight) * float(r.get("best_score", 0.0))
                + float(ce_weight) * ce_score
            )
            out.append(row)

        out.sort(key=lambda x: (float(x["rerank_score"]), float(x.get("best_score", 0.0)), int(x.get("count", 0))), reverse=True)
        return out

    def _rerank_me(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []

        if use_matched_pair_text:
            pairs: List[Tuple[str, str]] = []
            for r in rows:
                q_mode, q_effect = _get_me_qmode_qeffect(r)
                ex = r.get("example") or {}
                matched_mode = _safe(ex.get("matched_mode"))
                matched_effect = _safe(ex.get("matched_effect"))

                left = f"Failure mode: {q_mode}. Leads to: {q_effect}."
                right = f"Failure mode: {matched_mode}. Leads to: {matched_effect}."
                pairs.append((left, right))

            ce_scores_raw = list(cross_encoder_model.predict(pairs))
            ce_scores = _normalize_ce_scores([float(s) for s in ce_scores_raw])

            out: List[Dict[str, Any]] = []
            for r, ce_score in zip(rows, ce_scores):
                row = dict(r)
                row["ce_score"] = float(ce_score)
                row["rerank_score"] = (
                    float(original_weight) * float(r.get("best_score", 0.0))
                    + float(ce_weight) * float(ce_score)
                )
                out.append(row)

            out.sort(key=lambda x: (float(x["rerank_score"]), float(x.get("best_score", 0.0)), int(x.get("count", 0))), reverse=True)
            return out

        # 分别打 mode / effect 再平均
        pairs: List[Tuple[str, str]] = []
        pair_owner: List[Tuple[int, str]] = []

        for idx, r in enumerate(rows):
            q_mode, q_effect = _get_me_qmode_qeffect(r)
            ex = r.get("example") or {}
            matched_mode = _safe(ex.get("matched_mode"))
            matched_effect = _safe(ex.get("matched_effect"))

            pairs.append((q_mode, matched_mode))
            pair_owner.append((idx, "mode"))

            pairs.append((q_effect, matched_effect))
            pair_owner.append((idx, "effect"))

        ce_scores_raw = list(cross_encoder_model.predict(pairs))
        ce_scores = _normalize_ce_scores([float(s) for s in ce_scores_raw])

        agg: Dict[int, Dict[str, float]] = {}
        for (idx, tag), score in zip(pair_owner, ce_scores):
            agg.setdefault(idx, {})
            agg[idx][tag] = float(score)

        out = []
        for idx, r in enumerate(rows):
            mode_score = float((agg.get(idx) or {}).get("mode", 0.0))
            effect_score = float((agg.get(idx) or {}).get("effect", 0.0))
            ce_score = (mode_score + effect_score) / 2.0

            row = dict(r)
            row["ce_mode_score"] = mode_score
            row["ce_effect_score"] = effect_score
            row["ce_score"] = ce_score
            row["rerank_score"] = (
                float(original_weight) * float(r.get("best_score", 0.0))
                + float(ce_weight) * ce_score
            )
            out.append(row)

        out.sort(key=lambda x: (float(x["rerank_score"]), float(x.get("best_score", 0.0)), int(x.get("count", 0))), reverse=True)
        return out

    return {
        "MC": _rerank_mc(mc_rows),
        "ME": _rerank_me(me_rows),
    }


def print_reranked_semi_chains(
    reranked: Dict[str, List[Dict[str, Any]]],
    *,
    max_rows: int = 25,
) -> None:
    def _print_rows(rows: List[Dict[str, Any]], title: str, is_mc: bool) -> None:
        print("\n" + "=" * 90)
        print(f"{title}: {len(rows)} (showing up to {max_rows})")
        print("=" * 90)

        for i, r in enumerate(rows[:max_rows], 1):
            qc = list(r.get("query_combo") or [])
            ex = r.get("example") or {}

            print(
                f"[{i:02d}] rerank_score={float(r.get('rerank_score', 0.0)):.6f}  "
                f"ce_score={float(r.get('ce_score', 0.0)):.6f}  "
                f"best_score={float(r.get('best_score', 0.0)):.6f}  "
                f"count={int(r.get('count', 0))}"
            )

            if is_mc:
                q_mode, q_cause = (qc + ["", ""])[:2]
                print(f"     query_mode : {q_mode}")
                print(f"     query_cause: {q_cause}")
                print("     --- example ---")
                print(f"     matched_mode : {ex.get('matched_mode','')}")
                print(f"     matched_cause: {ex.get('matched_cause','')}")
            else:
                q_mode, q_effect = (qc + ["", ""])[:2]
                print(f"     query_mode  : {q_mode}")
                print(f"     query_effect: {q_effect}")
                print("     --- example ---")
                print(f"     matched_mode  : {ex.get('matched_mode','')}")
                print(f"     matched_effect: {ex.get('matched_effect','')}")

            print()

    _print_rows(reranked.get("MC", []) or [], "RERANKED MC semi-chains", is_mc=True)
    _print_rows(reranked.get("ME", []) or [], "RERANKED ME semi-chains", is_mc=False)

