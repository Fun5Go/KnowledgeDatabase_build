from typing import Dict, List, Union, Optional, Any, Tuple, Set
from collections import defaultdict
from pathlib import Path
from Retriever.failure_query_tools import _load_kb,query_semantic_kb
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
from .PPL_score import ChainPPLEvaluator
import json
from .entity import structure_input_motorcontrol, structure_input_powertrain
import re
BASE_DIR = Path(__file__).resolve().parent


# -------------------------
# Scoring config
# -------------------------
FIELD_WEIGHTS = {
    "mode": 1.2,
    "cause": 1.0,
    "effect": 1.0,
}

COMPLETE_CHAIN_BONUS = 1.4
GRAPH_CONNECTION_WEIGHT = 0.3
GRAPH_EXPAND_DEFAULT_SIM = 0.6



def generate_query_unique_chains(
    persist_dir: Union[str, Path],
    structure_input: Dict[str, Any],
    *,
    # semantic retrieval
    min_count: Optional[int] = None,
    min_similarity: float = 0.75,
    source_type: Optional[str] = None,
    top_k_per_field: int = 5,
    # scoring
    field_weights: Optional[Dict[str, float]] = None,
    graph_connection_weight: float = 0.15,
    edge_count_weight: float = 0.02,
    # join controls
    enable_query_mode_join: bool = True,
    query_mode_join_scope: str = "node",   # "node" or "global"  (node更安全：node_id::qmode)
    query_mode_join_topk: int = 10,        # 限制 query-mode join 时每个 key 的 topK（防止笛卡尔积爆炸）
    # limits
    top_n_complete: int = 50,
    top_n_partial: int = 50,
    # persist
    save_query_json: bool = True,
    query_match_log_name: str = "query_match_log.json",
) -> Dict[str, Any]:
    """
    1) 从 query 候选 nodes 两两拼接出 semi chains：
         MC: mode->cause  (必须 KB edge 存在)
         ME: mode->effect (必须 KB edge 存在)
       并对 (mode_id,cause_id)/(mode_id,effect_id) 做 unique 去重：只保留 best(score)。

    2) 生成 complete chain (MCE)：
       - 优先按 semantic mode_id join（join_type="semantic_id"）
       - 若拼不上，再按 query_mode_key join（join_type="query_mode_text"）

    3) 拼不上保留 partial (semi one)。

    4) 统计 unique query text 组合出现次数：
         MC combo: (query_mode, query_cause)
         ME combo: (query_mode, query_effect)
         MCE combo:(query_mode, query_cause, query_effect)
       count 是在“候选拼接且 edge 存在 / join 成功”时累加。

    返回：
      {
        "complete_chains": [...],
        "partial_chains": [...],
        "query_combo_stats": {"MC":[...], "ME":[...], "MCE":[...]},
        "per_query_matches": {...},
      }

    Requirements:
      - _load_kb(persist_dir)
      - query_semantic_kb(persist_dir, text, field_type, ...)
    """

    # -------------------------
    # defaults
    # -------------------------
    if field_weights is None:
        field_weights = {"mode": 0.4, "cause": 0.3, "effect": 0.3}

    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)

    field_store = getattr(kb, "field_store", {}) or {}
    edge_store = getattr(kb, "edge_store", {}) or {}

    mode_to_cause = edge_store.get("mode_to_cause", {}) or {}   # {mode_id:{cause_id:count}}
    mode_to_effect = edge_store.get("mode_to_effect", {}) or {} # {mode_id:{effect_id:count}}

    def _safe(x) -> str:
        return "" if x is None else str(x).strip()

    def _id_to_text(sid: Optional[str]) -> str:
        if not sid:
            return ""
        return _safe((field_store.get(sid) or {}).get("text"))

    def _norm_query_mode(text: str) -> str:
        t = (text or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^\w\s\-]+", "", t)  # remove punctuation (keep letters/numbers/_/space/-)
        return t

    # debug/audit
    query_match_map: Dict[str, Any] = {}

    # -------------------------
    # query combo stats
    # -------------------------
    combo_mc: Dict[Tuple[str, str], Dict[str, Any]] = {}
    combo_me: Dict[Tuple[str, str], Dict[str, Any]] = {}
    combo_mce: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def _update_combo(store: Dict[Tuple, Dict[str, Any]], key: Tuple, score: float, example: Dict[str, Any]) -> None:
        """Count occurrences; keep best_score and best example."""
        if not key:
            return
        if any(not _safe(k) for k in key):
            return
        cur = store.get(key)
        if cur is None:
            store[key] = {"count": 1, "best_score": float(score), "example": example}
            return
        cur["count"] += 1
        if float(score) > float(cur.get("best_score", 0.0)):
            cur["best_score"] = float(score)
            cur["example"] = example

    # -------------------------
    # semantic retrieval
    # -------------------------
    def semantic_nodes(query_text: str, field: str) -> List[Dict[str, Any]]:
        qt = _safe(query_text)
        if not qt:
            return []

        res = query_semantic_kb(
            persist_dir,
            qt,
            field_type=field,
            n_results=top_k_per_field,
            min_count=min_count,
            source_type=source_type,
        ) or {}

        docs = (res.get("documents") or [[]])[0] or []
        metas = (res.get("metadatas") or [[]])[0] or []
        dists = (res.get("distances") or [[]])[0] or []
        ids = (res.get("ids") or [[]])[0] or []

        out: List[Dict[str, Any]] = []
        for doc, meta, dist, sid in zip(docs, metas, dists, ids):
            try:
                sim = 1.0 - float(dist)
            except Exception:
                continue

            if sim < min_similarity:
                continue

            semantic_id = _safe(sid) or _safe((meta or {}).get("semantic_id"))
            matched_text = _id_to_text(semantic_id) or _safe(doc)

            out.append(
                {
                    "semantic_id": semantic_id,
                    "matched_text": matched_text,  # KB/historical text
                    "similarity": float(sim),
                    "query_text": qt,              # original query text
                    "field": field,
                }
            )

        query_match_map[qt] = {
            "field": field,
            "matched": [
                {
                    "matched_text": n["matched_text"],
                    "semantic_id": n["semantic_id"],
                    "similarity": round(n["similarity"], 4),
                }
                for n in out
            ],
        }
        return out

    def dedup_nodes_keep_best(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """dedup by semantic_id; keep best similarity"""
        best: Dict[str, Dict[str, Any]] = {}
        for n in nodes:
            sid = n["semantic_id"]
            if sid not in best or n["similarity"] > best[sid]["similarity"]:
                best[sid] = n
        return list(best.values())

    # -------------------------
    # Unique pair keep best by score
    # -------------------------
    def upsert_best(best_map: Dict[Tuple, Dict[str, Any]], key: Tuple, record: Dict[str, Any]) -> None:
        old = best_map.get(key)
        if old is None or float(record.get("score", 0.0)) > float(old.get("score", 0.0)):
            best_map[key] = record

    # =========================================================
    # MAIN
    # =========================================================
    all_complete: List[Dict[str, Any]] = []
    all_partial: List[Dict[str, Any]] = []

    for node in structure_input.get("nodes", []) or []:
        node_id = _safe(node.get("element_id"))
        failure_element_text = _safe(node.get("failure_element"))

        modes_txt = node.get("modes") or []
        causes_txt = node.get("causes") or []
        effects_txt = node.get("effects") or []

        # 1) retrieve candidates
        candidate_modes = dedup_nodes_keep_best([n for t in modes_txt for n in semantic_nodes(t, "mode")])
        candidate_causes = dedup_nodes_keep_best([n for t in causes_txt for n in semantic_nodes(t, "cause")])
        candidate_effects = dedup_nodes_keep_best([n for t in effects_txt for n in semantic_nodes(t, "effect")])

        mode_map = {n["semantic_id"]: n for n in candidate_modes}
        cause_map = {n["semantic_id"]: n for n in candidate_causes}
        effect_map = {n["semantic_id"]: n for n in candidate_effects}

        # 2) build unique semi-chains by pair, keep best score
        best_me: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (mode_id, effect_id) -> record
        best_mc: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (mode_id, cause_id)  -> record

        # --- ME pairs (mode x effect) only if edge exists ---
        for mode_id, mode_node in mode_map.items():
            effects_dict = (mode_to_effect.get(mode_id) or {})
            if not effects_dict:
                continue

            for effect_id, effect_node in effect_map.items():
                if effect_id not in effects_dict:
                    continue

                edge_cnt = int(effects_dict.get(effect_id, 0) or 0)
                sim_mode = float(mode_node["similarity"])
                sim_effect = float(effect_node["similarity"])

                score = (
                    sim_mode * field_weights["mode"]
                    + sim_effect * field_weights["effect"]
                    + graph_connection_weight
                    + edge_count_weight * edge_cnt
                )

                q_mode = _safe(mode_node["query_text"])
                q_eff = _safe(effect_node["query_text"])

                # query mode key for fallback join
                qkey = _norm_query_mode(q_mode)
                if query_mode_join_scope == "node":
                    qkey = f"{node_id}::{qkey}"

                # ✅统计 query combo（ME）
                _update_combo(
                    combo_me,
                    (q_mode, q_eff),
                    score,
                    example={
                        "query_mode": q_mode,
                        "query_effect": q_eff,
                        "matched_mode": mode_node["matched_text"],
                        "matched_effect": effect_node["matched_text"],
                        "edge_count": edge_cnt,
                    },
                )

                rec = {
                    "chain_type": "ME",
                    "node_id": node_id,
                    "failure_element": failure_element_text,

                    "query_mode": q_mode,
                    "query_effect": q_eff,
                    "query_mode_key": qkey,

                    "mode": mode_node["matched_text"],
                    "effect": effect_node["matched_text"],

                    "mode_id": mode_id,
                    "effect_id": effect_id,
                    "edge_count": edge_cnt,

                    "mode_similarity": round(sim_mode, 6),
                    "effect_similarity": round(sim_effect, 6),
                    "graph_connections": 1,

                    "score": round(float(score), 6),
                }

                upsert_best(best_me, (mode_id, effect_id), rec)

        # --- MC pairs (mode x cause) only if edge exists ---
        for mode_id, mode_node in mode_map.items():
            causes_dict = (mode_to_cause.get(mode_id) or {})
            if not causes_dict:
                continue

            for cause_id, cause_node in cause_map.items():
                if cause_id not in causes_dict:
                    continue

                edge_cnt = int(causes_dict.get(cause_id, 0) or 0)
                sim_mode = float(mode_node["similarity"])
                sim_cause = float(cause_node["similarity"])

                score = (
                    sim_mode * field_weights["mode"]
                    + sim_cause * field_weights["cause"]
                    + graph_connection_weight
                    + edge_count_weight * edge_cnt
                )

                q_mode = _safe(mode_node["query_text"])
                q_cau = _safe(cause_node["query_text"])

                qkey = _norm_query_mode(q_mode)
                if query_mode_join_scope == "node":
                    qkey = f"{node_id}::{qkey}"

                # ✅统计 query combo（MC）
                _update_combo(
                    combo_mc,
                    (q_mode, q_cau),
                    score,
                    example={
                        "query_mode": q_mode,
                        "query_cause": q_cau,
                        "matched_mode": mode_node["matched_text"],
                        "matched_cause": cause_node["matched_text"],
                        "edge_count": edge_cnt,
                    },
                )

                rec = {
                    "chain_type": "MC",
                    "node_id": node_id,
                    "failure_element": failure_element_text,

                    "query_mode": q_mode,
                    "query_cause": q_cau,
                    "query_mode_key": qkey,

                    "mode": mode_node["matched_text"],
                    "cause": cause_node["matched_text"],

                    "mode_id": mode_id,
                    "cause_id": cause_id,
                    "edge_count": edge_cnt,

                    "mode_similarity": round(sim_mode, 6),
                    "cause_similarity": round(sim_cause, 6),
                    "graph_connections": 1,

                    "score": round(float(score), 6),
                }

                upsert_best(best_mc, (mode_id, cause_id), rec)

        unique_me = list(best_me.values())
        unique_mc = list(best_mc.values())

        # =====================================================
        # 3) JOIN to complete chains
        #    3.1 semantic mode_id join (always)
        #    3.2 query_mode_key join (optional fallback)
        # =====================================================
        me_by_mode_id: Dict[str, List[Dict[str, Any]]] = {}
        mc_by_mode_id: Dict[str, List[Dict[str, Any]]] = {}

        me_by_qmode: Dict[str, List[Dict[str, Any]]] = {}
        mc_by_qmode: Dict[str, List[Dict[str, Any]]] = {}

        for rec in unique_me:
            me_by_mode_id.setdefault(rec["mode_id"], []).append(rec)
            me_by_qmode.setdefault(rec.get("query_mode_key", ""), []).append(rec)

        for rec in unique_mc:
            mc_by_mode_id.setdefault(rec["mode_id"], []).append(rec)
            mc_by_qmode.setdefault(rec.get("query_mode_key", ""), []).append(rec)

        used_me = set()  # (mode_id,effect_id)
        used_mc = set()  # (mode_id,cause_id)

        complete: List[Dict[str, Any]] = []
        seen_complete = set()

        # 用于去重：同一 query 三元组，只保留最佳（尤其针对 query_mode_text join）
        best_complete_by_query: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        # key = (node_id, query_mode_key, query_cause, query_effect)
        # 注：query_mode_key 里已经包含 node_id::norm_mode（当 scope="node" 时），这里再带 node_id 是更保险


        def _complete_key(me: Dict[str, Any], mc: Dict[str, Any], join_type: str) -> Tuple[str, str, str, str]:
            q_mode = _safe(me.get("query_mode") or mc.get("query_mode"))
            q_cau = _safe(mc.get("query_cause"))
            q_eff = _safe(me.get("query_effect"))

            # 只对 query_mode_text join 做强去重（按 query 文本三元组）
            if join_type == "query_mode_text":
                qkey = _safe(me.get("query_mode_key") or mc.get("query_mode_key") or "")
                return (node_id, qkey, q_cau, q_eff)

            # semantic_id join 保持更细粒度（避免误合并）
            # 仍然用 id 作为 key
            return (
                _safe(me.get("mode_id")),
                _safe(mc.get("cause_id")),
                _safe(me.get("effect_id")),
                "semantic",
            )


        def _emit_complete(me: Dict[str, Any], mc: Dict[str, Any], join_type: str) -> None:
            mode_id = me.get("mode_id")
            effect_id = me.get("effect_id")
            cause_id = mc.get("cause_id")

            score = float(me["score"]) + float(mc["score"])

            q_mode = _safe(me.get("query_mode") or mc.get("query_mode"))
            q_cau = _safe(mc.get("query_cause"))
            q_eff = _safe(me.get("query_effect"))

            rec = {
                "chain_type": "MCE",
                "join_type": join_type,
                "node_id": node_id,
                "failure_element": failure_element_text,

                "query_mode": q_mode,
                "query_cause": q_cau,
                "query_effect": q_eff,

                "mode": me.get("mode", ""),
                "cause": mc.get("cause", ""),
                "effect": me.get("effect", ""),

                "mode_id": mode_id,
                "cause_id": cause_id,
                "effect_id": effect_id,

                "mode_similarity": me.get("mode_similarity"),
                "cause_similarity": mc.get("cause_similarity"),
                "effect_similarity": me.get("effect_similarity"),

                "edge_count_mc": mc.get("edge_count", 0),
                "edge_count_me": me.get("edge_count", 0),
                "graph_connections": 2,

                "score": round(float(score), 6),
            }

            key = _complete_key(me, mc, join_type)

            # ✅去重：同 key 只保留 score 更高的
            old = best_complete_by_query.get(key)
            if old is None or float(rec["score"]) > float(old.get("score", 0.0)):
                best_complete_by_query[key] = rec

        # -------- 3.1 semantic_id join --------
        common_mode_ids = set(me_by_mode_id.keys()) & set(mc_by_mode_id.keys())
        for mode_id in common_mode_ids:
            for me in me_by_mode_id.get(mode_id, []):
                for mc in mc_by_mode_id.get(mode_id, []):
                    _emit_complete(me, mc, join_type="semantic_id")

        # -------- 3.2 query_mode_key join (fallback) --------
        if enable_query_mode_join:
            common_qkeys = set(me_by_qmode.keys()) & set(mc_by_qmode.keys())
            for qkey in common_qkeys:
                if not qkey:
                    continue

                me_list = [me for me in me_by_qmode.get(qkey, []) if (me["mode_id"], me["effect_id"]) not in used_me]
                mc_list = [mc for mc in mc_by_qmode.get(qkey, []) if (mc["mode_id"], mc["cause_id"]) not in used_mc]

                if not me_list or not mc_list:
                    continue

                me_list.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
                mc_list.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

                me_list = me_list[: max(1, int(query_mode_join_topk))]
                mc_list = mc_list[: max(1, int(query_mode_join_topk))]

                for me in me_list:
                    for mc in mc_list:
                        _emit_complete(me, mc, join_type="query_mode_text")

        complete = list(best_complete_by_query.values())
        complete.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        complete = complete[:top_n_complete]
        all_complete.extend(complete)

        # used_me/used_mc 也需要在最终 complete 里更新（否则 partial 会不准）
        used_me.clear()
        used_mc.clear()
        for c in complete:
            used_me.add((c.get("mode_id"), c.get("effect_id")))
            used_mc.add((c.get("mode_id"), c.get("cause_id")))
        # =====================================================
        # 4) partial: semi chains not used in complete
        # =====================================================
        partial: List[Dict[str, Any]] = []

        for rec in unique_me:
            key = (rec["mode_id"], rec["effect_id"])
            if key in used_me:
                continue
            rec2 = dict(rec)
            rec2.setdefault("cause", "")
            rec2.setdefault("cause_id", None)
            rec2.setdefault("query_cause", "")
            partial.append(rec2)

        for rec in unique_mc:
            key = (rec["mode_id"], rec["cause_id"])
            if key in used_mc:
                continue
            rec2 = dict(rec)
            rec2.setdefault("effect", "")
            rec2.setdefault("effect_id", None)
            rec2.setdefault("query_effect", "")
            partial.append(rec2)

        partial.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        partial = partial[:top_n_partial]
        all_partial.extend(partial)

    # -------------------------
    # save query match log
    # -------------------------
    if save_query_json:
        save_path = BASE_DIR/ query_match_log_name
        save_path.write_text(json.dumps(query_match_map, indent=2, ensure_ascii=False), encoding="utf-8")

    # global sort + cut
    all_complete.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    all_partial.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    # pack combo stats to list
    def _pack_combo(store: Dict[Tuple, Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
        rows = []
        for k, v in store.items():
            rows.append(
                {
                    "chain_type": kind,
                    "query_combo": list(k),
                    "count": int(v.get("count", 0) or 0),
                    "best_score": float(v.get("best_score", 0.0) or 0.0),
                    "example": v.get("example") or {},
                }
            )
        rows.sort(key=lambda r: (r["count"], r["best_score"]), reverse=True)
        return rows

    query_combo_stats = {
        "MC": _pack_combo(combo_mc, "MC"),
        "ME": _pack_combo(combo_me, "ME"),
        "MCE": _pack_combo(combo_mce, "MCE"),
    }

    return {
        "complete_chains": all_complete[:top_n_complete],
        "partial_chains": all_partial[:top_n_partial],
        "query_combo_stats": query_combo_stats,
        "per_query_matches": query_match_map,
    }

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

def print_strong_semi_chains_3parts_cartesian(
    results_graph: Dict[str, Any],
    *,
    min_count: int = 2,
    min_best_score: float = 0.5,
    max_rows: int = 50,
) -> None:
    """
    分三部分打印（先按 count+best_score 过滤 strong，再做连接）：
    1) 完整 chain：同一 query_mode 下 MC × ME 组合（再去重）
    2) 剩余未参与任何完整 chain 的 MC
    3) 剩余未参与任何完整 chain 的 ME

    依赖：
      - get_strong_semi_chains_from_query_combo_stats(results_graph, ..., keep_example=True)
    """

    strong = get_strong_semi_chains_from_query_combo_stats(
        results_graph,
        min_count=min_count,
        min_best_score=min_best_score,
        keep_example=True,
    )

    mc_rows: List[Dict[str, Any]] = strong.get("MC", []) or []
    me_rows: List[Dict[str, Any]] = strong.get("ME", []) or []

    def _rank_key(r: Dict[str, Any]) -> Tuple[float, int]:
        # higher best_score first, then higher count
        return (float(r.get("best_score", 0.0)), int(r.get("count", 0)))

    def _get_qc(r: Dict[str, Any]) -> List[str]:
        return (r.get("query_combo") or [])

    def _get_mc_qmode_qcause(r: Dict[str, Any]) -> Tuple[str, str]:
        qc = _get_qc(r)
        q_mode, q_cause = (qc + ["", ""])[:2]
        return q_mode, q_cause

    def _get_me_qmode_qeffect(r: Dict[str, Any]) -> Tuple[str, str]:
        qc = _get_qc(r)
        q_mode, q_effect = (qc + ["", ""])[:2]
        return q_mode, q_effect

    # ---- group by query_mode ----
    mc_by_qmode: Dict[str, List[Dict[str, Any]]] = {}
    me_by_qmode: Dict[str, List[Dict[str, Any]]] = {}

    for r in mc_rows:
        q_mode, _ = _get_mc_qmode_qcause(r)
        if q_mode:
            mc_by_qmode.setdefault(q_mode, []).append(r)

    for r in me_rows:
        q_mode, _ = _get_me_qmode_qeffect(r)
        if q_mode:
            me_by_qmode.setdefault(q_mode, []).append(r)

    # ---- PART1: cartesian join within same query_mode, then dedupe ----
    # dedupe key: (query_mode, query_cause, query_effect, matched_mode, matched_cause, matched_effect)
    # (如果你更想只按 query_* 去重，把 key 改成 (query_mode, query_cause, query_effect) 即可)
    full_chains: List[Dict[str, Any]] = []
    full_keys: Set[Tuple[str, str, str, str, str, str]] = set()

    used_mc_ids: Set[int] = set()
    used_me_ids: Set[int] = set()

    common_modes = sorted(set(mc_by_qmode.keys()) & set(me_by_qmode.keys()))
    for q_mode in common_modes:
        mcs = sorted(mc_by_qmode[q_mode], key=_rank_key, reverse=True)
        mes = sorted(me_by_qmode[q_mode], key=_rank_key, reverse=True)

        for mc in mcs:
            q_mode_mc, q_cause = _get_mc_qmode_qcause(mc)
            ex_mc = mc.get("example") or {}

            for me in mes:
                q_mode_me, q_effect = _get_me_qmode_qeffect(me)
                ex_me = me.get("example") or {}

                matched_mode = (ex_mc.get("matched_mode") or ex_me.get("matched_mode") or "")
                matched_cause = (ex_mc.get("matched_cause") or "")
                matched_effect = (ex_me.get("matched_effect") or "")

                key = (q_mode, q_cause, q_effect, matched_mode, matched_cause, matched_effect)
                if key in full_keys:
                    continue
                full_keys.add(key)

                used_mc_ids.add(id(mc))
                used_me_ids.add(id(me))

                full_chains.append(
                    {
                        "query_mode": q_mode,
                        "query_cause": q_cause,
                        "query_effect": q_effect,
                        "mc_count": mc.get("count", 0),
                        "mc_best_score": mc.get("best_score", 0.0),
                        "me_count": me.get("count", 0),
                        "me_best_score": me.get("best_score", 0.0),
                        "matched_mode": matched_mode,
                        "matched_cause": matched_cause,
                        "matched_effect": matched_effect,
                        "mc_edge_count": ex_mc.get("edge_count", ""),
                        "me_edge_count": ex_me.get("edge_count", ""),
                    }
                )

    # optional: sort full chains for nicer display
    full_chains.sort(
        key=lambda d: (
            d.get("query_mode", ""),
            -float(d.get("mc_best_score", 0.0)),
            -int(d.get("mc_count", 0)),
            -float(d.get("me_best_score", 0.0)),
            -int(d.get("me_count", 0)),
            d.get("query_cause", ""),
            d.get("query_effect", ""),
        )
    )

    # ---- PART2/3: remaining unconnected rows (not used in any full chain) ----
    remaining_mc = [r for r in mc_rows if id(r) not in used_mc_ids]
    remaining_me = [r for r in me_rows if id(r) not in used_me_ids]

    # ---- printing helpers ----
    def _print_full() -> None:
        print("\n" + "=" * 90)
        print(f"PART 1) FULL CHAINS (MC×ME within same query_mode, deduped): {len(full_chains)} (showing up to {max_rows})")
        print("=" * 90)

        for i, fc in enumerate(full_chains[:max_rows], 1):
            print(f"[{i:02d}] query_mode  : {fc['query_mode']}")
            print(f"     query_cause : {fc['query_cause']}")
            print(f"     query_effect: {fc['query_effect']}")
            print(f"     MC: count={fc['mc_count']} best_score={float(fc['mc_best_score']):.6f} edge_count={fc['mc_edge_count']}")
            print(f"     ME: count={fc['me_count']} best_score={float(fc['me_best_score']):.6f} edge_count={fc['me_edge_count']}")
            print("     --- example (representative matched entities) ---")
            print(f"     matched_mode  : {fc['matched_mode']}")
            print(f"     matched_cause : {fc['matched_cause']}")
            print(f"     matched_effect: {fc['matched_effect']}")
            print()

    def _print_mc(rows: List[Dict[str, Any]]) -> None:
        print("\n" + "=" * 90)
        print(f"PART 2) UNCONNECTED MC semi-chains: {len(rows)} (showing up to {max_rows})")
        print("=" * 90)

        for i, r in enumerate(rows[:max_rows], 1):
            q_mode, q_cause = _get_mc_qmode_qcause(r)
            ex = r.get("example") or {}
            print(f"[{i:02d}] count={r.get('count',0)}  best_score={float(r.get('best_score',0.0)):.6f}")
            print(f"     query_mode : {q_mode}")
            print(f"     query_cause: {q_cause}")
            if ex:
                print("     --- example ---")
                print(f"     matched_mode : {ex.get('matched_mode','')}")
                print(f"     matched_cause: {ex.get('matched_cause','')}")
                print(f"     edge_count   : {ex.get('edge_count','')}")
            print()

    def _print_me(rows: List[Dict[str, Any]]) -> None:
        print("\n" + "=" * 90)
        print(f"PART 3) UNCONNECTED ME semi-chains: {len(rows)} (showing up to {max_rows})")
        print("=" * 90)

        for i, r in enumerate(rows[:max_rows], 1):
            q_mode, q_effect = _get_me_qmode_qeffect(r)
            ex = r.get("example") or {}
            print(f"[{i:02d}] count={r.get('count',0)}  best_score={float(r.get('best_score',0.0)):.6f}")
            print(f"     query_mode  : {q_mode}")
            print(f"     query_effect: {q_effect}")
            if ex:
                print("     --- example ---")
                print(f"     matched_mode  : {ex.get('matched_mode','')}")
                print(f"     matched_effect: {ex.get('matched_effect','')}")
                print(f"     edge_count    : {ex.get('edge_count','')}")
            print()

    _print_full()
    _print_mc(remaining_mc)
    _print_me(remaining_me)

if  __name__ == "__main__":


    from pprint import pprint

    # -----------------------------------------------------
    # 1) KB Path
    # -----------------------------------------------------
    KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_miniLM\failure_kb"
    )

    # -----------------------------------------------------
    # 2) Structure Input

    FIELD_WEIGHTS = {
    "element":0.2,
    "mode": 1.2,
    "cause": 1.0,
    "effect": 1.0,
}
    def _fmt(x: Any) -> str:
        return "" if x is None else str(x)


    def print_query_chain_results(
        results: Dict[str, Any],
        *,
        # combo stats
        top_query_combos: int = 30,
        # chains
        top_complete: int = 20,
        top_partial: int = 30,
        # flags
        show_ids: bool = False,
        show_edge_counts: bool = True,
        show_join_type: bool = True,
    ) -> None:
        """
        Pretty print output of generate_query_unique_chains()

        Expects results dict keys:
        - complete_chains
        - partial_chains
        - query_combo_stats: {"MC":[...], "ME":[...], "MCE":[...]}
        """

        complete: List[Dict[str, Any]] = results.get("complete_chains", []) or []
        partial: List[Dict[str, Any]] = results.get("partial_chains", []) or []
        combo_stats: Dict[str, List[Dict[str, Any]]] = results.get("query_combo_stats", {}) or {}

        print("\n" + "=" * 92)
        print(f"✅ QUERY CHAIN RESULTS  |  complete={len(complete)}  partial={len(partial)}")
        print("=" * 92)

        # =========================================================
        # 1) UNIQUE QUERY COMBO STATS
        # =========================================================
        def _print_combos(kind: str, rows: List[Dict[str, Any]]) -> None:
            print("\n" + "-" * 92)
            print(f"📊 UNIQUE QUERY COMBOS ({kind})  Top {min(top_query_combos, len(rows))}/{len(rows)}")
            print("-" * 92)

            for i, r in enumerate(rows[:top_query_combos], 1):
                combo = r.get("query_combo") or []
                cnt = int(r.get("count", 0) or 0)
                best = float(r.get("best_score", 0.0) or 0.0)
                ex = r.get("example") or {}

                print(f"\n[{i:02d}] count={cnt} | best_score={best:.6f}")

                if kind == "MC":
                    # combo: [mode, cause]
                    print(f"  Query combo : MODE='{_fmt(combo[0])}'  CAUSE='{_fmt(combo[1])}'")
                    print(
                        f"  Example    : MODE='{_fmt(ex.get('matched_mode'))}'  "
                        f"CAUSE='{_fmt(ex.get('matched_cause'))}'  edge={_fmt(ex.get('edge_count'))}"
                    )
                elif kind == "ME":
                    # combo: [mode, effect]
                    print(f"  Query combo : MODE='{_fmt(combo[0])}'  EFFECT='{_fmt(combo[1])}'")
                    print(
                        f"  Example    : MODE='{_fmt(ex.get('matched_mode'))}'  "
                        f"EFFECT='{_fmt(ex.get('matched_effect'))}'  edge={_fmt(ex.get('edge_count'))}"
                    )
                else:
                    # MCE combo: [mode, cause, effect]
                    print(
                        f"  Query combo : MODE='{_fmt(combo[0])}'  "
                        f"CAUSE='{_fmt(combo[1])}'  EFFECT='{_fmt(combo[2])}'"
                    )
                    print(
                        f"  Example    : MODE='{_fmt(ex.get('matched_mode'))}'  "
                        f"CAUSE='{_fmt(ex.get('matched_cause'))}'  EFFECT='{_fmt(ex.get('matched_effect'))}'  "
                        f"MC={_fmt(ex.get('edge_count_mc'))} ME={_fmt(ex.get('edge_count_me'))}"
                    )

        if combo_stats:
            _print_combos("MC", combo_stats.get("MC", []) or [])
            _print_combos("ME", combo_stats.get("ME", []) or [])
            _print_combos("MCE", combo_stats.get("MCE", []) or [])
        else:
            print("\n" + "-" * 92)
            print("📊 UNIQUE QUERY COMBOS: (no combo stats found in results)")
            print("-" * 92)

        # =========================================================
        # 2) COMPLETE CHAINS (MCE)
        # =========================================================
        print("\n" + "-" * 92)
        print(f"🧩 COMPLETE CHAINS (MCE)  Top {min(top_complete, len(complete))}/{len(complete)}")
        print("-" * 92)

        for i, c in enumerate(complete[:top_complete], 1):
            score = float(c.get("score", 0.0) or 0.0)

            node_id = _fmt(c.get("node_id"))
            element = _fmt(c.get("failure_element"))
            join_type = _fmt(c.get("join_type"))  # semantic_id / query_mode_text

            print(f"\n[{i:02d}] score={score:.6f} | node_id={node_id} | element={element}")
            if show_join_type:
                print(f"  Join    : {join_type or 'semantic_id'}")

            print("  Query   :")
            print(f"    MODE  : {_fmt(c.get('query_mode'))}")
            print(f"    CAUSE : {_fmt(c.get('query_cause'))}")
            print(f"    EFFECT: {_fmt(c.get('query_effect'))}")

            print("  Matched :")
            print(f"    MODE  : {_fmt(c.get('mode'))}")
            print(f"    CAUSE : {_fmt(c.get('cause'))}")
            print(f"    EFFECT: {_fmt(c.get('effect'))}")

            if show_edge_counts:
                print(f"  Edges   : MC={_fmt(c.get('edge_count_mc'))}  ME={_fmt(c.get('edge_count_me'))}")

            if show_ids:
                print(
                    f"  IDs     : mode_id={_fmt(c.get('mode_id'))}  "
                    f"cause_id={_fmt(c.get('cause_id'))}  effect_id={_fmt(c.get('effect_id'))}"
                )

        # =========================================================
        # 3) PARTIAL CHAINS
        # =========================================================
        mc = [p for p in partial if p.get("chain_type") == "MC"]
        me = [p for p in partial if p.get("chain_type") == "ME"]

        def _print_partial(title: str, items: List[Dict[str, Any]], kind: str) -> None:
            print("\n" + "-" * 92)
            print(f"🧩 {title}  Top {min(top_partial, len(items))}/{len(items)}")
            print("-" * 92)

            for i, p in enumerate(items[:top_partial], 1):
                score = float(p.get("score", 0.0) or 0.0)
                node_id = _fmt(p.get("node_id"))
                element = _fmt(p.get("failure_element"))
                edge_cnt = p.get("edge_count", None)

                print(f"\n[{i:02d}] score={score:.6f} | node_id={node_id} | element={element}")

                if kind == "MC":
                    print("  Query   :")
                    print(f"    MODE  : {_fmt(p.get('query_mode'))}")
                    print(f"    CAUSE : {_fmt(p.get('query_cause'))}")
                    print("  Matched :")
                    print(f"    MODE  : {_fmt(p.get('mode'))}")
                    print(f"    CAUSE : {_fmt(p.get('cause'))}")
                else:
                    print("  Query   :")
                    print(f"    MODE  : {_fmt(p.get('query_mode'))}")
                    print(f"    EFFECT: {_fmt(p.get('query_effect'))}")
                    print("  Matched :")
                    print(f"    MODE  : {_fmt(p.get('mode'))}")
                    print(f"    EFFECT: {_fmt(p.get('effect'))}")

                if show_edge_counts and edge_cnt is not None:
                    print(f"  EdgeCnt : {int(edge_cnt)}")

                if show_ids:
                    if kind == "MC":
                        print(f"  IDs     : mode_id={_fmt(p.get('mode_id'))}  cause_id={_fmt(p.get('cause_id'))}")
                    else:
                        print(f"  IDs     : mode_id={_fmt(p.get('mode_id'))}  effect_id={_fmt(p.get('effect_id'))}")

        _print_partial("PARTIAL CHAINS (MC: mode → cause)", mc, "MC")
        _print_partial("PARTIAL CHAINS (ME: mode → effect)", me, "ME")

        print("\n" + "=" * 92)
        print("✅ End of results")
        print("=" * 92 + "\n")

    results_graph = generate_query_unique_chains(persist_dir=KB_PATH,structure_input=structure_input_powertrain,save_query_json=False,
                                                                  min_similarity=0.3,top_k_per_field=50)
    # print_query_chain_results(
    #     results_graph,
    #     top_query_combos=30,
    #     top_complete=20,
    #     top_partial=30,
    #     show_ids=False,        # True if you want semantic IDs printed
    #     show_edge_counts=True,
    # )
    print_strong_semi_chains_3parts_cartesian(results_graph, min_count=2, min_best_score=0.5, max_rows=15)