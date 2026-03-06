from typing import List, Dict, Any, Tuple, Set
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..graph_query import generate_query_unique_chains, get_strong_semi_chains_from_query_combo_stats
from ..entity import structure_input_powertrain, structure_input_motorcontrol

# 你已有
# def normalize_text(s: str) -> str: ...
def normalize_text(x: str) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split()).lower()

def load_gt(gt_path: Path, target_element: str) -> List[Dict]:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    target_norm = normalize_text(target_element)
    is_motor_control = target_norm == normalize_text("Motor control")

    gt_list = []
    for v in gt_raw.values():
        if normalize_text(v.get("failure_element_text")) != target_norm:
            continue
        if is_motor_control and v.get("productPnID") != 287883:
            continue
        gt_list.append(v)

    return gt_list


def build_gt_unique_semi_chains(gt_list: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
    """
    GT 去重规则同 pred：
      MC: mode + cause
      ME: mode + effect
    """
    out = {"MC": [], "ME": []}
    seen_mc = set()
    seen_me = set()

    for g in gt_list:
        mode   = (g.get("failure_mode_text") or "").strip()
        cause  = (g.get("failure_cause_text") or "").strip()
        effect = (g.get("failure_effect_text") or "").strip()

        nm = normalize_text(mode)
        if not nm:
            continue

        nc = normalize_text(cause)
        if nc:
            sig = (nm, nc)
            if sig not in seen_mc:
                seen_mc.add(sig)
                out["MC"].append({"failure_mode": mode, "failure_cause": cause})

        ne = normalize_text(effect)
        if ne:
            sig = (nm, ne)
            if sig not in seen_me:
                seen_me.add(sig)
                out["ME"].append({"failure_mode": mode, "failure_effect": effect})

    return out

def build_pred_from_strong_semi_stats(
    strong: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    strong: get_strong_semi_chains_from_query_combo_stats() 的输出

    Return:
      {
        "MC": [{"failure_mode":..., "failure_cause":..., "count":..., "best_score":...}, ...],
        "ME": [{"failure_mode":..., "failure_effect":..., "count":..., "best_score":...}, ...],
      }

    去重规则（你要求的）：只要 query_mode + query_cause/effect 一样就算重复
      MC: (normalize(mode), normalize(cause))
      ME: (normalize(mode), normalize(effect))
    """
    out = {"MC": [], "ME": []}

    seen_mc = set()
    seen_me = set()

    for kind in ("MC", "ME"):
        for r in (strong.get(kind) or []):
            combo = r.get("query_combo") or []
            if not isinstance(combo, list) or len(combo) < 2:
                continue

            mode = (combo[0] or "").strip()
            other = (combo[1] or "").strip()

            nm = normalize_text(mode)
            no = normalize_text(other)
            if not nm or not no:
                continue

            if kind == "MC":
                sig = (nm, no)
                if sig in seen_mc:
                    continue
                seen_mc.add(sig)
                out["MC"].append({
                    "failure_mode": mode,
                    "failure_cause": other,
                    "count": int(r.get("count", 0) or 0),
                    "best_score": float(r.get("best_score", 0.0) or 0.0),
                })
            else:
                sig = (nm, no)
                if sig in seen_me:
                    continue
                seen_me.add(sig)
                out["ME"].append({
                    "failure_mode": mode,
                    "failure_effect": other,
                    "count": int(r.get("count", 0) or 0),
                    "best_score": float(r.get("best_score", 0.0) or 0.0),
                })

    return out

from typing import Any, Dict, List, Tuple, Set

def build_pred_three_parts_from_results_graph(
    results_graph: Dict[str, Any],
    *,
    min_count: int = 2,
    min_best_score: float = 0.5,
    keep_example: bool = True,
) -> Dict[str, Any]:
    """
    严格复刻 print_strong_semi_chains_3parts_cartesian 的 Mode B 逻辑，
    但不打印，而是返回三部分数据：

    Return:
      {
        "strong": strong,
        "mc_rows": [...],                  # strong["MC"]
        "me_rows": [...],                  # strong["ME"]
        "full_chains": [...],              # PART 1
        "remaining_mc_rows": [...],        # PART 2
        "remaining_me_rows": [...],        # PART 3
      }
    """

    # 1) strong semi chains（同 print）
    strong = get_strong_semi_chains_from_query_combo_stats(
        results_graph,
        min_count=min_count,
        min_best_score=min_best_score,
        keep_example=keep_example,
    )

    mc_rows: List[Dict[str, Any]] = strong.get("MC", []) or []
    me_rows: List[Dict[str, Any]] = strong.get("ME", []) or []

    # 2) helpers（同 print）
    def _rank_key(r: Dict[str, Any]) -> Tuple[float, int]:
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

    # 3) group by query_mode（同 print）
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

    # 4) PART1: cartesian join + dedupe（同 print）
    full_chains: List[Dict[str, Any]] = []
    full_keys: Set[Tuple[str, str, str, str, str, str]] = set()

    used_mc_ids: Set[int] = set()
    used_me_ids: Set[int] = set()

    common_modes = sorted(set(mc_by_qmode.keys()) & set(me_by_qmode.keys()))
    for q_mode in common_modes:
        mcs = sorted(mc_by_qmode[q_mode], key=_rank_key, reverse=True)
        mes = sorted(me_by_qmode[q_mode], key=_rank_key, reverse=True)

        for mc in mcs:
            _, q_cause = _get_mc_qmode_qcause(mc)
            ex_mc = mc.get("example") or {}

            for me in mes:
                _, q_effect = _get_me_qmode_qeffect(me)
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

    # 5) 排序（同 print）
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

    # 6) 剩余（同 print）
    remaining_mc = [r for r in mc_rows if id(r) not in used_mc_ids]
    remaining_me = [r for r in me_rows if id(r) not in used_me_ids]

    return {
        "strong": strong,
        "mc_rows": mc_rows,
        "me_rows": me_rows,
        "full_chains": full_chains,
        "remaining_mc_rows": remaining_mc,
        "remaining_me_rows": remaining_me,
        "debug": {
            "common_modes": common_modes,
            "used_mc": len(used_mc_ids),
            "used_me": len(used_me_ids),
            "full_keys": len(full_keys),
        },
    }

def build_gt_unique_full_chains(gt_list: List[Dict]) -> List[Dict[str, Any]]:
    """
    GT FULL 去重： (mode, cause, effect) 归一化后去重
    """
    out = []
    seen = set()

    for g in gt_list:
        mode   = (g.get("failure_mode_text") or "").strip()
        cause  = (g.get("failure_cause_text") or "").strip()
        effect = (g.get("failure_effect_text") or "").strip()

        nm, nc, ne = normalize_text(mode), normalize_text(cause), normalize_text(effect)
        if not (nm and nc and ne):
            continue

        sig = (nm, nc, ne)
        if sig in seen:
            continue
        seen.add(sig)

        out.append({
            "failure_mode": mode,
            "failure_cause": cause,
            "failure_effect": effect,
        })

    return out

def evaluate_semi_strict(pred_list: List[Dict], gt_list: List[Dict], *, kind: str) -> Dict[str, Any]:
    """
    kind: "MC" or "ME"
      - MC compares (mode, cause)
      - ME compares (mode, effect)

    score in {0,1,2}
      2 = complete match (ONLY this counts as match)
      1 = partial match (no longer counted as match; only for assignment/component stats)
    """

    assert kind in ("MC", "ME")

    if kind == "MC":
        pred_key = "failure_cause"
        gt_key   = "failure_cause"
    else:
        pred_key = "failure_effect"
        gt_key   = "failure_effect"

    # -------------------------
    # 0) Dedup preds (by normalized pair)
    # -------------------------
    def sig_pred(p):
        return (normalize_text(p.get("failure_mode")), normalize_text(p.get(pred_key)))

    unique_pred, seen = [], set()
    for p in pred_list:
        s = sig_pred(p)
        if s[0] and s[1] and s not in seen:
            seen.add(s)
            unique_pred.append(p)
    pred_list = unique_pred

    # -------------------------
    # 0b) Dedup GT (by normalized pair)
    # -------------------------
    def sig_gt(g):
        return (normalize_text(g.get("failure_mode")), normalize_text(g.get(gt_key)))

    unique_gt, seen = [], set()
    for g in gt_list:
        s = sig_gt(g)
        if s[0] and s[1] and s not in seen:
            seen.add(s)
            unique_gt.append(g)
    gt_list = unique_gt

    total_pred = len(pred_list)
    total_gt = len(gt_list)
    if total_pred == 0 or total_gt == 0:
        return {
            "kind": kind,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "complete_match": 0,
            "partial_match": 0,          # deprecated
            "relaxed_match": 0,          # deprecated
            "no_match": total_pred,
        }

    # -------------------------
    # 1) score matrix + component matrices
    # -------------------------
    score_matrix = np.zeros((total_pred, total_gt))
    mode_matrix  = np.zeros((total_pred, total_gt), dtype=bool)
    other_matrix = np.zeros((total_pred, total_gt), dtype=bool)

    for i, p in enumerate(pred_list):
        pm = normalize_text(p.get("failure_mode"))
        po = normalize_text(p.get(pred_key))
        for j, g in enumerate(gt_list):
            gm = normalize_text(g.get("failure_mode"))
            go = normalize_text(g.get(gt_key))

            m_ok = (pm == gm) and bool(pm)
            o_ok = (po == go) and bool(po)

            mode_matrix[i, j] = m_ok
            other_matrix[i, j] = o_ok
            score_matrix[i, j] = (1 if m_ok else 0) + (1 if o_ok else 0)

    # -------------------------
    # 2) Hungarian (maximize score)
    # -------------------------
    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    complete_match = 0

    matched_pred_strict = set()  # score==2 only
    matched_gt_strict   = set()

    mode_tp = 0
    other_tp = 0  # cause_tp or effect_tp depending on kind

    print(f"\n================ {kind} MATCH DETAILS (STRICT ONLY: score==2) ================\n")

    # 2a) iterate assigned pairs: compute component TP, and collect strict matches
    for r, c in zip(row_ind, col_ind):
        score = int(score_matrix[r, c])

        # component TP (under assigned pairing)
        if mode_matrix[r, c]:
            mode_tp += 1
        if other_matrix[r, c]:
            other_tp += 1

        # STRICT match only
        if score == 2:
            matched_pred_strict.add(r)
            matched_gt_strict.add(c)
            complete_match += 1

    # 2b) print strict matched semi chains
    print(f"\n================ {kind} STRICT MATCHED SEMI CHAINS (score==2) ================\n")
    if not matched_pred_strict:
        print("(none)\n")
    else:
        # print them in assignment order for readability
        for r, c in zip(row_ind, col_ind):
            if r in matched_pred_strict:
                print("\n------------ STRICT MATCH ------------")
                print("[Prediction]")
                print(f"Mode  : {pred_list[r].get('failure_mode')}")
                print(f"{pred_key.replace('failure_', '').title():<6}: {pred_list[r].get(pred_key)}")
                if "count" in pred_list[r] or "best_score" in pred_list[r]:
                    print(f"count : {pred_list[r].get('count')}, best_score: {pred_list[r].get('best_score')}")
                print("\n[Ground Truth]")
                print(f"Mode  : {gt_list[c].get('failure_mode')}")
                print(f"{gt_key.replace('failure_', '').title():<6}: {gt_list[c].get(gt_key)}")
                print("--------------------------------------\n")

    # -------------------------
    # 3) Unmatched query chains (preds NOT strictly matched)
    # -------------------------
    # print(f"\n================ {kind} UNMATCHED QUERY CHAINS (NOT score==2) ================\n")
    # for i in range(total_pred):
    #     if i not in matched_pred_strict:
    #         print("\n------------ UNMATCHED PREDICTION ------------")
    #         print(f"Mode  : {pred_list[i].get('failure_mode')}")
    #         print(f"{pred_key.replace('failure_', '').title():<6}: {pred_list[i].get(pred_key)}")
    #         if "count" in pred_list[i] or "best_score" in pred_list[i]:
    #             print(f"count : {pred_list[i].get('count')}, best_score: {pred_list[i].get('best_score')}")
    #         print("------------------------------------------------\n")

    # -------------------------
    # 4) Unmatched GT (GT NOT strictly matched)
    # -------------------------
    print(f"\n================ {kind} UNMATCHED GROUND TRUTH (NOT score==2) ================\n")
    for j in range(total_gt):
        if j not in matched_gt_strict:
            print("\n------------ UNMATCHED GROUND TRUTH ------------")
            print(f"Mode  : {gt_list[j].get('failure_mode')}")
            print(f"{gt_key.replace('failure_', '').title():<6}: {gt_list[j].get(gt_key)}")
            print("------------------------------------------------\n")

    # -------------------------
    # 5) metrics (ONLY complete counts as match)
    # -------------------------
    def safe_f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # strict-only view
    no_match = total_pred - len(matched_pred_strict)

    precision_complete = complete_match / total_pred
    recall_complete    = complete_match / total_gt
    f1_complete        = safe_f1(precision_complete, recall_complete)

    hallucination_rate_strict = no_match / total_pred  # preds not strictly matched

    # component metrics (under assignment; unchanged)
    precision_mode  = mode_tp / total_pred
    recall_mode     = mode_tp / total_gt
    f1_mode         = safe_f1(precision_mode, recall_mode)

    precision_other = other_tp / total_pred
    recall_other    = other_tp / total_gt
    f1_other        = safe_f1(precision_other, recall_other)

    # keep old keys (partial/relaxed) but neutralize them to avoid breaking callers
    return {
        "kind": kind,
        "total_gt": total_gt,
        "total_pred": total_pred,

        # edge-level (strict-only)
        "complete_match": complete_match,     # score==2 only
        "no_match": no_match,                 # NOT score==2

        "precision_complete": precision_complete,
        "recall_complete": recall_complete,
        "f1_complete": f1_complete,

        # deprecated (mirror complete so downstream doesn't crash)

        # component-level
        "mode_tp": mode_tp,
        f"{pred_key.replace('failure_', '')}_tp": other_tp,

        "precision_mode": precision_mode,
        "recall_mode": recall_mode,
        "f1_mode": f1_mode,

        f"precision_{pred_key.replace('failure_', '')}": precision_other,
        f"recall_{pred_key.replace('failure_', '')}": recall_other,
        f"f1_{pred_key.replace('failure_', '')}": f1_other,
    }


def evaluate_full_strict(
    pred_full: List[Dict[str, Any]],
    gt_full: List[Dict[str, Any]],
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    pred_full item expects:
      - query_mode / query_cause / query_effect (作为“预测”三元组)
    gt_full item expects:
      - failure_mode / failure_cause / failure_effect

    score in {0,1,2,3} for (mode,cause,effect)
    STRICT match: score==3 only
    """

    # ---- build normalized triplets ----
    def pred_trip(p):
        return (
            normalize_text(p.get("query_mode")),
            normalize_text(p.get("query_cause")),
            normalize_text(p.get("query_effect")),
        )

    def gt_trip(g):
        return (
            normalize_text(g.get("failure_mode")),
            normalize_text(g.get("failure_cause")),
            normalize_text(g.get("failure_effect")),
        )

    # ---- dedup preds/gt ----
    pred_u, seen = [], set()
    for p in pred_full:
        t = pred_trip(p)
        if all(t) and t not in seen:
            seen.add(t)
            pred_u.append(p)
    pred_full = pred_u

    gt_u, seen = [], set()
    for g in gt_full:
        t = gt_trip(g)
        if all(t) and t not in seen:
            seen.add(t)
            gt_u.append(g)
    gt_full = gt_u

    total_pred, total_gt = len(pred_full), len(gt_full)
    if total_pred == 0 or total_gt == 0:
        return {
            "kind": "MCE",
            "total_gt": total_gt,
            "total_pred": total_pred,
            "complete_match": 0,
            "no_match": total_pred,
            "precision_complete": 0.0,
            "recall_complete": 0.0,
            "f1_complete": 0.0,
            "matched_pred_idx_strict": set(),
            "matched_gt_idx_strict": set(),
            "gt_edges_used_mc": set(),
            "gt_edges_used_me": set(),
        }

    score_matrix = np.zeros((total_pred, total_gt))
    mode_ok_mat  = np.zeros((total_pred, total_gt), dtype=bool)
    cause_ok_mat = np.zeros((total_pred, total_gt), dtype=bool)
    eff_ok_mat   = np.zeros((total_pred, total_gt), dtype=bool)

    for i, p in enumerate(pred_full):
        pm, pc, pe = pred_trip(p)
        for j, g in enumerate(gt_full):
            gm, gc, ge = gt_trip(g)

            m_ok = (pm == gm) and bool(pm)
            c_ok = (pc == gc) and bool(pc)
            e_ok = (pe == ge) and bool(pe)

            mode_ok_mat[i, j]  = m_ok
            cause_ok_mat[i, j] = c_ok
            eff_ok_mat[i, j]   = e_ok

            score_matrix[i, j] = (1 if m_ok else 0) + (1 if c_ok else 0) + (1 if e_ok else 0)

    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    matched_pred_strict = set()
    matched_gt_strict   = set()
    complete_match = 0

    # 记录 FULL 匹配到的 GT 边（用于后续 PART2/3 剔除）
    gt_edges_used_mc = set()  # (mode_norm, cause_norm)
    gt_edges_used_me = set()  # (mode_norm, effect_norm)

    for r, c in zip(row_ind, col_ind):
        if int(score_matrix[r, c]) == 3:
            matched_pred_strict.add(r)
            matched_gt_strict.add(c)
            complete_match += 1

            gm, gc, ge = gt_trip(gt_full[c])
            gt_edges_used_mc.add((gm, gc))
            gt_edges_used_me.add((gm, ge))

    def safe_f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    precision_complete = complete_match / total_pred
    recall_complete    = complete_match / total_gt
    f1_complete        = safe_f1(precision_complete, recall_complete)

    no_match = total_pred - len(matched_pred_strict)

    if verbose:
        print("\n================ MCE MATCH DETAILS (STRICT ONLY: score==3) ================\n")
        print(f"total_pred={total_pred} total_gt={total_gt} strict_match={complete_match}\n")

        print("\n================ MCE STRICT MATCHED FULL CHAINS (score==3) ================\n")
        if not matched_pred_strict:
            print("(none)\n")
        else:
            for r, c in zip(row_ind, col_ind):
                if r in matched_pred_strict:
                    p = pred_full[r]
                    g = gt_full[c]
                    print("\n------------ STRICT MATCH ------------")
                    print("[Prediction]")
                    print(f"MODE  : {p.get('query_mode')}")
                    print(f"CAUSE : {p.get('query_cause')}")
                    print(f"EFFECT: {p.get('query_effect')}")
                    print("\n[Ground Truth]")
                    print(f"MODE  : {g.get('failure_mode')}")
                    print(f"CAUSE : {g.get('failure_cause')}")
                    print(f"EFFECT: {g.get('failure_effect')}")
                    print("--------------------------------------\n")

    return {
        "kind": "MCE",
        "total_gt": total_gt,
        "total_pred": total_pred,

        "complete_match": complete_match,
        "no_match": no_match,

        "precision_complete": precision_complete,
        "recall_complete": recall_complete,
        "f1_complete": f1_complete,

        # 关键：返回 matched idx + GT edges used（用于 PART2/3 剔除）
        "matched_pred_idx_strict": matched_pred_strict,
        "matched_gt_idx_strict": matched_gt_strict,
        "gt_edges_used_mc": gt_edges_used_mc,
        "gt_edges_used_me": gt_edges_used_me,
    }

def strong_rows_to_pred_semi(rows: List[Dict[str, Any]], *, kind: str) -> List[Dict[str, Any]]:
    assert kind in ("MC", "ME")
    out = []
    seen = set()

    for r in rows:
        combo = r.get("query_combo") or []
        if not isinstance(combo, list) or len(combo) < 2:
            continue

        mode = (combo[0] or "").strip()
        other = (combo[1] or "").strip()

        nm = normalize_text(mode)
        no = normalize_text(other)
        if not (nm and no):
            continue

        sig = (nm, no)
        if sig in seen:
            continue
        seen.add(sig)

        if kind == "MC":
            out.append({
                "failure_mode": mode,
                "failure_cause": other,
                "count": int(r.get("count", 0) or 0),
                "best_score": float(r.get("best_score", 0.0) or 0.0),
            })
        else:
            out.append({
                "failure_mode": mode,
                "failure_effect": other,
                "count": int(r.get("count", 0) or 0),
                "best_score": float(r.get("best_score", 0.0) or 0.0),
            })

    return out

def evaluate_with_strong_semi_filters(
    *,
    results_graph: Dict[str, Any],
    gt_path: "Path",
    target_element: str,
    min_count: int = 2,
    min_best_score: float = 0.5,
    keep_example: bool = True,
) -> Dict[str, Any]:
    """
    - pred: 从 query_combo_stats 过滤 strong semi combos，再按 query pair 去重
    - gt: load_gt 过滤 element，再抽 unique semi edges (MC/ME)
    - eval: 分别对 MC / ME 做 Hungarian + component metrics
    """

    # 1) GT rows
    gt_rows = load_gt(gt_path, target_element)

    # 2) GT unique semi-chains
    gt_semi = build_gt_unique_semi_chains(gt_rows)

    # 3) Pred strong semi-chains (filtered)
    strong = get_strong_semi_chains_from_query_combo_stats(
        results_graph,
        min_count=min_count,
        min_best_score=min_best_score,
        keep_example=keep_example,
    )
    pred_semi = build_pred_from_strong_semi_stats(strong)

    # 4) Evaluate MC / ME
    mc_metrics = evaluate_semi_strict(pred_semi["MC"], gt_semi["MC"], kind="MC")
    me_metrics = evaluate_semi_strict(pred_semi["ME"], gt_semi["ME"], kind="ME")

    return {
        "settings": {
            "target_element": target_element,
            "min_count": min_count,
            "min_best_score": min_best_score,
            "keep_example": keep_example,
            "pred_mc": len(pred_semi["MC"]),
            "pred_me": len(pred_semi["ME"]),
            "gt_mc": len(gt_semi["MC"]),
            "gt_me": len(gt_semi["ME"]),
        },
        "MC": mc_metrics,
        "ME": me_metrics,
    }

def evaluate_three_parts_with_strong_semi_filters(
    *,
    results_graph: Dict[str, Any],
    gt_path: Path,
    target_element: str,
    min_count: int = 2,
    min_best_score: float = 0.5,
    keep_example: bool = True,
    verbose_full: bool = True,
) -> Dict[str, Any]:
    """
    1) 用 strong semi filters + cartesian join 生成三部分 pred
    2) GT: 同 element 过滤
       - full: (mode,cause,effect) 去重
       - semi: MC/ME 去重
    3) eval:
       - PART1 FULL vs GT_FULL（strict: score==3）
       - PART2 remaining MC vs (GT_MC - FULL已用MC边)
       - PART3 remaining ME vs (GT_ME - FULL已用ME边)
    """

    # ---- GT ----
    gt_rows = load_gt(gt_path, target_element)
    gt_full = build_gt_unique_full_chains(gt_rows)
    gt_semi = build_gt_unique_semi_chains(gt_rows)  # 你已有：{"MC":[...], "ME":[...]}

    # ---- Pred 3 parts ----
    parts = build_pred_three_parts_from_results_graph(
        results_graph,
        min_count=min_count,
        min_best_score=min_best_score,
        keep_example=keep_example,
    )
    pred_full = parts["full_chains"]
    pred_mc_remaining_rows = parts["remaining_mc_rows"]
    pred_me_remaining_rows = parts["remaining_me_rows"]

    # ---- PART1: FULL ----
    mce_metrics = evaluate_full_strict(pred_full, gt_full, verbose=verbose_full)

    # FULL 匹配用掉的 GT semi edges（用归一化对）
    used_mc_edges = mce_metrics["gt_edges_used_mc"]  # (mode_norm, cause_norm)
    used_me_edges = mce_metrics["gt_edges_used_me"]  # (mode_norm, effect_norm)

    # ---- build remaining GT semi (剔除 FULL 已用边，避免重复计数) ----
    gt_mc_remaining = []
    for g in gt_semi["MC"]:
        sig = (normalize_text(g.get("failure_mode")), normalize_text(g.get("failure_cause")))
        if sig[0] and sig[1] and sig not in used_mc_edges:
            gt_mc_remaining.append(g)

    gt_me_remaining = []
    for g in gt_semi["ME"]:
        sig = (normalize_text(g.get("failure_mode")), normalize_text(g.get("failure_effect")))
        if sig[0] and sig[1] and sig not in used_me_edges:
            gt_me_remaining.append(g)

    # ---- PART2: remaining MC ----
    pred_mc_remaining = strong_rows_to_pred_semi(pred_mc_remaining_rows, kind="MC")
    mc_metrics = evaluate_semi_strict(pred_mc_remaining, gt_mc_remaining, kind="MC")

    # ---- PART3: remaining ME ----
    pred_me_remaining = strong_rows_to_pred_semi(pred_me_remaining_rows, kind="ME")
    me_metrics = evaluate_semi_strict(pred_me_remaining, gt_me_remaining, kind="ME")

    return {
        "settings": {
            "target_element": target_element,
            "min_count": min_count,
            "min_best_score": min_best_score,
            "keep_example": keep_example,
            "pred_full": len(pred_full),
            "pred_mc_remaining": len(pred_mc_remaining),
            "pred_me_remaining": len(pred_me_remaining),
            "gt_full": len(gt_full),
            "gt_mc_remaining": len(gt_mc_remaining),
            "gt_me_remaining": len(gt_me_remaining),
        },
        "PART1_FULL_MCE": mce_metrics,
        "PART2_REMAINING_MC": mc_metrics,
        "PART3_REMAINING_ME": me_metrics,
    }

KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_discipline\failure_kb"
    )

GT_JSON = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_complete\failure_kb\entity_store.json")

FIELD_WEIGHTS = {
"element":0.2,
"mode": 1.2,
"cause": 1.1,
"effect": 0.9,
}
results_graph = generate_query_unique_chains(
    persist_dir=KB_PATH,
    structure_input=structure_input_powertrain,
    save_query_json=False,
    min_similarity=0.25,
    top_k_per_field=30,
    field_weights=None,
    hybrid=False
)

metrics = evaluate_with_strong_semi_filters(
    results_graph=results_graph,
    gt_path=GT_JSON,
    target_element="Power train",
    min_count=1,
    min_best_score=0.2,
)

print(metrics["MC"])
print(metrics["ME"])

# metrics_3parts = evaluate_three_parts_with_strong_semi_filters(
#     results_graph=results_graph,
#     gt_path=GT_JSON,
#     target_element="Power train",
#     min_count=2,
#     min_best_score=0.5,
#     keep_example=True,
#     verbose_full=True,   # FULL 部分会打印严格匹配详情
# )

# print(metrics_3parts["settings"])
# print(metrics_3parts["PART1_FULL_MCE"])
# print(metrics_3parts["PART2_REMAINING_MC"])
# print(metrics_3parts["PART3_REMAINING_ME"])