from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from ..graph_query import generate_query_unique_chains
from ..entity import structure_input_powertrain

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
    Return:
      {
        "MC": [{"failure_mode":..., "failure_cause":..., "failure_element":...}, ...],
        "ME": [{"failure_mode":..., "failure_effect":..., "failure_element":...}, ...],
      }
    Unique by normalized pair signature.
    """
    out = {"MC": [], "ME": []}

    seen_mc = set()
    seen_me = set()

    for g in gt_list:
        mode   = (g.get("failure_mode_text") or "").strip()
        cause  = (g.get("failure_cause_text") or "").strip()
        effect = (g.get("failure_effect_text") or "").strip()
        elem   = (g.get("failure_element_text") or "").strip()

        nm = normalize_text(mode)
        if nm:
            if cause and normalize_text(cause):
                sig_mc = (nm, normalize_text(cause))
                if sig_mc not in seen_mc:
                    seen_mc.add(sig_mc)
                    out["MC"].append({
                        "failure_element": elem,
                        "failure_mode": mode,
                        "failure_cause": cause,
                    })

            if effect and normalize_text(effect):
                sig_me = (nm, normalize_text(effect))
                if sig_me not in seen_me:
                    seen_me.add(sig_me)
                    out["ME"].append({
                        "failure_element": elem,
                        "failure_mode": mode,
                        "failure_effect": effect,
                    })

    return out

def build_pred_unique_semi_chains_from_results_graph(
    results_graph: Dict[str, Any],
    *,
    min_count: int = 1,
    min_best_score: float = 0.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    From results_graph["query_combo_stats"]["MC"/"ME"], build unique pred semi-chains.

    Output format mirrors GT builders:
      MC item: {"failure_mode":..., "failure_cause":..., "count":..., "best_score":...}
      ME item: {"failure_mode":..., "failure_effect":..., "count":..., "best_score":...}
    """
    stats = (results_graph or {}).get("query_combo_stats") or {}
    out: Dict[str, List[Dict[str, Any]]] = {"MC": [], "ME": []}

    seen_mc = set()
    seen_me = set()

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

            combo = r.get("query_combo") or []
            if not isinstance(combo, list) or len(combo) < 2:
                continue

            a = (combo[0] or "").strip()
            b = (combo[1] or "").strip()
            if not normalize_text(a) or not normalize_text(b):
                continue

            if kind == "MC":
                sig = (normalize_text(a), normalize_text(b))
                if sig in seen_mc:
                    continue
                seen_mc.add(sig)
                out["MC"].append({
                    "chain_type": "MC",
                    "failure_mode": a,
                    "failure_cause": b,
                    "count": c,
                    "best_score": s,
                })
            else:
                sig = (normalize_text(a), normalize_text(b))
                if sig in seen_me:
                    continue
                seen_me.add(sig)
                out["ME"].append({
                    "chain_type": "ME",
                    "failure_mode": a,
                    "failure_effect": b,
                    "count": c,
                    "best_score": s,
                })

    return out

def evaluate_semi_strict(pred_list: List[Dict], gt_list: List[Dict], *, kind: str) -> Dict[str, Any]:
    """
    kind: "MC" or "ME"
      - MC compares (mode, cause)
      - ME compares (mode, effect)

    score in {0,1,2}
      2 = complete match
      1 = partial match
    we treat score>=1 as "matched" for coverage reporting,
    and score==2 as strict correct edge.
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
            "partial_match": 0,
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
    # 2) Hungarian
    # -------------------------
    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    complete_match = 0
    partial_match  = 0

    matched_pred_strict = set()  # score==2
    matched_gt_strict   = set()

    matched_pred_relaxed = set()  # score>=1
    matched_gt_relaxed   = set()

    mode_tp = 0
    other_tp = 0  # cause_tp or effect_tp depending on kind

    print(f"\n================ {kind} MATCH DETAILS ================\n")

    for r, c in zip(row_ind, col_ind):
        score = int(score_matrix[r, c])

        # component TP (under assigned pairing)
        if mode_matrix[r, c]:
            mode_tp += 1
        if other_matrix[r, c]:
            other_tp += 1

        if score >= 1:
            matched_pred_relaxed.add(r)
            matched_gt_relaxed.add(c)

        if score == 2:
            matched_pred_strict.add(r)
            matched_gt_strict.add(c)
            complete_match += 1
        elif score == 1:
            partial_match += 1

            print("\n============ PARTIAL MATCH (1/2) ============")
            print("\n[Prediction]")
            print(f"Mode  : {pred_list[r].get('failure_mode')}")
            print(f"{pred_key.replace('failure_', '').title():<6}: {pred_list[r].get(pred_key)}")
            print("\n[Ground Truth]")
            print(f"Mode  : {gt_list[c].get('failure_mode')}")
            print(f"{gt_key.replace('failure_', '').title():<6}: {gt_list[c].get(gt_key)}")
            print("=============================================\n")

    # -------------------------
    # 3) Unmatched preds (relaxed view)
    # -------------------------
    print(f"\n================ {kind} NO MATCH PREDICTIONS (score==0) ================\n")
    for i in range(total_pred):
        if i not in matched_pred_relaxed:
            print("\n------------ UNMATCHED PREDICTION ------------")
            print(f"Mode  : {pred_list[i].get('failure_mode')}")
            print(f"{pred_key.replace('failure_', '').title():<6}: {pred_list[i].get(pred_key)}")
            if "count" in pred_list[i] or "best_score" in pred_list[i]:
                print(f"count : {pred_list[i].get('count')}, best_score: {pred_list[i].get('best_score')}")
            print("------------------------------------------------\n")

    # -------------------------
    # 4) Missed GT (relaxed view)
    # -------------------------
    print(f"\n================ {kind} MISSED GROUND TRUTH (no assigned score>=1) ================\n")
    for j in range(total_gt):
        if j not in matched_gt_relaxed:
            print("\n------------ MISSED GROUND TRUTH ------------")
            print(f"Mode  : {gt_list[j].get('failure_mode')}")
            print(f"{gt_key.replace('failure_', '').title():<6}: {gt_list[j].get(gt_key)}")
            print("------------------------------------------------\n")

    # -------------------------
    # 5) metrics
    # -------------------------
    def safe_f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    relaxed_match = complete_match + partial_match
    no_match = total_pred - len(matched_pred_relaxed)

    precision_complete = complete_match / total_pred
    recall_complete    = complete_match / total_gt
    f1_complete        = safe_f1(precision_complete, recall_complete)

    precision_relaxed  = relaxed_match / total_pred
    recall_relaxed     = relaxed_match / total_gt
    f1_relaxed         = safe_f1(precision_relaxed, recall_relaxed)

    hallucination_rate_relaxed = no_match / total_pred  # score==0 rate

    # component metrics (under assignment)
    precision_mode  = mode_tp / total_pred
    recall_mode     = mode_tp / total_gt
    f1_mode         = safe_f1(precision_mode, recall_mode)

    precision_other = other_tp / total_pred
    recall_other    = other_tp / total_gt
    f1_other        = safe_f1(precision_other, recall_other)

    return {
        "kind": kind,
        "total_gt": total_gt,
        "total_pred": total_pred,

        # edge-level (semi-chain)
        "complete_match": complete_match,   # score==2
        "partial_match": partial_match,     # score==1
        "relaxed_match": relaxed_match,     # score>=1
        "no_match": no_match,               # score==0

        "precision_complete": precision_complete,
        "recall_complete": recall_complete,
        "f1_complete": f1_complete,

        "precision_relaxed": precision_relaxed,
        "recall_relaxed": recall_relaxed,
        "f1_relaxed": f1_relaxed,

        "hallucination_rate_relaxed": hallucination_rate_relaxed,

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


def evaluate_graph_semi_chains_against_gt(
    *,
    results_graph: Dict[str, Any],
    gt_path: Path,
    target_element: str,
    min_count: int = 1,
    min_best_score: float = 0.0,
) -> Dict[str, Any]:
    # 1) load GT rows
    gt_rows = load_gt(gt_path, target_element)

    # 2) GT unique semi-chains
    gt_semi = build_gt_unique_semi_chains(gt_rows)

    # 3) Pred unique semi-chains
    pred_semi = build_pred_unique_semi_chains_from_results_graph(
        results_graph,
        min_count=min_count,
        min_best_score=min_best_score,
    )

    # 4) eval MC / ME
    mc_metrics = evaluate_semi_strict(pred_semi["MC"], gt_semi["MC"], kind="MC")
    me_metrics = evaluate_semi_strict(pred_semi["ME"], gt_semi["ME"], kind="ME")

    return {
        "MC": mc_metrics,
        "ME": me_metrics,
        "settings": {
            "target_element": target_element,
            "min_count": min_count,
            "min_best_score": min_best_score,
        }
    }

KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_miniLM\failure_kb"
    )

GT_JSON = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_MOTORCONTROL\failure_kb\entity_store.json")
results_graph = generate_query_unique_chains(
    persist_dir=KB_PATH,
    structure_input=structure_input_powertrain,
    save_query_json=False,
    min_similarity=0.25,
    top_k_per_field=30
)

metrics = evaluate_graph_semi_chains_against_gt(
    results_graph=results_graph,
    gt_path=GT_JSON,
    target_element="Power train",
    min_count=1,
    min_best_score=0.0,
)

print(metrics["MC"])
print(metrics["ME"])