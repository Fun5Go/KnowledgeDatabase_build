import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np 

# ============================================================
# CONFIG
# ============================================================

TARGET_ELEMENT_1 = "Motor control"

TARGET_ELEMENT_2 = "Power train"


# ============================================================
# Utils
# ============================================================

def normalize_text(x: str) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split()).lower()


def chain_signature(item: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        normalize_text(item.get("failure_mode")),
        normalize_text(item.get("failure_cause")),
        normalize_text(item.get("failure_effect")),
    )


# ============================================================
# Load Data
# ============================================================
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

def load_predictions(pred_path: Path, target_element: str) -> List[Dict]:

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_raw = json.load(f)


    failure_block = pred_raw.get("failure_candidates", {})

    if isinstance(failure_block, dict):
        pred_list = failure_block.get("failure_candidates", [])
    else:
        pred_list = []

    pred_list = [p for p in pred_list if isinstance(p, dict)]


    pred_filtered = [
        p for p in pred_list
        if normalize_text(p.get("failure_element", "")) ==
           normalize_text(target_element)
    ]

    return pred_filtered

def load_predictions_old(pred_path: Path, target_element: str) -> List[Dict]:
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_raw = json.load(f)

    pred_list = pred_raw.get("failure_candidates", [])

    pred_filtered = [
        p for p in pred_list
        if normalize_text(p.get("failure_element")) == normalize_text(target_element)
    ]

    return pred_filtered


# ============================================================
# Evaluation
# ============================================================

def evaluate_strict_dedup(pred_list: List[Dict], gt_list: List[Dict]) -> Dict[str, Any]:
    """
    与原 evaluate_strict 一样的逻辑，但：
      ✅ pred 去重 (mode,cause,effect)
      ✅ gt 也去重 (mode,cause,effect)
    然后在去重后的空间里做 Hungarian + partial/complete/no-match 打印与指标。
    """

    def pred_sig(p: Dict) -> Tuple[str, str, str]:
        return (
            normalize_text(p.get("failure_mode")),
            normalize_text(p.get("failure_cause")),
            normalize_text(p.get("failure_effect")),
        )

    def gt_sig(g: Dict) -> Tuple[str, str, str]:
        return (
            normalize_text(g.get("failure_mode_text")),
            normalize_text(g.get("failure_cause_text")),
            normalize_text(g.get("failure_effect_text")),
        )

    # --------------------------------------------------------
    # 0️⃣ Dedup predictions (by normalized triple)
    # --------------------------------------------------------
    uniq_pred, seen = [], set()
    for p in pred_list:
        s = pred_sig(p)
        if s not in seen:
            seen.add(s)
            uniq_pred.append(p)
    pred_list = uniq_pred

    # --------------------------------------------------------
    # 0️⃣b Dedup ground truth (by normalized triple)
    # --------------------------------------------------------
    uniq_gt, seen = [], set()
    for g in gt_list:
        s = gt_sig(g)
        if s not in seen:
            seen.add(s)
            uniq_gt.append(g)
    gt_list = uniq_gt

    total_pred = len(pred_list)
    total_gt = len(gt_list)
    if total_pred == 0 or total_gt == 0:
        return {}

    # --------------------------------------------------------
    # 1️⃣ Build score matrix + component matrices
    # --------------------------------------------------------
    score_matrix  = np.zeros((total_pred, total_gt))
    mode_matrix   = np.zeros((total_pred, total_gt), dtype=bool)
    cause_matrix  = np.zeros((total_pred, total_gt), dtype=bool)
    effect_matrix = np.zeros((total_pred, total_gt), dtype=bool)

    # 预先 normalize，避免重复计算
    pred_norm = [pred_sig(p) for p in pred_list]
    gt_norm   = [gt_sig(g) for g in gt_list]

    for i, (pm, pc, pe) in enumerate(pred_norm):
        for j, (gm, gc, ge) in enumerate(gt_norm):

            mode_match   = (pm == gm) and bool(pm)
            cause_match  = (pc == gc) and bool(pc)
            effect_match = (pe == ge) and bool(pe)

            mode_matrix[i, j] = mode_match
            cause_matrix[i, j] = cause_match
            effect_matrix[i, j] = effect_match
            score_matrix[i, j] = (1 if mode_match else 0) + (1 if cause_match else 0) + (1 if effect_match else 0)

    # --------------------------------------------------------
    # 2️⃣ Hungarian Algorithm
    # --------------------------------------------------------
    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    complete_match = 0
    partial_match = 0
    matched_pred = set()
    matched_gt = set()

    mode_tp = 0
    cause_tp = 0
    effect_tp = 0

    print("\n================ MATCH DETAILS (DEDUP SPACE) ================\n")

    for r, c in zip(row_ind, col_ind):
        score = int(score_matrix[r, c])

        # Component-level TP（按匈牙利配对结果统计）
        if mode_matrix[r, c]:
            mode_tp += 1
        if cause_matrix[r, c]:
            cause_tp += 1
        if effect_matrix[r, c]:
            effect_tp += 1

        # 只把 >=2 认为是真正匹配（与你原逻辑一致）
        if score >= 2:
            matched_pred.add(r)
            matched_gt.add(c)

        if score == 3:
            complete_match += 1
        elif score == 2:
            partial_match += 1

            print("\n============ PARTIAL MATCH (2/3) ============")
            print("\n[Prediction]")
            print(f"Element : {pred_list[r].get('failure_element')}")
            print(f"Mode    : {pred_list[r].get('failure_mode')}")
            print(f"Effect  : {pred_list[r].get('failure_effect')}")
            print(f"Cause   : {pred_list[r].get('failure_cause')}")

            print("\n[Ground Truth]")
            print(f"Element : {gt_list[c].get('failure_element_text')}")
            print(f"Mode    : {gt_list[c].get('failure_mode_text')}")
            print(f"Effect  : {gt_list[c].get('failure_effect_text')}")
            print(f"Cause   : {gt_list[c].get('failure_cause_text')}")
            print("=============================================\n")

    # --------------------------------------------------------
    # 3️⃣ Print No Match Predictions (in dedup space)
    # --------------------------------------------------------
    print("\n================ NO MATCH PREDICTIONS (DEDUP) ================\n")
    for i in range(total_pred):
        if i not in matched_pred:
            print("\n------------ UNMATCHED PREDICTION ------------")
            print(f"Element : {pred_list[i].get('failure_element')}")
            print(f"Mode    : {pred_list[i].get('failure_mode')}")
            print(f"Effect  : {pred_list[i].get('failure_effect')}")
            print(f"Cause   : {pred_list[i].get('failure_cause')}")
            print("------------------------------------------------\n")

    # --------------------------------------------------------
    # 4️⃣ Print Unmatched Ground Truth (in dedup space)
    # --------------------------------------------------------
    print("\n================ MISSED GROUND TRUTH (DEDUP) ================\n")
    for j in range(total_gt):
        if j not in matched_gt:
            print("\n------------ MISSED GROUND TRUTH ------------")
            print(f"Element : {gt_list[j].get('failure_element_text')}")
            print(f"Mode    : {gt_list[j].get('failure_mode_text')}")
            print(f"Effect  : {gt_list[j].get('failure_effect_text')}")
            print(f"Cause   : {gt_list[j].get('failure_cause_text')}")
            print("------------------------------------------------\n")

    # --------------------------------------------------------
    # 5️⃣ Chain-level Metrics
    # --------------------------------------------------------
    relaxed_match = complete_match + partial_match
    no_match = total_pred - len(matched_pred)

    def safe_f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    precision_complete = complete_match / total_pred
    recall_complete = complete_match / total_gt
    f1_complete = safe_f1(precision_complete, recall_complete)

    precision_partial = partial_match / total_pred
    recall_partial = partial_match / total_gt
    f1_partial = safe_f1(precision_partial, recall_partial)

    precision_relaxed = relaxed_match / total_pred
    recall_relaxed = relaxed_match / total_gt
    f1_relaxed = safe_f1(precision_relaxed, recall_relaxed)

    hallucination_rate = no_match / total_pred

    # --------------------------------------------------------
    # 6️⃣ Component-level Metrics
    # --------------------------------------------------------
    precision_mode = mode_tp / total_pred
    recall_mode = mode_tp / total_gt
    f1_mode = safe_f1(precision_mode, recall_mode)

    precision_cause = cause_tp / total_pred
    recall_cause = cause_tp / total_gt
    f1_cause = safe_f1(precision_cause, recall_cause)

    precision_effect = effect_tp / total_pred
    recall_effect = effect_tp / total_gt
    f1_effect = safe_f1(precision_effect, recall_effect)

    # 安全检查（dedup 后依然成立）
    assert complete_match <= min(total_pred, total_gt)
    assert partial_match <= min(total_pred, total_gt)

    return {
        "total_gt": total_gt,
        "total_pred": total_pred,

        "complete_match": complete_match,
        "partial_match": partial_match,
        "no_match": no_match,

        "precision_complete": precision_complete,
        "recall_complete": recall_complete,
        "f1_complete": f1_complete,

        "precision_partial": precision_partial,
        "recall_partial": recall_partial,
        "f1_partial": f1_partial,

        "precision_relaxed": precision_relaxed,
        "recall_relaxed": recall_relaxed,
        "f1_relaxed": f1_relaxed,

        "hallucination_rate": hallucination_rate,

        "mode_tp": mode_tp,
        "cause_tp": cause_tp,
        "effect_tp": effect_tp,

        "precision_mode": precision_mode,
        "recall_mode": recall_mode,
        "f1_mode": f1_mode,

        "precision_cause": precision_cause,
        "recall_cause": recall_cause,
        "f1_cause": f1_cause,

        "precision_effect": precision_effect,
        "recall_effect": recall_effect,
        "f1_effect": f1_effect,
    }

def evaluate_strict(pred_list: List[Dict], gt_list: List[Dict]):

    def chain_signature(x):
        return (
            normalize_text(x.get("failure_mode")),
            normalize_text(x.get("failure_cause")),
            normalize_text(x.get("failure_effect")),
        )

    # --------------------------------------------------------
    # 0️⃣ Deduplicate predictions
    # --------------------------------------------------------
    unique_pred = []
    seen = set()

    for p in pred_list:
        sig = chain_signature(p)
        if sig not in seen:
            seen.add(sig)
            unique_pred.append(p)

    pred_list = unique_pred

    total_pred = len(pred_list)
    total_gt = len(gt_list)

    if total_pred == 0 or total_gt == 0:
        return {}

    # --------------------------------------------------------
    # 1️⃣ Build score matrix + component matrices
    # --------------------------------------------------------
    score_matrix  = np.zeros((total_pred, total_gt))
    mode_matrix   = np.zeros((total_pred, total_gt), dtype=bool)
    cause_matrix  = np.zeros((total_pred, total_gt), dtype=bool)
    effect_matrix = np.zeros((total_pred, total_gt), dtype=bool)

    for i, pred in enumerate(pred_list):
        for j, gt in enumerate(gt_list):

            mode_match = normalize_text(pred["failure_mode"]) == normalize_text(gt["failure_mode_text"])
            cause_match = normalize_text(pred["failure_cause"]) == normalize_text(gt["failure_cause_text"])
            effect_match = normalize_text(pred["failure_effect"]) == normalize_text(gt["failure_effect_text"])

            mode_matrix[i, j] = mode_match
            cause_matrix[i, j] = cause_match
            effect_matrix[i, j] = effect_match

            score_matrix[i, j] = mode_match + cause_match + effect_match

    # --------------------------------------------------------
    # 2️⃣ Hungarian Algorithm
    # --------------------------------------------------------
    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    complete_match = 0
    partial_match = 0
    matched_pred = set()
    matched_gt = set()

    # ⭐ Component TP counters
    mode_tp = 0
    cause_tp = 0
    effect_tp = 0

    print("\n================ MATCH DETAILS ================\n")

    for r, c in zip(row_ind, col_ind):

        score = score_matrix[r, c]

        # Component-level TP (无需重复 normalize)
        if mode_matrix[r, c]:
            mode_tp += 1
        if cause_matrix[r, c]:
            cause_tp += 1
        if effect_matrix[r, c]:
            effect_tp += 1

        # 只把 >=2 认为是真正匹配
        if score >= 2:
            matched_pred.add(r)
            matched_gt.add(c)

        if score == 3:
            complete_match += 1

        elif score == 2:
            partial_match += 1

            print("\n============ PARTIAL MATCH (2/3) ============")

            print("\n[Prediction]")
            print(f"Element : {pred_list[r].get('failure_element')}")
            print(f"Mode    : {pred_list[r].get('failure_mode')}")
            print(f"Effect  : {pred_list[r].get('failure_effect')}")
            print(f"Cause   : {pred_list[r].get('failure_cause')}")

            print("\n[Ground Truth]")
            print(f"Element : {gt_list[c].get('failure_element_text')}")
            print(f"Mode    : {gt_list[c].get('failure_mode_text')}")
            print(f"Effect  : {gt_list[c].get('failure_effect_text')}")
            print(f"Cause   : {gt_list[c].get('failure_cause_text')}")

            print("=============================================\n")

    # --------------------------------------------------------
    # 3️⃣ Print No Match Predictions
    # --------------------------------------------------------
    print("\n================ NO MATCH PREDICTIONS ================\n")

    for i in range(total_pred):
        if i not in matched_pred:
            print("\n------------ UNMATCHED PREDICTION ------------")
            print(f"Element : {pred_list[i].get('failure_element')}")
            print(f"Mode    : {pred_list[i].get('failure_mode')}")
            print(f"Effect  : {pred_list[i].get('failure_effect')}")
            print(f"Cause   : {pred_list[i].get('failure_cause')}")
            print("------------------------------------------------\n")

    # --------------------------------------------------------
    # 4️⃣ Print Unmatched Ground Truth
    # --------------------------------------------------------
    print("\n================ MISSED GROUND TRUTH ================\n")

    for j in range(total_gt):
        if j not in matched_gt:
            print("\n------------ MISSED GROUND TRUTH ------------")
            print(f"Element : {gt_list[j].get('failure_element_text')}")
            print(f"Mode    : {gt_list[j].get('failure_mode_text')}")
            print(f"Effect  : {gt_list[j].get('failure_effect_text')}")
            print(f"Cause   : {gt_list[j].get('failure_cause_text')}")
            print("------------------------------------------------\n")

    # --------------------------------------------------------
    # 5️⃣ Chain-level Metrics
    # --------------------------------------------------------
    relaxed_match = complete_match + partial_match
    no_match = total_pred - len(matched_pred)

    def safe_f1(p, r):
        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    precision_complete = complete_match / total_pred
    recall_complete = complete_match / total_gt
    f1_complete = safe_f1(precision_complete, recall_complete)

    precision_partial = partial_match / total_pred
    recall_partial = partial_match / total_gt
    f1_partial = safe_f1(precision_partial, recall_partial)

    precision_relaxed = relaxed_match / total_pred
    recall_relaxed = relaxed_match / total_gt
    f1_relaxed = safe_f1(precision_relaxed, recall_relaxed)

    hallucination_rate = no_match / total_pred

    # --------------------------------------------------------
    # 6️⃣ Component-level Metrics
    # --------------------------------------------------------
    precision_mode = mode_tp / total_pred
    recall_mode = mode_tp / total_gt
    f1_mode = safe_f1(precision_mode, recall_mode)

    precision_cause = cause_tp / total_pred
    recall_cause = cause_tp / total_gt
    f1_cause = safe_f1(precision_cause, recall_cause)

    precision_effect = effect_tp / total_pred
    recall_effect = effect_tp / total_gt
    f1_effect = safe_f1(precision_effect, recall_effect)

    # 安全检查
    assert complete_match <= min(total_pred, total_gt)
    assert partial_match <= min(total_pred, total_gt)

    return {
        "total_gt": total_gt,
        "total_pred": total_pred,

        # Chain-level
        "complete_match": complete_match,
        "partial_match": partial_match,
        "no_match": no_match,

        "precision_complete": precision_complete,
        "recall_complete": recall_complete,
        "f1_complete": f1_complete,

        "precision_partial": precision_partial,
        "recall_partial": recall_partial,
        "f1_partial": f1_partial,

        "precision_relaxed": precision_relaxed,
        "recall_relaxed": recall_relaxed,
        "f1_relaxed": f1_relaxed,

        "hallucination_rate": hallucination_rate,

        # Component-level
        "mode_tp": mode_tp,
        "cause_tp": cause_tp,
        "effect_tp": effect_tp,

        "precision_mode": precision_mode,
        "recall_mode": recall_mode,
        "f1_mode": f1_mode,

        "precision_cause": precision_cause,
        "recall_cause": recall_cause,
        "f1_cause": f1_cause,

        "precision_effect": precision_effect,
        "recall_effect": recall_effect,
        "f1_effect": f1_effect,
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    GT_JSON = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_complete\failure_kb\entity_store.json")
    PREDICTION_JSON = Path(__file__).resolve().parents[2] / "Demonstration" / "failure_candidates_RAG.json"

    gt_list = load_gt(GT_JSON,target_element=TARGET_ELEMENT_2)
    pred_list = load_predictions_old(PREDICTION_JSON,target_element=TARGET_ELEMENT_2)

    results = evaluate_strict_dedup(pred_list, gt_list)
    print("\n========== FINAL METRICS ==========")
    print(json.dumps(results, indent=2))
