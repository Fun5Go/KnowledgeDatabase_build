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

def evaluate(pred_list: List[Dict], gt_list: List[Dict], verbose: bool = False):

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

    # --------------------------------------------------------
    # 1️⃣ Build score matrix
    # --------------------------------------------------------
    # score: 3 = strict, 2 = partial, 0/1 = no match
    match_matrix = []

    for i, pred in enumerate(pred_list):

        row = []

        for j, gt in enumerate(gt_list):

            mode_match = normalize_text(pred["failure_mode"]) == normalize_text(gt["failure_mode_text"])
            cause_match = normalize_text(pred["failure_cause"]) == normalize_text(gt["failure_cause_text"])
            effect_match = normalize_text(pred["failure_effect"]) == normalize_text(gt["failure_effect_text"])

            score = sum([mode_match, cause_match, effect_match])

            row.append(score)

        match_matrix.append(row)

    # --------------------------------------------------------
    # 2️⃣ Greedy bipartite matching (3 first, then 2)
    # --------------------------------------------------------
    matched_pred = set()
    matched_gt = set()

    complete_match = 0
    partial_match = 0

    # ---------- Strict Match (3/3) ----------
    for i in range(total_pred):
        for j in range(total_gt):
            if match_matrix[i][j] == 3:
                if i not in matched_pred and j not in matched_gt:
                    matched_pred.add(i)
                    matched_gt.add(j)
                    complete_match += 1

    # ---------- Partial Match (2/3) ----------
    for i in range(total_pred):
        if i in matched_pred:
            continue

        for j in range(total_gt):
            if match_matrix[i][j] == 2:
                if j not in matched_gt:
                    matched_pred.add(i)
                    matched_gt.add(j)
                    partial_match += 1

                    if verbose:
                        print("\n============ PARTIAL MATCH (2/3) ============")
                        print("\n[Prediction]")
                        print(json.dumps(pred_list[i], indent=2, ensure_ascii=False))
                        print("\n[Ground Truth]")
                        print(json.dumps(gt_list[j], indent=2, ensure_ascii=False))
                        print("=============================================\n")

                    break

    # --------------------------------------------------------
    #  Remaining unmatched predictions → hallucination
    # --------------------------------------------------------
    no_match = total_pred - len(matched_pred)

    # --------------------------------------------------------
    #  Metrics
    # --------------------------------------------------------

    precision_complete = complete_match / total_pred if total_pred else 0
    recall_complete = complete_match / total_gt if total_gt else 0
    f1_complete = (
        2 * precision_complete * recall_complete / (precision_complete + recall_complete)
        if precision_complete + recall_complete > 0 else 0
    )

    precision_partial = partial_match / total_pred if total_pred else 0
    recall_partial = partial_match / total_gt if total_gt else 0
    f1_partial = (
        2 * precision_partial * recall_partial / (precision_partial + recall_partial)
        if precision_partial + recall_partial > 0 else 0
    )

    hallucination_rate = no_match / total_pred if total_pred else 0

    # --------------------------------------------------------
    # 5️⃣ Summary
    # --------------------------------------------------------

    result = {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "complete_match": complete_match,
        "partial_match": partial_match,
        "no_match": no_match,
        "precision_complete": round(precision_complete, 4),
        "recall_complete": round(recall_complete, 4),
        "f1_complete": round(f1_complete, 4),
        "precision_partial": round(precision_partial, 4),
        "recall_partial": round(recall_partial, 4),
        "f1_partial": round(f1_partial, 4),
        "hallucination_rate": round(hallucination_rate, 4),
    }

    return result

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
    # 1️⃣ Build score matrix
    # --------------------------------------------------------
    score_matrix = np.zeros((total_pred, total_gt))

    for i, pred in enumerate(pred_list):
        for j, gt in enumerate(gt_list):

            mode_match = normalize_text(pred["failure_mode"]) == normalize_text(gt["failure_mode_text"])
            cause_match = normalize_text(pred["failure_cause"]) == normalize_text(gt["failure_cause_text"])
            effect_match = normalize_text(pred["failure_effect"]) == normalize_text(gt["failure_effect_text"])

            score_matrix[i, j] = sum([mode_match, cause_match, effect_match])

    # --------------------------------------------------------
    # 2️⃣ Hungarian Algorithm
    # --------------------------------------------------------
    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    complete_match = 0
    partial_match = 0
    matched_pred = set()
    matched_gt = set()

    print("\n================ MATCH DETAILS ================\n")

    for r, c in zip(row_ind, col_ind):

        score = score_matrix[r, c]

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
    # 5️⃣ Metrics
    # --------------------------------------------------------
    relaxed_match = complete_match + partial_match
    no_match = total_pred - len(matched_pred)

    precision_complete = complete_match / total_pred
    recall_complete = complete_match / total_gt
    f1_complete = (
        2 * precision_complete * recall_complete / (precision_complete + recall_complete)
        if precision_complete + recall_complete > 0 else 0
    )

    precision_partial = partial_match / total_pred
    recall_partial = partial_match / total_gt
    f1_partial = (
        2 * precision_partial * recall_partial / (precision_partial + recall_partial)
        if precision_partial + recall_partial > 0 else 0
    )

    precision_relaxed = relaxed_match / total_pred
    recall_relaxed = relaxed_match / total_gt
    f1_relaxed = (
        2 * precision_relaxed * recall_relaxed / (precision_relaxed + recall_relaxed)
        if precision_relaxed + recall_relaxed > 0 else 0
    )

    hallucination_rate = no_match / total_pred

    # 安全检查
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
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    GT_JSON = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_allPT_MC\failure_kb\entity_store.json")
    PREDICTION_JSON = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\failure_candidates_PURE_powertrain4.json")

    gt_list = load_gt(GT_JSON,target_element=TARGET_ELEMENT_2)
    pred_list = load_predictions(PREDICTION_JSON,target_element=TARGET_ELEMENT_2)

    results = evaluate_strict(pred_list, gt_list)
    print("\n========== FINAL METRICS ==========")
    print(json.dumps(results, indent=2))
