import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


# ============================================================
# CONFIG
# ============================================================

TARGET_ELEMENT = "Power train"


# ============================================================
# Utility
# ============================================================

def normalize_text(x: str) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split()).lower()


def build_signature(item: Dict[str, Any]) -> tuple:
    return (
        normalize_text(item.get("failure_element")),
        normalize_text(item.get("failure_mode")),
        normalize_text(item.get("failure_cause")),
        normalize_text(item.get("failure_effect")),
    )


# ============================================================
# Load Data
# ============================================================

def load_gt(gt_path: Path) -> List[Dict]:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    gt_list = []
    for v in gt_raw.values():
        if normalize_text(v.get("failure_element")) == normalize_text(TARGET_ELEMENT):
            gt_list.append(v)

    return gt_list


def load_predictions(pred_path: Path) -> List[Dict]:
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_raw = json.load(f)

    pred_list = pred_raw.get("failure_candidates", [])

    pred_filtered = [
        p for p in pred_list
        if normalize_text(p.get("failure_element")) == normalize_text(TARGET_ELEMENT)
    ]

    return pred_filtered


# ============================================================
# Evaluation
# ============================================================

def evaluate(pred_list: List[Dict], gt_list: List[Dict]):

    def chain_signature(x):
        return (
            normalize_text(x.get("failure_mode")),
            normalize_text(x.get("failure_cause")),
            normalize_text(x.get("failure_effect")),
        )

    # --------------------------------------------------------
    # 0️⃣ Deduplicate predictions (mode+cause+effect level)
    # --------------------------------------------------------
    unique_pred = []
    seen = set()

    for p in pred_list:
        sig = chain_signature(p)
        if sig not in seen:
            seen.add(sig)
            unique_pred.append(p)

    pred_list = unique_pred

    # --------------------------------------------------------
    # Attach GT flags
    # --------------------------------------------------------
    for gt in gt_list:
        gt["_matched_complete"] = False

    total_gt = len(gt_list)
    total_pred = len(pred_list)

    complete_match = 0
    partial_match = 0
    no_match = 0

    # --------------------------------------------------------
    # Evaluate each prediction
    # --------------------------------------------------------
    for pred in pred_list:

        best_match_score = 0
        best_gt = None

        for gt in gt_list:

            mode_match = normalize_text(pred["failure_mode"]) == normalize_text(gt["failure_mode"])
            cause_match = normalize_text(pred["failure_cause"]) == normalize_text(gt["failure_cause"])
            effect_match = normalize_text(pred["failure_effect"]) == normalize_text(gt["failure_effect"])

            match_count = sum([mode_match, cause_match, effect_match])

            if match_count > best_match_score:
                best_match_score = match_count
                best_gt = gt

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------
        if best_match_score == 3 and best_gt and not best_gt["_matched_complete"]:
            complete_match += 1
            best_gt["_matched_complete"] = True

        elif best_match_score == 2 and best_gt:
            partial_match += 1

            print("\n============ PARTIAL MATCH (2/3) ============")

            print("\n[Prediction]")
            print(json.dumps(pred, indent=2, ensure_ascii=False))

            print("\n[Ground Truth]")
            print(json.dumps(best_gt, indent=2, ensure_ascii=False))

            # Identify mismatch field
            mode_match = normalize_text(pred["failure_mode"]) == normalize_text(best_gt["failure_mode"])
            cause_match = normalize_text(pred["failure_cause"]) == normalize_text(best_gt["failure_cause"])
            effect_match = normalize_text(pred["failure_effect"]) == normalize_text(best_gt["failure_effect"])

            print("\nMismatch Field(s):")
            if not mode_match:
                print("❌ failure_mode")
            if not cause_match:
                print("❌ failure_cause")
            if not effect_match:
                print("❌ failure_effect")

            print("==============================================\n")

        else:
            no_match += 1

    # --------------------------------------------------------
    # Metrics
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
    # Print report
    # --------------------------------------------------------

    print("\n================ EVALUATION RESULT ================\n")

    print(f"Total GT chains      : {total_gt}")
    print(f"Total Pred chains    : {total_pred}  (after dedup)")
    print("-" * 50)

    print("---- COMPLETE MATCH (3/3 fields) ----")
    print(f"Matched              : {complete_match}")
    print(f"Precision            : {precision_complete:.4f}")
    print(f"Recall               : {recall_complete:.4f}")
    print(f"F1                   : {f1_complete:.4f}")
    print("-" * 50)

    print("---- PARTIAL MATCH (exactly 2/3 fields) ----")
    print(f"Matched              : {partial_match}")
    print(f"Precision            : {precision_partial:.4f}")
    print(f"Recall               : {recall_partial:.4f}")
    print(f"F1                   : {f1_partial:.4f}")
    print("-" * 50)

    print(f"No Match (Hallucination) : {no_match}")
    print(f"Hallucination Rate       : {hallucination_rate:.4f}")

    print("\n===================================================\n")

    return {
        "complete_match": complete_match,
        "partial_match": partial_match,
        "no_match": no_match,
        "precision_complete": precision_complete,
        "recall_complete": recall_complete,
        "f1_complete": f1_complete,
        "precision_partial": precision_partial,
        "recall_partial": recall_partial,
        "f1_partial": f1_partial,
        "hallucination_rate": hallucination_rate,
    }



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    GT_JSON = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\fmea_cause_store.json")
    PREDICTION_JSON = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\failure_candidates_RAG_FILL.json")

    gt_list = load_gt(GT_JSON)
    pred_list = load_predictions(PREDICTION_JSON)

    results = evaluate(pred_list, gt_list)
