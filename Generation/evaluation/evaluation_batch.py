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

def evaluate(pred_list: List[Dict], gt_list: List[Dict]):

    def chain_signature(x):
        return (
            normalize_text(x.get("failure_mode")),
            normalize_text(x.get("failure_cause")),
            normalize_text(x.get("failure_effect")),
        )

    # Deduplicate predictions
    unique_pred = []
    seen = set()

    for p in pred_list:
        sig = chain_signature(p)
        if sig not in seen:
            seen.add(sig)
            unique_pred.append(p)

    pred_list = unique_pred

    for gt in gt_list:
        gt["_matched_complete"] = False

    total_gt = len(gt_list)
    total_pred = len(pred_list)

    complete_match = 0
    partial_match = 0
    no_match = 0

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

        if best_match_score == 3 and best_gt and not best_gt["_matched_complete"]:
            complete_match += 1
            best_gt["_matched_complete"] = True

        elif best_match_score == 2:
            partial_match += 1
        else:
            no_match += 1

    # ---------------------------
    # Metrics
    # ---------------------------

    relaxed_match = complete_match + partial_match

    precision_complete = complete_match / total_pred if total_pred else 0
    recall_complete = complete_match / total_gt if total_gt else 0

    precision_relaxed = relaxed_match / total_pred if total_pred else 0
    recall_relaxed = relaxed_match / total_gt if total_gt else 0

    f1_complete = (
        2 * precision_complete * recall_complete / (precision_complete + recall_complete)
        if precision_complete + recall_complete > 0 else 0
    )

    f1_relaxed = (
        2 * precision_relaxed * recall_relaxed / (precision_relaxed + recall_relaxed)
        if precision_relaxed + recall_relaxed > 0 else 0
    )

    hallucination_rate = no_match / total_pred if total_pred else 0

    return {
        "complete_match": complete_match,
        "partial_match": partial_match,
        "relaxed_match": relaxed_match,
        "no_match": no_match,
        "precision_complete": precision_complete,
        "recall_complete": recall_complete,
        "f1_complete": f1_complete,
        "precision_relaxed": precision_relaxed,
        "recall_relaxed": recall_relaxed,
        "f1_relaxed": f1_relaxed,
        "hallucination_rate": hallucination_rate,
    }


if __name__ == "__main__":

    GT_JSON = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\fmea_cause_store.json"
    )

    PRED_FOLDER = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\batch_outputs\RAG"
    )

    gt_list = load_gt(GT_JSON)

    all_results = []

    print("\n========== BATCH EVALUATION ==========\n")

    for pred_file in sorted(PRED_FOLDER.glob("*.json")):

        pred_list = load_predictions(pred_file)
        result = evaluate(pred_list, gt_list.copy())

        result["file"] = pred_file.name
        all_results.append(result)

        print(
            f"{pred_file.name} | "
            f"F1_complete: {result['f1_complete']:.4f} | "
            f"F1_relaxed: {result['f1_relaxed']:.4f}"
        )

    # =====================================================
    # Average & STD
    # =====================================================

    if all_results:

        import statistics

        def avg(key):
            return sum(r[key] for r in all_results) / len(all_results)

        def std(key):
            return statistics.pstdev([r[key] for r in all_results])

        print("\n========== AVERAGE RESULTS ==========\n")

        print("---- STRICT (3/3) ----")
        print(f"Avg Precision : {avg('precision_complete'):.4f}")
        print(f"Avg Recall    : {avg('recall_complete'):.4f}")
        print(f"Avg F1        : {avg('f1_complete'):.4f}")
        print(f"F1 STD        : {std('f1_complete'):.4f}")

        print("\n---- RELAXED (≥2 fields) ----")
        print(f"Avg Precision : {avg('precision_relaxed'):.4f}")
        print(f"Avg Recall    : {avg('recall_relaxed'):.4f}")
        print(f"Avg F1        : {avg('f1_relaxed'):.4f}")
        print(f"F1 STD        : {std('f1_relaxed'):.4f}")

        print("\n---- HALLUCINATION ----")
        print(f"Avg Hallucination Rate : {avg('hallucination_rate'):.4f}")

