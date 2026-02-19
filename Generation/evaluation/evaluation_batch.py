import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt


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
    # 1️⃣ Build match matrix
    # --------------------------------------------------------
    match_matrix = []

    for pred in pred_list:
        row = []
        for gt in gt_list:

            mode_match = normalize_text(pred["failure_mode"]) == normalize_text(gt["failure_mode"])
            cause_match = normalize_text(pred["failure_cause"]) == normalize_text(gt["failure_cause"])
            effect_match = normalize_text(pred["failure_effect"]) == normalize_text(gt["failure_effect"])

            score = sum([mode_match, cause_match, effect_match])
            row.append(score)

        match_matrix.append(row)

    # --------------------------------------------------------
    # 2️⃣ Matching (strict first, then partial)
    # --------------------------------------------------------
    matched_pred = set()
    matched_gt = set()

    complete_match = 0
    partial_match = 0

    # strict (3/3)
    for i in range(total_pred):
        for j in range(total_gt):
            if match_matrix[i][j] == 3:
                if i not in matched_pred and j not in matched_gt:
                    matched_pred.add(i)
                    matched_gt.add(j)
                    complete_match += 1

    # partial (2/3)
    for i in range(total_pred):
        if i in matched_pred:
            continue

        for j in range(total_gt):
            if match_matrix[i][j] == 2 and j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(j)
                partial_match += 1
                break

    no_match = total_pred - len(matched_pred)

    # --------------------------------------------------------
    # 3️⃣ Metrics
    # --------------------------------------------------------

    # strict
    precision_complete = complete_match / total_pred if total_pred else 0
    recall_complete = complete_match / total_gt if total_gt else 0
    f1_complete = (
        2 * precision_complete * recall_complete / (precision_complete + recall_complete)
        if precision_complete + recall_complete > 0 else 0
    )

    # partial (only 2/3)
    precision_partial = partial_match / total_pred if total_pred else 0
    recall_partial = partial_match / total_gt if total_gt else 0
    f1_partial = (
        2 * precision_partial * recall_partial / (precision_partial + recall_partial)
        if precision_partial + recall_partial > 0 else 0
    )

    # relaxed (>=2/3)
    relaxed_match = complete_match + partial_match

    precision_relaxed = relaxed_match / total_pred if total_pred else 0
    recall_relaxed = relaxed_match / total_gt if total_gt else 0
    f1_relaxed = (
        2 * precision_relaxed * recall_relaxed / (precision_relaxed + recall_relaxed)
        if precision_relaxed + recall_relaxed > 0 else 0
    )

    hallucination_rate = no_match / total_pred if total_pred else 0

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

            mode_match = normalize_text(pred["failure_mode"]) == normalize_text(gt["failure_mode"])
            cause_match = normalize_text(pred["failure_cause"]) == normalize_text(gt["failure_cause"])
            effect_match = normalize_text(pred["failure_effect"]) == normalize_text(gt["failure_effect"])

            score_matrix[i, j] = sum([mode_match, cause_match, effect_match])

    # --------------------------------------------------------
    # 2️⃣ Hungarian Algorithm (maximize score)
    # --------------------------------------------------------
    # linear_sum_assignment minimizes cost, so we negate
    row_ind, col_ind = linear_sum_assignment(-score_matrix)

    complete_match = 0
    partial_match = 0
    matched_pred = set()

    for r, c in zip(row_ind, col_ind):
        score = score_matrix[r, c]

        if score == 3:
            complete_match += 1
            matched_pred.add(r)
        elif score == 2:
            partial_match += 1
            matched_pred.add(r)

    relaxed_match = complete_match + partial_match
    no_match = total_pred - len(matched_pred)

    # --------------------------------------------------------
    # 3️⃣ Metrics
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

    precision_relaxed = relaxed_match / total_pred if total_pred else 0
    recall_relaxed = relaxed_match / total_gt if total_gt else 0
    f1_relaxed = (
        2 * precision_relaxed * recall_relaxed / (precision_relaxed + recall_relaxed)
        if precision_relaxed + recall_relaxed > 0 else 0
    )

    hallucination_rate = no_match / total_pred if total_pred else 0

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

def plot_batch_counts(run_results, title="Batch - Match Counts"):
    """
    run_results: list[dict]，长度可为10或不足10
    每个 dict 至少包含 complete_match/partial_match/no_match，最好包含 file
    """
    n = len(run_results)
    runs = np.arange(1, n + 1)

    complete = np.array([r.get("complete_match", 0) for r in run_results], dtype=float)
    partial  = np.array([r.get("partial_match", 0)  for r in run_results], dtype=float)
    nomatch  = np.array([r.get("no_match", 0)       for r in run_results], dtype=float)

    labels = [r.get("file", f"run{i}") for i, r in enumerate(run_results, start=1)]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 5))
    ax.bar(runs, complete, label="complete_match")
    ax.bar(runs, partial, bottom=complete, label="partial_match")
    ax.bar(runs, nomatch, bottom=complete + partial, label="no_match")

    ax.set_xticks(runs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Run (file)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    totals = complete + partial + nomatch
    for x, t in zip(runs, totals):
        ax.text(x, t, f"{int(t)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_in_batches(all_results, batch_size=10, drop_last=False):
    """
    all_results: 你收集到的所有 evaluate_strict 结果 list
    batch_size: 每张图包含多少次（默认10）
    drop_last: True 则最后不足 batch_size 的不画
    """
    total = len(all_results)
    for start in range(0, total, batch_size):
        batch = all_results[start:start + batch_size]
        if drop_last and len(batch) < batch_size:
            break
        batch_id = start // batch_size + 1
        plot_batch_counts(batch, title=f"Batch #{batch_id} ({len(batch)} runs) - Match Counts")

if __name__ == "__main__":

    GT_JSON = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\RAG\KB_motor_drives\failure_kb\fmea_cause_store.json"
    )

    PRED_FOLDER = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\batch_outputs\RAG_FILL"
    )

    gt_list = load_gt(GT_JSON)

    all_results = []

    print("\n========== BATCH EVALUATION ==========\n")

    for pred_file in sorted(PRED_FOLDER.glob("*.json")):

        pred_list = load_predictions(pred_file)
        result = evaluate_strict(pred_list, gt_list.copy())

        result["file"] = pred_file.name
        all_results.append(result)

        print(
            f"{pred_file.name} | "
            f"F1_complete: {result['f1_complete']:.4f} | "
            f"F1_relaxed: {result['f1_relaxed']:.4f}"
        )
    plot_in_batches(all_results, batch_size=10, drop_last=False)

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

