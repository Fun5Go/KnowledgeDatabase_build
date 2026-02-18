import json
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("database/batch_outputs")
PURE_DIR = BASE_DIR / "PURE"
RAG_FILL_DIR = BASE_DIR / "RAG_FILL"

GT_JSON = Path("RAG/KB_motor_drives/failure_kb/fmea_cause_store.json")
TARGET_ELEMENT = "Power train"

# ============================================================
# Utility
# ============================================================

def normalize_text(x):
    if x is None:
        return ""
    return " ".join(str(x).strip().split()).lower()

def load_gt(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)

    return [
        v for v in gt_raw.values()
        if normalize_text(v.get("failure_element")) == normalize_text(TARGET_ELEMENT)
    ]

def load_predictions(pred_path):
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_raw = json.load(f)

    return [
        p for p in pred_raw.get("failure_candidates", [])
        if normalize_text(p.get("failure_element")) == normalize_text(TARGET_ELEMENT)
    ]

def evaluate_counts(pred_list, gt_list):

    unique_pred = []
    seen = set()

    for p in pred_list:
        sig = (
            normalize_text(p.get("failure_mode")),
            normalize_text(p.get("failure_cause")),
            normalize_text(p.get("failure_effect")),
        )
        if sig not in seen:
            seen.add(sig)
            unique_pred.append(p)

    pred_list = unique_pred

    for gt in gt_list:
        gt["_matched_complete"] = False

    complete = 0
    partial = 0
    no_match = 0

    for pred in pred_list:

        best_score = 0
        best_gt = None

        for gt in gt_list:

            mode_match = normalize_text(pred["failure_mode"]) == normalize_text(gt["failure_mode"])
            cause_match = normalize_text(pred["failure_cause"]) == normalize_text(gt["failure_cause"])
            effect_match = normalize_text(pred["failure_effect"]) == normalize_text(gt["failure_effect"])

            score = sum([mode_match, cause_match, effect_match])

            if score > best_score:
                best_score = score
                best_gt = gt

        if best_score == 3 and best_gt and not best_gt["_matched_complete"]:
            complete += 1
            best_gt["_matched_complete"] = True
        elif best_score == 2:
            partial += 1
        else:
            no_match += 1

    return complete, partial, no_match


# ============================================================
# Batch Evaluation
# ============================================================

pure_files = sorted(PURE_DIR.glob("*.json"))
rag_files = sorted(RAG_FILL_DIR.glob("*.json"))

runs = min(len(pure_files), len(rag_files))

pure_complete = []
pure_partial = []
pure_no = []

rag_complete = []
rag_partial = []
rag_no = []

for i in range(runs):

    gt_list = load_gt(GT_JSON)

    pc, pp, pn = evaluate_counts(load_predictions(pure_files[i]), gt_list.copy())
    pure_complete.append(pc)
    pure_partial.append(pp)
    pure_no.append(pn)

    gt_list = load_gt(GT_JSON)

    rc, rp, rn = evaluate_counts(load_predictions(rag_files[i]), gt_list.copy())
    rag_complete.append(rc)
    rag_partial.append(rp)
    rag_no.append(rn)

x = list(range(1, runs + 1))

# ============================================================
# Plot 1: Complete Match (3/3)
# ============================================================

plt.figure()
plt.plot(x, pure_complete)
plt.plot(x, rag_complete)
plt.xlabel("Run")
plt.ylabel("3/3 Match Count")
plt.title("Complete Match per Run (Stability)")
plt.legend(["PURE", "RAG_FILL"])
plt.show()

# ============================================================
# Plot 2: Partial Match (2/3)
# ============================================================

plt.figure()
plt.plot(x, pure_partial)
plt.plot(x, rag_partial)
plt.xlabel("Run")
plt.ylabel("2/3 Match Count")
plt.title("Partial Match per Run (Stability)")
plt.legend(["PURE", "RAG_FILL"])
plt.show()

# ============================================================
# Plot 3: No Match
# ============================================================

plt.figure()
plt.plot(x, pure_no)
plt.plot(x, rag_no)
plt.xlabel("Run")
plt.ylabel("No Match Count")
plt.title("No Match per Run (Stability)")
plt.legend(["PURE", "RAG_FILL"])
plt.show()
