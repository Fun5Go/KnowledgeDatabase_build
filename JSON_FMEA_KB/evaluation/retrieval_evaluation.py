import json
from pathlib import Path
from typing import List, Dict, Optional, Set

from query_fmea import retrieve_failures, retrieve_causes


# =========================================================
# Utilities
# =========================================================
def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON parse error in {path} at line {lineno}:\n{line}"
                ) from e
    return rows


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def key_of(row: Dict) -> str:
    # 同一 failure / cause pair 作为 anchor
    return f"{row.get('failure_id')}/{row.get('cause_id')}"


# =========================================================
# Adapters (use YOUR retrieval code)
# =========================================================
def failure_retrieve_adapter(
    failure_mode: Optional[str],
    failure_element: Optional[str],
    failure_effect: Optional[str],
    top_k: int = 3,
) -> List[str]:
    failure_ids, _ = retrieve_failures(
        failure_mode=failure_mode,
        failure_element=failure_element,
        failure_effect=failure_effect,
        top_k=top_k,
    )
    return list(failure_ids)


def cause_retrieve_adapter(
    cause_query: str,
    failure_id: str,
    top_k: int = 5,
) -> List[str]:
    cause_ids, _ = retrieve_causes(
        cause_query=cause_query,
        failure_id=failure_id,
        top_k=top_k,
    )
    return list(cause_ids)


# =========================================================
# Core Evaluation (Failure Accuracy Focus)
# =========================================================
def eval_pair(
    orig_row: Dict,
    reph_row: Dict,
    k_failure: int = 3,
) -> Dict:

    gt_failure_id = orig_row.get("failure_id")

    f_orig = failure_retrieve_adapter(
        failure_mode=orig_row.get("failure_mode"),
        failure_element=orig_row.get("failure_element"),
        failure_effect=orig_row.get("failure_effect"),
        top_k=k_failure,
    )

    f_reph = failure_retrieve_adapter(
        failure_mode=reph_row.get("failure_mode"),
        failure_element=reph_row.get("failure_element"),
        failure_effect=reph_row.get("failure_effect"),
        top_k=k_failure,
    )

    return {
        "failure": {
            "gt_failure_id": gt_failure_id,

            "orig_ids": f_orig,
            "orig_top1_correct": bool(f_orig and f_orig[0] == gt_failure_id),
            "orig_topk_hit": gt_failure_id in f_orig,

            "reph_ids": f_reph,
            "reph_top1_correct": bool(f_reph and f_reph[0] == gt_failure_id),
            "reph_topk_hit": gt_failure_id in f_reph,
        }
    }


# =========================================================
# Batch Runner
# =========================================================
def run_evaluation(
    orig_path: str,
    reph_path: str,
    out_path: str = "eval_report.jsonl",
    k_failure: int = 3,
):
    orig_rows = load_jsonl(orig_path)
    reph_rows = load_jsonl(reph_path)

    orig_map = {key_of(r): r for r in orig_rows}
    reph_map = {key_of(r): r for r in reph_rows}

    keys = sorted(orig_map.keys() & reph_map.keys())
    if not keys:
        raise RuntimeError("No matching pairs found.")

    stats = {
        "n": 0,
        "orig_failure_top1_acc": 0,
        "orig_failure_topk_recall": 0,
        "reph_failure_top1_acc": 0,
        "reph_failure_topk_recall": 0,
        "rephrase_regression": 0,
    }

    reports = []

    for k in keys:
        orig_row = orig_map[k]
        reph_row = reph_map[k]

        res = eval_pair(orig_row, reph_row, k_failure=k_failure)
        f = res["failure"]

        stats["n"] += 1
        stats["orig_failure_top1_acc"] += int(f["orig_top1_correct"])
        stats["orig_failure_topk_recall"] += int(f["orig_topk_hit"])
        stats["reph_failure_top1_acc"] += int(f["reph_top1_correct"])
        stats["reph_failure_topk_recall"] += int(f["reph_topk_hit"])

        # -------- PRINT FAILURE MISMATCH --------
        if not f["orig_top1_correct"]:
            print("\n[FAILURE MISMATCH - ORIGINAL]")
            print("Key:", k)
            print("GT failure:", f["gt_failure_id"])
            print("Retrieved:", f["orig_ids"])
            print("Mode:", orig_row.get("failure_mode"))
            print("Element:", orig_row.get("failure_element"))
            print("Effect:", orig_row.get("failure_effect"))

        if f["orig_top1_correct"] and not f["reph_top1_correct"]:
            stats["rephrase_regression"] += 1
            print("\n[REGRESSION AFTER REPHRASE]")
            print("Key:", k)
            print("GT failure:", f["gt_failure_id"])
            print("Original retrieved OK")
            print("Rephrased retrieved:", f["reph_ids"])
            print("Rephrased mode:", reph_row.get("failure_mode"))
            print("Rephrased element:", reph_row.get("failure_element"))
            print("Rephrased effect:", reph_row.get("failure_effect"))

        reports.append({
            "key": k,
            "result": res,
        })

    # normalize
    n = stats["n"]
    for k in list(stats.keys()):
        if k != "n":
            stats[k] /= n

    # write output
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": stats}, ensure_ascii=False) + "\n")
        for r in reports:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nDetailed report written to: {out_path}")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    run_evaluation(
        orig_path=BASE_DIR / "origninal_sample_20_failure_symptom_cause.jsonl",
        reph_path=BASE_DIR / "rephrased_sample_20_failure_symptom_cause.jsonl",
        out_path="eval_report.jsonl",
        k_failure=3,
    )
