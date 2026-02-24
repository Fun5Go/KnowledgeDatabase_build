import json
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
from itertools import islice

ROOT = Path(__file__).resolve().parents[1] 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ..failure_query_tools import retrieve_similar_failures_from_entity
from ..bm25_baseline import BM25FailureRetriever


GT_JSON_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\sample_10pct_rephrased.json"
KB_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb")


TOP_K = 5
# N_RESULTS_EACH_ROLE = 25  # role-level retrieval size (can tune)


def precision_at_k(pred: List[str], gt: str, k: int) -> float:
    return (1.0 if gt in pred[:k] else 0.0) / float(k)

def reciprocal_rank(pred: List[str], gt: str) -> float:
    for i, p in enumerate(pred, start=1):
        if p == gt:
            return 1.0 / i
    return 0.0

def recall_at_k(pred: List[str], gt: str, k: int) -> float:
    # Single ground-truth per query
    return 1.0 if gt in pred[:k] else 0.0

def build_entity(item: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Map json item to chunks input expected by query_failure_kb_by_chunks."""
    return {
        "failure_mode_text": item.get("failure_mode_text"),
        "failure_element_text": "",
        "failure_effect_text": item.get("failure_effect_text"),
        "failure_cause_text": item.get("failure_cause_text"),
    }

def evaluate(
    gt_data: Dict[str, Dict[str, Any]],
    persist_dir: str,
    top_k: int = TOP_K,
    method: str = "dense",   # "dense" | "bm25"
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Evaluate retrieval performance.

    method:
        - "dense"  : semantic retriever
        - "bm25"   : lexical baseline

    Returns:
        avg_mrr
        avg_recall_at_k
        details
    """

    mrrs: List[float] = []
    recalls: List[float] = []
    details: List[Dict[str, Any]] = []

    # --------------------------------------------------
    # Build retrievers ONCE
    # --------------------------------------------------
    kb = FMEAFailureKB(Path(persist_dir))

    bm25_retriever = None
    if method == "bm25":
        bm25_retriever = BM25FailureRetriever(kb)

    # --------------------------------------------------
    # Loop over GT samples
    # --------------------------------------------------
    for gt_failure_id, item in islice(gt_data.items(), 100):

        entity = build_entity(item)

        # ------------------------------
        # Retrieve
        # ------------------------------
        if method == "dense":
            ranked = retrieve_similar_failures_from_entity(
                persist_dir=persist_dir,
                failure_entity=entity,
                top_n=30,
                min_similarity=0.4,
                top_k_per_field=15,
            ) or []

        elif method == "bm25":
            ranked = bm25_retriever.query(entity, top_k=30)

        else:
            raise ValueError(f"Unknown method: {method}")

        # ------------------------------
        # Extract top_k predictions
        # ------------------------------
        pred_topk_failure_ids = [
            r.get("failure_id")
            for r in ranked[:top_k]
            if r.get("failure_id") is not None
        ]

        final_score = [
            r.get("score")
            for r in ranked[:top_k]
        ]

        # ------------------------------
        # Metrics
        # ------------------------------
        rr = reciprocal_rank(pred_topk_failure_ids, gt_failure_id)
        hit = gt_failure_id in pred_topk_failure_ids
        r_at_k = 1.0 if hit else 0.0

        mrrs.append(rr)
        recalls.append(r_at_k)

        details.append({
            "item_key": gt_failure_id,
            "gt_failure_id": gt_failure_id,
            "pred_topk_failure_ids": pred_topk_failure_ids,
            "score": final_score,
            "hit_at_k": hit,
            "reciprocal_rank": rr,
            "recall_at_k": r_at_k,
            "method": method,
        })

    # --------------------------------------------------
    # Final Metrics
    # --------------------------------------------------
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    return avg_mrr, avg_recall, details


def main():
    with open(GT_JSON_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    dense_mrr, dense_recall, _ = evaluate(
        gt_data,
        persist_dir=KB_PATH,
        top_k=10,
        method="dense"
    )

    bm25_mrr, bm25_recall, _ = evaluate(
        gt_data,
        persist_dir=KB_PATH,
        top_k=10,
        method="bm25"
    )

    print("Dense  MRR:", dense_mrr)
    print("Dense  Recall@10:", dense_recall)

    print("BM25   MRR:", bm25_mrr)
    print("BM25   Recall@10:", bm25_recall)

    # out_path = Path(f"eval_top{TOP_K}_details_multiple.json")
    # out_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    # print(f"Saved details to: {out_path.resolve()}")

if __name__ == "__main__":
    main()