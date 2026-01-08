from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import chromadb
from chromadb.utils import embedding_functions


# =========================================================
# Config
# =========================================================

ROLES = ["failure_element", "failure_mode", "failure_effect"]

DEFAULT_K = 5

# Outlier rule: mean + sigma * factor
SIGMA_FACTOR = 2.0


# =========================================================
# Helpers
# =========================================================

def mean_std(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


# =========================================================
# Core Semantic Evaluation
# =========================================================

class SemanticEvaluator:
    """
    Semantic distance evaluation for KB construction stage.
    Read-only, no mutation.
    """

    def __init__(self, failure_kb_dir: Path):
        self.failure_kb_dir = Path(failure_kb_dir)

        self.client = chromadb.PersistentClient(
            path=str(self.failure_kb_dir)
        )

        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_collection(
            name="failure_kb",
            embedding_function=self.embedder,
        )

    # -----------------------------------------------------
    # SD1: Intra-role semantic cohesion
    # -----------------------------------------------------

    def evaluate_role_cohesion(
        self,
        role: str,
        k: int = DEFAULT_K,
    ) -> Dict[str, Any]:
        """
        For each item under a role, compute avg distance to k nearest
        neighbors under the SAME role (excluding itself).
        """

        res = self.collection.get(
            where={"role": role},
            include=["documents", "metadatas"],
        )

        ids = res["ids"]
        docs = res["documents"]

        items = []
        distances_all = []

        for _id, text in zip(ids, docs):
            q = self.collection.query(
                query_texts=[text],
                n_results=k + 1,  # include itself
                where={"role": role},
            )

            dists = q["distances"][0]

            # drop self (distance ~ 0)
            dists = [float(d) for d in dists if d > 1e-6]

            avg_dist = safe_mean(dists)
            if avg_dist is not None:
                distances_all.append(avg_dist)

            items.append({
                "id": _id,
                "text": text,
                "avg_neighbor_distance": avg_dist,
            })

        mean_d, std_d = mean_std(distances_all)

        threshold = None
        if mean_d is not None and std_d is not None:
            threshold = mean_d + SIGMA_FACTOR * std_d

        # mark outliers
        outliers = []
        for it in items:
            if (
                threshold is not None
                and it["avg_neighbor_distance"] is not None
                and it["avg_neighbor_distance"] > threshold
            ):
                outliers.append(it)

        return {
            "role": role,
            "count": len(items),
            "mean_distance": mean_d,
            "std_distance": std_d,
            "outlier_threshold": threshold,
            "items": items,
            "outliers": outliers,
        }

    # -----------------------------------------------------
    # SD2: Cross-role semantic confusion scan
    # -----------------------------------------------------

    def evaluate_role_confusion_all(
        self,
        k: int = 3,
        margin: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Scan all failure embeddings and detect role confusion.
        margin: how much better the best role must be than current role
        """

        # 1) load all items
        all_items = []
        for role in ROLES:
            res = self.collection.get(
                where={"role": role},
                include=["documents", "metadatas"],
            )
            for _id, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                all_items.append({
                    "id": _id,
                    "text": text,
                    "current_role": role,
                    "failure_id": meta.get("failure_id"),
                })

        confusions = []

        # 2) for each item, check best role
        for it in all_items:
            text = it["text"]
            if not isinstance(text, str) or not text.strip():
                continue

            scores = {}
            for role in ROLES:
                q = self.collection.query(
                    query_texts=[text],
                    n_results=k,
                    where={"role": role},
                )
                if q["distances"] and q["distances"][0]:
                    scores[role] = float(min(q["distances"][0]))
                else:
                    scores[role] = None

            # find best role
            best_role = None
            best_dist = float("inf")
            for r, d in scores.items():
                if d is not None and d < best_dist:
                    best_dist = d
                    best_role = r

            cur_role = it["current_role"]
            cur_dist = scores.get(cur_role)

            if (
                best_role
                and best_role != cur_role
                and cur_dist is not None
                and (cur_dist - best_dist) > margin
            ):
                confusions.append({
                    "id": it["id"],
                    "failure_id": it["failure_id"],
                    "text": text,
                    "current_role": cur_role,
                    "best_role": best_role,
                    "scores": scores,
                    "delta": round(cur_dist - best_dist, 6),
                })

        return {
            "total_scanned": len(all_items),
            "confusion_count": len(confusions),
            "margin": margin,
            "confusions": confusions,
        }

    # -----------------------------------------------------
    # SD3: Extreme isolation scan (embedding usability)
    # -----------------------------------------------------

    def extreme_distance_scan(
        self,
        role: str,
        k: int = DEFAULT_K,
        hard_threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """
        Flag items that are far from ANY neighbor under same role.
        """
        cohesion = self.evaluate_role_cohesion(role, k)
        anomalies = []

        for it in cohesion["items"]:
            d = it["avg_neighbor_distance"]
            if d is not None and d > hard_threshold:
                anomalies.append(it)

        return anomalies

class CauseSemanticEvaluator:
    def __init__(self, cause_kb_dir: Path):
        self.cause_kb_dir = Path(cause_kb_dir)

        self.client = chromadb.PersistentClient(
            path=str(self.cause_kb_dir)
        )

        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_collection(
            name="cause_kb",
            embedding_function=self.embedder,
        )

    def evaluate_cohesion(
        self,
        k: int = DEFAULT_K,
    ) -> Dict[str, Any]:

        res = self.collection.get(include=["documents"])
        items = []
        distances_all = []

        for _id, text in zip(res["ids"], res["documents"]):
            q = self.collection.query(
                query_texts=[text],
                n_results=k + 1,
            )
            dists = [float(d) for d in q["distances"][0] if d > 1e-6]
            avg_dist = safe_mean(dists)

            if avg_dist is not None:
                distances_all.append(avg_dist)

            items.append({
                "id": _id,
                "text": text,
                "avg_neighbor_distance": avg_dist,
            })

        mean_d, std_d = mean_std(distances_all)
        threshold = None
        if mean_d is not None and std_d is not None:
            threshold = mean_d + SIGMA_FACTOR * std_d

        outliers = [
            it for it in items
            if it["avg_neighbor_distance"] is not None
            and threshold is not None
            and it["avg_neighbor_distance"] > threshold
        ]

        return {
            "count": len(items),
            "mean_distance": mean_d,
            "std_distance": std_d,
            "outlier_threshold": threshold,
            "outliers": outliers,
        }


# =========================================================
# Runner / Report
# =========================================================

def run_semantic_evaluation(
    failure_kb_dir: Path,
    cause_kb_dir: Path,
    output_path: Path,
    k: int = DEFAULT_K,
):
    evaluator = SemanticEvaluator(failure_kb_dir)

    report: Dict[str, Any] = {
        "config": {
            "k": k,
            "sigma_factor": SIGMA_FACTOR,
        },
        "failure": {},
        "cause": {},
    }

    # ---- Failure SD1 + SD3 ----
    for role in ROLES:
        cohesion = evaluator.evaluate_role_cohesion(role, k=k)
        extreme = evaluator.extreme_distance_scan(role, k=k)

        report["failure"][role] = {
            "cohesion": {
                "count": cohesion["count"],
                "mean_distance": cohesion["mean_distance"],
                "std_distance": cohesion["std_distance"],
                "outlier_threshold": cohesion["outlier_threshold"],
                "outliers": cohesion["outliers"],
            },
            "extreme_distance_anomalies": extreme,
        }

    # ---- Failure SD2 ----
    report["failure"]["role_confusion"] = evaluator.evaluate_role_confusion_all(
        k=3,
        margin=0.05,
    )

    # ---- Cause semantic evaluation ----
    cause_eval = CauseSemanticEvaluator(cause_kb_dir)
    report["cause"]["cohesion"] = cause_eval.evaluate_cohesion(k=k)

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Semantic evaluation report written to: {output_path}")
    return report

# =========================================================
# CLI entry
# =========================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    KB_DATA_ROOT = BASE_DIR.parent / "kb_data"
    FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"
    CAUSE_KB_DIR = KB_DATA_ROOT / "cause_kb"

    OUTPUT = BASE_DIR / "semantic_evaluation_report.json"

    run_semantic_evaluation(
        failure_kb_dir=FAILURE_KB_DIR,
        cause_kb_dir=CAUSE_KB_DIR,
        output_path=OUTPUT,
        k=5,
    )
