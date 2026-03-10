from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any
from pathlib import Path
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
import numpy as np


# -----------------------------
# Utils
# -----------------------------
def simple_tokenize(text: Any) -> List[str]:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def _safe_text(x: Any) -> str:
    return "" if x is None else str(x)


# -----------------------------
# Field-aware BM25
# -----------------------------
class BM25SemanticFieldRetriever:
    """
    BM25 retriever built on field_store (semantic-level, deduplicated).
    """
    def __init__(
        self,
        kb,
        weight_element: float = 0.5,
        weight_mode: float = 1.5,
        weight_cause: float = 1.5,
        weight_effect: float = 1.5,
    ):
        self.kb = kb

        self.weight_element = weight_element
        self.weight_mode = weight_mode
        self.weight_cause = weight_cause
        self.weight_effect = weight_effect

        self.semantic_ids = {
            "element": [],
            "mode": [],
            "cause": [],
            "effect": [],
        }

        self.field_text = {
            "element": {},
            "mode": {},
            "cause": {},
            "effect": {},
        }

        corpus = {
            "element": [],
            "mode": [],
            "cause": [],
            "effect": [],
        }

        print("Building semantic-level BM25 from field_store...")

        # ✅ 正确遍历平铺结构
        for sid, node in kb.field_store.items():

            field = node.get("field_type")

            if field not in self.semantic_ids:
                continue

            text = _safe_text(node.get("text", ""))

            self.semantic_ids[field].append(sid)
            self.field_text[field][sid] = text
            corpus[field].append(simple_tokenize(text))

        # Debug
        for f in corpus:
            print(f"{f} node count:", len(corpus[f]))

        # Build BM25
        self.bm25 = {
            f: BM25Okapi(corpus[f]) if corpus[f] else None
            for f in corpus
        }

        print("Semantic BM25 built successfully.")



    # -----------------------------
    # Query
    # -----------------------------
    def query(
        self,
        failure_entity: Dict[str, Any],
        top_k: int = 30,
    ) -> List[Dict[str, Any]]:

        queries = {
            "element": failure_entity.get("failure_element_text", ""),
            "mode": failure_entity.get("failure_mode_text", ""),
            "cause": failure_entity.get("failure_cause_text", ""),
            "effect": failure_entity.get("failure_effect_text", ""),
        }

        weights = {
            "element": self.weight_element,
            "mode": self.weight_mode,
            "cause": self.weight_cause,
            "effect": self.weight_effect,
        }

        # semantic-level score aggregation
        semantic_scores = {}

        for field, query_text in queries.items():

            if not query_text or self.bm25[field] is None:
                continue

            tokens = simple_tokenize(_safe_text(query_text))
            scores = self.bm25[field].get_scores(tokens)

            for idx, score in enumerate(scores):

                if score <= 0:
                    continue

                sid = self.semantic_ids[field][idx]

                semantic_scores.setdefault(sid, 0.0)
                semantic_scores[sid] += weights[field] * float(score)

        # rank semantic nodes
        ranked = sorted(
            semantic_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for sid, score in ranked:

            # detect field from prefix if needed
            field = sid.split(":")[0]

            results.append({
                "semantic_id": sid,
                "score": float(score),
                "field": field,
                "text": self.field_text[field][sid],
                "failure_ids": self.kb.field_store[field][sid].get("failure_ids", [])
            })

        return results
    def query_single_field(
        self,
        query_text: str,
        field: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:

        if field not in self.bm25 or self.bm25[field] is None:
            return []

        tokens = simple_tokenize(_safe_text(query_text))
        scores = self.bm25[field].get_scores(tokens)

        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_idx:

            sid = self.semantic_ids[field][idx]
            score = float(scores[idx])

            results.append({
                "semantic_id": sid,
                "score": score,
                "field": field,
                "text": self.field_text[field][sid],

                "failure_ids": self.kb.field_store[sid].get("failure_ids", [])
            })

        return results

# -----------------------------
# Example standalone usage
# -----------------------------
if __name__ == "__main__":
    KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_discipline\failure_kb"
    )

    kb = FMEAFailureKB(KB_PATH)

    bm25_retriever = BM25SemanticFieldRetriever(kb)

#     Example query
#     FAILURE_ENTITY = {
#         "failure_mode_text": "Soft-start time too long",
#         "failure_element_text": "Motor control",
#         "failure_effect_text": "Motor overcurrent",
#         "failure_cause_text": "Overvoltage from motor disconnect"
#   }

#     results = bm25_retriever.query(FAILURE_ENTITY, top_k=10)

#     for r in results:
#         print(r)
    results = bm25_retriever.query_single_field(
    query_text="Motor can not provide enough torque",
    field="cause",
    top_k=10
    )
    # print(results)

    for r in results:
        print(r)