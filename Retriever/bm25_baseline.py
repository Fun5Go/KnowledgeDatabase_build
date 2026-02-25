from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any
from pathlib import Path
from JSON_FMEA_KB.kb_structure import FMEAFailureKB


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
class BM25FailureRetriever:
    def __init__(
        self,
        kb,
        weight_element: float = 0.5,
        weight_mode: float = 1.5,
        weight_cause: float = 1.5,
        weight_effect: float = 1.5,
    ):
        self.kb = kb
        self.failure_ids: List[str] = []

        self.weight_element = weight_element
        self.weight_mode = weight_mode
        self.weight_cause = weight_cause
        self.weight_effect = weight_effect

        # Separate corpora per field
        corpus_element = []
        corpus_mode = []
        corpus_cause = []
        corpus_effect = []
        self.field_text = {
            "element": {},
            "mode": {},
            "cause": {},
            "effect": {},
        }

        for fid, entity in kb.entity_store.items():
            self.failure_ids.append(fid)

            corpus_element.append(
                simple_tokenize(_safe_text(entity.get("failure_element_text")))
            )
            corpus_mode.append(
                simple_tokenize(_safe_text(entity.get("failure_mode_text")))
            )
            corpus_cause.append(
                simple_tokenize(_safe_text(entity.get("failure_cause_text")))
            )
            corpus_effect.append(
                simple_tokenize(_safe_text(entity.get("failure_effect_text")))
            )

        print(f"Building field-wise BM25 for {len(self.failure_ids)} failures...")

        self.bm25_element = BM25Okapi(corpus_element)
        self.bm25_mode = BM25Okapi(corpus_mode)
        self.bm25_cause = BM25Okapi(corpus_cause)
        self.bm25_effect = BM25Okapi(corpus_effect)
        print("Field-wise BM25 index built successfully.")

        for fid, entity in kb.entity_store.items():
            self.failure_ids.append(fid)

            element_text = _safe_text(entity.get("failure_element_text"))
            mode_text = _safe_text(entity.get("failure_mode_text"))
            cause_text = _safe_text(entity.get("failure_cause_text"))
            effect_text = _safe_text(entity.get("failure_effect_text"))

            # cache raw text
            self.field_text["element"][fid] = element_text
            self.field_text["mode"][fid] = mode_text
            self.field_text["cause"][fid] = cause_text
            self.field_text["effect"][fid] = effect_text

            corpus_element.append(simple_tokenize(element_text))
            corpus_mode.append(simple_tokenize(mode_text))
            corpus_cause.append(simple_tokenize(cause_text))
            corpus_effect.append(simple_tokenize(effect_text))



    # -----------------------------
    # Query
    # -----------------------------
    def query(self, failure_entity: Dict[str, Any], top_k: int = 20):

        q_element = simple_tokenize(
            _safe_text(failure_entity.get("failure_element_text"))
        )
        q_mode = simple_tokenize(
            _safe_text(failure_entity.get("failure_mode_text"))
        )
        q_cause = simple_tokenize(
            _safe_text(failure_entity.get("failure_cause_text"))
        )
        q_effect = simple_tokenize(
            _safe_text(failure_entity.get("failure_effect_text"))
        )

        # Get per-field scores
        scores_element = self.bm25_element.get_scores(q_element)
        scores_mode = self.bm25_mode.get_scores(q_mode)
        scores_cause = self.bm25_cause.get_scores(q_cause)
        scores_effect = self.bm25_effect.get_scores(q_effect)

        # Weighted fusion
        final_scores = []
        for i in range(len(self.failure_ids)):
            score = (
                self.weight_element * scores_element[i] +
                self.weight_mode * scores_mode[i] +
                self.weight_cause * scores_cause[i] +
                self.weight_effect * scores_effect[i]
            )
            final_scores.append(score)

        ranked = sorted(
            zip(self.failure_ids, final_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            {"failure_id": fid, "score": float(score)}
            for fid, score in ranked
        ]
    def query_single_field(
    self,
    query_text: str,
    field: str,
    top_k: int = 20
):
        """
        Single-field BM25 retrieval.

        Args:
            query_text: raw text query
            field: one of ["element", "mode", "cause", "effect"]
            top_k: number of results

        Returns:
            List[{"failure_id": str, "score": float}]
        """

        tokens = simple_tokenize(_safe_text(query_text))

        if field == "element":
            scores = self.bm25_element.get_scores(tokens)

        elif field == "mode":
            scores = self.bm25_mode.get_scores(tokens)

        elif field == "cause":
            scores = self.bm25_cause.get_scores(tokens)

        elif field == "effect":
            scores = self.bm25_effect.get_scores(tokens)

        else:
            raise ValueError(
                "field must be one of ['element', 'mode', 'cause', 'effect']"
            )

        ranked = sorted(
            zip(self.failure_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            {
                "failure_id": fid,
                "score": float(score),
                "field": field,
                "text": self.field_text[field][fid],  # specific text
            }
            for fid, score in ranked
        ]


# -----------------------------
# Example standalone usage
# -----------------------------
if __name__ == "__main__":
    KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb"
    )

    kb = FMEAFailureKB(KB_PATH)

    bm25_retriever = BM25FailureRetriever(kb)

    # Example query
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
    query_text="Motor overheat",
    field="mode",
    top_k=10
)

    for r in results:
        print(r)