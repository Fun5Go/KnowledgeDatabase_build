from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any
from pathlib import Path
from JSON_FMEA_KB.kb_structure import FMEAFailureKB


# -----------------------------
# simple tokenizer
# -----------------------------
def simple_tokenize(text: Any) -> List[str]:
    # 允许 None / 非 str 输入
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def _safe_text(x: Any) -> str:
    return "" if x is None else str(x)


class BM25FailureRetriever:
    def __init__(self, kb: FMEAFailureKB):
        self.kb = kb
        self.failure_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []

        for fid, entity in kb.entity_store.items():
            # entity.get(...) 可能返回 None，必须兜底成 ""
            doc = " ".join([
                _safe_text(entity.get("failure_element_text")),
                _safe_text(entity.get("failure_mode_text")),
                _safe_text(entity.get("failure_cause_text")),
                _safe_text(entity.get("failure_effect_text")),
            ])

            tokens = simple_tokenize(doc)
            self.failure_ids.append(fid)
            self.corpus_tokens.append(tokens)

        print(f"BM25 indexing {len(self.failure_ids)} failures...")
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print("BM25 index built successfully.")

    def query(self, failure_entity: Dict[str, Any], top_k: int = 20):
        query_text = " ".join([
            _safe_text(failure_entity.get("failure_element_text")),
            _safe_text(failure_entity.get("failure_mode_text")),
            _safe_text(failure_entity.get("failure_cause_text")),
            _safe_text(failure_entity.get("failure_effect_text")),
        ])

        query_tokens = simple_tokenize(query_text)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.failure_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [{"failure_id": fid, "score": float(score)} for fid, score in ranked]


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
    FAILURE_ENTITY = {
        "failure_mode_text": "Soft-start time too long",
        "failure_element_text": "Motor control",
        "failure_effect_text": "Motor overcurrent",
        "failure_cause_text": "Overvoltage from motor disconnect"
  }

    results = bm25_retriever.query(FAILURE_ENTITY, top_k=10)

    for r in results:
        print(r)