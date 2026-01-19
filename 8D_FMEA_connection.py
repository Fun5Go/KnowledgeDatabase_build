import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from JSON_FMEA_KB.kb_structure import FMEACauseKB
from JSON8D_KB.kb_structure import CauseKB


def match_8d_cause_to_fmea(
    cause_kb: CauseKB,
    fmea_kb: FMEACauseKB,
    cause_id_8d: str,
    similarity_threshold: float = 0.75,
):
    # 8D cause vectors
    cause_8d = cause_kb.collection.get(
        ids=[cause_id_8d],
        include=["embeddings", "metadatas"],
    )
    vec_8d = np.array(cause_8d["embeddings"][0])
    meta_8d = cause_8d["metadatas"][0]

    # FMEA cause vectors
    fmea_all = fmea_kb.collection.get(
        include=["embeddings", "metadatas", "ids"],
    )

    matches = []

    for fmea_id, vec_fmea, meta_fmea in zip(
        fmea_all["ids"],
        fmea_all["embeddings"],
        fmea_all["metadatas"],
    ):
        # hard filter
        if meta_fmea.get("discipline") != meta_8d.get("discipline"):
            continue

        sim = cosine_similarity(
            [vec_8d],
            [vec_fmea],
        )[0][0]

        if sim >= similarity_threshold:
            matches.append({
                "fmea_cause_id": fmea_id,
                "similarity": float(sim),
                "failure_id": meta_fmea.get("failure_id"),
            })

    # rank
    matches.sort(key=lambda x: x["similarity"], reverse=True)

    return matches

def main():
    BASE_DIR = Path(__file__).resolve().parent

    # Root folder containing many FMEA JSON files
    FMEA_CAUSE_KB_PATH = BASE_DIR / "JSON_FMEA_KB"/"kb_data"/"fmea_cause_kb"
    D_CAUSE_KB_PATH = BASE_DIR / "JSON8D_KB"/"kb_data"/"cause_kb"

    cause_kb_fmea = CauseKB(FMEA_CAUSE_KB_PATH)
    cause_kb_8D = CauseKB(D_CAUSE_KB_PATH)

    match_8d_cause_to_fmea(
        cause_kb_fmea,
        cause_kb_8D,
    )



if __name__ == "__main__":
    main()