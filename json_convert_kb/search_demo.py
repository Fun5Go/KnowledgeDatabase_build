import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# Capsule store utils
# =========================================================
def load_capsule_store(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Vector store
# =========================================================
def create_vector_store(embeddings, persist_dir: Path) -> Chroma:
    return Chroma(
        collection_name="capsule_kb",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def upsert_capsules_to_chroma(vector_store: Chroma, capsules: List[Dict[str, Any]]):
    """
    Safe for re-run:
    - Use add_texts with ids; Chroma will upsert by id in most setups.
    - Embeds ONLY similarity_text.
    """
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for c in capsules:
        sim_text = (c.get("similarity_text") or "").strip()
        if not sim_text:
            continue

        capsule_id = c.get("capsule_id")
        if not capsule_id:
            continue

        meta = {
            # keep whatever you already store
            "capsule_id": capsule_id,
            "failure_id": c.get("failure_id"),
            "product": c.get("product_name") or c.get("product"),
            "discipline": c.get("discipline"),
            "cause_level": c.get("cause_level"),
            "confidence": c.get("confidence"),
            "failure_element": c.get("failure_element"),
            "failure_mode": c.get("failure_mode"),
        }

        texts.append(sim_text)
        metadatas.append(meta)
        ids.append(capsule_id)

    if texts:
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)


# =========================================================
# Similarity-only search (no capsule_store usage)
# =========================================================
def search_similarity_only(
    vector_store: Chroma,
    query_text: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Any, float]]:
    """
    PURE vector similarity search.
    Prints Chroma-returned docs and metadata only.
    """
    results = vector_store.similarity_search_with_score(
        query_text,
        k=k,
        filter=filters,
    )

    print("\n================ SEARCH RESULTS ================\n")
    print("[QUERY]", query_text)
    print("[FILTERS]", filters)

    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\n#{rank}  score={score}")
        print(">>> METADATA:")
        print(doc.metadata)
        print("\n>>> TEXT:")
        print(doc.page_content)
        print("-" * 70)

    return results


# =========================================================
# Demo main
# =========================================================
if __name__ == "__main__":

    # -------- Paths --------
    kb_dir = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\chroma_capsule_kb"
    )
    kb_dir.mkdir(parents=True, exist_ok=True)

    capsule_store_path = kb_dir / "capsule_store.json"

    # -------- Load capsule store (optional; only needed if you want to upsert) --------
    capsule_store = load_capsule_store(capsule_store_path)
    print(f"[INFO] Capsule store loaded: {len(capsule_store)} capsules")

    # -------- Embeddings --------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------- Vector store --------
    vector_store = create_vector_store(embeddings, kb_dir)

    # -------- OPTIONAL: (re)upsert everything from capsule_store into chroma --------
    # If your chroma is already built, you can comment this out.
    # capsules = list(capsule_store.values())
    # upsert_capsules_to_chroma(vector_store, capsules)
    # print("[INFO] Upsert done")

    # -------- Query --------
    query = "show me possible circuit failure"
    filters = {"discipline": "HW"}

    search_similarity_only(
        vector_store,
        query_text=query,
        k=3,
        filters=filters,
    )
