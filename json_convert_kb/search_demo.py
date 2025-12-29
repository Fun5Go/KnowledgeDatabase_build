import json
from pathlib import Path
from typing import Dict

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# Load Capsule Store
# =========================================================
def load_capsule_store(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Capsule store not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Load Vector Store (READ-ONLY)
# =========================================================
def load_vector_store(persist_dir: Path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        collection_name="fmea_evidence_capsules",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


# =========================================================
# Similar Failure Search + Reasoning Display
# =========================================================
def search_similar_failures(
    vector_store,
    capsule_store: Dict[str, Dict],
    query_text: str,
    k: int = 5,
    filters: Dict = None,
):
    docs = vector_store.similarity_search(
        query_text,
        k=k,
        filter=filters,
    )

    for rank, d in enumerate(docs, start=1):
        cid = d.metadata.get("capsule_id")
        cap = capsule_store.get(cid)

        print("=" * 100)
        print(f"Rank {rank} | Capsule ID: {cid}")

        print("\n>>> EMBEDDED TEXT:")
        print(d.page_content)

        if cap is None:
            print("\n[WARN] Capsule not found in capsule_store")
            print("Metadata:", d.metadata)
            continue

        print("\n---- Reasoning text ----")
        print(cap["reasoning_text"])

        print("\n>>> METADATA:")
        print(d.metadata)

    return docs


# =========================================================
# Standalone Search Test
# =========================================================
if __name__ == "__main__":

    kb_dir = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\chroma_capsule_kb"
    )

    capsule_store_path = kb_dir / "capsule_store.json"

    # -------- Load KB --------
    capsule_store = load_capsule_store(capsule_store_path)
    vector_store = load_vector_store(kb_dir)

    print(f"[INFO] Capsule store size: {len(capsule_store)}")

    # -------- Test Query --------
    query = "show me the current failure"

    filters = {
        "discipline": "HW",
    }

    search_similar_failures(
        vector_store=vector_store,
        capsule_store=capsule_store,
        query_text=query,
        k=3,
        filters=filters,
    )
