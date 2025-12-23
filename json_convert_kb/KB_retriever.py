# kb_retrieval.py
from pathlib import Path
from typing import Dict
import json

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# Load KB
# =========================================================
def load_kb(persist_dir: Path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="fmea_evidence_capsules",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    capsule_store_path = persist_dir / "capsules.json"
    with open(capsule_store_path, "r", encoding="utf-8") as f:
        capsule_store = json.load(f)

    return vector_store, capsule_store


# =========================================================
# Retrieval
# =========================================================
def search_similar_failures(
    vector_store,
    capsule_store: Dict,
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
        cid = d.metadata["capsule_id"]
        cap = capsule_store.get(cid, {})

        print("=" * 90)
        print(f"Rank {rank} | Capsule ID: {cid}")
        print(">>> Similarity basis:")
        print(d.page_content)
        print("\n---- Reasoning text ----")
        print(cap.get("reasoning_text", "[missing reasoning_text]"))
        print("\n>>> Metadata:")
        print(d.metadata)

    return docs


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    kb_dir = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\chroma_capsule_kb"
    )

    vector_store, capsule_store = load_kb(kb_dir)

    search_similar_failures(
        vector_store,
        capsule_store,
        query_text="the current failure in electronics domain",
        k=3,
    )



