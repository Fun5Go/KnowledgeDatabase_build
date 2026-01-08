# test_search_chroma.py
# Purpose:
# 1) Verify whether a specific field-embedding (e.g., failure_element) exists in Chroma
# 2) Inspect the stored document text (to catch " ", "-", "N/A", etc.)
# 3) Optionally run a small similarity query under a specific role

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions


def make_client_and_collection(persist_dir: Path, collection_name: str):
    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(f"Persist dir not found: {persist_dir}")

    client = chromadb.PersistentClient(path=str(persist_dir))

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Use get_collection (NOT create) so the test fails loudly if name is wrong
    col = client.get_collection(
        name=collection_name,
        embedding_function=embedder,
    )
    return client, col


def test_get_by_ids(col, fid: str):
    """Check if the 3 role-split IDs exist and print their documents."""
    ids = [
        f"{fid}::failure_element",
        f"{fid}::failure_mode",
        f"{fid}::failure_effect",
    ]
    res = col.get(ids=ids, include=["documents", "metadatas"])

    print("\n=== GET BY IDS ===")
    if not res["ids"]:
        print("No IDs found (none of the 3 exist).")
        return

    for _id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        shown = doc.replace("\n", "\\n") if isinstance(doc, str) else doc
        print(f"\nID: {_id}")
        print(f"DOCUMENT: {repr(shown)}")
        print(f"ROLE(meta): {meta.get('role')}")
        print(f"FAILURE_ID(meta): {meta.get('failure_id')}")
        # Print full meta if you want:
        # print("META:", meta)


def test_role_query(col, query_text: str, role: str, k: int = 5):
    """Run a similarity query restricted to a role and print top-k hits."""
    res = col.query(
        query_texts=[query_text],
        n_results=k,
        where={"role": role},
        include=["documents", "metadatas", "distances"],
    )

    print("\n=== ROLE QUERY ===")
    print(f"QUERY: {repr(query_text)} | ROLE: {role} | k={k}")

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not ids:
        print("No results.")
        return

    for rank, (_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
        shown = doc.replace("\n", "\\n") if isinstance(doc, str) else doc
        print(f"\n#{rank}")
        print(f"ID: {_id}")
        print(f"DISTANCE: {dist:.6f}  (lower = more similar)")
        print(f"DOCUMENT: {repr(shown)}")
        print(f"FAILURE_ID(meta): {meta.get('failure_id')}")
        print(f"ROLE(meta): {meta.get('role')}")


if __name__ == "__main__":
    # ---- Adjust these to your project ----
    # This should be the SAME folder that contains chroma.sqlite3 for the FailureKB
    FAILURE_KB_DIR = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\JSON_FMEA_KB\kb_data\failure_kb")

    # Use your actual collection name
    COLLECTION_NAME = "fmea_failure_kb"

    # Pick a failure_id that you want to inspect
    FID = "FMEA6022160004R03__F23"

    # ---- Create collection handle ----
    _, collection = make_client_and_collection(FAILURE_KB_DIR, COLLECTION_NAME)

    # ---- 1) Check whether the split IDs exist and what text is stored ----
    test_get_by_ids(collection, FID)

    # ---- 2) Optional: run role-restricted similarity query ----
    # Example: check what "empty-ish" queries retrieve
    test_role_query(collection, query_text="motor assembly", role="failure_element", k=5)
    test_role_query(collection, query_text="-", role="failure_mode", k=5)
