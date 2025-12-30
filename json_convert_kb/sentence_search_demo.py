import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions


# ======================
# Config
# ======================
KB_ROOT = Path(
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\kb"
).resolve()

SENTENCE_KB_DIR = (KB_ROOT / "sentence_kb").resolve()
FAILURE_INDEX_PATH = (KB_ROOT / "failure_index_kb.jsonl").resolve()

TOP_K = 5
MIN_FAITHFUL = 90


# ======================
# Load Failure Index KB
# ======================
def load_failure_index(path: Path):
    failures = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            failures[obj["failure_id"]] = obj
    return failures


# ======================
# Load Sentence KB
# ======================
def load_sentence_kb(persist_dir):
    client = chromadb.PersistentClient(
        path=str(persist_dir)
    )

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="sentences",
        embedding_function=embedder
    )
    return collection


# ======================
# Search Demo
# ======================
def search(query: str):
    print("=" * 60)
    print("QUERY:", query)
    print("=" * 60)

    # 1) Load KBs
    collection = load_sentence_kb(SENTENCE_KB_DIR)
    print("Total sentences in KB:", collection.count())
    failure_index = load_failure_index(FAILURE_INDEX_PATH)

    # 2) Sentence-level similarity search
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        where={"faithful_score": {"$gte": MIN_FAITHFUL}}
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not ids:
        print("No matching sentences found.")
        return

    # 3) Print results + map to failure
    for i, sid in enumerate(ids):
        meta = metas[i]
        dist = dists[i]

        failure_id = meta.get("failure_id")
        failure = failure_index.get(failure_id)

        print(f"\n[{i+1}] Similar Sentence")
        print("-" * 40)
        print("Sentence ID :", sid)
        print("Text        :", docs[i])
        print("Distance    :", round(dist, 4))
        print("Case ID     :", meta.get("case_id"))
        print("Section     :", meta.get("source_section"))
        print("Assertion   :", meta.get("assertion_level"))
        print("Faithful    :", meta.get("faithful_score"))

        if failure:
            print("→ Failure ID:", failure["failure_id"])
            print("  Element   :", failure.get("failure_element"))
            print("  Mode      :", failure.get("failure_mode"))
            print("  Status    :", failure.get("status"))
        else:
            print("→ Failure   : (not indexed)")

    print("\nDone.")


# ======================
# Main
# ======================
if __name__ == "__main__":
    query = "device blew up during voltage dip test"
    search(query)
