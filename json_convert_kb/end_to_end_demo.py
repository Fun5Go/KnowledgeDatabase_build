import json
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# 1) Build Evidence Capsules (NO LLM)
# =========================================================
def build_evidence_capsules(data: Dict) -> List[Dict]:
    capsules = []

    doc0 = (data.get("documents") or [{}])[0]
    product_name = doc0.get("product_name")

    failure = data.get("failure") or {}
    failure_id = failure.get("failure_ID")
    failure_mode = failure.get("failure_mode")
    failure_element = failure.get("failure_element")
    failure_effect = failure.get("failure_effect")

    # ========== NEW supporting_entities ==========
    supporting_entities = failure.get("supporting_entities", [])

    # -------- Failure-level facts --------
    failure_symptoms = []
    failure_conditions = []
    failure_occurrences = []
    failure_investigations = []

    for ent in supporting_entities:
        text = ent.get("text")
        if not text:
            continue

        etype = ent.get("entity_type")

        if etype == "symptom":
            failure_symptoms.append(text)
        elif etype == "condition":
            failure_conditions.append(text)
        elif etype == "occurrence":
            failure_occurrences.append(text)
        elif etype == "root_cause_evidence":
            failure_investigations.append(text)

    # -------- Root causes -> capsules --------
    for rc in failure.get("root_causes", []):
        capsule_id = rc.get("cause_ID")
        root_cause = rc.get("failure_cause")
        cause_level = rc.get("cause_level")
        discipline = rc.get("discipline_type")
        confidence = rc.get("confidence")

        # -------- Root-cause evidence (NEW) --------
        rc_evidence_items = []
        rc_evidence_texts = []

        for ent in supporting_entities:
            if ent.get("entity_type") == "root_cause_evidence":
                rc_evidence_items.append({
                    "id": ent.get("id"),
                    "text": ent.get("text"),
                    "source_section": ent.get("source_section"),
                    "assertion_level": ent.get("assertion_level"),
                })
                rc_evidence_texts.append(ent.get("text"))

        # =====================
        # similarity_text (NO assertion_level)
        # =====================
        similarity_lines = [
            f"Failure mode: {failure_mode}",
            f"Failure element: {failure_element}",
            f"Failure effect: {failure_effect}",
            f"Root cause: {root_cause}",
        ]

        for txt in rc_evidence_texts[:3]:
            similarity_lines.append(f"Evidence: {txt}")

        similarity_text = "\n".join(similarity_lines)

        # =====================
        # reasoning_text (ASSERTION-AWARE)
        # =====================
        reasoning_lines = ["Failure description:"]

        for s in failure_symptoms:
            reasoning_lines.append(f"- {s}")

        if failure_conditions:
            reasoning_lines.append("\nConditions:")
            for s in failure_conditions:
                reasoning_lines.append(f"- {s}")

        if failure_occurrences:
            reasoning_lines.append("\nOccurrences:")
            for s in failure_occurrences:
                reasoning_lines.append(f"- {s}")

        if failure_investigations:
            reasoning_lines.append("\nInvestigations:")
            for s in failure_investigations:
                reasoning_lines.append(f"- {s}")

        if failure_effect:
            reasoning_lines.append(f"\nFailure effect:\n- {failure_effect}")

        reasoning_lines.append("\nRoot cause:")
        reasoning_lines.append(root_cause)

        reasoning_lines.append("\nSupporting evidence:")
        for ev in rc_evidence_items:
            lvl = ev.get("assertion_level", "unknown")
            reasoning_lines.append(f"- [{lvl}] {ev['text']}")

        reasoning_lines.append(f"\nConfidence: {confidence}")

        capsules.append({
            "capsule_id": capsule_id,
            "source_type": "8D",
            "product": product_name,
            "failure_id": failure_id,
            "failure_mode": failure_mode,
            "failure_element": failure_element,
            "root_cause": root_cause,
            "cause_level": cause_level,
            "discipline": discipline,
            "confidence": confidence,
            "similarity_text": similarity_text,
            "reasoning_text": "\n".join(reasoning_lines),

            # 保留原子事实（为下一步 sentence-level 做准备）
            "root_cause_evidence": rc_evidence_items,
        })

    return capsules



# =========================================================
# 2) Capsule Store (GLOBAL KB)
# =========================================================
def load_capsule_store(path: Path) -> Dict[str, Dict]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_capsule_store(store: Dict[str, Dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


# =========================================================
# 3) Vector Store (Chroma)
# =========================================================
def create_vector_store(embeddings, persist_dir: Path):
    return Chroma(
        collection_name="fmea_evidence_capsules",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def add_capsules_to_chroma(vector_store, capsules):
    texts = []
    metadatas = []
    ids = []

    for cap in capsules:
        cid = cap["capsule_id"]

        texts.append(cap["similarity_text"])
        metadatas.append({
            "capsule_id": cid,
            "failure_id": cap["failure_id"],
            "failure_element": cap["failure_element"],
            "discipline": cap["discipline"],
            "cause_level": cap["cause_level"],
            "confidence": cap["confidence"],
            "product": cap["product"],
        })
        ids.append(cid)

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
    )


# =========================================================
# 4) Similar Failure Search + Reasoning Display
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
# 5) End-to-End Demo (FULL KB MODE)
# =========================================================
if __name__ == "__main__":

    # -------- Paths --------
    json_path = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\eightD_json_raw\8D6318110135R04.json"
    )

    kb_dir = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\chroma_capsule_kb"
    )
    kb_dir.mkdir(parents=True, exist_ok=True)

    capsule_store_path = kb_dir / "capsule_store.json"

    # -------- Load 8D JSON --------
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------- Build capsules --------
    capsules = build_evidence_capsules(data)
    print(f"[INFO] Generated {len(capsules)} capsules from this file")

    # -------- Load + merge GLOBAL capsule store --------
    capsule_store = load_capsule_store(capsule_store_path)
    for c in capsules:
        capsule_store[c["capsule_id"]] = c
    save_capsule_store(capsule_store, capsule_store_path)

    print(f"[INFO] Global capsule store size: {len(capsule_store)}")

    # -------- Embeddings --------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------- Vector store --------
    vector_store = create_vector_store(embeddings, kb_dir)

    # -------- Add to Chroma (id-based, safe for re-run) --------
    add_capsules_to_chroma(vector_store, capsules)
    print("[INFO] Capsules added to Chroma")

    # -------- Query --------
    query = "show me the current failure"
    print("\n[QUERY]", query)

    filters = {
        "discipline": "HW",
    }

    search_similar_failures(
        vector_store,
        capsule_store,
        query_text=query,
        k=3,
        filters=filters,
    )
