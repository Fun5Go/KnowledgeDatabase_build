import json
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# 1) Build Evidence Capsules (NO LLM)
# =========================================================
def build_evidence_capsules(data: Dict) -> List[Dict]:
    """
    Build evidence capsules from 8D JSON.
    Rules:
    - 1 root cause -> 1 capsule
    - similarity_text: used for embedding & similarity retrieval
    - reasoning_text: used for explanation / UI
    """

    capsules = []

    doc0 = (data.get("documents") or [{}])[0]
    product_name = doc0.get("product_name")

    failure = data.get("failure") or {}
    failure_id = failure.get("failure_ID")
    failure_mode = failure.get("failure_mode")
    failure_element = failure.get("failure_element")
    failure_effect = failure.get("failure_effect")

    # -----------------------
    # Failure-level entities
    # -----------------------
    failure_symptoms = []
    failure_actions = []

    for ent in failure.get("supporting_entities", []):
        if not ent.get("text"):
            continue
        if ent.get("entity_type") == "symptom":
            failure_symptoms.append(ent["text"])
        elif ent.get("entity_type") == "action":
            failure_actions.append(ent["text"])

    # -----------------------
    # Root causes -> capsules
    # -----------------------
    for rc in failure.get("root_causes", []):
        cause_id = rc.get("cause_ID")
        root_cause = rc.get("failure_cause")
        cause_level = rc.get("cause_level")
        discipline = rc.get("discipline_type")
        confidence = rc.get("confidence")
        inferred_insight = rc.get("inferred_insight")

        rc_evidence_texts = []
        rc_evidence_items = []

        for ent in rc.get("supporting_entities", []):
            if not ent.get("text"):
                continue
            rc_evidence_texts.append(ent["text"])
            rc_evidence_items.append({
                "text": ent.get("text"),
                "source_section": ent.get("source_section"),
                "signal_id": ent.get("id"),
                "entity_type": ent.get("entity_type"),
            })

        # =====================
        # similarity_text (EMBED)
        # =====================
        similarity_lines = [
            f"Failure mode: {failure_mode}",
            f"Failure element: {failure_element}",
            f"Failure effect: {failure_effect}",
            f"Root cause: {root_cause}",
        ]

        for txt in rc_evidence_texts[:3]:  # limit noise
            similarity_lines.append(f"Evidence: {txt}")

        similarity_text = "\n".join(similarity_lines)

        # =====================
        # reasoning_text (EXPLAIN)
        # =====================
        reasoning_lines = []

        reasoning_lines.append("Failure description:")
        seen = set()
        for s in failure_symptoms:
            if s not in seen:
                reasoning_lines.append(f"- {s}")
                seen.add(s)

        if failure_effect:
            reasoning_lines.append(f"- Failure effect: {failure_effect}")

        reasoning_lines.append("\nRoot cause:")
        reasoning_lines.append(root_cause)

        if inferred_insight:
            reasoning_lines.append(f"\nInsight:\n{inferred_insight}")

        reasoning_lines.append("\nSupporting evidence:")
        seen = set()
        for txt in rc_evidence_texts:
            if txt not in seen:
                reasoning_lines.append(f"- {txt}")
                seen.add(txt)

        if failure_actions:
            reasoning_lines.append("\nRelated actions / workarounds:")
            seen = set()
            for act in failure_actions:
                if act not in seen:
                    reasoning_lines.append(f"- {act}")
                    seen.add(act)

        reasoning_lines.append(f"\nConfidence: {confidence}")

        reasoning_text = "\n".join(reasoning_lines)

        capsules.append({
            "capsule_id": cause_id,
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
            "reasoning_text": reasoning_text,
            "root_cause_evidence": rc_evidence_items,
            "failure_symptoms": failure_symptoms,
            "failure_actions": failure_actions,
        })

    return capsules


# =========================================================
# 2) Create / Load Chroma Vector Store
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
        texts.append(cap["similarity_text"])
        metadatas.append({
            "capsule_id": cap["capsule_id"],
            "failure_id": cap["failure_id"],
            "failure_element": cap["failure_element"],
            "discipline": cap["discipline"],
            "cause_level": cap["cause_level"],
            "confidence": cap["confidence"],
            "product": cap["product"],
        })
        ids.append(cap["capsule_id"])  # unique ID for each capsule

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )


# =========================================================
# 3) Similar Failure Search + Reasoning Display
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
        cid = d.metadata["capsule_id"]
        cap = capsule_store[cid]

        print("=" * 90)
        print(f"Rank {rank} | Capsule ID: {cid}")
        print(">>> EMBEDDED TEXT (page_content):")
        print(d.page_content)
        print("\n---- Reasoning text ----")
        print(cap["reasoning_text"])

        print("=" * 90)

        print(">>> METADATA:")
        print(d.metadata)

    return docs


# =========================================================
# 4) End-to-End Demo
# =========================================================
if __name__ == "__main__":

    # -------- Paths --------
    json_path = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\eightD_json_raw\8D620721025401.json")
    kb_dir = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\chroma_capsule_kb"
    )

    # -------- Load 8D JSON --------
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -------- Build capsules --------
    capsules = build_evidence_capsules(data)
    print(f"[INFO] Generated {len(capsules)} evidence capsules")

    # -------- Capsule store (for reasoning lookup) --------
    capsule_store = {c["capsule_id"]: c for c in capsules}

    # -------- Embeddings --------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------- Vector store --------
    vector_store = create_vector_store(embeddings, kb_dir)

    # Add capsules only once (comment out after first run if needed)
    add_capsules_to_chroma(vector_store, capsules)
    print("[INFO] Capsules added to Chroma vector store")

    # -------- Query --------
    query = "PCB failure"
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
