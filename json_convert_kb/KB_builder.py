# kb_builder.py
import json
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def merge_and_save_capsule_store(
    capsules: List[Dict],
    capsule_store_path: Path,
):
    """
    Merge capsules into existing capsule_store.json
    - capsule_id is the primary key
    - existing capsules are updated
    - new capsules are appended
    """

    # 1️⃣ Load existing store (if exists)
    if capsule_store_path.exists():
        with open(capsule_store_path, "r", encoding="utf-8") as f:
            capsule_store = json.load(f)
    else:
        capsule_store = {}

    # 2️⃣ Merge (upsert)
    for cap in capsules:
        cid = cap["capsule_id"]
        capsule_store[cid] = cap   # upsert

    # 3️⃣ Write back
    with open(capsule_store_path, "w", encoding="utf-8") as f:
        json.dump(
            capsule_store,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[CAPSULE STORE] Total capsules: {len(capsule_store)} "
        f"(+{len(capsules)} ingested)"
    )


# =========================================================
# Build Evidence Capsules
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

    failure_symptoms = []
    failure_actions = []

    for ent in failure.get("supporting_entities", []):
        if not ent.get("text"):
            continue
        if ent.get("entity_type") == "symptom":
            failure_symptoms.append(ent["text"])
        elif ent.get("entity_type") == "action":
            failure_actions.append(ent["text"])

    for rc in failure.get("root_causes", []):
        cause_id = rc.get("cause_ID")
        root_cause = rc.get("failure_cause")
        cause_level = rc.get("cause_level")
        discipline = rc.get("discipline_type")
        confidence = rc.get("confidence")

        rc_evidence_texts = [
            ent["text"]
            for ent in rc.get("supporting_entities", [])
            if ent.get("text")
        ]

        # ---------- similarity_text ----------
        similarity_lines = [
            f"Failure mode: {failure_mode}",
            f"Failure element: {failure_element}",
            f"Failure effect: {failure_effect}",
            f"Root cause: {root_cause}",
        ]
        for txt in rc_evidence_texts[:3]:
            similarity_lines.append(f"Evidence: {txt}")

        similarity_text = "\n".join(similarity_lines)

        # ---------- reasoning_text ----------
        reasoning_lines = []

        reasoning_lines.append("Failure description:")
        for s in set(failure_symptoms):
            reasoning_lines.append(f"- {s}")

        if failure_effect:
            reasoning_lines.append(f"- Failure effect: {failure_effect}")

        reasoning_lines.append("\nRoot cause:")
        reasoning_lines.append(root_cause)

        reasoning_lines.append("\nSupporting evidence:")
        for txt in set(rc_evidence_texts):
            reasoning_lines.append(f"- {txt}")

        if failure_actions:
            reasoning_lines.append("\nRelated actions:")
            for act in set(failure_actions):
                reasoning_lines.append(f"- {act}")

        reasoning_lines.append(f"\nConfidence: {confidence}")

        capsules.append({
            "capsule_id": cause_id,
            "failure_id": failure_id,
            "product": product_name,
            "failure_mode": failure_mode,
            "failure_element": failure_element,
            "root_cause": root_cause,
            "cause_level": cause_level,
            "discipline": discipline,
            "confidence": confidence,
            "similarity_text": similarity_text,
            "reasoning_text": "\n".join(reasoning_lines),
        })

    return capsules


# =========================================================
# Ingest Capsules into Chroma + Save Capsule Store
# =========================================================
def ingest_capsules_to_kb(capsules: List[Dict], persist_dir: Path):
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="fmea_evidence_capsules",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    vector_store.add_texts(
        texts=[c["similarity_text"] for c in capsules],
        metadatas=[
            {
                "capsule_id": c["capsule_id"],
                "discipline": c["discipline"],
                "failure_element": c["failure_element"],
                "cause_level": c["cause_level"],
                "confidence": c["confidence"],
            }
            for c in capsules
        ],
        ids=[c["capsule_id"] for c in capsules],
    )

    # ⭐ 保存 capsule_store
    capsule_store_path = persist_dir / "capsules.json"

    merge_and_save_capsule_store(
        capsules=capsules,
        capsule_store_path=capsule_store_path,
    )

    print(f"[KB BUILDER] Ingested {len(capsules)} capsules")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    json_path = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\eightD_json_raw\8D6318110147R01.json")
    kb_dir = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\chroma_capsule_kb"
    )


    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    capsules = build_evidence_capsules(data)
    ingest_capsules_to_kb(capsules, kb_dir)
