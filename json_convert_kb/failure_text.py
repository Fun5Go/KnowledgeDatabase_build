import json
from pathlib import Path
from typing import List, Dict


def build_evidence_capsules(data: Dict) -> List[Dict]:
    """
    Build evidence capsules from 8D JSON.
    - One capsule per root cause
    - Root-cause evidence drives similarity
    - Failure-level evidence enriches reasoning only
    """

    capsules = []

    # ------------------------
    # Document & failure info
    # ------------------------
    doc0 = (data.get("documents") or [{}])[0]
    product_name = doc0.get("product_name")

    failure = data.get("failure") or {}

    failure_id = failure.get("failure_ID")
    failure_mode = failure.get("failure_mode")
    failure_element = failure.get("failure_element")
    failure_effect = failure.get("failure_effect")

    # ------------------------
    # Failure-level entities
    # ------------------------
    failure_symptoms = []
    failure_actions = []

    for ent in failure.get("supporting_entities", []):
        etype = ent.get("entity_type")
        text = ent.get("text")
        if not text:
            continue

        if etype == "symptom":
            failure_symptoms.append(text)
        elif etype == "action":
            failure_actions.append(text)

    # ------------------------
    # Root cause → capsule
    # ------------------------
    for rc in failure.get("root_causes", []):

        cause_id = rc.get("cause_ID")
        root_cause = rc.get("failure_cause")
        cause_level = rc.get("cause_level")
        discipline = rc.get("discipline_type")
        confidence = rc.get("confidence")
        inferred_insight = rc.get("inferred_insight")

        # ---- Root-cause evidence ----
        rc_evidence_texts = []
        rc_evidence_items = []

        for ent in rc.get("supporting_entities", []):
            text = ent.get("text")
            if not text:
                continue

            rc_evidence_texts.append(text)
            rc_evidence_items.append({
                "text": text,
                "source_section": ent.get("source_section"),
                "signal_id": ent.get("id"),
                "entity_type": ent.get("entity_type")
            })

        # ============================
        # Similarity text (EMBED THIS)
        # ============================
        similarity_lines = [
            f"Failure mode: {failure_mode}",
            f"Failure element: {failure_element}",
            f"Root cause: {root_cause}"
        ]

        # add only high-signal evidence (limit to 3)
        for txt in rc_evidence_texts[:3]:
            similarity_lines.append(f"Evidence: {txt}")

        similarity_text = "\n".join(similarity_lines)

        # ============================
        # Reasoning text (UI / explain)
        # ============================
        reasoning_lines = []

        # Failure context
        reasoning_lines.append("Failure description:")
        if failure_symptoms:
            for s in failure_symptoms:
                reasoning_lines.append(f"- {s}")
        else:
            reasoning_lines.append(f"- {failure_mode}")

        if failure_effect:
            reasoning_lines.append(f"- Failure effect: {failure_effect}")

        # Root cause
        reasoning_lines.append("\nRoot cause:")
        reasoning_lines.append(root_cause)

        if inferred_insight:
            reasoning_lines.append(f"\nInsight:\n{inferred_insight}")

        # Evidence
        reasoning_lines.append("\nSupporting evidence:")
        if rc_evidence_texts:
            for txt in rc_evidence_texts:
                reasoning_lines.append(f"- {txt}")
        else:
            reasoning_lines.append("- No explicit evidence listed")

        # Actions (linked but NOT causal)
        if failure_actions:
            reasoning_lines.append("\nRelated actions / workarounds:")
            for act in failure_actions:
                reasoning_lines.append(f"- {act}")

        reasoning_lines.append(f"\nConfidence: {confidence}")

        reasoning_text = "\n".join(reasoning_lines)

        # ------------------------
        # Capsule object
        # ------------------------
        capsule = {
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

            # texts
            "similarity_text": similarity_text,
            "reasoning_text": reasoning_text,

            # traceability
            "root_cause_evidence": rc_evidence_items,
            "failure_symptoms": failure_symptoms,
            "failure_actions": failure_actions,
        }

        capsules.append(capsule)

    return capsules

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":

    json_path = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\eightD_json_raw\8D620721025401.json")


    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    capsules = build_evidence_capsules(data)

    # ---- output preview ----
    print(f"Generated {len(capsules)} evidence capsules\n")

    for i, cap in enumerate(capsules, start=1):
        print("=" * 80)
        print(f"Capsule {i}: {cap['capsule_id']}")
        print("---- Similarity text (EMBED) ----")
        print(cap["similarity_text"])
        print("\n---- Reasoning text (EXPLAIN) ----")
        print(cap["reasoning_text"])

    # ---- optional: save to file ----
    out_path = json_path.with_name("8D_evidence_capsules_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(capsules, f, indent=2, ensure_ascii=False)

    print(f"\nCapsules saved to: {out_path}")
