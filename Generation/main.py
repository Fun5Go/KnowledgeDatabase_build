from .LLM_function import  failure_inference_generation
from Retriever.SA_query import build_failure_chains_from_structure
from typing import Dict, List, Optional
from pathlib import Path
from langsmith import traceable
from pprint import pprint
import json

def build_structure_analysis_input(structure_input: Dict,):
    """
    Convert structure analysis dict to LLM-readable formatted text.
    """

    lines = []
    lines.append(f"Product Domain: {structure_input.get('product_domain', '')}\n")

    for node in structure_input.get("nodes", []):
        lines.append(f"Element ID: {node.get('element_id', '')}")
        lines.append(f"Failure Element: {node.get('failure_element', '')}")

        lines.append("Possible Failure Modes:")
        for m in node.get("modes", []):
            lines.append(f"  - {m}")

        lines.append("Possible Failure Causes:")
        for c in node.get("causes", []):
            lines.append(f"  - {c}")

        lines.append("Possible Failure Effects:")
        for e in node.get("effects", []):
            lines.append(f"  - {e}")

        lines.append("-" * 50)

    return "\n".join(lines)
    

def build_ground_truth_input(
    results: List[Dict],
    target_n: Optional[int] = None,
    strict_unique: bool = False,
) -> str:
    """
    Build LLM-readable GT examples.

    Args:
        results: retrieved chains
        target_n: desired number of output cases
        strict_unique:
            False → backfill duplicates if unique not enough
            True  → do NOT backfill, return fewer and warn
    """

    if not results:
        return "No similar failure chains were retrieved from the knowledge base."

    if target_n is None:
        target_n = len(results)

    def norm(x):
        if x is None:
            return ""
        return " ".join(str(x).strip().split()).lower()

    def sig(r):
        return (
            norm(r.get("element")),
            norm(r.get("function")),
            norm(r.get("mode")),
            norm(r.get("effect")),
            norm(r.get("cause")),
        )

    # -------------------------------------------------
    # 1️⃣ Collect unique chains
    # -------------------------------------------------
    selected = []
    seen = set()
    duplicates = []

    for r in results:
        s = sig(r)
        if s in seen:
            duplicates.append(r)
            continue
        seen.add(s)
        selected.append(r)
        if len(selected) >= target_n:
            break

    # -------------------------------------------------
    # 2️⃣ Backfill only if NOT strict
    # -------------------------------------------------
    if not strict_unique and len(selected) < target_n:
        need = target_n - len(selected)
        selected.extend(duplicates[:need])

    # In strict mode, do nothing (may be fewer)

    # -------------------------------------------------
    # 3️⃣ Format
    # -------------------------------------------------
    lines = []
    lines.append("Retrieved Similar FMEA Failure Chains:\n")

    if strict_unique and len(selected) < target_n:
        lines.append(
            f"⚠ WARNING: Only {len(selected)} unique chains available "
            f"(requested {target_n}). No duplicate backfilling applied.\n"
        )

    for idx, r in enumerate(selected, start=1):
        lines.append(f"Rank {idx}")
        lines.append(f"Failure ID: {r.get('failure_id')}")
        lines.append(f"Relevance Score: {r.get('score')}")
        lines.append(f"Matched Fields: {', '.join(r.get('matched_fields', []))}")

        lines.append("Failure Chain:")
        lines.append(f"  Element : {r.get('element')}")
        lines.append(f"  Function: {r.get('function')}")
        lines.append(f"  Mode    : {r.get('mode')}")
        lines.append(f"  Effect  : {r.get('effect')}")
        lines.append(f"  Cause   : {r.get('cause')}")

        # match_detail = r.get("match_detail", {}) or {}
        # if isinstance(match_detail, dict):
        #     for field_type, matches in match_detail.items():
        #         if matches:
        #             lines.append(f"  {str(field_type).upper()}:")
        #             for m in matches:
        #                 lines.append(f"    - {m}")

        lines.append("-" * 60)

    return "\n".join(lines)

def save_failure_candidates_to_json(result: dict, output_path: Path):
    """
    Save failure candidates to JSON file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"\n Failure candidates saved to: {output_path}")

@traceable(name="RAG")
def RAG_pipeline(structure_input: Dict, KB_PATH: str):

    similar_failure = build_failure_chains_from_structure(
        persist_dir=KB_PATH,
        structure_input=structure_input,
        top_k_per_field=5,
        minimum_field_match=2,
        top_n=20,
    )

    structure_input = build_structure_analysis_input(structure_input)

    failure_example = build_ground_truth_input(similar_failure,target_n=10, strict_unique=True)

    failure_candidates = failure_inference_generation.invoke({
        "data": {
            "structure_analysis": structure_input, # Sentences with annotations
            "gt_example": failure_example, # Similar FMEA cases in text format
        }
    })
    return failure_candidates

if __name__ == "__main__":
        # -----------------------------------------------------
    # 1) KB Path
    # -----------------------------------------------------
    KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb"
    )
    OUTPUT_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\failure_candidates.json"
    )

    # -----------------------------------------------------
    # 2) Structure Input
    # -----------------------------------------------------
    # structure_input = {
    #     "product_domain": "motor_drives",
    #     "nodes": [
    #         {
    #             "element_id": "E1",
    #             "failure_element": "power train",
    #             "modes": [
    #             "unstable control behavior",
    #             "Motor failure / overheating",
    #             "Insufficient torque output",
    #             "No voltage applied",
    #             "creates too much noise",

    #             ],
    #             "causes": [
    #                 "Incorrect control parameter settings",
    #                 "Thermal protection malfunction",
    #                 "Startup motor current exceeds component limits",
    #                 "Overvoltage caused by motor disconnection",
    #                 "Embedded SW migration issue (e3.2)",
                    
    #             ],
    #             "effects": [
    #                 "Improper gear shifting",
    #                 "Gear does not engage",
    #                 "too much noise",
    #                 "Motor not shorted while device is not powered",
    #                 "No connection to RC",
    #                 "Unstable cadence setting"
    #             ]
    #         }
    #     ]
    # }

    structure_input = {
    "product_domain": "motor_drives",
    "nodes": [
        {
            "element_id": "E1",
            "failure_element": "power train",
            "modes": [
                "unstable control performance",
                "motor failure or overheating",
                "insufficient torque delivery",
                "no voltage supplied",
                "excessive noise generation",
            ],
            "causes": [
                "incorrect control parameter configuration",
                "thermal protection system malfunction",
                "startup current exceeding component limits",
                "overvoltage due to motor disconnection",
                "embedded software migration issue (e3.2)",
            ],
            "effects": [
                "improper gear shifting",
                "gear fails to engage",
                "excessive noise",
                "motor not shorted when device is unpowered",
                "no connection to RC",
                "unstable cadence control",
            ]
        }
    ]
}

    result = RAG_pipeline(structure_input=structure_input, KB_PATH=KB_PATH)
    print("\n================ FAILURE CANDIDATES ================\n")
    # print(json.dumps(result, indent=4))
    save_failure_candidates_to_json(result, OUTPUT_PATH)
