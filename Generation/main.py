from .LLM_function import  failure_inference_generation_RAG, failure_inference_generation_PURE
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

    if not strict_unique and len(selected) < target_n:
        need = target_n - len(selected)
        selected.extend(duplicates[:need])

    def extract_matched_inputs(match_detail: dict) -> dict:
        """
        Return {field_type: 'a | b | c'} extracted from hit['structure_text'].
        Dedup + preserve order.
        """
        out = {}
        for ft in ["element", "mode", "cause", "effect"]:
            hits = (match_detail or {}).get(ft, []) or []
            seen_txt = set()
            texts = []
            for h in hits:
                t = (h.get("structure_text") or "").strip()
                if not t:
                    continue
                key = norm(t)
                if key in seen_txt:
                    continue
                seen_txt.add(key)
                texts.append(t)
            if texts:
                out[ft] = " | ".join(texts)
        return out

    lines = []
    lines.append("GROUND TRUTH FAILURE PATTERN:\n")

    for idx, r in enumerate(selected, start=1):
        element = r.get("element")
        function = r.get("function") or "N/A"
        mode = r.get("mode") or "N/A"
        effect = r.get("effect") or "N/A"
        cause = r.get("cause") or "N/A"

        lines.append(f"Pattern {idx}")
        lines.append(f"Failure ID: {r.get('failure_id')}")
        # lines.append(f"Relevance Score: {r.get('score')}")
        # lines.append(f"Matched Fields: {', '.join(r.get('matched_fields', []))}")

        lines.append(
            f'In element "{element}", the function "{function}" results in effect "{effect}" '
            f'when failure mode "{mode}" occurs, which is caused by "{cause}".'
        )
        # Only add a compact "Matched input: ..." line
        matched_inputs = extract_matched_inputs(r.get("match_detail", {}))
        if matched_inputs:
            # keep stable order
            parts = []
            for ft in ["element", "mode", "cause", "effect"]:
                if ft in matched_inputs:
                    parts.append(f'{ft} "{matched_inputs[ft]}"')
            lines.append("Matched structrure analysis text: " + ", ".join(parts))

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
def RAG_pipeline(structure_input: Dict, KB_PATH: str, top_n: int = 25,top_k_per_field: int = 10, 
    max_hits_per_field_per_failure: int = 2, require_cause: bool = False,
    require_cause_plus: bool = False, RAG: bool = True, ):



    # structure_input = build_structure_analysis_input(structure_input)
    structure_input_json = json.dumps(structure_input, ensure_ascii=False,indent=2)

    if RAG == True:
        similar_failure = build_failure_chains_from_structure(
        persist_dir=KB_PATH,
        structure_input=structure_input,
        top_k_per_field=top_k_per_field,
        top_n=top_n,
        max_hits_per_field_per_failure = max_hits_per_field_per_failure,
        require_cause = require_cause,
        require_cause_plus = require_cause_plus,
    )
        failure_example = build_ground_truth_input(similar_failure,target_n=30, strict_unique=True)
        failure_candidates = failure_inference_generation_RAG.invoke({
            "data": {
                "structure_analysis": structure_input_json, # Sentences with annotations
                "gt_example": failure_example, # Similar FMEA cases in text format
            }
        })
        OUTPUT_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\failure_candidates_RAG.json"
    )
    else:
       failure_candidates = failure_inference_generation_PURE.invoke({
                       "data": {
                "structure_analysis": structure_input_json, # Sentences with annotations
            }
       })
       OUTPUT_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\failure_candidates_pure.json")
    return failure_candidates,OUTPUT_PATH

if __name__ == "__main__":
        # -----------------------------------------------------
    # 1) KB Path
    # -----------------------------------------------------
    KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb"
    )

    # -----------------------------------------------------
    # 2) Structure Input
    # -----------------------------------------------------
    structure_input = {
        "product_domain": "motor_drives",
        "nodes": [
            {
                "element_id": "E1",
                "failure_element": "Power train",
                "modes": [
                    "Incorrect",
                    "No pulses seen",
                    "No voltage applied",
                    "Incorrect torque applied",
                    "Not enough torque",
                    "Motor breaks/overheats (e.g. resulting in demagnetisation)",
                    "Unstable regulation",
                    "High loss in torque transfer",
                    "Gear train breaks/wears out",
                    "Transmission ratio drifts",
                    "creates too much noise"
                ],
                "causes": [
                    "Gears loose on motor shaft (slips)",
                    "External force on spline",
                    "Motor can not provide enough torque",
                    "Too much friction in gear train",
                    "Gears material/design choice",
                    "Manufacturing tolerances of gears",
                    "Lubrication choice (e.g. degradation)",
                    "Motor design (temperature spec, actuation length/duty cycle)",
                    "Encoder circuit crosstalk",
                    "HW cannot supply enough power",
                    "ADC measurements incorrect (incl. bandwidth)",
                    "Wrong motor driver dimension (current rating etc.)",
                    "Overcurrent detection incorrect (threshold etc.)",
                    "Incorrect control loop (bandwidth)",
                    "Motor not shorted while device is not powered",
                    "Control parameters incorrect",
                    "Thermal protection fails (e.g. I2T)"
                ],
                "effects": [
                    "Does not shift gear",
                    "Incorrect gear shift",
                    "Incorrect cadence (offset)",
                    "Unstable cadence setting",
                    "Incorrect cadence (fixed gear ratio)",
                    "Incorrect ratio (offset)",
                    "Unstable ratio setting",
                    "Does not enter limp home mode",
                    "Sets wrong gear ratio",
                    "Gear ratio drifts when battery is empty",
                    "Firmware update not possible/fails",
                    "Device bricked",
                    "Update takes too much time (>5 minutes)"
                ]
            }
        ]
    }

#     structure_input = {
#     "product_domain": "motor_drives",
#     "nodes": [
#         {
#             "element_id": "E1",
#             "failure_element": "power train",
#             "modes": [
#                 "unstable control performance",
#                 "motor failure or overheating",
#                 "insufficient torque delivery",
#                 "no voltage supplied",
#                 "excessive noise generation",
#             ],
#             "causes": [
#                 "incorrect control parameter configuration",
#                 "thermal protection system malfunction",
#                 "startup current exceeding component limits",
#                 "overvoltage due to motor disconnection",
#                 "embedded software migration issue (e3.2)",
#             ],
#             "effects": [
#                 "improper gear shifting",
#                 "gear fails to engage",
#                 "excessive noise",
#                 "motor not shorted when device is unpowered",
#                 "no connection to RC",
#                 "unstable cadence control",
#             ]
#         }
#     ]
# }

    result,OUTPUT_PATH = RAG_pipeline(structure_input=structure_input, KB_PATH=KB_PATH, top_k_per_field=20, top_n=50, max_hits_per_field_per_failure=8, RAG = True)
    print("\n================ FAILURE CANDIDATES ================\n")
    # print(json.dumps(result, indent=4))
    save_failure_candidates_to_json(result, OUTPUT_PATH)
