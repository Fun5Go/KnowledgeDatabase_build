from .LLM_function import  failure_inference_generation_RAG, failure_inference_generation_PURE, failure_inference_generation_RAG_FILL
from Retriever.SA_query import build_failure_chains_from_structure,generate_failure_chains_from_structure
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
        return "GROUND TRUTH FAILURE PATTERNS (Structured Reference Only)\n\n[]"

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

    # -----------------------------
    # 1️⃣ Deduplication
    # -----------------------------
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

    # -----------------------------
    # 2️⃣ Extract matched SA fields
    # -----------------------------
    def extract_matched_inputs(match_detail: dict) -> dict:
        """
        Return structured dict:
        {
            "element": "...",
            "mode": "...",
            ...
        }
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

    # -----------------------------
    # 3️⃣ Build structured GT list
    # -----------------------------
    structured_patterns = []

    for r in selected:
        gt_pattern = {
            "failure_id": r.get("failure_id"),
            "gt_causal_pattern": {
                "cause": r.get("cause") or "N/A",
                "mode": r.get("mode") or "N/A",
                "effect": r.get("effect") or "N/A",
            },
        }

        matched_inputs = extract_matched_inputs(r.get("match_detail", {}))

        if matched_inputs:
            gt_pattern["matched_structure_fields"] = matched_inputs

        structured_patterns.append(gt_pattern)

    # -----------------------------
    # 4️⃣ Return final formatted block
    # -----------------------------
    return (
        "GROUND TRUTH FAILURE PATTERNS (Structured Reference Only)\n\n"
        + json.dumps(structured_patterns, indent=2, ensure_ascii=False)
    )

def build_fill_entity(
    results: List[Dict],
    target_n: Optional[int] = None,
    strict_unique: bool = False,
) -> str:

    if not results:
        return "No similar failure chains were retrieved from the knowledge base."

    if target_n is None:
        target_n = len(results)

    # ---------------------------
    # Utility functions
    # ---------------------------

    def norm(x):
        if x is None:
            return ""
        return " ".join(str(x).strip().split()).lower()

    def sig(r):
        """Signature for duplicate filtering"""
        return (
            norm(r.get("element")),
            norm(r.get("function")),
            norm(r.get("mode")),
            norm(r.get("effect")),
            norm(r.get("cause")),
        )

    def with_tag(text: str, tag: str) -> str:
        """
        Build display text with tag.
        Example:
        ADC measurements incorrect [STRUCTURE]
        """
        text = (text or "N/A").strip()
        tag = (tag or "KB").strip().upper()
        return f"{text} [{tag}]"

    def get_tagged_text(r: dict, field: str, fallback_text: str):
        """
        Extract text and tag from r["tagged"][field]
        Fallback to plain r[field]
        Default tag = KB
        """
        tagged = (r.get("tagged") or {}).get(field) or {}
        text = tagged.get("text") or r.get(field) or fallback_text or "N/A"
        tag = tagged.get("tag") or "KB"
        return text, tag

    # ---------------------------
    # Select unique results
    # ---------------------------

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

    # ---------------------------
    # Build structured entities
    # ---------------------------

    entities = []
    llm_entities = []

    for r in selected:
        function = r.get("function") or "N/A"

        element_text, element_tag = get_tagged_text(r, "element", r.get("element"))
        mode_text, mode_tag = get_tagged_text(r, "mode", r.get("mode"))
        cause_text, cause_tag = get_tagged_text(r, "cause", r.get("cause"))
        effect_text, effect_tag = get_tagged_text(r, "effect", r.get("effect"))

        # -------------------------
        # Internal structured entity (for your system logic)
        # -------------------------
        entity = {
            "failure_id": r.get("failure_id"),
            "failure_element": element_text,
            "failure_element_tag": element_tag,
            "failure_function": function,
            "given_failure_mode": mode_text,
            "given_failure_mode_tag": mode_tag,
            "given_failure_cause": cause_text,
            "given_failure_cause_tag": cause_tag,
            "given_failure_effect": effect_text,
            "given_failure_effect_tag": effect_tag,
        }

        entities.append(entity)

        # -------------------------
        # LLM-only payload (display only)
        # -------------------------
        llm_entity = {
            "failure_id": r.get("failure_id"),
            "failure_function": function,
            "failure_element": with_tag(element_text, element_tag),
            "failure_mode": with_tag(mode_text, mode_tag),
            "failure_cause": with_tag(cause_text, cause_tag),
            "failure_effect": with_tag(effect_text, effect_tag),
        }

        llm_entities.append(llm_entity)

    # 
    return json.dumps(llm_entities, indent=2, ensure_ascii=False)



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
    require_cause_plus: bool = False, RAG: bool = True, FILL: bool=True):



    # structure_input = build_structure_analysis_input(structure_input)
    structure_input_json = json.dumps(structure_input, ensure_ascii=False,indent=2)

    if RAG:
        similar_failure = generate_failure_chains_from_structure(
        persist_dir=KB_PATH,
        structure_input=structure_input,
        top_k_per_field=top_k_per_field,
        top_n=top_n,
        max_hits_per_field_per_failure = max_hits_per_field_per_failure,
        require_cause = require_cause,
        require_cause_plus = require_cause_plus,
        replace=False
    )
        if not FILL:
            failure_example = build_ground_truth_input(similar_failure,target_n=25, strict_unique=True)
            failure_candidates = failure_inference_generation_RAG.invoke({
                "data": {
                    "structure_analysis": structure_input_json, # Sentences with annotations
                    "gt_example": failure_example, # Similar FMEA cases in text format
                }
            })
            OUTPUT_PATH = Path(
            r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\failure_candidates_RAG.json")
        else:
            semi_candidates = generate_failure_chains_from_structure(
                persist_dir=KB_PATH,
                structure_input=structure_input,
                top_k_per_field=top_k_per_field,
                top_n=top_n,
                max_hits_per_field_per_failure = max_hits_per_field_per_failure,
                require_cause = require_cause,
                require_cause_plus = require_cause_plus,
            )
            semi_candidates =  build_fill_entity(semi_candidates,target_n=25,strict_unique=True)
            failure_candidates = failure_inference_generation_RAG_FILL.invoke({
                "data": {
                    "structure_analysis": structure_input_json, # Sentences with annotations
                    "fill_failure": semi_candidates
                }
            })
            OUTPUT_PATH = Path(
            r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\failure_candidates_RAG_FILL.json"
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
    # result,OUTPUT_PATH = RAG_pipeline(structure_input=structure_input, KB_PATH=KB_PATH, top_k_per_field=20, top_n=50, 
    #                                   max_hits_per_field_per_failure=8, RAG = True, FILL = False)
    # print("\n================ FAILURE CANDIDATES ================\n")
    # # print(json.dumps(result, indent=4))
    # save_failure_candidates_to_json(result, OUTPUT_PATH)

    # -----------------------------------------------------
    # 3) Batch Settings
    # -----------------------------------------------------
    NUM_RUNS = 10  

    BASE_SAVE_DIR = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\batch_outputs"
    )

    MODES = [
        # {"name": "PURE", "RAG": False, "FILL": False},
        {"name": "RAG", "RAG": True, "FILL": False},
        # {"name": "RAG_FILL", "RAG": True, "FILL": True},
    ]

    # -----------------------------------------------------
    # 4) Run Loop
    # -----------------------------------------------------
    for mode in MODES:

        mode_name = mode["name"]
        save_folder = BASE_SAVE_DIR / mode_name
        save_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n================ RUNNING MODE: {mode_name} =================\n")

        for i in range(1, NUM_RUNS+1):

            print(f"\n--- Run {i} ---\n")

            result, _ = RAG_pipeline(
                structure_input=structure_input,
                KB_PATH=KB_PATH,
                top_k_per_field=20,
                top_n=50,
                max_hits_per_field_per_failure=8,
                RAG=mode["RAG"],
                FILL=mode["FILL"],
            )

            output_path = save_folder / f"failure_candidates_{mode_name.lower()}_{i}.json"

            save_failure_candidates_to_json(result, output_path)


