from .LLM_function import  failure_inference_generation_single
from KnowledgeBase.Retriever.SA_query import build_failure_chains_from_structure,generate_failure_chains_from_structure
from KnowledgeBase.Retriever.sentence_query_tools import query_sentence_kb_from_structure,build_8d_failure_context_from_grouped
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from langsmith import traceable
from pprint import pprint
import json
import re
from KnowledgeBase.Retriever.graph_query import generate_query_unique_chains, print_query_chain_results
from .utils import build_semi_chain_query_text_from_graph_results, build_semi_chain_query_text_from_graph_results_PPL

ENTITY_PATH = Path(
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_miniLM\failure_kb\entity_store.json"
)

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
def _safe_filename(s: str) -> str:
    # 避免文件名包含空格/特殊字符（Windows 更稳）
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "_", s)   # 只保留 字母数字下划线连字符，其余变 _
    return s

def build_llm_structure_input(structure_input: Dict) -> Tuple[str, List[str]]:
    """
    causes: keep dict(category->list[str]) (normalized)
    """
    nodes = structure_input.get("nodes") or []

    minimal_nodes = []
    element_list: List[str] = []

    for n in nodes:
        fe = (n.get("failure_element") or "").strip()
        if not fe:
            continue

        element_list.append(fe)

        causes_raw = n.get("causes") or {}
        causes_grouped = {}

        if isinstance(causes_raw, dict):
            for k, items in causes_raw.items():
                k2 = str(k).strip()
                if not k2:
                    continue
                if isinstance(items, list):
                    causes_grouped[k2] = [str(x).strip() for x in items if str(x).strip()]
                else:
                    causes_grouped[k2] = []
        elif isinstance(causes_raw, list):
            # fallback: no category info
            causes_grouped["uncategorized"] = [str(x).strip() for x in causes_raw if str(x).strip()]

        minimal_nodes.append(
            {
                "failure_element": fe,
                "failure_modes": [str(x).strip() for x in (n.get("modes") or []) if str(x).strip()],
                "failure_causes": causes_grouped,
                "failure_effects": [str(x).strip() for x in (n.get("effects") or []) if str(x).strip()],
            }
        )

    return json.dumps(minimal_nodes, ensure_ascii=False, indent=2), element_list

def add_candidates_to_semi_chain_prompt(
    prompt_text: str,
    structure_input_json_min: List[Dict[str, Any]],
    max_candidates: int = 20,
) -> str:
    """
    Post-process semi chain prompt text and add candidates
    without modifying build_semi_chain_query_text_from_graph_results.

    MC  -> add effect_candidates
    ME  -> add cause_candidates
    FULL -> no candidates
    """

    if not structure_input_json_min:
        return prompt_text

    data = structure_input_json_min[0]

    mode_candidates = data.get("failure_modes", [])[:max_candidates]

    cause_candidates = []
    for v in (data.get("failure_causes") or {}).values():
        cause_candidates.extend(v)
    cause_candidates = cause_candidates[:max_candidates]

    effect_candidates = data.get("failure_effects", [])[:max_candidates]

    def format_candidates(name, cands):
        if not cands:
            return ""
        lines = [f"{name}:"]
        for i, c in enumerate(cands, 1):
            lines.append(f"  ({i}) {c}")
        return "\n".join(lines)

    effect_block = format_candidates("effect_candidates", effect_candidates)
    cause_block = format_candidates("cause_candidates", cause_candidates)

    lines = prompt_text.split("\n")
    output = []

    current_type = None

    for line in lines:

        # detect chain type
        if "[MC_SEMI_CHAIN" in line:
            current_type = "MC"

        elif "[ME_SEMI_CHAIN" in line:
            current_type = "ME"

        elif "[FULL_CHAIN" in line:
            current_type = "FULL"

        output.append(line)

        # inject after blank slot
        if "failure_effect:" in line and "____" in line and current_type == "MC":
            output.append(effect_block)

        if "failure_cause:" in line and "____" in line and current_type == "ME":
            output.append(cause_block)

    return "\n".join(output)

def split_semi_chains(semi_text: str):
    """
    Split MC/ME semi chains into individual blocks.
    """
    pattern = r"(\[(MC|ME)_SEMI_CHAIN\s+\d+\][\s\S]*?)(?=\n\[(MC|ME)_SEMI_CHAIN|\Z)"
    matches = re.findall(pattern, semi_text)

    chains = [m[0].strip() for m in matches]

    return chains


def parse_semi_chain(chain_text: str):
    chain_id_match = re.search(r"\[(MC|ME)_SEMI_CHAIN\s+\d+\]", chain_text)
    chain_id = chain_id_match.group(0).strip("[]") if chain_id_match else ""

    mode = re.search(r"failure_mode:\s*(.*)", chain_text)
    cause = re.search(r"failure_cause:\s*(.*)", chain_text)
    effect = re.search(r"failure_effect:\s*(.*)", chain_text)

    return {
        "chain_id": chain_id,
        "failure_mode": mode.group(1).strip() if mode else "",
        "failure_cause": cause.group(1).strip() if cause else "",
        "failure_effect": effect.group(1).strip() if effect else "",
    }

def save_failure_candidates(output_path: str, candidates: list):

    payload = {
        "failure_candidates": candidates
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def build_single_semi_chain_prompt(
    semi_chain_text: str,
    structure_input_json_min,
) -> str:
    """
    Build prompt content for a single semi chain.
    """

    if not structure_input_json_min:
        return semi_chain_text

    # 如果是字符串，需要解析
    if isinstance(structure_input_json_min, str):
        structure_input_json_min = json.loads(structure_input_json_min)

    data = structure_input_json_min[0]

    mode_candidates = data.get("failure_modes", [])

    cause_candidates = []
    for v in (data.get("failure_causes") or {}).values():
        cause_candidates.extend(v)

    effect_candidates = data.get("failure_effects", [])

    def format_candidates(name, cands):
        lines = [f"{name}:"]
        for i, c in enumerate(cands, 1):
            lines.append(f"{i}. {c}")
        return "\n".join(lines)

    prompt_parts = []

    prompt_parts.append(
        "You are given a partial FMEA failure chain. "
        "Select the most relevant candidate to fill the blank."
    )

    prompt_parts.append("\n--- Semi Chain ---\n")
    prompt_parts.append(semi_chain_text)

    # detect which blank
    if "failure_effect: ____" in semi_chain_text:
        prompt_parts.append("\n--- Effect Candidates ---\n")
        prompt_parts.append(format_candidates("effect_candidates", effect_candidates))

    if "failure_cause: ____" in semi_chain_text:
        prompt_parts.append("\n--- Cause Candidates ---\n")
        prompt_parts.append(format_candidates("cause_candidates", cause_candidates))

    prompt_parts.append(
        """
--- Task ---

Choose the most relevant candidate to fill the blank.

Rules:
- Choose ONLY one candidate
- Keep the original candidate text
- Base your reasoning on motor drive systems and FMEA logic
- Do not invent new candidates

--- Output JSON format ---


{
  "selected_failure_cause": "... or null",
  "selected_failure_effect": "... or null",
  "confidence": 0.0,
  "inference_reason": "brief reason"
}
"""
    )

    return "\n".join(prompt_parts)

@traceable(name="RAG-single")
def RAG_pipeline(
    structure_input: Dict,
    failure_KB_PATH: str,
    sentence_KB_PATH: str,
    top_n: int = 25,
    top_k_per_field: int = 10,
    require_cause: bool = False,
    weight_element: float = 0.3,
    target_n: int = 30,
    min_similarity: float = 0.55,
    require_cause_plus: bool = False,
    RAG: bool = True,
    FILL: bool = True,
    object: str = "powertrain",
):

    structure_input_json_min, structure_elements = build_llm_structure_input(structure_input)

    product_pnID = structure_input.get("product_pnID")
    obj = _safe_filename(object)

    graph_semi_chains = generate_query_unique_chains(
        persist_dir=failure_KB_PATH,
        structure_input=structure_input,
        save_query_json=False,
        min_similarity=0.3,
        top_k_per_field=50,
    )

    semi_candidates = build_semi_chain_query_text_from_graph_results(
        graph_semi_chains,
        structure_element=structure_elements,
        target_n_mc=15,
        target_n_me=15,
        join_cartesian=False,
        min_best_score= 0.25
    )
    print(f"semi candidates: {semi_candidates}")

    chains = split_semi_chains(semi_candidates)

    # print(f"chains: {semi_candidates}")

    results = []

    for chain in chains:

        chain_info = parse_semi_chain(chain)

        prompt_content = build_single_semi_chain_prompt(
            chain,
            structure_input_json_min
        )

        resp = failure_inference_generation_single.invoke(
            {
                "data": {
                    "semi_entity": prompt_content,
                }
            }
        )

        # LLM返回
        selected_cause = resp.get("selected_failure_cause")
        selected_effect = resp.get("selected_failure_effect")
        confidence = resp.get("confidence", 0.8)
        reason = resp.get("inference_reason", "")

        final_cause = chain_info["failure_cause"]
        final_effect = chain_info["failure_effect"]

        if final_cause == "____":
            final_cause = selected_cause

        if final_effect == "____":
            final_effect = selected_effect

        candidate = {
            "failure_element": structure_elements[0] if structure_elements else "",
            "failure_function": "",
            "failure_mode": chain_info["failure_mode"],
            "failure_effect": final_effect,
            "failure_cause": final_cause,
            "confidence": "high" if confidence >= 0.8 else "medium",
            "fill_from_id": [chain_info["chain_id"]],
            "inference_reason": reason,
        }

        results.append(candidate)

    output_path = f"./outputs/failure_candidates_{obj}.json"

    save_failure_candidates(output_path, results)

    return results

    
if __name__ == "__main__":
    # -----------------------------------------------------
    # 1) KB Path
    # -----------------------------------------------------
    Failure_KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_discipline\failure_kb"
    )
    Sentence_KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_miniLM\sentence_kb"
    )

    # -----------------------------------------------------
    # 2) Structure Input
    # -----------------------------------------------------
    structure_input_powertrain = {
        "product_domain": "motor_drives",
        "product_pnID": 133427,
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
                "causes": {
                    "mechanics": [
                    "Gears loose on motor shaft (slips)",
                    "External force on spline",
                    "Motor can not provide enough torque",
                    "Too much friction in gear train",
                    "Gears material/design choice",
                    "Manufacturing tolerances of gears",
                    "Lubrication choice (e.g. degradation)",
                    "Motor design (temperature spec, actuation length/duty cycle)"
                    ],
                    "hardware": [
                    "Encoder circuit crosstalk",
                    "HW cannot supply enough power",
                    "ADC measurements incorrect (incl. bandwidth)",
                    "Wrong motor driver dimension (current rating etc.)",
                    "Overcurrent detection incorrect (threshold etc.)",
                    "Incorrect control loop (bandwidth)",
                    "Motor not shorted while device is not powered"
                    ],
                    "software": [
                    "Control parameters incorrect",
                    "Thermal protection fails (e.g. I2T)"
                    ]
                },
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
                    "Update takes too much time (>5 minutes)",
                    "Too much noise",
                ]
            }
        ]
    }

    structure_input_motorcontrol = {
    "product_domain": "motor_drives",
    "nodes": [
        {
            "element_id": "E1",
            "failure_element": "Motor control",
            "modes": [
                "Component break-down",
                "Unbalanced motor currents",
                "Incorrect interpretation zero-crossing",
                "Soft start too long",
                "No detection",
                "Welded relay",
                "Relay cannot close",
                "False turn-on / turn-off"
            ],
            "causes": [
                "Cooling insufficient",
                "Compressor vibrations",
                "(Starting) Motor current too high for chosen components",
                "Overvoltage due to motor disconnect",
                "Under Voltage due to incorrect triggering",
                "Live switching of relays",
                "Priority zero-crossing interrupt too low",
                "Open loop control",
                "No (correctly designed) snubber design",
                "Too high dT junction as a result of power cycling of component"
            ],
            "effects": [
                "Motor cannot start",
                "Overcurrent towards motor",
                "Motor starts without soft start",
                #Extra
                # "(Final) Pressure deviates from setpoints",
                # "Overpressure",
                # "No pressure build-up",
                # "No user control",
            ]
        }
    ]
}
    RAG_pipeline(structure_input=structure_input_powertrain, failure_KB_PATH=Failure_KB_PATH, sentence_KB_PATH = Sentence_KB_PATH,
                  top_k_per_field=30, top_n=50,object = "single")
    
    

            # print(semi_candidates)

    # -----------------------------------------------------
    # 3) Batch Settings
    # -----------------------------------------------------
    # NUM_RUNS = 15  

    # BASE_SAVE_DIR = Path(
    #     r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\batch_outputs"
    # )

    # MODES = [
    #     {"name": "PURE", "RAG": False, "FILL": False},
    #     {"name": "RAG", "RAG": True, "FILL": False},
    #     {"name": "RAG_FILL", "RAG": True, "FILL": True},
    # ]

    # # -----------------------------------------------------
    # # 4) Run Loop
    # # -----------------------------------------------------
    # for mode in MODES:

    #     mode_name = mode["name"]
    #     save_folder = BASE_SAVE_DIR / mode_name / "25_2_powertrain"
    #     save_folder.mkdir(parents=True, exist_ok=True)

    #     print(f"\n================ RUNNING MODE: {mode_name} =================\n")

    #     for i in range(1, NUM_RUNS+1):

    #         print(f"\n--- Run {i} ---\n")

    #         result, _ = RAG_pipeline(
    #             structure_input=structure_input_powertrain,
    #             KB_PATH=KB_PATH,
    #             top_k_per_field=30,
    #             top_n=50,
    #             min_similarity=0.45,
    #             RAG=mode["RAG"],
    #             FILL=mode["FILL"],
    #         )

    #         output_path = save_folder / f"failure_candidates_{mode_name.lower()}_{i}.json"

    #         save_failure_candidates_to_json(result, output_path)


