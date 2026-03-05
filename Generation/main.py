from .LLM_function import  failure_inference_generation_RAG, failure_inference_generation_PURE, failure_inference_generation_RAG_FILL
from Retriever.SA_query import build_failure_chains_from_structure,generate_failure_chains_from_structure
from Retriever.sentence_query_tools import query_sentence_kb_from_structure,build_8d_failure_context_from_grouped
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from langsmith import traceable
from pprint import pprint
import json
import re
from Retriever.graph_query import generate_query_unique_chains, print_query_chain_results
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

def build_llm_structure_input(
    structure_input: Dict,
) -> Tuple[str, List[str]]:
    """
    For LLM input: keep ONLY failure_element / modes / causes / effects.
    Also return the list of failure_element strings (from structure nodes),
    so you can feed them into build_semi_chain_query_text (element will be fixed to these).
    """
    nodes = structure_input.get("nodes") or []

    minimal_nodes = []
    element_list: List[str] = []

    for n in nodes:
        fe = (n.get("failure_element") or "").strip()
        if not fe:
            continue

        element_list.append(fe)

        minimal_nodes.append(
            {
                "failure_element": fe,
                "failure_modes": list(n.get("modes") or []),
                "failure_causes": list(n.get("causes") or []),
                "failure_effects": list(n.get("effects") or []),
            }
        )

    # minimal_payload = {
    #     "product_domain": structure_input.get("product_domain"),
    #     "product_pnID": structure_input.get("product_pnID"),
    #     "nodes": minimal_nodes,
    # }

    return json.dumps(minimal_nodes, ensure_ascii=False, indent=2), element_list


def build_semi_chain_query_text(
    results: List[Dict],
    structure_elements: List[str],
    target_n: Optional[int] = None,
    strict_unique: bool = False,
    blank: str = "____",
) -> str:
    """
    Build LLM query text as "semi chains":
    - failure_element is ALWAYS from structure_elements (round-robin if multiple nodes)
    - mode/cause/effect: ONLY keep STRUCTURE-tagged texts; otherwise blank for LLM to fill
    """

    if not results:
        return "No similar failure chains were retrieved."

    if not structure_elements:
        raise ValueError("structure_elements is empty. Provide failure_element(s) from structure input.")

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
        return (
            norm(r.get("mode")),
            norm(r.get("effect")),
            norm(r.get("cause")),
        )

    def _tagged(r: dict, field: str) -> dict:
        return ((r.get("tagged") or {}).get(field) or {}) if isinstance(r, dict) else {}

    def get_structure_text(r: dict, field: str) -> str:
        t = _tagged(r, field)
        text = (t.get("text") or "").strip()
        tag = (t.get("tag") or "").strip().upper()
        return text if (text and tag == "STRUCTURE") else ""

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
    # Build query text semi chains
    # ---------------------------
    blocks = []
    for i, r in enumerate(selected, start=1):
        failure_id = r.get("failure_id")

        # element fixed from structure input
        element = structure_elements[(i - 1) % len(structure_elements)]

        # only keep STRUCTURE-tagged; else blank
        mode = get_structure_text(r, "mode")
        cause = get_structure_text(r, "cause")
        effect = get_structure_text(r, "effect")

        mode_q = mode if mode else blank
        cause_q = cause if cause else blank
        effect_q = effect if effect else blank

        blocks.append(
            f"[SEMI_CHAIN {i}]"
            + (f" (failure_id={failure_id})" if failure_id is not None else "")
            + "\n"
            f"failure_element: {element}\n"
            f"failure_mode: {mode_q}\n"
            f"failure_cause: {cause_q}\n"
            f"failure_effect: {effect_q}\n"
        )

    return "\n".join(blocks)


def format_sentences_grouped_by_case_id(
    sentence_results: List[Dict[str, Any]],
    unknown_case_label: str = "unknown_case",
    max_cases: Optional[int] = None,
    max_sentences_per_case: Optional[int] = None,
) -> str:
    """
    Reformat sentence retrieval output for LLM prompt:
    - keep only sentence_id + text
    - group by metadata.case_id
    - return a prompt-friendly string

    sentence_results item example:
      {
        "sentence_id": "...",
        "text": "...",
        "metadata": {"case_id": "...", ...},
        ...
      }
    """

    # ---- group by case_id ----
    grouped: Dict[str, List[Tuple[str, str]]] = {}

    for r in sentence_results or []:
        sid = (r.get("sentence_id") or "").strip()
        text = (r.get("text") or "").strip()
        if not sid or not text:
            continue

        meta = r.get("metadata") or {}
        case_id = meta.get("case_id") if isinstance(meta, dict) else None
        case_id = str(case_id).strip() if case_id is not None else ""
        if not case_id:
            case_id = unknown_case_label

        grouped.setdefault(case_id, []).append((sid, text))

    if not grouped:
        return ""

    # ---- optionally limit cases by "how many sentences they have" ----
    case_ids = sorted(grouped.keys(), key=lambda cid: len(grouped[cid]), reverse=True)
    if max_cases is not None:
        case_ids = case_ids[: max(0, int(max_cases))]

    # ---- build prompt text ----
    lines: List[str] = []
    for cid in case_ids:
        sents = grouped[cid]
        if max_sentences_per_case is not None:
            sents = sents[: max(0, int(max_sentences_per_case))]

        lines.append(f"## case_id: {cid}")
        for sid, text in sents:
            # 一行一个句子，便于LLM引用
            lines.append(f"- [{sid}] {text}")
        lines.append("")  # blank line between cases

    return "\n".join(lines).strip()


def build_llm_case_context(
    sentence_results: List[Dict[str, Any]],
    unknown_case_label: str = "unknown_case",
    sort_cases_by_sentence_count: bool = True,
    max_cases: Optional[int] = None,
    max_sentences_per_case: Optional[int] = None,
    dedup_within_case: bool = True,
    truncate_text_chars: Optional[int] = None,
) -> str:
    """
    Integrated function:
    1. Group sentences by metadata.case_id
    2. Keep only sentence_id + text
    3. Format into LLM-friendly structured block

    sentence_results item example:
        {
            "sentence_id": "...",
            "text": "...",
            "metadata": {"case_id": "..."}
        }
    """

    if not sentence_results:
        return ""

    # -------------------------------------------------
    # 1️⃣ Group by case_id
    # -------------------------------------------------
    grouped: Dict[str, List[Dict[str, str]]] = {}

    for r in sentence_results:
        sid = str(r.get("sentence_id", "")).strip()
        text = str(r.get("text", "")).strip()

        if not sid or not text:
            continue

        meta = r.get("metadata") or {}
        case_id = None
        if isinstance(meta, dict):
            case_id = meta.get("case_id")

        case_id = str(case_id).strip() if case_id else unknown_case_label

        grouped.setdefault(case_id, []).append({
            "sentence_id": sid,
            "text": text
        })

    if not grouped:
        return ""

    # -------------------------------------------------
    # 2️⃣ Sort case order
    # -------------------------------------------------
    case_ids = list(grouped.keys())

    if sort_cases_by_sentence_count:
        case_ids.sort(key=lambda cid: len(grouped[cid]), reverse=True)
    else:
        case_ids.sort()

    if max_cases is not None:
        case_ids = case_ids[: max(0, int(max_cases))]

    # -------------------------------------------------
    # 3️⃣ Helper: truncate
    # -------------------------------------------------
    def _truncate(s: str) -> str:
        if truncate_text_chars is None:
            return s
        n = int(truncate_text_chars)
        if n <= 0 or len(s) <= n:
            return s
        return s[: n - 1].rstrip() + "…"

    # -------------------------------------------------
    # 4️⃣ Build final prompt block
    # -------------------------------------------------
    output_lines: List[str] = []

    for idx, cid in enumerate(case_ids, start=1):

        sentences = grouped[cid]

        # optional de-dup inside case
        if dedup_within_case:
            seen: set[Tuple[str, str]] = set()
            unique = []
            for s in sentences:
                key = (s["sentence_id"], s["text"])
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            sentences = unique

        if max_sentences_per_case is not None:
            sentences = sentences[: max(0, int(max_sentences_per_case))]

        output_lines.append(f"# Case {idx}: {cid}")
        output_lines.append(f"- Sentence count: {len(sentences)}")

        for i, s in enumerate(sentences, start=1):
            sid = s["sentence_id"]
            text = _truncate(s["text"])
            output_lines.append(f"  {i}. [{sid}] {text}")

        output_lines.append("")  # blank line between cases

    return "\n".join(output_lines).strip()


# def save_failure_candidates_to_json(result: dict, output_path: Path):
#     """
#     Save failure candidates to JSON file.
#     """

#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     with output_path.open("w", encoding="utf-8") as f:
#         json.dump(result, f, indent=4)

#     print(f"\n Failure candidates saved to: {output_path}")

def find_me_partial_rows(graph_results, q_mode, q_effect):
    def _safe(x): return "" if x is None else str(x).strip()
    partial = graph_results.get("partial_chains") or []
    hits = []
    for r in partial:
        if _safe(r.get("chain_type")) != "ME":
            continue
        # 注意：这里同时用 query_effect 和 effect 做 fallback
        rm = _safe(r.get("query_mode"))
        re = _safe(r.get("query_effect")) or _safe(r.get("effect"))
        if rm == q_mode and re == q_effect:
            hits.append(r)
    print("hits:", len(hits))
    for h in hits[:10]:
        print("  element=", h.get("failure_element"),
              "| query_mode=", h.get("query_mode"),
              "| query_effect=", h.get("query_effect"),
              "| effect=", h.get("effect"),
              "| score=", h.get("score"))

@traceable(name="RAG")
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
    # structure_input_json = json.dumps(structure_input, ensure_ascii=False, indent=2)
    structure_input_json_min, structure_elements = build_llm_structure_input(structure_input)
    product_pnID = structure_input.get("product_pnID")
    obj = _safe_filename(object)
   
    # To store the retrieval results
    retrieval_payload = None

    if RAG:
        if product_pnID !=None:
            sentences =  query_sentence_kb_from_structure(persist_dir=sentence_KB_PATH,structure_input=structure_input, use_role_separation=False,
                                                        productPnID=product_pnID,top_k=15,similarity_threshold=0.3,n_results_per_query=5)
            structred_sentences = build_llm_case_context(sentences)
            D_failures=  build_8d_failure_context_from_grouped(structred_sentences,entity_store_path=ENTITY_PATH, max_cases=4)
        if not FILL:
            similar_failure = generate_failure_chains_from_structure(
                persist_dir=failure_KB_PATH,
                structure_input=structure_input,
                top_k_per_field=top_k_per_field,
                top_n=top_n,
                min_similarity=min_similarity,
                require_cause=require_cause,
                require_cause_plus=require_cause_plus,
                weight_element=weight_element,
                replace=False,
                hybrid_score=False,
                source_type=["new_fmea","old_fmea"]
            )
            retrieval_payload = similar_failure

            failure_example = build_ground_truth_input(
                similar_failure,
                target_n=target_n,
                strict_unique=True,
            )

            failure_candidates = failure_inference_generation_RAG.invoke(
                {
                    "data": {
                        "structure_analysis": structure_input_json_min,
                        "gt_example": failure_example,
                        "sentences": D_failures,
                    }
                }
            )

            OUTPUT_PATH = Path(
                fr"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\single_test\RAG_{obj}.json"
            )

        else:
            # semi_candidates = generate_failure_chains_from_structure(
            #     persist_dir=failure_KB_PATH,
            #     structure_input=structure_input,
            #     top_k_per_field=top_k_per_field,
            #     top_n=top_n,
            #     min_similarity=min_similarity,
            #     require_cause=require_cause,
            #     require_cause_plus=require_cause_plus,
            #     weight_element=weight_element,
            #     replace=True,
            #     source_type=["new_fmea","old_fmea"],
            #     hybrid_score=False,
            # )

            # semi_candidates = build_semi_chain_query_text(
            #     semi_candidates,
            #     target_n=target_n,
            #     strict_unique=True,
            #     structure_elements=structure_elements
            # )
            graph_semi_chains = generate_query_unique_chains(persist_dir=failure_KB_PATH,structure_input=structure_input,save_query_json=False,
                                                                  min_similarity=0.25,top_k_per_field=30)
            semi_candidates = build_semi_chain_query_text_from_graph_results(graph_semi_chains,structure_elements, target_n_mc=15, target_n_me=15,min_count=2, min_best_score=0.5, join_cartesian=False)
            # semi_candidates = build_semi_chain_query_text_from_graph_results_PPL(graph_semi_chains,structure_element=structure_elements, target_n_mc=15, target_n_me=15,min_count=2, 
            #                                                              structure_input= structure_input, ppl_top_k=10,ppl_max_show=5, 
            #                                                              min_best_score=0.5, join_cartesian=False,enable_ppl_candidates=True)
            print(semi_candidates)
            # failure_candidates = failure_inference_generation_RAG_FILL.invoke(
            #     {
            #         "data": {
            #             "structure_analysis": structure_input_json_min,
            #             "fill_failure": semi_candidates,
            #         }
            #     }
            # )
            OUTPUT_PATH = Path(
                fr"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\single_test\RAG_FILL_{obj}.json"
            )

    else:
        failure_candidates = failure_inference_generation_PURE.invoke(
            {
                "data": {
                    "structure_analysis": structure_input_json_min,
                }
            }
        )
        OUTPUT_PATH = Path(
            fr"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\single_test\PURE_{obj}.json"
        )

    # -------- 序列化工具：避免 LangChain / Pydantic 对象无法 json.dump ----------
    def _to_jsonable(x: Any):
        if x is None:
            return None
        # 常见：Pydantic v2
        if hasattr(x, "model_dump"):
            try:
                return x.model_dump()
            except Exception:
                pass
        # 常见：Pydantic v1
        if hasattr(x, "dict"):
            try:
                return x.dict()
            except Exception:
                pass
        # 常见：LangChain message / generation
        if hasattr(x, "to_dict"):
            try:
                return x.to_dict()
            except Exception:
                pass
        # 兜底：字符串化（至少不炸）
        try:
            json.dumps(x, ensure_ascii=False)
            return x
        except Exception:
            return str(x)

    # -------- 最终落盘内容：把 retrieval 一起写进 json ----------
    bundle = {
        "failure_candidates": _to_jsonable(failure_candidates),          # LLM OUTPUT
        "meta": {
            "RAG": RAG,
            "FILL": FILL,
            "top_n": top_n,
            "top_k_per_field": top_k_per_field,
            "min_similarity": min_similarity,
            "require_cause": require_cause,
            "require_cause_plus": require_cause_plus,
            "weight_element": weight_element,
            "target_n": target_n,
        },
        "structure_analysis": structure_input,           # 原始结构输入
        "retrieval_candidates": _to_jsonable(retrieval_payload),  # ✅ similar_failure / semi_candidates

    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

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
                  top_k_per_field=30, top_n=50,object = "seperate_candidate_powertrain_discipline_1",
                 target_n = 15, weight_element = 0.5, min_similarity=0.45, RAG = True, FILL = True)
    

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


