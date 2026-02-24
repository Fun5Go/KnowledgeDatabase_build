from Generation.LLM_function import  failure_inference_generation_RAG, failure_inference_generation_PURE, failure_inference_generation_RAG_FILL
from Retriever.SA_query import build_failure_chains_from_structure,generate_failure_chains_from_structure
from typing import Dict, List, Optional
from pathlib import Path
from langsmith import traceable
from pprint import pprint
import json


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
    # Deduplication
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
    # Extract matched SA fields
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
    # Build structured GT list
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
    # Return final formatted block
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
                 require_cause: bool = False,weight_element: float = 0.3, target_n: int =30,
                 min_similarity: float=0.55,require_cause_plus: bool = False, RAG: bool = True, FILL: bool=True):



    # structure_input = build_structure_analysis_input(structure_input)
    structure_input_json = json.dumps(structure_input, ensure_ascii=False,indent=2)

    if RAG:
        similar_failure = generate_failure_chains_from_structure(
        persist_dir=KB_PATH,
        structure_input=structure_input,
        top_k_per_field=top_k_per_field,
        top_n=top_n,
        min_similarity=min_similarity,
        require_cause = require_cause,
        require_cause_plus = require_cause_plus,
        weight_element=weight_element,
        replace=False
    )
        if not FILL:
            failure_example = build_ground_truth_input(similar_failure,target_n=target_n, strict_unique=True)
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
                require_cause = require_cause,
                min_similarity=min_similarity,
                require_cause_plus = require_cause_plus,
                replace = True,
            )
            semi_candidates =  build_fill_entity(semi_candidates,target_n=target_n,strict_unique=True)
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