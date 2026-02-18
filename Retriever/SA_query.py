from typing import Dict, List, Union, Optional, Any
from collections import defaultdict
from pathlib import Path
from Retriever.failure_query_tools import _load_kb,query_semantic_kb
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
import math


# =========================================================
# Structure → Failure Chain Retrieval
# =========================================================

# =========================================================
# Structure → Failure Chain Retrieval (Soft-AND Optimized)
# =========================================================

def build_failure_chains_from_structure(
    persist_dir: Union[str, Path],
    structure_input: Dict[str, Any],
    top_k_per_field: int = 25,
    min_count: Optional[int] = None,
    weight_element: float = 1.0,
    weight_mode: float = 1.5,
    weight_cause: float = 1.5,
    weight_effect: float = 1.5,
    top_n: Optional[int] = 50,
    source_type: Optional[str] = None,
    # ---- graph constraints / scoring controls ----
    require_cause: bool = False,
    require_cause_plus: bool = False,  # require (mode + cause) or (mode + effect) if True
    min_similarity: float = 0.45,     # ignore weak semantic hits
    max_hits_per_field_per_failure: int = 10,  # cap to prevent score explosion from many near-duplicates
    normalize_by_hits: bool = False,  # for graph-constrained retrieval, default False
) -> List[Dict[str, Any]]:
    """
    Graph-constrained (failure_id-level) retrieval for FMEA chains.

    Key changes vs soft-AND global aggregation:
    1) Node-scoped: each structure node builds its own candidate pool (prevents cross-node contamination).
    2) Failure_id constrained: all outputs are complete KB entities (element/mode/cause/effect come from same failure_id).
    3) Structural constraints:
        - require_mode: only keep failures with mode hit
        - require_mode_plus: additionally require (cause or effect) hit
    4) Similarity threshold + per-field hit cap for robustness.

    Output: list of chains (KB failure entities) with node_id, matched_fields, and match_detail.
    """

    persist_dir = Path(persist_dir)

    kb = _load_kb(persist_dir)
    product_domain = structure_input.get("product_domain")

    all_results: List[Dict[str, Any]] = []

    nodes = structure_input.get("nodes", []) or []
    for node in nodes:
        node_id = node.get("element_id")

        element_text = (node.get("failure_element") or "").strip()
        modes = [x for x in (node.get("modes", []) or []) if str(x).strip()]
        causes = [x for x in (node.get("causes", []) or []) if str(x).strip()]
        effects = [x for x in (node.get("effects", []) or []) if str(x).strip()]

        # ------------------------------
        # Node-scoped candidate scores
        # ------------------------------
        candidate_scores = defaultdict(lambda: {
            "score": 0.0,
            "field_hits": set(),
            "matched": {
                "element": [],
                "mode": [],
                "cause": [],
                "effect": []
            }
        })

        # ---- ELEMENT ----
        if element_text:
            res = query_semantic_kb(
                persist_dir,
                element_text,
                field_type="element",
                n_results=top_k_per_field,
                min_count=min_count,
                source_type=source_type,
            )
            _accumulate_candidate_scores(
                kb=kb,
                semantic_query_result=res,
                candidate_scores=candidate_scores,
                query_text=element_text,
                field_type="element",
                weight=weight_element,
                min_similarity=min_similarity,
                max_hits_per_field_per_failure=max_hits_per_field_per_failure,
            )

        # ---- MODES ----
        for m in modes:
            res = query_semantic_kb(
                persist_dir,
                m,
                field_type="mode",
                n_results=top_k_per_field,
                min_count=min_count,
                source_type=source_type,
            )
            _accumulate_candidate_scores(
                kb=kb,
                semantic_query_result=res,
                candidate_scores=candidate_scores,
                query_text=m,
                field_type="mode",
                weight=weight_mode,
                min_similarity=min_similarity,
                max_hits_per_field_per_failure=max_hits_per_field_per_failure,
            )

        # ---- CAUSES ----
        for c in causes:
            res = query_semantic_kb(
                persist_dir,
                c,
                field_type="cause",
                n_results=top_k_per_field,
                min_count=min_count,
                source_type=source_type,
            )
            _accumulate_candidate_scores(
                kb=kb,
                semantic_query_result=res,
                candidate_scores=candidate_scores,
                query_text=c,
                field_type="cause",
                weight=weight_cause,
                min_similarity=min_similarity,
                max_hits_per_field_per_failure=max_hits_per_field_per_failure,
            )

        # ---- EFFECTS ----
        for e in effects:
            res = query_semantic_kb(
                persist_dir,
                e,
                field_type="effect",
                n_results=top_k_per_field,
                min_count=min_count,
                source_type=source_type,
            )
            _accumulate_candidate_scores(
                kb=kb,
                semantic_query_result=res,
                candidate_scores=candidate_scores,
                query_text=e,
                field_type="effect",
                weight=weight_effect,
                min_similarity=min_similarity,
                max_hits_per_field_per_failure=max_hits_per_field_per_failure,
            )

        # --------------------------------------------
        # Graph Constrained Filtering (failure_id-level)
        # --------------------------------------------
        for fid, info in candidate_scores.items():
            fields = info["field_hits"]

            if require_cause and ("cause" not in fields):
                continue

            if require_cause_plus and ("cause"in fields) and not (("mode" in fields) or ("effect" in fields) or ("element" in fields)):
                continue

            entity = kb.entity_store.get(fid)
            if not entity:
                continue

            if product_domain and entity.get("product_domain") != product_domain:
                continue

            score = info["score"]
            if normalize_by_hits:
                # Optional: mild normalization; keep stable denominator (max 4 fields)
                denom = max(1, min(len(fields), 4))
                score = score / denom

            chain = {
                "node_id": node_id,
                "failure_id": fid,
                "element": entity.get("failure_element_text"),
                "function": entity.get("function"),
                "mode": entity.get("failure_mode_text"),
                "cause": entity.get("failure_cause_text"),
                "effect": entity.get("failure_effect_text"),
                "score": round(float(score), 4),
                "matched_fields": sorted(list(fields)),
                "match_detail": info["matched"],
            }
            all_results.append(chain)

    # ------------------------------
    # Global sort + truncate
    # ------------------------------
    all_results.sort(key=lambda x: x["score"], reverse=True)
    if top_n is not None:
        all_results = all_results[: int(top_n)]

    return all_results


def _accumulate_candidate_scores(
    kb: Any,
    semantic_query_result: Dict[str, Any],
    candidate_scores: Dict[str, Any],
    query_text: str,
    field_type: str,
    weight: float,
    min_similarity: float = 0.35,
    max_hits_per_field_per_failure: int = 3,
) -> None:
    """
    Accumulate failure_id scores from semantic hits with graph constraint:
    semantic_id -> failure_ids (from kb.field_store) -> candidate_scores[failure_id]

    Improvements:
    - min_similarity threshold to drop weak matches
    - cap hits per field per failure_id to prevent score explosion
    - dedupe semantic_id per failure_id per field
    """

    ids = (semantic_query_result.get("ids", [[]]) or [[]])[0] or []
    dists = (semantic_query_result.get("distances", [[]]) or [[]])[0] or []

    for sid, dist in zip(ids, dists):
        # Chroma distances are typically [0..2] depending on metric;
        # keep your original conversion but guard.
        try:
            similarity = max(0.0, 1.0 - float(dist))
        except Exception:
            continue

        if similarity < min_similarity:
            continue

        node = kb.field_store.get(sid, {}) or {}
        failure_ids = node.get("failure_ids", []) or []
        def _get_kb_text(node: dict, field_type: str) -> str:
            for k in ("text",):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""
        if not failure_ids:
            continue

        for fid in failure_ids:
            matched_list = candidate_scores[fid]["matched"][field_type]

            # ---- cap per-field hits per failure ----
            if max_hits_per_field_per_failure is not None and len(matched_list) >= int(max_hits_per_field_per_failure):
                continue

            # ---- prevent duplicate semantic_id scoring ----
            existing_ids = {m["semantic_id"] for m in matched_list}
            if sid in existing_ids:
                continue

            candidate_scores[fid]["score"] += similarity * float(weight)
            candidate_scores[fid]["field_hits"].add(field_type)
            kb_text = _get_kb_text(node, field_type)

            matched_list.append({
                "semantic_id": sid,
                "similarity": round(float(similarity), 4),
                "structure_text": query_text,
                "kb_text": kb_text
            })


def generate_failure_chains_from_structure(
    persist_dir: Union[str, Path],
    structure_input: Dict[str, Any],
    top_k_per_field: int = 25,
    min_count: Optional[int] = None,
    weight_element: float = 0.5,
    weight_mode: float = 1.5,
    weight_cause: float = 1.5,
    weight_effect: float = 1.5,
    top_n: Optional[int] = 50,
    source_type: Optional[str] = None,
    require_cause: bool = False,
    require_cause_plus: bool = False,
    min_similarity: float = 0.45,
    max_hits_per_field_per_failure: int = 10,
    normalize_by_hits: bool = False,
) -> List[Dict[str, Any]]:
    """
    Graph-constrained retrieval + structure-native FMEA chain generation.

    Output chains:
    - Relation pattern from KB
    - Semantic text from structure analysis
    - KB used only as fallback completion
    """

    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)
    product_domain = structure_input.get("product_domain")

    all_results: List[Dict[str, Any]] = []
    nodes = structure_input.get("nodes", []) or []

    for node in nodes:
        node_id = node.get("element_id")

        element_text = (node.get("failure_element") or "").strip()
        modes = [x.strip() for x in (node.get("modes") or []) if str(x).strip()]
        causes = [x.strip() for x in (node.get("causes") or []) if str(x).strip()]
        effects = [x.strip() for x in (node.get("effects") or []) if str(x).strip()]

        candidate_scores = defaultdict(lambda: {
            "score": 0.0,
            "field_hits": set(),
            "matched": {
                "element": [],
                "mode": [],
                "cause": [],
                "effect": []
            }
        })

        # -------------------------
        # SEMANTIC SEARCH
        # -------------------------

        def query_and_accumulate(text: str, field: str, weight: float):
            res = query_semantic_kb(
                persist_dir,
                text,
                field_type=field,
                n_results=top_k_per_field,
                min_count=min_count,
                source_type=source_type,
            )
            _accumulate_candidate_scores(
                kb=kb,
                semantic_query_result=res,
                candidate_scores=candidate_scores,
                query_text=text,
                field_type=field,
                weight=weight,
                min_similarity=min_similarity,
                max_hits_per_field_per_failure=max_hits_per_field_per_failure,
            )

        if element_text:
            query_and_accumulate(element_text, "element", weight_element)

        for m in modes:
            query_and_accumulate(m, "mode", weight_mode)

        for c in causes:
            query_and_accumulate(c, "cause", weight_cause)

        for e in effects:
            query_and_accumulate(e, "effect", weight_effect)

        # -------------------------
        # GRAPH-CONSTRAINED FILTER
        # -------------------------

        for fid, info in candidate_scores.items():
            fields = info["field_hits"]

            if require_cause and ("cause" not in fields):
                continue

            if require_cause_plus and ("cause" in fields) and not (
                ("mode" in fields) or ("effect" in fields) or ("element" in fields)
            ):
                continue

            entity = kb.entity_store.get(fid)
            if not entity:
                continue

            if product_domain and entity.get("product_domain") != product_domain:
                continue

            score = info["score"]

            # -------------------------
            # CONNECTION BONUS (mode/cause/effect)
            # -------------------------

            mce_fields = {"mode", "cause", "effect"}
            mce_hit_count = len(fields.intersection(mce_fields))

            # bonus strategy
            if mce_hit_count >= 2:
                score *= 0.7          # 强化双命中
            elif mce_hit_count == 1:
                score *= 1         # 轻微惩罚
            else:
                score *= 1          # 严重惩罚


            if normalize_by_hits:
                denom = max(1, min(len(fields), 4))
                score = score / denom

            # -------------------------
            # STRUCTURE TEXT REPLACEMENT
            # -------------------------
            def pick_best_structure_text(matched_list):
                if not matched_list:
                    return None
                best = sorted(matched_list, key=lambda x: x["similarity"], reverse=True)[0]
                return best.get("structure_text")

            structure_element = pick_best_structure_text(info["matched"]["element"])
            structure_mode    = pick_best_structure_text(info["matched"]["mode"])
            structure_cause   = pick_best_structure_text(info["matched"]["cause"])
            structure_effect  = pick_best_structure_text(info["matched"]["effect"])

            # final values (structure first, kb fallback)
            final_element = structure_element or entity.get("failure_element_text")
            final_mode    = structure_mode    or entity.get("failure_mode_text")
            final_cause   = structure_cause   or entity.get("failure_cause_text")
            final_effect  = structure_effect  or entity.get("failure_effect_text")

            # tag per field
            tag_element = "STRUCTURE" if structure_element else "KB"
            tag_mode    = "STRUCTURE" if structure_mode else "KB"
            tag_cause   = "STRUCTURE" if structure_cause else "KB"
            tag_effect  = "STRUCTURE" if structure_effect else "KB"

            chain = {
                "node_id": node_id,
                "failure_id": fid,

                # keep plain fields for downstream compatibility
                "element": final_element,
                "function": entity.get("function"),
                "mode": final_mode,
                "cause": final_cause,
                "effect": final_effect,

                # NEW: tagged display fields
                "tagged": {
                    "element": {"text": final_element, "tag": tag_element},
                    "mode":    {"text": final_mode,    "tag": tag_mode},
                    "cause":   {"text": final_cause,   "tag": tag_cause},
                    "effect":  {"text": final_effect,  "tag": tag_effect},
                },

                "score": round(float(score), 4),
                "matched_fields": sorted(list(fields)),
                "match_detail": info["matched"],  # 保持你build_ground_truth_input能print
            }
            all_results.append(chain)

    # -------------------------
    # GLOBAL SORT
    # -------------------------
    all_results.sort(key=lambda x: x["score"], reverse=True)

    if top_n is not None:
        all_results = all_results[: int(top_n)]

    return all_results



if  __name__ == "__main__":


    from pprint import pprint

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

    # -----------------------------------------------------
    # 3) Run Retrieval
    # -----------------------------------------------------
    results = build_failure_chains_from_structure(
        persist_dir=KB_PATH,
        structure_input=structure_input,
        top_k_per_field=15,
        # minimum_field_match=2,
        top_n=70,
        # source_type=["8D","8D,old_fmea","8D,new_fmea","8D,new_fmea,old_fmea"],
    )
    # re
    # results = generate_failure_chains_from_structure(
    #     persist_dir=KB_PATH,
    #     structure_input=structure_input,
    #     top_k_per_field=15,
    #     # minimum_field_match=2,
    #     top_n=50,
    #     source_type=["8D","8D,old_fmea","8D,new_fmea","8D,new_fmea,old_fmea"],
    # )
    # results = generate_failure_chains_from_structure(
    #     persist_dir=KB_PATH,
    #     structure_input=structure_input
    # )
    # print(results)

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
            tagged = r.get("tagged") or {}
            def fmt(field_name: str, fallback_key: str):
                obj = tagged.get(field_name)
                if isinstance(obj, dict) and "text" in obj:
                    return f"{obj.get('text')}  [{obj.get('tag', 'UNKNOWN')}]"
                return f"{r.get(fallback_key)}  [UNKNOWN]"

            lines.append(f"  Element : {fmt('element', 'element')}")
            lines.append(f"  Function: {r.get('function')}  [KB]")  # function通常来自KB
            lines.append(f"  Mode    : {fmt('mode', 'mode')}")
            lines.append(f"  Effect  : {fmt('effect', 'effect')}")
            lines.append(f"  Cause   : {fmt('cause', 'cause')}")
            match_detail = r.get("match_detail", {}) or {}
            if isinstance(match_detail, dict):
                for field_type, matches in match_detail.items():
                    if matches:
                        lines.append(f"  {str(field_type).upper()}:")
                        for m in matches:
                            display = dict(m)
                            if display.get("kb_text"):
                                display["structure_text"] = display["kb_text"]
                            display.pop("kb_text", None)

                            lines.append(f"    - {display}")
            lines.append("-" * 60)

        return "\n".join(lines)


    results = build_ground_truth_input(results,target_n=25,strict_unique=True)
    print(results)
