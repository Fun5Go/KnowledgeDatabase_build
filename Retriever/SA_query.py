from typing import Dict, List, Union, Optional, Any
from collections import defaultdict
from pathlib import Path
from Retriever.failure_query_tools import _load_kb,query_semantic_kb
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
import math
from .PPL_score import ChainPPLEvaluator
import random
import json
BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# Structure → Failure Chain Retrieval
# =========================================================
def distance_to_similarity_exp(
    distance: float,
    alpha: float = 1.0,
    max_distance: float = 10.0,
    min_similarity: float = 1e-6,
) -> float:
    """
    Stable exponential distance → similarity conversion.

    similarity = exp(-alpha * distance)

    Enhancements:
    - clip negative distance
    - cap very large distance
    - floor minimal similarity
    """

    try:
        d = float(distance)
    except Exception:
        return 0.0

    if d < 0:
        d = 0.0

    # avoid extreme overflow in exp
    d = min(d, max_distance)

    sim = math.exp(-alpha * d)

    return max(sim, min_similarity)
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

FIELD_STATS = {
    "mode": {"p50": 0.11, "p95": 0.33},
    "cause": {"p50": 0.11, "p95": 0.33},
    "effect": {"p50": 0.13, "p95": 0.47},
    "element":{"p50": 0.13, "p95": 0.36}
}
def normalize_similarity(sim, field):
    stats = FIELD_STATS[field]
    p50 = stats["p50"]
    p95 = stats["p95"]

    if sim <= p50:
        return 0.0

    return (sim - p50) / (p95 - p50)


def _accumulate_candidate_scores(
    kb: Any,
    semantic_query_result: Dict[str, Any],
    candidate_scores: Dict[str, Any],
    query_text: str,
    field_type: str,
    weight: float,
    min_similarity: float = 0.35,
) -> None:
    """
    - threshold uses semantic similarity
    - "better" uses semantic similarity
    - score accumulation uses hybrid_score if available else semantic
    - stored 'similarity' is semantic (for replace/pick_best)
    """

    ids = (semantic_query_result.get("ids", [[]]) or [[]])[0] or []
    dists = (semantic_query_result.get("distances", [[]]) or [[]])[0] or []

    hyb_scores = (semantic_query_result.get("hybrid_scores", [[]]) or [[]])[0]
    has_hybrid = isinstance(hyb_scores, list) and len(hyb_scores) == len(ids)

    for idx, (sid, dist) in enumerate(zip(ids, dists)):
        try:
            sem_sim = max(0.0, 1.0 - float(dist))
        except Exception:
            continue

        # threshold by semantic
        if sem_sim < min_similarity:
            continue

        # scoring uses hybrid if available, else semantic
        used_score = float(hyb_scores[idx]) if has_hybrid else float(sem_sim)

        node = kb.field_store.get(sid, {}) or {}
        failure_ids = node.get("failure_ids", []) or []
        if not failure_ids:
            continue

        kb_text = node.get("text", "").strip() if isinstance(node.get("text"), str) else ""

        for fid in failure_ids:
            matched_list = candidate_scores[fid]["matched"][field_type]

            existing = None
            for m in matched_list:
                if m.get("semantic_id") == sid:
                    existing = m
                    break

            if existing:
                old_sem = float(existing.get("similarity", 0.0))          # semantic
                old_used = float(existing.get("hybrid_score", old_sem))   # used for scoring

                # better judged by semantic only
                if sem_sim > old_sem:
                    delta = max(0.0, (used_score - old_used) * float(weight))
                    candidate_scores[fid]["score"] += delta

                    existing["similarity"] = round(float(sem_sim), 4)         # semantic (for replace)
                    existing["hybrid_score"] = round(float(used_score), 4)    # for ranking/score
                    existing["structure_text"] = query_text
                    existing["kb_text"] = kb_text
                continue

            # new hit
            candidate_scores[fid]["score"] += used_score * float(weight)
            candidate_scores[fid]["field_hits"].add(field_type)

            matched_list.append({
                "semantic_id": sid,
                "similarity": round(float(sem_sim), 4),        # semantic only (for replace)
                "hybrid_score": round(float(used_score), 4),   # for ranking/score
                "structure_text": query_text,
                "kb_text": kb_text,
            })

def _apply_controlled_reinforcement(
    score: float,
    matched: Dict[str, List[Dict[str, Any]]],
    *,
    reinforce_cross_field: float = 0.15,
    max_reinforce_ratio: float = 0.4,
) -> float:
    """
    Controlled "duplicate reinforcement" for generate-ranking.

    - Within-field: if a field has multiple distinct semantic hits, add small bonus
    - Cross-field: if multiple fields hit (element/mode/cause/effect), add bonus
    - Cap: reinforcement <= base_score * max_reinforce_ratio
    """
    base = float(score)
    if base <= 0:
        return base

    reinforcement = 0.0

    # 2) cross-field reinforcement (reward completeness)
    active_fields = [f for f, hits in (matched or {}).items() if hits]
    if len(active_fields) >= 2:
        reinforcement += (len(active_fields) - 1) * float(reinforce_cross_field)

    # 3) cap
    reinforcement = min(reinforcement, base * float(max_reinforce_ratio))
    return base + reinforcement


def generate_failure_chains_from_structure(
    persist_dir: Union[str, Path],
    structure_input: Dict[str, Any],
    top_k_per_field: int = 25,
    min_count: Optional[int] = None,
    weight_element: float = 0.4,
    weight_mode: float = 1.0,
    weight_cause: float = 1.0,
    weight_effect: float = 1.0,
    top_n: Optional[int] = 50,
    source_type: Optional[str] = None,
    require_cause: bool = False,
    require_cause_plus: bool = False,
    min_similarity: float = 0.45,
    normalize_by_hits: bool = False,

    # ---- duplicate reinforcement ----
    allow_reinforcement: bool = True,
    reinforce_cross_field: float = 0.0,
    max_reinforce_ratio: float = 0.4,

    # ---- connection shaping ----
    apply_connection_bonus: bool = True,
    connection_factor_ge2: float = 1.0,
    connection_factor_eq1: float = 1.0,
    connection_factor_eq0: float = 1.0,
    hybrid_score: bool = True,
    replace: bool = True,
) -> List[Dict[str, Any]]:

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
            "matched": {"element": [], "mode": [], "cause": [], "effect": []},
        })

        # -------------------------
        # SEMANTIC SEARCH
        # -------------------------
        def query_and_accumulate(text: str, field: str, weight: float,hybrid_score):
            res = query_semantic_kb(
                persist_dir,
                text,
                field_type=field,
                n_results=top_k_per_field,
                min_count=min_count,
                source_type=source_type,
                hybrid=hybrid_score,
                alpha = 0.8,
            )
            _accumulate_candidate_scores(
                kb=kb,
                semantic_query_result=res,
                candidate_scores=candidate_scores,
                query_text=text,
                field_type=field,
                weight=weight,
                min_similarity=min_similarity,
            )

        if element_text:
            query_and_accumulate(element_text, "element", weight_element, hybrid_score)
        for m in modes:
            query_and_accumulate(m, "mode", weight_mode, hybrid_score)
        for c in causes:
            query_and_accumulate(c, "cause", weight_cause, hybrid_score)
        for e in effects:
            query_and_accumulate(e, "effect", weight_effect, hybrid_score)

        # -------------------------
        # PROCESS CANDIDATES
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

            score = float(info["score"])

            # -------------------------
            # CONTROLLED REINFORCEMENT
            # -------------------------
            if allow_reinforcement:
                score = _apply_controlled_reinforcement(
                    score,
                    info.get("matched") or {},
                    reinforce_cross_field=reinforce_cross_field,
                    max_reinforce_ratio=max_reinforce_ratio,
                )

            # -------------------------
            # CONNECTION SHAPING
            # -------------------------
            if apply_connection_bonus:
                mce_fields = {"mode", "cause", "effect"}
                mce_hit_count = len(fields.intersection(mce_fields))

                if mce_hit_count >= 2:
                    score *= float(connection_factor_ge2)
                elif mce_hit_count == 1:
                    score *= float(connection_factor_eq1)
                else:
                    score *= float(connection_factor_eq0)

            # -------------------------
            # OPTIONAL NORMALIZATION
            # -------------------------
            if normalize_by_hits:
                denom = max(1, min(len(fields), 4))
                score = score / denom

            # -------------------------
            # PICK BEST STRUCTURE TEXT
            # -------------------------
            def pick_best_structure_text(matched_list: List[Dict[str, Any]]) -> Optional[str]:
                if not matched_list:
                    return None
                best = max(matched_list, key=lambda x: float(x.get("similarity", 0.0)))
                return best.get("structure_text")
            if replace:
                structure_element = pick_best_structure_text(info["matched"]["element"])
                structure_mode = pick_best_structure_text(info["matched"]["mode"])
                structure_cause = pick_best_structure_text(info["matched"]["cause"])
                structure_effect = pick_best_structure_text(info["matched"]["effect"])

                final_element = structure_element or entity.get("failure_element_text")
                final_mode = structure_mode or entity.get("failure_mode_text")
                final_cause = structure_cause or entity.get("failure_cause_text")
                final_effect = structure_effect or entity.get("failure_effect_text")

                tag_element = "STRUCTURE" if structure_element else "KB"
                tag_mode = "STRUCTURE" if structure_mode else "KB"
                tag_cause = "STRUCTURE" if structure_cause else "KB"
                tag_effect = "STRUCTURE" if structure_effect else "KB"

                chain = {
                    "node_id": node_id,
                    "failure_id": fid,
                    "element": final_element,
                    "function": entity.get("function"),
                    "mode": final_mode,
                    "cause": final_cause,
                    "effect": final_effect,
                    "tagged": {
                        "element": {"text": final_element, "tag": tag_element},
                        "mode": {"text": final_mode, "tag": tag_mode},
                        "cause": {"text": final_cause, "tag": tag_cause},
                        "effect": {"text": final_effect, "tag": tag_effect},
                    },
                    "score": round(float(score), 4),
                    "matched_fields": sorted(list(fields)),
                    "match_detail": info["matched"],
                }
                all_results.append(chain)

            else:
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

    # -------------------------
    # SORT + TOP N
    # -------------------------
    all_results.sort(key=lambda x: x["score"], reverse=True)

    if top_n:
        all_results = all_results[:top_n]

    return all_results

# PPL Evaluator
evaluator = ChainPPLEvaluator("gpt2")
def attach_ppl_scores(results: List[Dict]) -> List[Dict]:
    for r in results:
        cause = r.get("cause")
        mode = r.get("mode")
        effect = r.get("effect")


        if not cause or not mode or not effect:
            r["forward_ppl"] = None
            r["reverse_ppl"] = None
            r["delta_ppl"] = None
            continue

        ppl_result = evaluator.evaluate_chain(
            cause=cause,
            mode=mode,
            effect=effect
        )

        r["forward_ppl"] = ppl_result["forward_ppl"]
        r["reverse_ppl"] = ppl_result["reverse_ppl"]
        r["delta_ppl"] = ppl_result["delta_ppl"]

    return results

if  __name__ == "__main__":


    from pprint import pprint

    # -----------------------------------------------------
    # 1) KB Path
    # -----------------------------------------------------
    KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_miniLM\failure_kb"
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
                    # "Incorrect cadence (offset)",
                    # "Unstable cadence setting",
                    # "Incorrect cadence (fixed gear ratio)",
                    # "Incorrect ratio (offset)",
                    # "Unstable ratio setting",
                    # "Does not enter limp home mode",
                    # "Sets wrong gear ratio",
                    # "Gear ratio drifts when battery is empty",
                    # "Firmware update not possible/fails",
                    # "Device bricked",
                    # "Update takes too much time (>5 minutes)",
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
                ]
            }
        ]
    }

    # -----------------------------------------------------
    # 3) Run Retrieval
    # -----------------------------------------------------
    # results = build_failure_chains_from_structure(
    #     persist_dir=KB_PATH,
    #     structure_input=structure_input,
    #     top_k_per_field=15,
    #     # minimum_field_match=2,
    #     top_n=70,
    #     # source_type=["8D","8D,old_fmea","8D,new_fmea","8D,new_fmea,old_fmea"],
    # )
    # re
    results = generate_failure_chains_from_structure(
        persist_dir=KB_PATH,
        structure_input=structure_input,
        weight_element=0.2,
        top_k_per_field=30,
        # minimum_field_match=2,
        min_similarity=0.45,
        top_n=100,
        replace=True,
        hybrid_score=False,
    )
    results = attach_ppl_scores(results)

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
            if r.get("delta_ppl") is not None:
                lines.append(f"PPL Forward: {round(r['forward_ppl'], 2)}")
                lines.append(f"PPL Reverse: {round(r['reverse_ppl'], 2)}")
                lines.append(f"Delta PPL  : {round(r['delta_ppl'], 2)}")
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


    results = build_ground_truth_input(results,target_n=15,strict_unique=True)
    print(results)


