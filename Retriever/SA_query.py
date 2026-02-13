from typing import Dict, List, Union, Optional
from collections import defaultdict
from pathlib import Path
from Retriever.failure_query_tools import _load_kb,query_semantic_kb
from JSON_FMEA_KB.kb_structure import FMEAFailureKB


# =========================================================
# Structure → Failure Chain Retrieval
# =========================================================

# =========================================================
# Structure → Failure Chain Retrieval (Soft-AND Optimized)
# =========================================================

def build_failure_chains_from_structure(
    persist_dir: Union[str, Path],
    structure_input: Dict,
    top_k_per_field: int = 10,
    min_count: Optional[int] = None,
    weight_element: float = 1.0,
    weight_mode: float = 2.0,
    weight_cause: float = 1.5,
    weight_effect: float = 1.5,
    minimum_field_match: int = 2,
    top_n: Optional[int] = 50,
    source_type: Optional[str] = None,
):
    """
    Soft-AND based retrieval.
    Independent field search + score aggregation.
    """

    kb = _load_kb(persist_dir)
    product_domain = structure_input.get("product_domain")

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

    # =====================================================
    # 1) Traverse structure
    # =====================================================
    for node in structure_input.get("nodes", []):

        element_text = node.get("failure_element")
        modes = node.get("modes", []) or []
        causes = node.get("causes", []) or []
        effects = node.get("effects", []) or []

        # ---------- ELEMENT ----------
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
                kb, res, candidate_scores,
                field_type="element",
                weight=weight_element
            )

        # ---------- MODES ----------
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
                kb, res, candidate_scores,
                field_type="mode",
                weight=weight_mode
            )

        # ---------- CAUSES ----------
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
                kb, res, candidate_scores,
                field_type="cause",
                weight=weight_cause
            )

        # ---------- EFFECTS ----------
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
                kb, res, candidate_scores,
                field_type="effect",
                weight=weight_effect
            )

    # =====================================================
    # 2) Reconstruct full failure chains
    # =====================================================
    results = []

    for fid, info in candidate_scores.items():

        # ---- minimum field match filter ----
        if len(info["field_hits"]) < minimum_field_match:
            continue

        entity = kb.entity_store.get(fid)
        if not entity:
            continue

        # ---- product_domain filter ----
        if product_domain:
            if entity.get("product_domain") != product_domain:
                continue

        # ---- normalize score (avoid explosion) ----
        normalized_score = info["score"] / max(len(info["field_hits"]), 1)

        chain = {
            "failure_id": fid,
            "element": entity.get("failure_element_text"),
            "function": entity.get("function"),
            "mode": entity.get("failure_mode_text"),
            "effect": entity.get("failure_effect_text"),
            "cause": entity.get("failure_cause_text"),
            "score": round(normalized_score, 4),
            "matched_fields": list(info["field_hits"]),
            "match_detail": info["matched"],
        }

        results.append(chain)

    # =====================================================
    # 3) Sort
    # =====================================================
    results.sort(key=lambda x: x["score"], reverse=True)

    if top_n:
        results = results[:top_n]

    return results


# =========================================================
# Helper
# =========================================================

def _accumulate_candidate_scores(
    kb,
    semantic_query_result,
    candidate_scores,
    field_type: str,
    weight: float,
):
    ids = semantic_query_result.get("ids", [[]])[0]
    dists = semantic_query_result.get("distances", [[]])[0]

    for sid, dist in zip(ids, dists):

        similarity = max(0.0, 1 - float(dist))

        node = kb.field_store.get(sid, {})
        failure_ids = node.get("failure_ids", []) or []

        for fid in failure_ids:

            # ---- prevent duplicate semantic_id scoring ----
            existing_ids = {
                m["semantic_id"]
                for m in candidate_scores[fid]["matched"][field_type]
            }

            if sid in existing_ids:
                continue

            candidate_scores[fid]["score"] += similarity * weight
            candidate_scores[fid]["field_hits"].add(field_type)

            candidate_scores[fid]["matched"][field_type].append({
                "semantic_id": sid,
                "similarity": round(similarity, 4)
            })

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
                "failure_element": "",
                "modes": [
                "unstable control behavior",
                "Motor failure / overheating",
                "Insufficient torque output",
                "No voltage applied",
                "creates too much noise",

                ],
                "causes": [
                    "Incorrect control parameter settings",
                    "Thermal protection malfunction",
                    "Startup motor current exceeds component limits",
                    "Overvoltage caused by motor disconnection",
                    "Embedded SW migration issue (e3.2)",
                    
                ],
                "effects": [
                    "Improper gear shifting",
                    "Gear does not engage",
                    "too much noise",
                    "Motor not shorted while device is not powered",
                    "No connection to RC",
                    "Unstable cadence setting"
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
        top_k_per_field=5,
        minimum_field_match=2,
        top_n=50,
        # source_type="old_fmea",
    )
    from typing import List, Dict, Tuple


    from typing import List, Dict, Tuple, Optional


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

            match_detail = r.get("match_detail", {}) or {}
            if isinstance(match_detail, dict):
                for field_type, matches in match_detail.items():
                    if matches:
                        lines.append(f"  {str(field_type).upper()}:")
                        for m in matches:
                            lines.append(f"    - {m}")

            lines.append("-" * 60)

        return "\n".join(lines)


    results = build_ground_truth_input(results,target_n=30,strict_unique=True)
    print(results)

    # # -----------------------------------------------------
    # # 4) Print Results
    # # -----------------------------------------------------
    # print("\n" + "=" * 100)
    # print(f"Total Retrieved Chains: {len(results)}")
    # print("=" * 100)

    # for i, r in enumerate(results):
    #     print(f"\nRank {i+1}")
    #     print("-" * 80)
    #     print(f"Failure ID : {r['failure_id']}")
    #     print(f"Score      : {r['score']}")
    #     print(f"Element    : {r['element']}")
    #     print(f"Mode       : {r['mode']}")
    #     print(f"Effect     : {r['effect']}")
    #     print(f"Cause      : {r['cause']}")
    #     print(f"Matched    : {r['matched_fields']}")

    # print("\nDone.")
