from typing import Dict, List, Union, Optional, Any
from collections import defaultdict
from pathlib import Path
from Retriever.failure_query_tools import _load_kb,query_semantic_kb

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
    Accumulate failure_id scores from semantic hits with graph constraint.

    Upgrade:
    - Keep highest similarity per (failure_id, field_type, semantic_id)
    - Update score by delta if better similarity found
    """

    ids = (semantic_query_result.get("ids", [[]]) or [[]])[0] or []
    dists = (semantic_query_result.get("distances", [[]]) or [[]])[0] or []

    for sid, dist in zip(ids, dists):

        try:
            similarity = max(0.0, 1.0 - float(dist))
        except Exception:
            continue

        if similarity < min_similarity:
            continue

        node = kb.field_store.get(sid, {}) or {}
        failure_ids = node.get("failure_ids", []) or []

        if not failure_ids:
            continue

        kb_text = node.get("text", "").strip() if isinstance(node.get("text"), str) else ""

        for fid in failure_ids:

            matched_list = candidate_scores[fid]["matched"][field_type]

            # ------------------------------------------------
            # Check if this semantic_id already exists
            # ------------------------------------------------
            existing = None
            for m in matched_list:
                if m["semantic_id"] == sid:
                    existing = m
                    break

            if existing:
                old_sim = float(existing.get("similarity", 0.0))

                # Keep only if new similarity is higher
                if similarity > old_sim:
                    delta = (similarity - old_sim) * float(weight)
                    candidate_scores[fid]["score"] += delta

                    existing["similarity"] = round(float(similarity), 4)
                    existing["structure_text"] = query_text
                    existing["kb_text"] = kb_text

                continue  # done handling this sid for this fid

            # ------------------------------------------------
            # New semantic hit for this field
            # ------------------------------------------------
            candidate_scores[fid]["score"] += similarity * float(weight)
            candidate_scores[fid]["field_hits"].add(field_type)

            matched_list.append({
                "semantic_id": sid,
                "similarity": round(float(similarity), 4),
                "structure_text": query_text,
                "kb_text": kb_text
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
            lines.append(f"  Function: {r.get('function')}  [KB]") 
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