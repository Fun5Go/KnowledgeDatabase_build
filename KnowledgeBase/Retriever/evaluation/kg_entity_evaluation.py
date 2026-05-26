import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_REPHRASED_PATH = ROOT / "sample_10pct_rephrased.json"
DEFAULT_GROUND_TRUTH_PATH = ROOT / "sample_10pct.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "kg_entity_eval_details.json"
TOP_K = 10
_SA_QUERY_MODULE = None
DEFAULT_FIELD_WEIGHTS = {
    "element": 1.0,
    "function": 1.0,
    "mode": 1.0,
    "cause": 1.0,
    "effect": 1.0,
}


FIELD_ALIASES = {
    "element": (
        "element",
        "element_text",
        "failure_element",
        "failure_element_text",
    ),
    "effect": (
        "effect",
        "effect_text",
        "failure_effect",
        "failure_effect_text",
    ),
    "mode": (
        "mode",
        "mode_text",
        "failure_mode",
        "failure_mode_text",
    ),
    "cause": (
        "cause",
        "cause_text",
        "failure_cause",
        "failure_cause_text",
    ),
}


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_sa_query_module():
    """Import SA_query lazily because it initializes the embedding model."""
    global _SA_QUERY_MODULE
    if _SA_QUERY_MODULE is None:
        from KnowledgeGraph import SA_query

        _SA_QUERY_MODULE = SA_query
    return _SA_QUERY_MODULE


def load_json_data(path: Path) -> Any:
    """Load JSON from disk with a clear error if the file is absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ordered_items(data: Any) -> List[Tuple[Optional[str], Dict[str, Any]]]:
    """Return records from either dict-shaped or list-shaped JSON."""
    if isinstance(data, dict):
        return [
            (str(key), value if isinstance(value, dict) else {"value": value})
            for key, value in data.items()
        ]

    if isinstance(data, list):
        return [
            (None, value if isinstance(value, dict) else {"value": value})
            for value in data
        ]

    raise ValueError(f"Expected JSON object or array, got {type(data).__name__}")


def align_samples(
    rephrased_data: Any,
    ground_truth_data: Any,
) -> List[Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]]:
    """
    Align query samples and ground truth.

    Prefer shared dict keys when both inputs are dicts and all query keys exist in
    the ground truth. Otherwise fall back to list/order alignment.
    """
    if isinstance(rephrased_data, dict) and isinstance(ground_truth_data, dict):
        query_keys = list(rephrased_data.keys())
        gt_keys = set(ground_truth_data.keys())
        if query_keys and all(key in gt_keys for key in query_keys):
            return [
                (
                    str(key),
                    rephrased_data[key] if isinstance(rephrased_data[key], dict) else {},
                    ground_truth_data[key] if isinstance(ground_truth_data[key], dict) else {},
                )
                for key in query_keys
            ]

    query_items = _ordered_items(rephrased_data)
    gt_items = _ordered_items(ground_truth_data)
    n = min(len(query_items), len(gt_items))
    return [
        (query_items[i][0] or gt_items[i][0], query_items[i][1], gt_items[i][1])
        for i in range(n)
    ]


def get_text_field(item: Dict[str, Any], field: str) -> str:
    """Read a failure attribute while tolerating small schema differences."""
    for key in FIELD_ALIASES[field]:
        if key in item:
            return safe_text(item.get(key))
    return ""


def build_query_from_rephrased_attributes(item: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build the multi-field query expected by KnowledgeGraph.SA_query."""
    element = get_text_field(item, "element")
    effect = get_text_field(item, "effect")
    mode = get_text_field(item, "mode")
    cause = get_text_field(item, "cause")

    return {
        "elements": [element] if element else [],
        "functions": [],
        "modes": [mode] if mode else [],
        "causes": [cause] if cause else [],
        "effects": [effect] if effect else [],
    }


def make_entity_text(element: str, effect: str, mode: str, cause: str) -> str:
    return " - ".join([safe_text(element), safe_text(effect), safe_text(mode), safe_text(cause)])


def make_entity_from_item(item: Dict[str, Any]) -> Dict[str, str]:
    element = get_text_field(item, "element")
    effect = get_text_field(item, "effect")
    mode = get_text_field(item, "mode")
    cause = get_text_field(item, "cause")
    return {
        "element": element,
        "effect": effect,
        "mode": mode,
        "cause": cause,
        "entity_text": make_entity_text(element, effect, mode, cause),
    }


def normalize_text(text: Any) -> str:
    """Normalize text for exact candidate/ground-truth comparison."""
    text = safe_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text.casefold()).strip()
    return text


def entity_key(entity: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (
        normalize_text(entity.get("element", "")),
        normalize_text(entity.get("effect", "")),
        normalize_text(entity.get("mode", "")),
        normalize_text(entity.get("cause", "")),
    )


def entity_matches_ground_truth(candidate: Dict[str, Any], gt: Dict[str, str]) -> bool:
    """
    Match a retrieved entity against ground truth.

    Ground-truth fields can be empty in the sampled JSON even when the KG can
    recover a more complete entity. Treat empty GT fields as unknown/wildcard
    fields, while requiring every non-empty GT field to match exactly after
    normalization.
    """
    compared = 0
    for field in ("element", "effect", "mode", "cause"):
        gt_value = normalize_text(gt.get(field, ""))
        if not gt_value:
            continue

        compared += 1
        if normalize_text(candidate.get(field, "")) != gt_value:
            return False

    return compared > 0


def _nonempty_or_blank(values: Iterable[Any]) -> List[str]:
    cleaned = []
    seen = set()
    for value in values or []:
        text = safe_text(value)
        key = normalize_text(text)
        if key and key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned or [""]


def _weighted_match_score(
    matches: Sequence[Dict[str, Any]],
    field: str,
    expected_text: Optional[str] = None,
    field_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Score only the match evidence that belongs to this concrete entity."""
    if field_weights is None:
        field_weights = DEFAULT_FIELD_WEIGHTS

    expected_norm = normalize_text(expected_text) if expected_text is not None else None
    score = 0.0
    for match in matches or []:
        if expected_norm is not None and normalize_text(match.get("matched_text", "")) != expected_norm:
            continue
        score += field_weights.get(field, 1.0) * float(match.get("score", 0.0))
    return score


def score_complete_entity_candidate_with_breakdown(
    result: Dict[str, Any],
    effect: str,
    cause: str,
    field_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Re-score a concrete element-effect-mode-cause entity.

    SA_query scores at the Element+Function+Mode bucket level and attaches all
    connected causes/effects as context. Once we split that context into complete
    entities, cause/effect evidence must be filtered to the selected cause/effect
    so unrelated combinations do not inherit the same score.
    """
    breakdown = {
        "element": _weighted_match_score(result.get("element_matches", []), "element", None, field_weights),
        "function": _weighted_match_score(result.get("function_matches", []), "function", None, field_weights),
        "mode": _weighted_match_score(result.get("mode_matches", []), "mode", None, field_weights),
        "cause": _weighted_match_score(result.get("cause_matches", []), "cause", cause, field_weights),
        "effect": _weighted_match_score(result.get("effect_matches", []), "effect", effect, field_weights),
    }
    score = round(sum(breakdown.values()), 4)
    rounded_breakdown = {key: round(value, 4) for key, value in breakdown.items()}
    return score, rounded_breakdown


def score_complete_entity_candidate(
    result: Dict[str, Any],
    effect: str,
    cause: str,
    field_weights: Optional[Dict[str, float]] = None,
) -> float:
    score, _ = score_complete_entity_candidate_with_breakdown(result, effect, cause, field_weights)
    return score


def _texts_from_nodes(nodes: Iterable[Any]) -> List[str]:
    texts = []
    seen = set()
    for node in nodes or []:
        text = safe_text(node.get("text") if node else "")
        key = normalize_text(text)
        if key and key not in seen:
            texts.append(text)
            seen.add(key)
    return texts


def _texts_for_trigger_field(expanded: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Choose causes/effects for the concrete entity expansion.

    If a cause/effect node triggered this expansion, only that concrete
    cause/effect receives this field evidence. Broader fields such as element or
    mode are expanded across all connected causes/effects.
    """
    field = expanded.get("trigger_field")
    trigger_text = safe_text(expanded.get("trigger_node_text"))

    causes = _texts_from_nodes(expanded.get("cause_nodes", []))
    effects = _texts_from_nodes(expanded.get("effect_nodes", []))

    if field == "cause" and trigger_text:
        causes = [trigger_text]
    if field == "effect" and trigger_text:
        effects = [trigger_text]

    return causes or [""], effects or [""]


def _add_field_evidence_to_entity(
    candidate_map: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    expanded: Dict[str, Any],
    field_weights: Dict[str, float],
) -> None:
    field = expanded.get("trigger_field")
    raw_score = float(expanded.get("trigger_score", 0.0))
    weighted_score = field_weights.get(field, 1.0) * raw_score

    element = safe_text(expanded.get("element_text"))
    mode = safe_text(expanded.get("mode_text"))
    causes, effects = _texts_for_trigger_field(expanded)

    for effect in effects:
        for cause in causes:
            entity = {
                "element": element,
                "effect": effect,
                "mode": mode,
                "cause": cause,
            }
            key = entity_key(entity)
            if key not in candidate_map:
                candidate_map[key] = {
                    **entity,
                    "entity_text": make_entity_text(element, effect, mode, cause),
                    "score": 0.0,
                    "score_breakdown": {
                        "element": 0.0,
                        "function": 0.0,
                        "mode": 0.0,
                        "cause": 0.0,
                        "effect": 0.0,
                    },
                    "source_candidate_ids": set(),
                    "_evidence_keys": set(),
                }

            bucket = candidate_map[key]
            evidence_key = (
                field,
                safe_text(expanded.get("trigger_query")),
                safe_text(expanded.get("trigger_node_sid")),
            )
            if evidence_key in bucket["_evidence_keys"]:
                continue

            bucket["_evidence_keys"].add(evidence_key)
            bucket["score"] += weighted_score
            bucket["score_breakdown"][field] += weighted_score
            bucket["source_candidate_ids"].add(expanded.get("candidate_id"))


def expand_failure_candidates_from_retrieved_nodes(
    session: Any,
    field: str,
    retrieved_items: Sequence[Tuple[Any, float, str]],
) -> List[Dict[str, Any]]:
    """
    Expand retrieved nodes through Failure-centered relationships.

    Some records, including 8D and FMEA rows without a complete Function
    backbone, are connected as:
        Failure -> HAS_ELEMENT / HAS_MODE / HAS_CAUSE / HAS_EFFECT
    instead of:
        Element -> Function -> Mode, Cause -> Mode, Mode -> Effect

    This keeps all source types eligible without requiring Element -> Function
    -> Mode and without adding any source_type filter.
    """
    if not retrieved_items:
        return []

    field_to_cypher = {
        "element": """
            MATCH (f:Failure)-[:HAS_ELEMENT]->(e:Element {semantic_id:$sid})
            OPTIONAL MATCH (f)-[:HAS_MODE]->(m:Mode)
            OPTIONAL MATCH (f)-[:HAS_CAUSE]->(c:Cause)
            OPTIONAL MATCH (f)-[:HAS_EFFECT]->(ef:Effect)
            WITH f, e, m, collect(DISTINCT c) AS cause_nodes, collect(DISTINCT ef) AS effect_nodes
            WHERE m IS NOT NULL
            RETURN
                f.failure_id AS failure_id,
                e.semantic_id AS element_sid,
                e.text AS element_text,
                m.semantic_id AS mode_sid,
                m.text AS mode_text,
                cause_nodes,
                effect_nodes
        """,
        "mode": """
            MATCH (f:Failure)-[:HAS_MODE]->(m:Mode {semantic_id:$sid})
            OPTIONAL MATCH (f)-[:HAS_ELEMENT]->(e:Element)
            OPTIONAL MATCH (f)-[:HAS_CAUSE]->(c:Cause)
            OPTIONAL MATCH (f)-[:HAS_EFFECT]->(ef:Effect)
            WITH f, e, m, collect(DISTINCT c) AS cause_nodes, collect(DISTINCT ef) AS effect_nodes
            WHERE e IS NOT NULL
            RETURN
                f.failure_id AS failure_id,
                e.semantic_id AS element_sid,
                e.text AS element_text,
                m.semantic_id AS mode_sid,
                m.text AS mode_text,
                cause_nodes,
                effect_nodes
        """,
        "cause": """
            MATCH (f:Failure)-[:HAS_CAUSE]->(c0:Cause {semantic_id:$sid})
            OPTIONAL MATCH (f)-[:HAS_ELEMENT]->(e:Element)
            OPTIONAL MATCH (f)-[:HAS_MODE]->(m:Mode)
            OPTIONAL MATCH (f)-[:HAS_CAUSE]->(c:Cause)
            OPTIONAL MATCH (f)-[:HAS_EFFECT]->(ef:Effect)
            WITH f, e, m, collect(DISTINCT c) AS cause_nodes, collect(DISTINCT ef) AS effect_nodes
            WHERE e IS NOT NULL AND m IS NOT NULL
            RETURN
                f.failure_id AS failure_id,
                e.semantic_id AS element_sid,
                e.text AS element_text,
                m.semantic_id AS mode_sid,
                m.text AS mode_text,
                cause_nodes,
                effect_nodes
        """,
        "effect": """
            MATCH (f:Failure)-[:HAS_EFFECT]->(ef0:Effect {semantic_id:$sid})
            OPTIONAL MATCH (f)-[:HAS_ELEMENT]->(e:Element)
            OPTIONAL MATCH (f)-[:HAS_MODE]->(m:Mode)
            OPTIONAL MATCH (f)-[:HAS_CAUSE]->(c:Cause)
            OPTIONAL MATCH (f)-[:HAS_EFFECT]->(ef:Effect)
            WITH f, e, m, collect(DISTINCT c) AS cause_nodes, collect(DISTINCT ef) AS effect_nodes
            WHERE e IS NOT NULL AND m IS NOT NULL
            RETURN
                f.failure_id AS failure_id,
                e.semantic_id AS element_sid,
                e.text AS element_text,
                m.semantic_id AS mode_sid,
                m.text AS mode_text,
                cause_nodes,
                effect_nodes
        """,
    }

    if field not in field_to_cypher:
        return []

    expanded = []
    for node, score, query_text in retrieved_items:
        sid = node.get("semantic_id")
        node_text = node.get("text", "")
        if not sid:
            continue

        for row in session.run(field_to_cypher[field], sid=sid):
            mode_sid = row["mode_sid"]
            failure_id = row["failure_id"]
            candidate_id = f"{failure_id}|{mode_sid}"
            expanded.append(
                {
                    "candidate_id": candidate_id,
                    "element_sid": row["element_sid"],
                    "element_text": row["element_text"],
                    "function_sid": "",
                    "function_text": "",
                    "mode_sid": mode_sid,
                    "mode_text": row["mode_text"],
                    "cause_nodes": [x for x in row["cause_nodes"] if x],
                    "effect_nodes": [x for x in row["effect_nodes"] if x],
                    "trigger_field": field,
                    "trigger_score": float(score),
                    "trigger_query": query_text,
                    "trigger_node_sid": sid,
                    "trigger_node_text": node_text,
                }
            )

    return expanded


def _candidate_combinations(result: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    element = safe_text(result.get("element"))
    mode = safe_text(result.get("mode"))
    effects = _nonempty_or_blank(result.get("effects", []))
    causes = _nonempty_or_blank(result.get("causes", []))

    for effect in effects:
        for cause in causes:
            score, score_breakdown = score_complete_entity_candidate_with_breakdown(
                result, effect, cause
            )
            yield {
                "element": element,
                "effect": effect,
                "mode": mode,
                "cause": cause,
                "entity_text": make_entity_text(element, effect, mode, cause),
                "score": score,
                "score_breakdown": score_breakdown,
                "bucket_score": result.get("score", 0.0),
                "source_candidate_id": result.get("candidate_id"),
            }


def retrieve_top5_kg_entity_candidates(
    rephrased_item: Dict[str, Any],
    top_k: int = TOP_K,
    search_top_k: int = 20,
    retrieval_k_each: int = 100,
    retrieval_pool_k: int = 300,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Retrieve ranked complete FMEA entities from the KG.

    This follows the SA_query retrieval pieces directly:
    1. retrieve field-level vector hits for element/effect/mode/cause;
    2. expand each field hit through real KG relationships;
    3. add the field score to each connected complete entity;
    4. rank complete element-effect-mode-cause entities.
    """
    query = build_query_from_rephrased_attributes(rephrased_item)
    if not any(query.values()):
        return []

    sa_query = get_sa_query_module()
    field_specs = {
        "element": {
            "label": "Element",
            "index_name": "element_embedding",
            "prefix": "Element",
            "query_texts": query.get("elements", []),
        },
        "mode": {
            "label": "Mode",
            "index_name": "mode_embedding",
            "prefix": "Failure mode",
            "query_texts": query.get("modes", []),
        },
        "cause": {
            "label": "Cause",
            "index_name": "cause_embedding",
            "prefix": "Failure cause",
            "query_texts": query.get("causes", []),
        },
        "effect": {
            "label": "Effect",
            "index_name": "effect_embedding",
            "prefix": "Failure effect",
            "query_texts": query.get("effects", []),
        },
    }

    candidate_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    with sa_query.driver.session(database=sa_query.NEO4J_DATABASE) as session:
        for field, spec in field_specs.items():
            retrieved = sa_query.retrieve_candidates(
                session=session,
                label=spec["label"],
                index_name=spec["index_name"],
                prefix=spec["prefix"],
                query_texts=spec["query_texts"],
                top_k_each=retrieval_k_each,
                pool_k=retrieval_pool_k,
                min_score=min_score,
                keep_group_or_single_only=True,
            )
            expanded = sa_query.expand_candidates_from_retrieved_nodes(session, field, retrieved)
            expanded.extend(
                expand_failure_candidates_from_retrieved_nodes(session, field, retrieved)
            )
            for expanded_candidate in expanded:
                _add_field_evidence_to_entity(
                    candidate_map,
                    expanded_candidate,
                    DEFAULT_FIELD_WEIGHTS,
                )

    candidates = []
    for candidate in candidate_map.values():
        candidate["score"] = round(float(candidate["score"]), 4)
        candidate["score_breakdown"] = {
            key: round(float(value), 4)
            for key, value in candidate["score_breakdown"].items()
        }
        candidate["source_candidate_ids"] = sorted(
            sid for sid in candidate["source_candidate_ids"] if sid
        )
        candidate.pop("_evidence_keys", None)
        candidates.append(candidate)

    return sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:top_k]


def reciprocal_rank_at_k(candidates: Sequence[Dict[str, Any]], gt: Dict[str, str], k: int) -> float:
    for rank, candidate in enumerate(candidates[:k], start=1):
        if entity_matches_ground_truth(candidate, gt):
            return 1.0 / rank
    return 0.0


def recall_at_k(candidates: Sequence[Dict[str, Any]], gt: Dict[str, str], k: int) -> float:
    return 1.0 if reciprocal_rank_at_k(candidates, gt, k) > 0.0 else 0.0


def find_correct_rank(candidates: Sequence[Dict[str, Any]], gt: Dict[str, str], k: int) -> Optional[int]:
    for rank, candidate in enumerate(candidates[:k], start=1):
        if entity_matches_ground_truth(candidate, gt):
            return rank
    return None


def evaluate(
    rephrased_data: Any,
    ground_truth_data: Any,
    top_k: int = TOP_K,
    search_top_k: int = 20,
    retrieval_k_each: int = 100,
    retrieval_pool_k: int = 300,
    min_score: float = 0.4,
    limit: Optional[int] = None,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    samples = align_samples(rephrased_data, ground_truth_data)
    if limit is not None:
        samples = samples[:limit]

    recalls: List[float] = []
    reciprocal_ranks: List[float] = []
    details: List[Dict[str, Any]] = []

    for idx, (sample_key, rephrased_item, gt_item) in enumerate(samples, start=1):
        gt_entity = make_entity_from_item(gt_item)
        candidates = retrieve_top5_kg_entity_candidates(
            rephrased_item,
            top_k=top_k,
            search_top_k=search_top_k,
            retrieval_k_each=retrieval_k_each,
            retrieval_pool_k=retrieval_pool_k,
            min_score=min_score,
        )

        rank = find_correct_rank(candidates, gt_entity, top_k)
        recall = 1.0 if rank is not None else 0.0
        rr = 1.0 / rank if rank is not None else 0.0

        recalls.append(recall)
        reciprocal_ranks.append(rr)
        details.append(
            {
                "sample_index": idx - 1,
                "sample_key": sample_key,
                "rephrased_input_attributes": make_entity_from_item(rephrased_item),
                "ground_truth_entity_text": gt_entity["entity_text"],
                "top5_retrieved_entity_candidates": candidates[:top_k],
                "correct_rank": rank,
                "recall_at_5_success": bool(rank),
                "reciprocal_rank_at_5": rr,
            }
        )

    recall_at_5 = sum(recalls) / len(recalls) if recalls else 0.0
    mrr_at_5 = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return recall_at_5, mrr_at_5, details


def save_details(details: List[Dict[str, Any]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FMEA KG retrieval as complete element-effect-mode-cause entities."
    )
    parser.add_argument("--rephrased-path", type=Path, default=DEFAULT_REPHRASED_PATH)
    parser.add_argument("--ground-truth-path", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--search-top-k", type=int, default=20)
    parser.add_argument("--retrieval-k-each", type=int, default=100)
    parser.add_argument("--retrieval-pool-k", type=int, default=300)
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rephrased_data = load_json_data(args.rephrased_path)
    ground_truth_data = load_json_data(args.ground_truth_path)

    try:
        recall, mrr, details = evaluate(
            rephrased_data=rephrased_data,
            ground_truth_data=ground_truth_data,
            top_k=args.top_k,
            search_top_k=args.search_top_k,
            retrieval_k_each=args.retrieval_k_each,
            retrieval_pool_k=args.retrieval_pool_k,
            min_score=args.min_score,
            limit=args.limit,
        )
    finally:
        if _SA_QUERY_MODULE is not None:
            _SA_QUERY_MODULE.driver.close()

    save_details(details, args.output_path)
    print(f"Recall@{args.top_k}: {recall:.4f}")
    print(f"MRR@{args.top_k}: {mrr:.4f}")
    print(f"Saved details to: {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
