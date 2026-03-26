import os
import hashlib
from typing import List, Dict, Any, Optional, Literal
import re

import torch
from neo4j import GraphDatabase
from dotenv import load_dotenv
from chromadb.utils import embedding_functions


# =========================================================
# 0. USER INPUT
# =========================================================

structure_input_powertrain = {
    "product_domain": "motor_drives",
    "nodes": [
        {
            "element_id": "E1",
            "failure_element": "Power train",
            "modes": [
                "No voltage applied",
                "Incorrect torque applied",
                "Not enough torque",
                "Motor breaks/overheats (e.g. resulting in demagnetisation)",
                "Unstable regulation",
                "High loss in torque transfer",
                "Gear train breaks/wears out",
                "Tranmission ratio drifts",
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
                "Too much noise"
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
            "modes": {
                "Soft starter": [
                    "Component break-down",
                    "Unbalanced motor currents",
                ],
                "Zero-crossing detection": [
                    "Incorrect interpretation zero-crossing",
                    "Soft start too long",
                    "No detection",
                ],
                "Relay switching": [
                    "Welded relay",
                    "Relay cannot close",
                    "False turn-on / turn-off"
                ],
            },
            "causes": {
                "mechanics": [
                    "Cooling insufficient",
                    "Compressor vibrations"
                ],
                "hardware": [
                    "(Starting) Motor current too high for chosen components",
                    "Overvoltage due to motor disconnect",
                    "Under Voltage due to incorrect triggering",
                    "Live switching of relays"
                ],
                "software": [
                    "Priority zero-crossing interrupt too low",
                    "Open loop control"
                ],
                "other": [
                    "No (correctly designed) snubber design",
                    "Too high dT junction as a result of power cycling of component"
                ]
            },
            "effects": [
                "Motor cannot start",
                "Overcurrent towards motor",
                "Motor starts without soft start",
                "Short-circuit",
            ]
        }
    ]
}


# =========================================================
# 1. CONFIG
# =========================================================

FINAL_TOP_K  = 5
POOL_K = 200
FUSION_TOP_K = 100  
# Dense threshold is meaningful for cosine similarity
MIN_SCORE_DENSE = 0.7

# BM25/fulltext scores are not directly comparable across queries
MIN_SCORE_BM25 = 2.5

# Hybrid score is normalized/fused, so do not use 0.7 here by default
MIN_SCORE_HYBRID = 0.3

HYBRID_ALPHA = 0.65

RetrievalMethod = Literal["dense", "bm25", "hybrid"]
ReviewMode = Literal["none", "human", "llm"]


# =========================================================
# 2. DB CLIENT
# =========================================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class Neo4jKGClient:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def session(self):
        return self.driver.session(database=self.database)


# =========================================================
# 3. UTILS
# =========================================================

def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def deduplicate_mapped_nodes(mapped_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for item in mapped_nodes:
        kg_id = item["kg_id"]
        if kg_id not in best or float(item["score"]) > float(best[kg_id]["score"]):
            best[kg_id] = item
    return sorted(best.values(), key=lambda x: float(x["score"]), reverse=True)


def _build_common_where_clause(
    use_discipline_filter: bool,
    discipline: Optional[str],
):
    where_clauses = []

    if use_discipline_filter and discipline is not None:
        where_clauses.append("""
            coalesce(node.discipline, "unknown") IN $allowed_disciplines
        """)

    if where_clauses:
        return "WHERE " + "\n AND ".join(where_clauses)
    return ""

LUCENE_SPECIAL_CHARS = r'(\+|-|&&|\|\||!|\(|\)|\{|\}|\[|\]|\^|"|~|\*|\?|:|\\|/)'

def escape_lucene_query(text: str) -> str:
    text = safe_text(text)
    if not text:
        return ""
    return re.sub(LUCENE_SPECIAL_CHARS, r'\\\1', text)


def fetch_candidate_context(
    session,
    semantic_id: str,
    node_label: str,
    max_neighbors_each: int = 5,
) -> Dict[str, List[Dict[str, str]]]:
    if node_label == "Cause":
        cypher = """
        MATCH (c:Cause {semantic_id: $semantic_id})
        OPTIONAL MATCH (c)-[:CAUSES]->(m:Mode)
        WITH c, collect(DISTINCT {
            kg_id: m.semantic_id,
            kg_text: coalesce(m.text, m.name, m.semantic_id)
        })[..$limit] AS modes
        OPTIONAL MATCH (c)-[:CAUSES]->(:Mode)-[:LEADS_TO]->(e:Effect)
        RETURN
            [] AS connected_causes,
            modes AS connected_modes,
            collect(DISTINCT {
                kg_id: e.semantic_id,
                kg_text: coalesce(e.text, e.name, e.semantic_id)
            })[..$limit] AS connected_effects
        """

    elif node_label == "Mode":
        cypher = """
        MATCH (m:Mode {semantic_id: $semantic_id})
        OPTIONAL MATCH (c:Cause)-[:CAUSES]->(m)
        WITH m, collect(DISTINCT {
            kg_id: c.semantic_id,
            kg_text: coalesce(c.text, c.name, c.semantic_id)
        })[..$limit] AS causes
        OPTIONAL MATCH (m)-[:LEADS_TO]->(e:Effect)
        RETURN
            causes AS connected_causes,
            [] AS connected_modes,
            collect(DISTINCT {
                kg_id: e.semantic_id,
                kg_text: coalesce(e.text, e.name, e.semantic_id)
            })[..$limit] AS connected_effects
        """

    elif node_label == "Effect":
        cypher = """
        MATCH (e:Effect {semantic_id: $semantic_id})
        OPTIONAL MATCH (m:Mode)-[:LEADS_TO]->(e)
        WITH e, collect(DISTINCT {
            kg_id: m.semantic_id,
            kg_text: coalesce(m.text, m.name, m.semantic_id)
        })[..$limit] AS modes
        OPTIONAL MATCH (c:Cause)-[:CAUSES]->(:Mode)-[:LEADS_TO]->(e)
        RETURN
            collect(DISTINCT {
                kg_id: c.semantic_id,
                kg_text: coalesce(c.text, c.name, c.semantic_id)
            })[..$limit] AS connected_causes,
            modes AS connected_modes,
            [] AS connected_effects
        """

    else:
        return {
            "connected_causes": [],
            "connected_modes": [],
            "connected_effects": []
        }

    result = session.run(
        cypher,
        semantic_id=semantic_id,
        limit=max_neighbors_each
    ).single()

    if result is None:
        return {
            "connected_causes": [],
            "connected_modes": [],
            "connected_effects": []
        }

    return {
        "connected_causes": result["connected_causes"] or [],
        "connected_modes": result["connected_modes"] or [],
        "connected_effects": result["connected_effects"] or [],
    }

def enrich_candidates_with_context(
    session,
    candidates: List[Dict[str, Any]],
    node_label: str,
    max_neighbors_each: int = 5,
) -> List[Dict[str, Any]]:
    enriched = []

    for c in candidates:
        item = dict(c)
        item["context"] = fetch_candidate_context(
            session=session,
            semantic_id=c["kg_id"],
            node_label=node_label,
            max_neighbors_each=max_neighbors_each,
        )
        enriched.append(item)

    return enriched
# =========================================================
# 4. QUERY EXTRACTION
# =========================================================

def extract_cause_queries_with_disciplines(structure_input: Dict[str, Any]) -> List[Dict[str, str]]:
    cause_items = []

    for node in structure_input.get("nodes", []):
        causes = node.get("causes", {})
        if not isinstance(causes, dict):
            continue

        for discipline, items in causes.items():
            d = safe_text(discipline) or "unknown"
            if not isinstance(items, list):
                continue

            for cause_text in items:
                txt = safe_text(cause_text)
                if txt:
                    cause_items.append({
                        "query_text": txt,
                        "discipline": d
                    })

    return cause_items


def extract_mode_queries(structure_input: Dict[str, Any]) -> List[str]:
    all_modes = []

    for node in structure_input.get("nodes", []):
        modes = node.get("modes", [])

        if isinstance(modes, list):
            for x in modes:
                txt = safe_text(x)
                if txt:
                    all_modes.append(txt)

        elif isinstance(modes, dict):
            for _, mode_list in modes.items():
                if isinstance(mode_list, list):
                    for x in mode_list:
                        txt = safe_text(x)
                        if txt:
                            all_modes.append(txt)

    return all_modes


def extract_effect_queries(structure_input: Dict[str, Any]) -> List[str]:
    all_effects = []

    for node in structure_input.get("nodes", []):
        effects = node.get("effects", [])
        if isinstance(effects, list):
            for x in effects:
                txt = safe_text(x)
                if txt:
                    all_effects.append(txt)

    return all_effects


def extract_structure_queries(structure_input: Dict[str, Any]):
    all_causes = extract_cause_queries_with_disciplines(structure_input)
    all_modes = extract_mode_queries(structure_input)
    all_effects = extract_effect_queries(structure_input)
    return all_causes, all_modes, all_effects


# =========================================================
# 5. EMBEDDING
# =========================================================

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_embedding_cache: Dict[str, list] = {}


def embed(text: str):
    text = safe_text(text)
    if not text:
        return None
    if text in _embedding_cache:
        return _embedding_cache[text]
    vec = embedder([text])[0]
    _embedding_cache[text] = vec
    return vec


# =========================================================
# 6. RETRIEVAL
# =========================================================

def retrieve_dense_candidates(
    session,
    label: str,
    index_name: str,
    query_text: str,
    top_k: int = 10,
    pool_k: int = 100,
    min_score: float = 0.0,
    keep_group_or_single_only: bool = True,
    use_discipline_filter: bool = False,
    discipline: Optional[str] = None,
) -> List[Dict[str, Any]]:
    emb = embed(query_text)
    if emb is None:
        return []

    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().tolist()

    params = {
        "embedding": emb,
        "k": top_k,
    }
    if use_discipline_filter and discipline is not None:
        params["allowed_disciplines"] = [discipline, "unknown"]

    filter_clause = _build_common_where_clause(
        use_discipline_filter=use_discipline_filter,
        discipline=discipline,
    )

    cypher = f"""
    CALL db.index.vector.queryNodes(
        '{index_name}',
        {pool_k},
        $embedding
    )
    YIELD node, score

    WITH node, vector.similarity.cosine(node.embedding, $embedding) AS final_score
    {filter_clause}
    RETURN
        node.semantic_id AS kg_id,
        coalesce(node.text, node.name, node.semantic_id) AS kg_text,
        coalesce(node.discipline, "unknown") AS discipline,
        final_score,
        "dense" AS source
    ORDER BY final_score DESC
    LIMIT $k
    """

    query_result = session.run(cypher, **params)

    candidates = []
    for r in query_result:
        score = float(r["final_score"])
        if score >= min_score:
            candidates.append({
                "kg_id": r["kg_id"],
                "kg_text": r["kg_text"],
                "discipline": r["discipline"],
                "score": score,
                "source": "dense"
            })

    return deduplicate_mapped_nodes(candidates)


def retrieve_bm25_candidates(
    session,
    label: str,
    fulltext_index_name: str,
    query_text: str,
    top_k: int = 10,
    min_score: float = 0.0,
    keep_group_or_single_only: bool = True,
    use_discipline_filter: bool = False,
    discipline: Optional[str] = None,
) -> List[Dict[str, Any]]:
    escaped_query = escape_lucene_query(query_text)

    params = {
        "query_text": escaped_query,
        "k": top_k,
    }
    if use_discipline_filter and discipline is not None:
        params["allowed_disciplines"] = [discipline, "unknown"]

    filter_clause = _build_common_where_clause(
        use_discipline_filter=use_discipline_filter,
        discipline=discipline,
    )

    cypher = f"""
    CALL db.index.fulltext.queryNodes(
        '{fulltext_index_name}',
        $query_text
    )
    YIELD node, score

    WITH node, score AS final_score
    {filter_clause}
    RETURN
        node.semantic_id AS kg_id,
        coalesce(node.text, node.name, node.semantic_id) AS kg_text,
        coalesce(node.discipline, "unknown") AS discipline,
        final_score,
        "bm25" AS source
    ORDER BY final_score DESC
    LIMIT $k
    """

    query_result = session.run(cypher, **params)

    candidates = []
    for r in query_result:
        score = float(r["final_score"])
        if score >= min_score:
            candidates.append({
                "kg_id": r["kg_id"],
                "kg_text": r["kg_text"],
                "discipline": r["discipline"],
                "score": score,
                "source": "bm25"
            })

    return deduplicate_mapped_nodes(candidates)


def _minmax_normalize(cands: List[Dict[str, Any]], score_key: str = "score"):
    if not cands:
        return cands

    # avoid in-place mutation side effects
    out = [dict(x) for x in cands]

    scores = [float(x[score_key]) for x in out]
    s_min, s_max = min(scores), max(scores)

    for x in out:
        if s_max - s_min < 1e-12:
            x["norm_score"] = 1.0
        else:
            x["norm_score"] = (float(x[score_key]) - s_min) / (s_max - s_min)

    return out


def merge_hybrid_candidates(
    dense_candidates: List[Dict[str, Any]],
    bm25_candidates: List[Dict[str, Any]],
    alpha: float = 0.7,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    dense_candidates = _minmax_normalize(dense_candidates)
    bm25_candidates = _minmax_normalize(bm25_candidates)

    merged = {}

    for c in dense_candidates:
        kg_id = c["kg_id"]
        merged[kg_id] = {
            "kg_id": c["kg_id"],
            "kg_text": c["kg_text"],
            "discipline": c.get("discipline", "unknown"),
            "dense_score": float(c["score"]),
            "bm25_score": 0.0,
            "dense_norm": float(c["norm_score"]),
            "bm25_norm": 0.0,
            "score": alpha * float(c["norm_score"]),
            "source": "hybrid"
        }

    for c in bm25_candidates:
        kg_id = c["kg_id"]
        if kg_id not in merged:
            merged[kg_id] = {
                "kg_id": c["kg_id"],
                "kg_text": c["kg_text"],
                "discipline": c.get("discipline", "unknown"),
                "dense_score": 0.0,
                "bm25_score": float(c["score"]),
                "dense_norm": 0.0,
                "bm25_norm": float(c["norm_score"]),
                "score": (1 - alpha) * float(c["norm_score"]),
                "source": "hybrid"
            }
        else:
            merged[kg_id]["bm25_score"] = float(c["score"])
            merged[kg_id]["bm25_norm"] = float(c["norm_score"])
            merged[kg_id]["score"] = (
                alpha * float(merged[kg_id]["dense_norm"]) +
                (1 - alpha) * float(c["norm_score"])
            )

    final_candidates = list(merged.values())
    final_candidates.sort(key=lambda x: float(x["score"]), reverse=True)
    return final_candidates[:top_k]


def retrieve_candidates(
    session,
    label: str,
    index_name: str,
    fulltext_index_name: Optional[str],
    query_text: str,
    method: RetrievalMethod = "dense",
    final_k: int = 10,
    fusion_k: int = 50,
    pool_k: int = 100,
    dense_min_score: float = MIN_SCORE_DENSE,
    bm25_min_score: float = MIN_SCORE_BM25,
    hybrid_min_score: float = MIN_SCORE_HYBRID,
    use_discipline_filter: bool = False,
    discipline: Optional[str] = None,
    hybrid_alpha: float = HYBRID_ALPHA,
) -> List[Dict[str, Any]]:

    if method == "dense":
        return retrieve_dense_candidates(
            session=session,
            label=label,
            index_name=index_name,
            query_text=query_text,
            top_k=final_k,
            pool_k=pool_k,
            min_score=dense_min_score,
            use_discipline_filter=use_discipline_filter,
            discipline=discipline,
        )

    if method == "bm25":
        if not fulltext_index_name:
            raise ValueError("fulltext_index_name is required for BM25 retrieval")

        return retrieve_bm25_candidates(
            session=session,
            label=label,
            fulltext_index_name=fulltext_index_name,
            query_text=query_text,
            top_k=final_k,
            min_score=bm25_min_score,
            use_discipline_filter=use_discipline_filter,
            discipline=discipline,
        )

    if method == "hybrid":
        if not fulltext_index_name:
            raise ValueError("fulltext_index_name is required for hybrid retrieval")

        # 先各自保留更大的候选池用于融合
        dense_candidates = retrieve_dense_candidates(
            session=session,
            label=label,
            index_name=index_name,
            query_text=query_text,
            top_k=fusion_k,
            pool_k=pool_k,
            min_score=0.0,
            use_discipline_filter=use_discipline_filter,
            discipline=discipline,
        )

        bm25_candidates = retrieve_bm25_candidates(
            session=session,
            label=label,
            fulltext_index_name=fulltext_index_name,
            query_text=query_text,
            top_k=fusion_k,
            min_score=0.0,
            use_discipline_filter=use_discipline_filter,
            discipline=discipline,
        )

        hybrid = merge_hybrid_candidates(
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            alpha=hybrid_alpha,
            top_k=final_k,   # 注意：这里才截断成最终输出数量
        )

        if hybrid_min_score > 0:
            hybrid = [c for c in hybrid if float(c["score"]) >= hybrid_min_score]

        return hybrid

    raise ValueError(f"Unsupported retrieval method: {method}")


# =========================================================
# 7. REVIEW / PRINT
# =========================================================

def print_similar_candidates(
    query_text: str,
    candidates: List[Dict[str, Any]],
    allowed_disciplines: Optional[List[str]] = None,
    title: str = "CANDIDATES"
):
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"Query Text: {query_text}")
    if allowed_disciplines is not None:
        print(f"Allowed Disciplines: {allowed_disciplines}")

    if not candidates:
        print("  No candidates found.")
        return

    for i, c in enumerate(candidates, 1):
        print(f"  [{i}]")
        print(f"    KG ID      : {c.get('kg_id')}")
        print(f"    KG Text    : {c.get('kg_text')}")
        print(f"    Discipline : {c.get('discipline')}")
        print(f"    Score      : {float(c.get('score', 0.0)):.4f}")
        if "dense_score" in c:
            print(f"    Dense      : {float(c.get('dense_score', 0.0)):.4f}")
        if "bm25_score" in c:
            print(f"    BM25       : {float(c.get('bm25_score', 0.0)):.4f}")
        print(f"    Source     : {c.get('source', 'unknown')}")

        ctx = c.get("context", {})
        causes = ctx.get("connected_causes", [])
        modes = ctx.get("connected_modes", [])
        effects = ctx.get("connected_effects", [])

        if causes:
            print("    Connected Causes:")
            for x in causes:
                print(f"      - {x.get('kg_text')} ({x.get('kg_id')})")

        if modes:
            print("    Connected Modes:")
            for x in modes:
                print(f"      - {x.get('kg_text')} ({x.get('kg_id')})")

        if effects:
            print("    Connected Effects:")
            for x in effects:
                print(f"      - {x.get('kg_text')} ({x.get('kg_id')})")


def human_select_candidates(
    query_text: str,
    candidates: List[Dict[str, Any]],
    allow_skip: bool = True,
    allow_all: bool = True,
) -> List[Dict[str, Any]]:
    print_similar_candidates(query_text, candidates, title="HUMAN REVIEW")

    if not candidates:
        return []

    while True:
        prompt = f"Select candidate indices for '{query_text}' (e.g. 1,2,4"
        if allow_all:
            prompt += ", or 'all'"
        if allow_skip:
            prompt += ", Enter to skip"
        prompt += "): "

        raw = input(prompt).strip()

        if raw == "" and allow_skip:
            return []

        if allow_all and raw.lower() == "all":
            return candidates

        try:
            parts = [x.strip() for x in raw.split(",") if x.strip()]
            if not parts:
                raise ValueError

            indices = []
            for p in parts:
                if not p.isdigit():
                    raise ValueError
                idx = int(p)
                if not (1 <= idx <= len(candidates)):
                    raise ValueError
                indices.append(idx)

            # 去重但保持输入顺序
            seen = set()
            selected = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    selected.append(candidates[idx - 1])

            return selected

        except ValueError:
            print("Invalid input. Use e.g. 1,2,4 or all.")


def llm_select_candidate(
    agent,
    query_text: str,
    candidates: List[Dict[str, Any]],
    node_type: str,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    simplified_candidates = []
    for i, c in enumerate(candidates, 1):
        simplified_candidates.append({
            "index": i,
            "kg_id": c["kg_id"],
            "kg_text": c["kg_text"],
            "discipline": c.get("discipline", "unknown"),
            "score": c["score"],
            "source": c.get("source", "unknown"),
        })

    decision = agent.select_best_match(
        query_text=query_text,
        node_type=node_type,
        candidates=simplified_candidates,
    )

    selected_index = decision.get("selected_index")
    if selected_index is None:
        return None

    if 1 <= selected_index <= len(candidates):
        return candidates[selected_index - 1]

    return None


def review_and_select_candidate(
    query_text: str,
    candidates: List[Dict[str, Any]],
    review_mode: ReviewMode = "none",
    node_type: str = "node",
    agent=None,
    auto_select_if_single: bool = True,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    if auto_select_if_single and len(candidates) == 1:
        return candidates[0]

    if review_mode == "none":
        return candidates[0]

    if review_mode == "human":
        return human_select_candidates(query_text, candidates)

    if review_mode == "llm":
        if agent is None:
            raise ValueError("agent is required when review_mode='llm'")
        return llm_select_candidate(
            agent=agent,
            query_text=query_text,
            candidates=candidates,
            node_type=node_type,
        )

    raise ValueError(f"Unsupported review mode: {review_mode}")


# =========================================================
# 8. MAPPING
# =========================================================

def map_query_texts_to_nodes(
    session,
    label: str,
    index_name: str,
    fulltext_index_name: Optional[str],
    query_texts: List[Any],
    top_k_each: int = 5,
    fusion_k_each: int = 50,
    pool_k: int = 100,
    use_discipline_filter: bool = False,
    retrieval_method: RetrievalMethod = "dense",
    review_mode: ReviewMode = "none",
    review_agent=None,
    print_candidates: bool = True,
    hybrid_alpha: float = HYBRID_ALPHA,
) -> List[Dict[str, Any]]:
    results = []

    for item in query_texts:
        if isinstance(item, dict):
            text = safe_text(item.get("query_text"))
            discipline = safe_text(item.get("discipline")) or "unknown"
        else:
            text = safe_text(item)
            discipline = None

        candidates = retrieve_candidates(
            session=session,
            label=label,
            index_name=index_name,
            fulltext_index_name=fulltext_index_name,
            query_text=text,
            method=retrieval_method,
            final_k=top_k_each,
            fusion_k=fusion_k_each,
            pool_k=pool_k,
            dense_min_score=MIN_SCORE_DENSE,
            bm25_min_score=MIN_SCORE_BM25,
            hybrid_min_score=MIN_SCORE_HYBRID,
            use_discipline_filter=use_discipline_filter,
            discipline=discipline,
            hybrid_alpha=hybrid_alpha,
        )
        candidates = enrich_candidates_with_context(
            session=session,
            candidates=candidates,
            node_label=label,
            max_neighbors_each=5,
        )

        allowed_disciplines = [discipline, "unknown"] if discipline is not None else None

        if print_candidates:
            print_similar_candidates(
                query_text=text,
                candidates=candidates,
                allowed_disciplines=allowed_disciplines,
                title=f"{label.upper()} RETRIEVAL ({retrieval_method})"
            )

        selected_node = review_and_select_candidate(
            query_text=text,
            candidates=candidates,
            review_mode=review_mode,
            node_type=label.lower(),
            agent=review_agent,
            auto_select_if_single=True,
        )

        out = {
            "query_text": text,
            "mapped_nodes": candidates,
            "selected_node": selected_node,
        }
        if discipline is not None:
            out["allowed_disciplines"] = allowed_disciplines

        results.append(out)

    return results


def map_structure_input_to_kg(
    session,
    structure_input: Dict[str, Any],
    retrieval_method: RetrievalMethod = "dense",
    review_mode: ReviewMode = "none",
    review_agent=None,
    print_candidates: bool = True,
):
    causes, modes, effects = extract_structure_queries(structure_input)

    cause_maps = map_query_texts_to_nodes(
        session=session,
        label="Cause",
        index_name="cause_embedding",
        fulltext_index_name="cause_fulltext",
        query_texts=causes,
        top_k_each=FINAL_TOP_K,
        pool_k=POOL_K,
        use_discipline_filter=True,
        retrieval_method=retrieval_method,
        review_mode=review_mode,
        review_agent=review_agent,
        print_candidates=print_candidates,
    )

    mode_maps = map_query_texts_to_nodes(
        session=session,
        label="Mode",
        index_name="mode_embedding",
        fulltext_index_name="mode_fulltext",
        query_texts=modes,
        top_k_each=FINAL_TOP_K,
        pool_k=POOL_K,
        retrieval_method=retrieval_method,
        review_mode=review_mode,
        review_agent=review_agent,
        print_candidates=print_candidates,
    )

    effect_maps = map_query_texts_to_nodes(
        session=session,
        label="Effect",
        index_name="effect_embedding",
        fulltext_index_name="effect_fulltext",
        query_texts=effects,
        top_k_each=FINAL_TOP_K,
        pool_k=POOL_K,
        retrieval_method=retrieval_method,
        review_mode=review_mode,
        review_agent=review_agent,
        print_candidates=print_candidates,
    )

    return {
        "causes": cause_maps,
        "modes": mode_maps,
        "effects": effect_maps
    }


def print_mapping_results(title: str, mapping_list: List[Dict[str, Any]]):
    print(f"\n{'#' * 100}")
    print(title)
    print(f"{'#' * 100}")

    for item in mapping_list:
        print(f"\nQuery Text: {item['query_text']}")

        if "allowed_disciplines" in item:
            print(f"Allowed Disciplines: {item['allowed_disciplines']}")

        mapped_nodes = item.get("mapped_nodes", [])
        if not mapped_nodes:
            print("  No mapped nodes.")
        else:
            for i, node in enumerate(mapped_nodes, 1):
                print(f"  [{i}]")
                print(f"    KG ID      : {node.get('kg_id')}")
                print(f"    KG Text    : {node.get('kg_text')}")
                print(f"    Discipline : {node.get('discipline')}")
                print(f"    Score      : {float(node.get('score', 0.0)):.4f}")
                if "dense_score" in node:
                    print(f"    Dense      : {float(node.get('dense_score', 0.0)):.4f}")
                if "bm25_score" in node:
                    print(f"    BM25       : {float(node.get('bm25_score', 0.0)):.4f}")
                print(f"    Source     : {node.get('source', 'unknown')}")

        selected = item.get("selected_node")
        if selected is not None:
            print("  --> SELECTED")
            print(f"      KG ID      : {selected.get('kg_id')}")
            print(f"      KG Text    : {selected.get('kg_text')}")
            print(f"      Discipline : {selected.get('discipline')}")
            print(f"      Score      : {float(selected.get('score', 0.0)):.4f}")


# =========================================================
# 9. MAIN
# =========================================================

if __name__ == "__main__":
    kg_client = Neo4jKGClient(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )

    try:
        with kg_client.session() as session:
            mapped = map_structure_input_to_kg(
                session=session,
                structure_input=structure_input_motorcontrol,
                retrieval_method="hybrid",   # "dense" / "bm25" / "hybrid"
                review_mode="human",          # "none" / "human" / "llm"
                review_agent=None,
                print_candidates=True,
            )

            print_mapping_results("CAUSE MAPPING", mapped["causes"])
            print_mapping_results("MODE MAPPING", mapped["modes"])
            print_mapping_results("EFFECT MAPPING", mapped["effects"])

    finally:
        kg_client.close()