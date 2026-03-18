import os
import hashlib
from typing import Dict, List, Tuple, Any

from neo4j import GraphDatabase
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

load_dotenv()

# =========================================================
# CONFIG
# =========================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# =========================================================
# HELPERS
# =========================================================
def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()

def flatten_causes(causes):
    """
    支持两种格式：
    1. list[str]
    2. dict[str, list[str]]   # 你当前的格式
    """
    if not causes:
        return []

    if isinstance(causes, list):
        return [safe_text(x) for x in causes if safe_text(x)]

    if isinstance(causes, dict):
        all_causes = []
        for _, items in causes.items():
            if isinstance(items, list):
                all_causes.extend([safe_text(x) for x in items if safe_text(x)])
        return all_causes

    return []



# =========================================================
# EMBEDDING
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
# DEDUPLICATION
# =========================================================
def deduplicate_by_semantic_id(node_score_list):
    """
    输入支持:
        (node, score, query_text)

    对同一个 semantic_id:
    - 保留最高 score
    - 同时保留对应 query_text
    """
    best = {}

    for node, score, query_text in node_score_list:
        sid = node.get("semantic_id")
        if not sid:
            continue

        if sid not in best or float(score) > float(best[sid][1]):
            best[sid] = (node, float(score), query_text)

    return list(best.values())


def deduplicate_match_list(match_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对 matched query list 去重:
    同一个 matched_sid + query，保留最高分
    """
    best = {}
    for item in match_list:
        key = (
            item.get("matched_sid"),
            item.get("query"),
            item.get("field")
        )
        if key not in best or item["score"] > best[key]["score"]:
            best[key] = item
    return list(best.values())


# =========================================================
# PRINT
# =========================================================
def print_results(results):
    print("\n" + "=" * 100)
    print("Semantic Historical Failure Search Results")
    print("=" * 100)

    if not results:
        print("\nNo matched historical failures found.")
        return

    for i, item in enumerate(results, 1):
        print(f"\n[{i}] Weighted Score : {item['score']}")
        print(f"    Candidate ID    : {item['candidate_id']}")
        print(f"    Element         : {item['element']}")
        print(f"    Function        : {item['function']}")
        print(f"    Mode            : {item['mode']}")

        print("\n    Causes:")
        if item["causes"]:
            for c in item["causes"]:
                print(f"      - {c}")
        else:
            print("      - None")

        print("\n    Effects:")
        if item["effects"]:
            for ef in item["effects"]:
                print(f"      - {ef}")
        else:
            print("      - None")

        print("\n    Matched Queries:")

        if item["element_matches"]:
            print("      Element Match:")
            for m in item["element_matches"]:
                print(
                    f"        - query=\"{m['query']}\""
                    f" | matched=\"{m['matched_text']}\""
                    f" | score={m['score']}"
                )

        if item["function_matches"]:
            print("      Function Match:")
            for m in item["function_matches"]:
                print(
                    f"        - query=\"{m['query']}\""
                    f" | matched=\"{m['matched_text']}\""
                    f" | score={m['score']}"
                )

        if item["mode_matches"]:
            print("      Mode Match:")
            for m in item["mode_matches"]:
                print(
                    f"        - query=\"{m['query']}\""
                    f" | matched=\"{m['matched_text']}\""
                    f" | score={m['score']}"
                )

        if item["cause_matches"]:
            print("      Cause Match:")
            for m in item["cause_matches"]:
                print(
                    f"        - query=\"{m['query']}\""
                    f" | matched=\"{m['matched_text']}\""
                    f" | score={m['score']}"
                )

        if item["effect_matches"]:
            print("      Effect Match:")
            for m in item["effect_matches"]:
                print(
                    f"        - query=\"{m['query']}\""
                    f" | matched=\"{m['matched_text']}\""
                    f" | score={m['score']}"
                )


# =========================================================
# GENERIC RETRIEVAL
# =========================================================
def retrieve_candidates(
    session,
    label: str,
    index_name: str,
    prefix: str,
    query_texts: List[str],
    top_k_each: int = 5,
    pool_k: int = 100,
    min_score: float = 0.75,
    keep_group_or_single_only: bool = True
):
    """
    通用向量召回:
    返回 [(node, refined_score, query_text), ...]
    """

    all_results = []

    for text in query_texts:
        emb = embed(f"{prefix}: {text}")
        if emb is None:
            continue

        if keep_group_or_single_only:
            filter_clause = f"""
            WHERE
                coalesce(node.is_group, false) = true
                OR NOT (node)-[:BELONGS_TO]->(:{label})
            """
        else:
            filter_clause = ""

        cypher = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            {pool_k},
            $embedding
        )
        YIELD node, score

        WITH node, vector.similarity.cosine(node.embedding, $embedding) AS refined_score
        {filter_clause}
        RETURN node, refined_score
        ORDER BY refined_score DESC
        LIMIT $k
        """

        result = session.run(cypher, embedding=emb, k=top_k_each)

        for r in result:
            score = float(r["refined_score"])
            if score >= min_score:
                all_results.append((r["node"], score, text))

    return deduplicate_by_semantic_id(all_results)


# =========================================================
# EXPAND RETRIEVED NODE -> FAILURE CANDIDATE
# =========================================================
def expand_candidates_from_retrieved_nodes(session, field: str, retrieved_items):
    """
    把每个字段召回结果扩展成完整 failure candidate:
        candidate = Element + Function + Mode + all Causes + all Effects

    输入 retrieved_items:
        [(node, score, query_text), ...]

    返回:
        [
            {
                "candidate_id": ...,
                "element_sid": ...,
                "element_text": ...,
                "function_sid": ...,
                "function_text": ...,
                "mode_sid": ...,
                "mode_text": ...,
                "causes": [...],
                "effects": [...],
                "trigger_field": field,
                "trigger_score": ...,
                "trigger_query": ...,
                "trigger_node_sid": ...,
                "trigger_node_text": ...
            }
        ]
    """

    if not retrieved_items:
        return []

    candidates = []

    field_to_cypher = {
        "element": """
            MATCH (e:Element {semantic_id:$sid})-[:HAS_FUNCTION]->(f:Function)-[:HAS_MODE]->(m:Mode)
            OPTIONAL MATCH (m)-[:CAUSED_BY]->(c:Cause)
            OPTIONAL MATCH (m)-[:LEADS_TO]->(ef:Effect)
            RETURN
                e.semantic_id AS element_sid,
                e.text AS element_text,
                f.semantic_id AS function_sid,
                f.text AS function_text,
                m.semantic_id AS mode_sid,
                m.text AS mode_text,
                collect(DISTINCT c) AS cause_nodes,
                collect(DISTINCT ef) AS effect_nodes
        """,
        "function": """
            MATCH (e:Element)-[:HAS_FUNCTION]->(f:Function {semantic_id:$sid})-[:HAS_MODE]->(m:Mode)
            OPTIONAL MATCH (m)-[:CAUSED_BY]->(c:Cause)
            OPTIONAL MATCH (m)-[:LEADS_TO]->(ef:Effect)
            RETURN
                e.semantic_id AS element_sid,
                e.text AS element_text,
                f.semantic_id AS function_sid,
                f.text AS function_text,
                m.semantic_id AS mode_sid,
                m.text AS mode_text,
                collect(DISTINCT c) AS cause_nodes,
                collect(DISTINCT ef) AS effect_nodes
        """,
        "mode": """
            MATCH (e:Element)-[:HAS_FUNCTION]->(f:Function)-[:HAS_MODE]->(m:Mode {semantic_id:$sid})
            OPTIONAL MATCH (m)-[:CAUSED_BY]->(c:Cause)
            OPTIONAL MATCH (m)-[:LEADS_TO]->(ef:Effect)
            RETURN
                e.semantic_id AS element_sid,
                e.text AS element_text,
                f.semantic_id AS function_sid,
                f.text AS function_text,
                m.semantic_id AS mode_sid,
                m.text AS mode_text,
                collect(DISTINCT c) AS cause_nodes,
                collect(DISTINCT ef) AS effect_nodes
        """,
        "cause": """
            MATCH (e:Element)-[:HAS_FUNCTION]->(f:Function)-[:HAS_MODE]->(m:Mode)-[:CAUSED_BY]->(c:Cause {semantic_id:$sid})
            OPTIONAL MATCH (m)-[:CAUSED_BY]->(allc:Cause)
            OPTIONAL MATCH (m)-[:LEADS_TO]->(ef:Effect)
            RETURN
                e.semantic_id AS element_sid,
                e.text AS element_text,
                f.semantic_id AS function_sid,
                f.text AS function_text,
                m.semantic_id AS mode_sid,
                m.text AS mode_text,
                collect(DISTINCT allc) AS cause_nodes,
                collect(DISTINCT ef) AS effect_nodes
        """,
        "effect": """
            MATCH (e:Element)-[:HAS_FUNCTION]->(f:Function)-[:HAS_MODE]->(m:Mode)-[:LEADS_TO]->(ef0:Effect {semantic_id:$sid})
            OPTIONAL MATCH (m)-[:CAUSED_BY]->(c:Cause)
            OPTIONAL MATCH (m)-[:LEADS_TO]->(allef:Effect)
            RETURN
                e.semantic_id AS element_sid,
                e.text AS element_text,
                f.semantic_id AS function_sid,
                f.text AS function_text,
                m.semantic_id AS mode_sid,
                m.text AS mode_text,
                collect(DISTINCT c) AS cause_nodes,
                collect(DISTINCT allef) AS effect_nodes
        """
    }

    cypher = field_to_cypher[field]

    for node, score, query_text in retrieved_items:
        sid = node.get("semantic_id")
        node_text = node.get("text", "")
        if not sid:
            continue

        rows = session.run(cypher, sid=sid)

        for r in rows:
            cause_nodes = [x for x in r["cause_nodes"] if x]
            effect_nodes = [x for x in r["effect_nodes"] if x]

            candidate_id = f"{r['element_sid']}|{r['function_sid']}|{r['mode_sid']}"

            candidates.append({
                "candidate_id": candidate_id,

                "element_sid": r["element_sid"],
                "element_text": r["element_text"],

                "function_sid": r["function_sid"],
                "function_text": r["function_text"],

                "mode_sid": r["mode_sid"],
                "mode_text": r["mode_text"],

                "cause_nodes": cause_nodes,
                "effect_nodes": effect_nodes,

                "trigger_field": field,
                "trigger_score": float(score),
                "trigger_query": query_text,
                "trigger_node_sid": sid,
                "trigger_node_text": node_text
            })

    return candidates


# =========================================================
# CANDIDATE SCORE FUSION
# =========================================================
def init_candidate_bucket(expanded_candidate):
    return {
        "candidate_id": expanded_candidate["candidate_id"],

        "element_sid": expanded_candidate["element_sid"],
        "element": expanded_candidate["element_text"],

        "function_sid": expanded_candidate["function_sid"],
        "function": expanded_candidate["function_text"],

        "mode_sid": expanded_candidate["mode_sid"],
        "mode": expanded_candidate["mode_text"],

        "causes": [],
        "effects": [],

        "score": 0.0,

        "element_matches": [],
        "function_matches": [],
        "mode_matches": [],
        "cause_matches": [],
        "effect_matches": []
    }


def attach_full_context(bucket, expanded_candidate):
    bucket["causes"] = sorted(
        list({
            safe_text(n.get("text"))
            for n in expanded_candidate["cause_nodes"]
            if n and safe_text(n.get("text"))
        })
    )
    bucket["effects"] = sorted(
        list({
            safe_text(n.get("text"))
            for n in expanded_candidate["effect_nodes"]
            if n and safe_text(n.get("text"))
        })
    )


def add_match_to_bucket(bucket, expanded_candidate, field_weights):
    field = expanded_candidate["trigger_field"]
    score = float(expanded_candidate["trigger_score"])
    query_text = expanded_candidate["trigger_query"]
    matched_sid = expanded_candidate["trigger_node_sid"]
    matched_text = expanded_candidate["trigger_node_text"]

    weight = field_weights.get(field, 1.0)
    bucket["score"] += weight * score

    match_info = {
        "field": field,
        "query": query_text,
        "matched_sid": matched_sid,
        "matched_text": matched_text,
        "score": round(score, 4)
    }

    if field == "element":
        bucket["element_matches"].append(match_info)
    elif field == "function":
        bucket["function_matches"].append(match_info)
    elif field == "mode":
        bucket["mode_matches"].append(match_info)
    elif field == "cause":
        bucket["cause_matches"].append(match_info)
    elif field == "effect":
        bucket["effect_matches"].append(match_info)


# =========================================================
# MAIN SEARCH
# =========================================================
def semantic_failure_search(
    query,
    top_k=5,
    retrieval_k_each=100,
    retrieval_pool_k=200,
    min_score=0.85,
    field_weights=None
):
    """
    Multi-field semantic search on Vector KG

    query 示例:
    {
        "elements": [...],
        "functions": [...],
        "modes": [...],
        "causes": [...],
        "effects": [...]
    }

    核心逻辑:
    1) 分字段独立向量召回候选 node
    2) 按 KG 边把候选 node 扩展成完整 historical failure candidate
    3) 对 candidate 做加权融合
    4) 返回完整 chain + matched query evidence
    """

    if field_weights is None:
        field_weights = {
            "element": 0.7,
            "function": 0.9,
            "mode": 1.2,
            "cause": 1.0,
            "effect": 1.0
        }

    element_texts = [safe_text(x) for x in query.get("elements", []) if safe_text(x)]
    function_texts = [safe_text(x) for x in query.get("functions", []) if safe_text(x)]
    mode_texts = [safe_text(x) for x in query.get("modes", []) if safe_text(x)]
    cause_texts = [safe_text(x) for x in query.get("causes", []) if safe_text(x)]
    effect_texts = [safe_text(x) for x in query.get("effects", []) if safe_text(x)]

    with driver.session(database=NEO4J_DATABASE) as session:
        # =====================================================
        # 1. RETRIEVE CANDIDATES FOR EACH FIELD
        # =====================================================
        element_results = retrieve_candidates(
            session=session,
            label="Element",
            index_name="element_embedding",
            prefix="Element",
            query_texts=element_texts,
            top_k_each=retrieval_k_each,
            pool_k=retrieval_pool_k,
            min_score=min_score,
            keep_group_or_single_only=True
        )

        function_results = retrieve_candidates(
            session=session,
            label="Function",
            index_name="function_embedding",
            prefix="Function",
            query_texts=function_texts,
            top_k_each=retrieval_k_each,
            pool_k=retrieval_pool_k,
            min_score=min_score,
            keep_group_or_single_only=True
        )

        mode_results = retrieve_candidates(
            session=session,
            label="Mode",
            index_name="mode_embedding",
            prefix="Failure mode",
            query_texts=mode_texts,
            top_k_each=retrieval_k_each,
            pool_k=retrieval_pool_k,
            min_score=min_score,
            keep_group_or_single_only=True
        )

        cause_results = retrieve_candidates(
            session=session,
            label="Cause",
            index_name="cause_embedding",
            prefix="Failure cause",
            query_texts=cause_texts,
            top_k_each=retrieval_k_each,
            pool_k=retrieval_pool_k,
            min_score=min_score,
            keep_group_or_single_only=True
        )

        effect_results = retrieve_candidates(
            session=session,
            label="Effect",
            index_name="effect_embedding",
            prefix="Failure effect",
            query_texts=effect_texts,
            top_k_each=retrieval_k_each,
            pool_k=retrieval_pool_k,
            min_score=min_score,
            keep_group_or_single_only=True
        )

        # =====================================================
        # 2. EXPAND EACH RETRIEVED FIELD TO FULL FAILURE CANDIDATE
        # =====================================================
        expanded_candidates = []
        expanded_candidates.extend(
            expand_candidates_from_retrieved_nodes(session, "element", element_results)
        )
        expanded_candidates.extend(
            expand_candidates_from_retrieved_nodes(session, "function", function_results)
        )
        expanded_candidates.extend(
            expand_candidates_from_retrieved_nodes(session, "mode", mode_results)
        )
        expanded_candidates.extend(
            expand_candidates_from_retrieved_nodes(session, "cause", cause_results)
        )
        expanded_candidates.extend(
            expand_candidates_from_retrieved_nodes(session, "effect", effect_results)
        )

        # =====================================================
        # 3. FUSE BY REAL KG-CONNECTED CANDIDATE
        # =====================================================
        candidate_map = {}

        for exp_cand in expanded_candidates:
            cid = exp_cand["candidate_id"]

            if cid not in candidate_map:
                candidate_map[cid] = init_candidate_bucket(exp_cand)
                attach_full_context(candidate_map[cid], exp_cand)

            add_match_to_bucket(candidate_map[cid], exp_cand, field_weights)

        # 去重 matched query list
        for cid in candidate_map:
            candidate_map[cid]["element_matches"] = deduplicate_match_list(
                candidate_map[cid]["element_matches"]
            )
            candidate_map[cid]["function_matches"] = deduplicate_match_list(
                candidate_map[cid]["function_matches"]
            )
            candidate_map[cid]["mode_matches"] = deduplicate_match_list(
                candidate_map[cid]["mode_matches"]
            )
            candidate_map[cid]["cause_matches"] = deduplicate_match_list(
                candidate_map[cid]["cause_matches"]
            )
            candidate_map[cid]["effect_matches"] = deduplicate_match_list(
                candidate_map[cid]["effect_matches"]
            )

        # =====================================================
        # 4. SORT
        # =====================================================
        results = sorted(
            candidate_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        for item in results:
            item["score"] = round(float(item["score"]), 4)

            # 为了打印稳定性，也可以按 score 排序 matched evidence
            for field_name in [
                "element_matches",
                "function_matches",
                "mode_matches",
                "cause_matches",
                "effect_matches"
            ]:
                item[field_name] = sorted(
                    item[field_name],
                    key=lambda x: x["score"],
                    reverse=True
                )

        return results


############################################
# QUERY BUILDER
############################################

def build_query_from_structure_input(structure_input):
    """
    从你的结构化输入中提取搜索字段
    默认只取第一个 node
    """
    nodes = structure_input.get("nodes", [])
    if not nodes:
        return {"modes": [], "causes": [], "effects": []}

    node = nodes[0]

    modes = [safe_text(x) for x in node.get("modes", []) if safe_text(x)]
    causes = flatten_causes(node.get("causes", {}))
    effects = [safe_text(x) for x in node.get("effects", []) if safe_text(x)]

    return {
        "modes": modes,
        "causes": causes,
        "effects": effects
    }


############################################
# MAIN
############################################

if __name__ == "__main__":
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

    try:
        query = build_query_from_structure_input(structure_input_powertrain)

        print("Query Summary")
        print("-" * 80)
        print(f"Modes   : {len(query['modes'])}")
        print(f"Causes  : {len(query['causes'])}")
        print(f"Effects : {len(query['effects'])}")

        results = semantic_failure_search(query, top_k=20,retrieval_k_each=100, retrieval_pool_k= 300, min_score= 0.82)
        print_results(results)

    finally:
        driver.close()