import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from chromadb.utils import embedding_functions


############################################
# ENV
############################################

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    raise ValueError("NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD 未正确配置到 .env 文件中")


############################################
# EMBEDDING
############################################

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_embedding_cache = {}


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def embed(text):
    text = safe_text(text)
    if not text:
        return None

    if text in _embedding_cache:
        return _embedding_cache[text]

    vec = embedder([text])[0]
    _embedding_cache[text] = vec
    return vec


############################################
# DRIVER
############################################

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)


############################################
# HELPERS
############################################

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


def deduplicate_by_semantic_id(node_score_list):
    """
    node_score_list: [(node, score), ...]
    对同一个 semantic_id 保留最高分
    """
    best = {}
    for node, score in node_score_list:
        sid = node.get("semantic_id")
        if not sid:
            continue
        if sid not in best or score > best[sid][1]:
            best[sid] = (node, score)
    return list(best.values())


def print_results(results, title="Search Results"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for i, item in enumerate(results, 1):
        print(f"\n[{i}] Score   : {item.get('score', 0)}")
        print(f"    Element : {item.get('element', '')}")
        print(f"    Function: {item.get('function', '')}")
        print(f"    Mode    : {item.get('mode', '')}")

        print("    Causes  :")
        causes = item.get("causes", [])
        if causes:
            for c in causes:
                print(f"      - {c}")
        else:
            print("      - None")

        print("    Effects :")
        effects = item.get("effects", [])
        if effects:
            for ef in effects:
                print(f"      - {ef}")
        else:
            print("      - None")


############################################
# CORE SEARCH
############################################

def semantic_failure_search(query, top_k=5):
    """
    Multi-field semantic search on Vector KG

    query 示例:
    {
        "modes": [...],
        "causes": [...],    # 可以是 flatten 后的 list
        "effects": [...]
    }
    """

    mode_texts = [safe_text(m) for m in query.get("modes", []) if safe_text(m)]
    cause_texts = [safe_text(c) for c in query.get("causes", []) if safe_text(c)]
    effect_texts = [safe_text(e) for e in query.get("effects", []) if safe_text(e)]

    mode_embs = [embed(f"Failure mode: {m}") for m in mode_texts]
    cause_embs = [embed(f"Failure cause: {c}") for c in cause_texts]
    effect_embs = [embed(f"Failure effect: {e}") for e in effect_texts]

    mode_embs = [x for x in mode_embs if x is not None]
    cause_embs = [x for x in cause_embs if x is not None]
    effect_embs = [x for x in effect_embs if x is not None]

    with driver.session() as session:

        # ===============================
        # 1. MODE SEARCH
        # ===============================
        mode_results = []
        for emb in mode_embs:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    'mode_embedding',
                    $k,
                    $embedding
                )
                YIELD node, score
                RETURN node, score
            """, k=top_k, embedding=emb)

            mode_results.extend([(r["node"], r["score"]) for r in result])

        mode_results = deduplicate_by_semantic_id(mode_results)

        # ===============================
        # 2. CAUSE SEARCH -> MODE
        # ===============================
        cause_modes = []
        for emb in cause_embs:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    'cause_embedding',
                    $k,
                    $embedding
                )
                YIELD node, score

                MATCH (m:Mode)-[:CAUSED_BY]->(node)
                RETURN m AS mode, score
            """, k=top_k, embedding=emb)

            cause_modes.extend([(r["mode"], r["score"]) for r in result])

        cause_modes = deduplicate_by_semantic_id(cause_modes)

        # ===============================
        # 3. EFFECT SEARCH -> MODE
        # ===============================
        effect_modes = []
        for emb in effect_embs:
            result = session.run("""
                CALL db.index.vector.queryNodes(
                    'effect_embedding',
                    $k,
                    $embedding
                )
                YIELD node, score

                MATCH (m:Mode)-[:LEADS_TO]->(node)
                RETURN m AS mode, score
            """, k=top_k, embedding=emb)

            effect_modes.extend([(r["mode"], r["score"]) for r in result])

        effect_modes = deduplicate_by_semantic_id(effect_modes)

        # ===============================
        # 4. SCORE FUSION
        # ===============================
        mode_score_map = {}

        def accumulate(modes, weight):
            for m, score in modes:
                mid = m.get("semantic_id")
                if not mid:
                    continue

                if mid not in mode_score_map:
                    mode_score_map[mid] = {"node": m, "score": 0.0}

                mode_score_map[mid]["score"] += weight * float(score)

        accumulate(mode_results, 1.0)
        accumulate(cause_modes, 0.8)
        accumulate(effect_modes, 0.8)

        # ===============================
        # 5. TOP MODES
        # ===============================
        top_modes = sorted(
            mode_score_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        # ===============================
        # 6. EXPAND TO FAILURE CHAIN
        # ===============================
        results = []

        for item in top_modes:
            mode_node = item["node"]
            score = item["score"]

            chain_result = session.run("""
                MATCH (f:Function)-[:HAS_MODE]->(m:Mode {semantic_id:$id})
                MATCH (e:Element)-[:HAS_FUNCTION]->(f)

                OPTIONAL MATCH (m)-[:CAUSED_BY]->(c:Cause)
                OPTIONAL MATCH (m)-[:LEADS_TO]->(ef:Effect)

                RETURN
                    e.text AS element,
                    f.text AS function,
                    m.text AS mode,
                    collect(DISTINCT c.text) AS causes,
                    collect(DISTINCT ef.text) AS effects
            """, id=mode_node["semantic_id"]).single()

            if chain_result:
                results.append({
                    "element": chain_result["element"],
                    "function": chain_result["function"],
                    "mode": chain_result["mode"],
                    "causes": [x for x in chain_result["causes"] if x],
                    "effects": [x for x in chain_result["effects"] if x],
                    "score": round(score, 4)
                })

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

        results = semantic_failure_search(query, top_k=20)
        print_results(results, title="Semantic Failure Search Results")

    finally:
        driver.close()