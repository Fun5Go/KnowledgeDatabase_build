import ast
from typing import List, Dict, Any, Tuple
import torch
import pandas as pd
from neo4j import GraphDatabase
import hashlib
from RGCN_inference import predict_tail, load_model, build_graph, build_relations, load_nodes, load_triples
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

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
                #Extra
                # "(Final) Pressure deviates from setpoints",
                # "Overpressure",
                # "No pressure build-up",
                # "No user control",
            ]
        }
    ]
}


# =========================================================
# 1. CONFIG
# =========================================================
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

NODE_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\nodes.tsv"
TRIPLE_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\triples.tsv"
MODEL_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\rgcn_model.pt"

HIDDEN_DIM = 128
OUT_DIM = 128

TOP_K_MAP = 3
MIN_SIM = 0.80
POOL_K = 50

TOP_K_PRED = 3
PRED_SCORE_THRESHOLD = 0.0



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
# 6. NEO4J
# =========================================================

class Neo4jKGClient:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def session(self):
        return self.driver.session(database=self.database)


# =========================================================
# 7. HELPERS
# =========================================================

def deduplicate_mapped_nodes(mapped_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep best score per kg_id.
    """
    best = {}
    for item in mapped_nodes:
        kg_id = item["kg_id"]
        if kg_id not in best or item["score"] > best[kg_id]["score"]:
            best[kg_id] = item
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def extract_structure_queries(structure_input: Dict[str, Any]):
    all_causes = []
    all_modes = []
    all_effects = []

    for node in structure_input.get("nodes", []):
        all_modes.extend(node.get("modes", []))
        all_effects.extend(node.get("effects", []))

        causes_dict = node.get("causes", {})
        for _, items in causes_dict.items():
            all_causes.extend(items)

    return all_causes, all_modes, all_effects


def collect_mapped_ids(mapping_list: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for item in mapping_list:
        for m in item["mapped_nodes"]:
            ids.append(m["kg_id"])
    return list(dict.fromkeys(ids))


def build_kg_text_lookup(mapped: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    lookup = {}
    for group in ["causes", "modes", "effects"]:
        for item in mapped[group]:
            for m in item["mapped_nodes"]:
                lookup[m["kg_id"]] = m["kg_text"]
    return lookup

def build_kg_to_query_texts(mapping_list):
    """
    mapping_list:
        [
            {
                "query_text": "...",
                "mapped_nodes": [
                    {"kg_id": "...", "kg_text": "...", "score": ...},
                    ...
                ]
            },
            ...
        ]

    return:
        {
            "kg_id_1": ["query text a", "query text b"],
            "kg_id_2": ["query text c"]
        }
    """
    lookup = {}

    for item in mapping_list:
        query_text = item["query_text"]
        for m in item["mapped_nodes"]:
            kg_id = m["kg_id"]
            if kg_id not in lookup:
                lookup[kg_id] = []
            if query_text not in lookup[kg_id]:
                lookup[kg_id].append(query_text)

    return lookup

def format_query_texts(query_texts):
    if not query_texts:
        return "N/A"
    return " | ".join(query_texts)


# =========================================================
# 8. KG RETRIEVAL / VECTOR MAPPING
# =========================================================

def map_query_texts_to_nodes(
    session,
    label: str,
    index_name: str,
    prefix: str,
    query_texts: List[str],
    top_k_each: int = 3,
    pool_k: int = 100,
    min_score: float = 0.80,
    keep_group_or_single_only: bool = True
) -> List[Dict[str, Any]]:
    """
    For each query text:
      - embed query
      - search Neo4j vector index
      - keep top-k matches with score >= min_score
      - preserve query text

    Return format:
    [
      {
        "query_text": "...",
        "mapped_nodes": [
          {"kg_id": "...", "kg_text": "...", "score": 0.95},
          ...
        ]
      }
    ]
    """

    results = []

    for text in query_texts:
        emb = embed(f"{prefix}: {text}")
        if emb is None:
            results.append({
                "query_text": text,
                "mapped_nodes": []
            })
            continue

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().tolist()

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
        RETURN
            node.semantic_id AS kg_id,
            coalesce(node.canonical_text, node.text, node.name, node.semantic_id) AS kg_text,
            refined_score
        ORDER BY refined_score DESC
        LIMIT $k
        """

        query_result = session.run(cypher, embedding=emb, k=top_k_each)

        mapped_nodes = []
        for r in query_result:
            score = float(r["refined_score"])
            if score >= min_score:
                mapped_nodes.append({
                    "kg_id": r["kg_id"],
                    "kg_text": r["kg_text"],
                    "score": score
                })

        mapped_nodes = deduplicate_mapped_nodes(mapped_nodes)

        results.append({
            "query_text": text,
            "mapped_nodes": mapped_nodes
        })

    return results


def map_structure_input_to_kg(session, structure_input: Dict[str, Any]):
    causes, modes, effects = extract_structure_queries(structure_input)

    cause_maps = map_query_texts_to_nodes(
        session=session,
        label="Cause",
        index_name="cause_embedding",
        prefix="Failure cause",
        query_texts=causes,
        top_k_each=TOP_K_MAP,
        pool_k=POOL_K,
        min_score=MIN_SIM
    )

    mode_maps = map_query_texts_to_nodes(
        session=session,
        label="Mode",
        index_name="mode_embedding",
        prefix="Failure mode",
        query_texts=modes,
        top_k_each=TOP_K_MAP,
        pool_k=POOL_K,
        min_score=MIN_SIM
    )

    effect_maps = map_query_texts_to_nodes(
        session=session,
        label="Effect",
        index_name="effect_embedding",
        prefix="Failure effect",
        query_texts=effects,
        top_k_each=TOP_K_MAP,
        pool_k=POOL_K,
        min_score=MIN_SIM
    )

    return {
        "causes": cause_maps,
        "modes": mode_maps,
        "effects": effect_maps
    }


# =========================================================
# 9. PREDICTION
# =========================================================


@torch.no_grad()
def infer_cause_to_mode(
    model, x, edge_index, edge_type,
    node2id, id2node, id2type, rel2id,
    cause_node_id: str,
    candidate_mode_node_ids: List[str] = None,
    top_k: int = 10
):
    if cause_node_id not in node2id:
        return []

    head_id = node2id[cause_node_id]

    if "CAUSES" not in rel2id:
        raise KeyError("Relation 'CAUSES' not found in rel2id.")

    relation_id = rel2id["CAUSES"]

    if candidate_mode_node_ids is None:
        candidates = [nid for nid, t in id2type.items() if t == "Mode"]
    else:
        candidates = [
            node2id[mid]
            for mid in candidate_mode_node_ids
            if mid in node2id and id2type[node2id[mid]] == "Mode"
        ]

    if not candidates:
        return []

    results = predict_tail(
        model, x, edge_index, edge_type,
        head_id, relation_id, candidates, top_k
    )
    results = [(nid, score) for nid, score in results if score >= PRED_SCORE_THRESHOLD]

    return [(id2node[nid], score) for nid, score in results]


@torch.no_grad()
def infer_mode_to_effect(
    model, x, edge_index, edge_type,
    node2id, id2node, id2type, rel2id,
    mode_node_id: str,
    candidate_effect_node_ids: List[str] = None,
    top_k: int = 10
):
    if mode_node_id not in node2id:
        return []

    head_id = node2id[mode_node_id]

    if "LEADS_TO" not in rel2id:
        raise KeyError("Relation 'LEADS_TO' not found in rel2id.")

    relation_id = rel2id["LEADS_TO"]

    if candidate_effect_node_ids is None:
        candidates = [nid for nid, t in id2type.items() if t == "Effect"]
    else:
        candidates = [
            node2id[eid]
            for eid in candidate_effect_node_ids
            if eid in node2id and id2type[node2id[eid]] == "Effect"
        ]

    if not candidates:
        return []

    results = predict_tail(
        model, x, edge_index, edge_type,
        head_id, relation_id, candidates, top_k
    )
    results = [(nid, score) for nid, score in results if score >= PRED_SCORE_THRESHOLD]
    return [(id2node[nid], score) for nid, score in results]


# =========================================================
# 10. PRINTING
# =========================================================

def print_mapping_results(title: str, mapping_list: List[Dict[str, Any]]):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")

    for item in mapping_list:
        print(f"\nQuery Text: {item['query_text']}")
        if not item["mapped_nodes"]:
            print("  No mapped KG nodes.")
            continue

        for rank, m in enumerate(item["mapped_nodes"], 1):
            print(f"  [{rank}]")
            print(f"    KG ID   : {m['kg_id']}")
            print(f"    KG Text : {m['kg_text']}")
            print(f"    Sim     : {m['score']:.4f}")


def print_cause_to_mode_predictions(
    query_text,
    mapped_cause,
    results,
    kg_text_lookup,
    kg_mode_to_query_texts
):
    print(f"\n{'-' * 80}")
    print("Cause -> Mode Prediction")
    print(f"{'-' * 80}")
    print(f"Query Cause Text : {query_text}")
    print(f"Mapped Cause ID  : {mapped_cause['kg_id']}")
    print(f"Mapped Cause Text: {mapped_cause['kg_text']}")
    print(f"Cause Similarity : {mapped_cause['score']:.4f}")

    if not results:
        print("No predicted modes.")
        return

    for rank, (mode_id, pred_score) in enumerate(results, 1):
        mode_text = kg_text_lookup.get(mode_id, mode_id)
        matched_query_modes = kg_mode_to_query_texts.get(mode_id, [])

        print(f"\n[{rank}]")
        print(f"  Pred Mode ID         : {mode_id}")
        print(f"  Pred Mode Text       : {mode_text}")
        print(f"  Matched Query Mode   : {format_query_texts(matched_query_modes)}")
        print(f"  Pred Score           : {pred_score:.4f}")


def print_mode_to_effect_predictions(
    query_text,
    mapped_mode,
    results,
    kg_text_lookup,
    kg_effect_to_query_texts
):
    print(f"\n{'-' * 80}")
    print("Mode -> Effect Prediction")
    print(f"{'-' * 80}")
    print(f"Query Mode Text  : {query_text}")
    print(f"Mapped Mode ID   : {mapped_mode['kg_id']}")
    print(f"Mapped Mode Text : {mapped_mode['kg_text']}")
    print(f"Mode Similarity  : {mapped_mode['score']:.4f}")

    if not results:
        print("No predicted effects.")
        return

    for rank, (effect_id, pred_score) in enumerate(results, 1):
        effect_text = kg_text_lookup.get(effect_id, effect_id)
        matched_query_effects = kg_effect_to_query_texts.get(effect_id, [])

        print(f"\n[{rank}]")
        print(f"  Pred Effect ID       : {effect_id}")
        print(f"  Pred Effect Text     : {effect_text}")
        print(f"  Matched Query Effect : {format_query_texts(matched_query_effects)}")
        print(f"  Pred Score           : {pred_score:.4f}")


# =========================================================
# 11. PIPELINE
# =========================================================

def run_structure_mapping_and_inference(
    session,
    structure_input,
    model, x, edge_index, edge_type,
    node2id, id2node, id2type, rel2id
):
    # Step 1: map input texts to KG nodes
    mapped = map_structure_input_to_kg(session, structure_input)


    print_mapping_results("CAUSE MAPPING", mapped["causes"])
    print_mapping_results("MODE MAPPING", mapped["modes"])
    print_mapping_results("EFFECT MAPPING", mapped["effects"])

    # Build lookup
    kg_text_lookup = build_kg_text_lookup(mapped)
    kg_mode_to_query_texts = build_kg_to_query_texts(mapped["modes"])
    kg_effect_to_query_texts = build_kg_to_query_texts(mapped["effects"])

    # Restricted candidate sets
    mapped_mode_ids = collect_mapped_ids(mapped["modes"])
    mapped_effect_ids = collect_mapped_ids(mapped["effects"])

    print(f"\nMapped mode candidate count   : {len(mapped_mode_ids)}")
    print(f"Mapped effect candidate count : {len(mapped_effect_ids)}")

    # # Step 2A: cause -> mode
    # print(f"\n{'#' * 80}")
    # print("RUNNING CAUSE -> MODE INFERENCE")
    # print(f"{'#' * 80}")

    # for cause_item in mapped["causes"]:
    #     query_text = cause_item["query_text"]

    #     if not cause_item["mapped_nodes"]:
    #         print(f"\nSkip cause query (no mapping): {query_text}")
    #         continue

    #     for mapped_cause in cause_item["mapped_nodes"]:
    #         cause_id = mapped_cause["kg_id"]

    #         results = infer_cause_to_mode(
    #             model, x, edge_index, edge_type,
    #             node2id, id2node, id2type, rel2id,
    #             cause_node_id=cause_id,
    #             candidate_mode_node_ids=mapped_mode_ids,
    #             top_k=TOP_K_PRED
    #         )

    #         print_cause_to_mode_predictions(
    #             query_text=query_text,
    #             mapped_cause=mapped_cause,
    #             results=results,
    #             kg_text_lookup=kg_text_lookup,
    #             kg_mode_to_query_texts=kg_mode_to_query_texts
    #         )

    # # Step 2B: mode -> effect
    # print(f"\n{'#' * 80}")
    # print("RUNNING MODE -> EFFECT INFERENCE")
    # print(f"{'#' * 80}")

    # for mode_item in mapped["modes"]:
    #     query_text = mode_item["query_text"]

    #     if not mode_item["mapped_nodes"]:
    #         print(f"\nSkip mode query (no mapping): {query_text}")
    #         continue

    #     for mapped_mode in mode_item["mapped_nodes"]:
    #         mode_id = mapped_mode["kg_id"]

    #         results = infer_mode_to_effect(
    #             model, x, edge_index, edge_type,
    #             node2id, id2node, id2type, rel2id,
    #             mode_node_id=mode_id,
    #             candidate_effect_node_ids=mapped_effect_ids,
    #             top_k=TOP_K_PRED
    #         )

    #         print_mode_to_effect_predictions(
    #             query_text=query_text,
    #             mapped_mode=mapped_mode,
    #             results=results,
    #             kg_text_lookup=kg_text_lookup,
    #             kg_effect_to_query_texts=kg_effect_to_query_texts
    #         )


# =========================================================
# 12. MAIN
# =========================================================

def main():
    # load graph data
    node2id, id2node, id2type, id2text, x = load_nodes(NODE_FILE)
    triples_raw, rel_list_base = load_triples(TRIPLE_FILE, node2id)
    rel2id, id2rel = build_relations(rel_list_base)
    edge_index, edge_type = build_graph(triples_raw, rel2id)

    # load model
    model = load_model(
        MODEL_PATH,
        in_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_relations=len(rel2id)
    )

    # neo4j
    kg_client = Neo4jKGClient(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )

    try:
        with kg_client.session() as session:
            run_structure_mapping_and_inference(
                session=session,
                structure_input=structure_input_motorcontrol,
                model=model,
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                node2id=node2id,
                id2node=id2node,
                id2type=id2type,
                rel2id=rel2id
            )
    finally:
        kg_client.close()


if __name__ == "__main__":
    main()