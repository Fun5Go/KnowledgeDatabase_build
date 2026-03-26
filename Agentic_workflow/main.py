from .LLMs.select_agent import FMEASelectionAgent
from .KG.SA_infer import load_nodes, load_triples, build_relations, build_graph,load_model,Neo4jKGClient, run_structure_mapping_and_inference, structure_input_motorcontrol, structure_input_powertrain
import os
import ast
import hashlib
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import torch
import json
from langsmith import traceable

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
MODEL_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\rgcn_best_model.pt"

HIDDEN_DIM = 256
EMD_DIM = 256

TOP_K_MAP = 5
MIN_SIM = 0.80
POOL_K = 100

TOP_K_PRED = 2
PRED_SCORE_THRESHOLD = 0.0

WEIGHTING_METHOD = "square"   # ["linear", "square", "uniform"]
APPLY_SIGMOID_TO_PAIR_SCORE = True
SHOW_TOP_PAIR_DETAILS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Helpers 
# ================================
def build_minimal_result_for_llm(results, structure_input):
    """
    Rebuild final inference result for LLM prompt with structure context.
    Keep:
      - element_text
      - function_text (for mode)
      - discipline (for cause, optional)
      - query cause/mode text
      - predicted mode/effect query text
      - rank
    """

    simplified = {
        "cause_to_mode": [],
        "mode_to_effect": []
    }

    # ----------------------------
    # Build structure context lookup
    # ----------------------------
    element_text = ""
    mode_to_function = {}
    cause_to_discipline = {}

    nodes = structure_input.get("nodes", [])
    if nodes:
        node = nodes[0]   # 当前结构里只有一个 E1
        element_text = node.get("failure_element", "")

        # modes: {function: [mode1, mode2, ...]}
        for function_text, modes in node.get("modes", {}).items():
            for mode_text in modes:
                mode_to_function[mode_text] = function_text

        # causes: {discipline: [cause1, cause2, ...]}
        for discipline, causes in node.get("causes", {}).items():
            for cause_text in causes:
                cause_to_discipline[cause_text] = discipline

    # ----------------------------
    # Simplify cause -> mode
    # ----------------------------
    for cause_item in results.get("cause_to_mode", []):
        query_cause_text = cause_item.get("query_cause_text", "")

        simplified_cause_item = {
            "element_text": element_text,
            "cause_discipline": cause_to_discipline.get(query_cause_text, ""),
            "query_cause_text": query_cause_text,
            "predicted_modes": []
        }

        for mode_item in cause_item.get("predicted_modes", []):
            mode_query_text = mode_item.get("mode_query_text", "")
            simplified_cause_item["predicted_modes"].append({
                "rank": mode_item.get("rank"),
                "mode_query_text": mode_query_text,
                "function_text": mode_to_function.get(mode_query_text, "")
            })

        simplified["cause_to_mode"].append(simplified_cause_item)

    # ----------------------------
    # Simplify mode -> effect
    # ----------------------------
    for mode_item in results.get("mode_to_effect", []):
        query_mode_text = mode_item.get("query_mode_text", "")

        simplified_mode_item = {
            "element_text": element_text,
            "function_text": mode_to_function.get(query_mode_text, ""),
            "query_mode_text": query_mode_text,
            "predicted_effects": []
        }

        for effect_item in mode_item.get("predicted_effects", []):
            simplified_mode_item["predicted_effects"].append({
                "rank": effect_item.get("rank"),
                "effect_query_text": effect_item.get("effect_query_text", "")
            })

        simplified["mode_to_effect"].append(simplified_mode_item)

    return simplified

def pretty_print_selection_result(result: dict):
    print("\n" + "=" * 80)
    print("FMEA SELECTION RESULT")
    print("=" * 80)

    print("\n[1] Cause -> Mode selection")
    for item in result.get("cause_to_mode_selection", []):
        print(f"\nCause: {item.get('cause', '')}")
        for mode in item.get("selected_modes", []):
            print(f"  - Mode: {mode.get('mode_text', '')}")
            print(f"    Selected  : {mode.get('selected', True)}")
            print(f"    Reason    : {mode.get('reason', '')}")
            print(f"    Confidence: {mode.get('confidence', '')}")
        print(f"  Summary: {item.get('global_reasoning', '')}")

    print("\n[2] Mode -> Effect selection")
    for mode_text, item in result.get("mode_to_effect_selection", {}).items():
        print(f"\nMode: {mode_text}")
        for eff in item.get("selected_effects", []):
            print(f"  - Effect: {eff.get('effect_text', '')}")
            print(f"    Selected  : {eff.get('selected', True)}")
            print(f"    Reason    : {eff.get('reason', '')}")
            print(f"    Confidence: {eff.get('confidence', '')}")
        print(f"  Summary: {item.get('global_reasoning', '')}")

    print("\n[3] Final chains")
    for i, chain in enumerate(result.get("final_chains", []), 1):
        print(f"  [{i}] {chain['chain_text']}")

    print("=" * 80)


# ================================
# Functions 
# ================================
def MAPandPRED(structure_input):
    node2id, id2node, id2type, id2text, x = load_nodes(NODE_FILE)
    triples_raw, rel_list_base = load_triples(TRIPLE_FILE, node2id)
    rel2id, id2rel = build_relations(rel_list_base)
    edge_index, edge_type = build_graph(triples_raw, rel2id)

    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_type = edge_type.to(DEVICE)

    model = load_model(
        path=MODEL_PATH,
        in_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
        emb_dim=EMD_DIM,
        num_relations=len(rel2id),
        dropout=0.1,
        num_bases=4,
        device=DEVICE
    )

    kg_client = Neo4jKGClient(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )

    try:
        with kg_client.session() as session:
            results = run_structure_mapping_and_inference(
                is_print=False,
                session=session,
                structure_input=structure_input,
                model=model,
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                node2id=node2id,
                id2node=id2node,
                id2type=id2type,
                id2text=id2text,
                rel2id=rel2id,
            )
    finally:
        kg_client.close()
    return results

@traceable(
    run_type="chain",
    name="fmea_experiment-integration",
    tags=["fmea", "exp-powertrain", "prompt_v1"]
)
def run_fmea_experiment(agent, simplified_result):
    return agent.select_all(
        simplified_result=simplified_result,
        top_n_modes=4,
        max_effects_per_mode=4,
    )

    
if __name__ == "__main__":

    results = MAPandPRED(structure_input_powertrain)
    simplified_results = build_minimal_result_for_llm(results,structure_input_powertrain)
    # print(json.dumps(simplified_results, indent=2, ensure_ascii=False))


    agent = FMEASelectionAgent(
    backend=os.getenv("LLM_BACKEND", "openai"),
    model=os.getenv("LLM_MODEL", "azure/gpt-4.1"),
    )

    result = run_fmea_experiment(agent,simplified_results)
    pretty_print_selection_result(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))