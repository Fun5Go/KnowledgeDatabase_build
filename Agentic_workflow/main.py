from .LLMs.select_agent import FMEASelectionAgent
from .KG.SA_infer import load_nodes, load_triples, build_relations, build_graph,load_model,Neo4jKGClient, run_structure_mapping_and_inference, structure_input_motorcontrol
import os
import ast
import hashlib
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import torch

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
                rel2id=rel2id
            )
    finally:
        kg_client.close()
    return results

    
if __name__ == "__main__":
    results = MAPandPRED(structure_input_motorcontrol)
    # print(results)