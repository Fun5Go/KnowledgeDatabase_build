import os
import ast
import hashlib
from typing import List, Dict, Any, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from neo4j import GraphDatabase
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from torch_geometric.nn import RGCNConv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                "Short-circuit",
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
MODEL_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\rgcn_best_model.pt"

HIDDEN_DIM = 256
EMD_DIM = 256

TOP_K_MAP = 3
MIN_SIM = 0.85
POOL_K = 100

TOP_K_PRED = 3
PRED_SCORE_THRESHOLD = 0.0

WEIGHTING_METHOD = "linear"   # ["linear", "square", "uniform"]
APPLY_SIGMOID_TO_PAIR_SCORE = True
SHOW_TOP_PAIR_DETAILS = 5

# =========================================================
# 2. HELPERS
# =========================================================

def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def flatten_causes(causes):
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


def deduplicate_mapped_nodes(mapped_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        all_modes.extend([safe_text(x) for x in node.get("modes", []) if safe_text(x)])
        all_effects.extend([safe_text(x) for x in node.get("effects", []) if safe_text(x)])
        all_causes.extend(flatten_causes(node.get("causes", {})))

    return all_causes, all_modes, all_effects


def build_kg_text_lookup(mapped: Dict[str, List[Dict[str, Any]]], id2text: Dict[int, str], node2id: Dict[str, int]) -> Dict[str, str]:
    lookup = {}

    for group in ["causes", "modes", "effects"]:
        for item in mapped[group]:
            for m in item["mapped_nodes"]:
                lookup[m["kg_id"]] = m["kg_text"]

    for kg_id, nid in node2id.items():
        if kg_id not in lookup:
            lookup[kg_id] = id2text.get(nid, kg_id)

    return lookup


# =========================================================
# 3. EMBEDDING
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
# 4. LOAD DATA
# =========================================================

def load_nodes(node_file: str):
    df = pd.read_csv(node_file, sep="\t")

    node_ids = df["node_id"].tolist()
    node_types = df["node_type"].tolist()

    if "text" in df.columns:
        node_texts = df["text"].fillna("").tolist()
    elif "canonical_text" in df.columns:
        node_texts = df["canonical_text"].fillna("").tolist()
    elif "name" in df.columns:
        node_texts = df["name"].fillna("").tolist()
    else:
        node_texts = node_ids[:]

    embeddings = []
    for emb_str in df["embedding"]:
        emb = torch.tensor(ast.literal_eval(emb_str), dtype=torch.float)
        embeddings.append(emb)

    x = torch.stack(embeddings)

    node2id = {nid: i for i, nid in enumerate(node_ids)}
    id2node = {i: nid for nid, i in node2id.items()}
    id2type = {node2id[nid]: t for nid, t in zip(node_ids, node_types)}
    id2text = {node2id[nid]: txt for nid, txt in zip(node_ids, node_texts)}

    return node2id, id2node, id2type, id2text, x


def load_triples(triple_file: str, node2id: Dict[str, int]):
    df = pd.read_csv(triple_file, sep="\t")

    triples = []
    relations = set()

    for _, row in df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]

        if h not in node2id or t not in node2id:
            continue

        triples.append((node2id[h], r, node2id[t]))
        relations.add(r)

    return triples, sorted(list(relations))


# =========================================================
# 5. GRAPH
# =========================================================

def build_relations(rel_list_base: List[str]):
    rel_list_full = rel_list_base + [r + "_REV" for r in rel_list_base]
    rel2id = {r: i for i, r in enumerate(rel_list_full)}
    id2rel = {v: k for k, v in rel2id.items()}
    return rel2id, id2rel


def build_graph(triples_raw: List[Tuple[int, str, int]], rel2id: Dict[str, int]):
    edge_index = []
    edge_type = []

    for h, r, t in triples_raw:
        edge_index.append([h, t])
        edge_type.append(rel2id[r])

        rev_r = r + "_REV"
        if rev_r not in rel2id:
            raise KeyError(f"Reverse relation missing in rel2id: {rev_r}")

        edge_index.append([t, h])
        edge_type.append(rel2id[rev_r])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    return edge_index, edge_type


# =========================================================
# 6. MODEL
# =========================================================

class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, dropout=0.3, num_bases=4):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class ComplExDecoder(nn.Module):
    """
    z shape: [num_nodes, 2 * emb_dim]
    first half = real part
    second half = imaginary part
    """
    def __init__(self, num_relations, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.rel = nn.Parameter(torch.empty(num_relations, 2 * emb_dim))
        nn.init.xavier_uniform_(self.rel)

    def forward(self, z, triples):
        s = z[triples[:, 0]]
        r = self.rel[triples[:, 1]]
        o = z[triples[:, 2]]

        s_re, s_im = torch.chunk(s, 2, dim=-1)
        r_re, r_im = torch.chunk(r, 2, dim=-1)
        o_re, o_im = torch.chunk(o, 2, dim=-1)

        score = (
            s_re * r_re * o_re
            + s_im * r_re * o_im
            + s_re * r_im * o_im
            - s_im * r_im * o_re
        ).sum(dim=-1)

        return score


class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim, num_relations, dropout=0.3, num_bases=4):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)

        self.rgcn = RGCN(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=2 * emb_dim,
            num_relations=num_relations,
            dropout=dropout,
            num_bases=num_bases
        )

        self.decoder = ComplExDecoder(num_relations, emb_dim)

    def encode(self, x, edge_index, edge_type):
        x = self.input_proj(x)
        x = F.relu(x)
        z = self.rgcn(x, edge_index, edge_type)
        return z

    def score(self, z, triples):
        return self.decoder(z, triples)

    def forward(self, x, edge_index, edge_type, triples):
        z = self.encode(x, edge_index, edge_type)
        return self.score(z, triples)


def load_model(path, in_dim, hidden_dim, emb_dim, num_relations, dropout=0.3, num_bases=4, device="cpu"):
    model = Model(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        num_relations=num_relations,
        dropout=dropout,
        num_bases=num_bases
    ).to(device)

    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Loaded model from: {path}")
    return model


# =========================================================
# 7. NEO4J
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
# 9. WEIGHTED QUERY REPRESENTATION + PREDICTION
# =========================================================

@torch.no_grad()
def encode_graph_once(model, x, edge_index, edge_type):
    model.eval()
    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_type = edge_type.to(DEVICE)
    return model.encode(x, edge_index, edge_type)


@torch.no_grad()
def build_weighted_node_list(
    mapped_nodes: List[Dict[str, Any]],
    node2id: Dict[str, int],
    weighting: str = "linear",
    min_sim: float = 0.0
):
    valid_items = []

    for m in mapped_nodes:
        kg_id = m["kg_id"]
        sim = float(m["score"])
        if kg_id in node2id and sim >= min_sim:
            valid_items.append((kg_id, sim))

    if not valid_items:
        return []

    sims = torch.tensor([sim for _, sim in valid_items], dtype=torch.float)

    if weighting == "linear":
        raw_weights = sims.clone()
    elif weighting == "square":
        raw_weights = sims ** 2
    elif weighting == "uniform":
        raw_weights = torch.ones_like(sims)
    else:
        raise ValueError(f"Unsupported weighting method: {weighting}")

    weight_sum = raw_weights.sum().item()
    if weight_sum <= 0:
        raw_weights = torch.ones_like(raw_weights)
        weight_sum = raw_weights.sum().item()

    norm_weights = raw_weights / weight_sum

    weighted_nodes = []
    for i, (kg_id, sim) in enumerate(valid_items):
        weighted_nodes.append({
            "kg_id": kg_id,
            "node_id": node2id[kg_id],
            "raw_sim": float(sim),
            "weight": float(norm_weights[i].item())
        })

    return weighted_nodes


@torch.no_grad()
def complex_score_from_embeddings(head_emb, rel_emb, tail_emb):
    """
    Standard ComplEx score for one triple.
    head_emb, rel_emb, tail_emb: shape [2 * emb_dim]
    """
    h_re, h_im = torch.chunk(head_emb, 2, dim=-1)
    r_re, r_im = torch.chunk(rel_emb, 2, dim=-1)
    t_re, t_im = torch.chunk(tail_emb, 2, dim=-1)

    score = (
        h_re * r_re * t_re
        + h_im * r_re * t_im
        + h_re * r_im * t_im
        - h_im * r_im * t_re
    ).sum()

    return score


@torch.no_grad()
def score_weighted_node_sets(
    model,
    z: torch.Tensor,
    relation_id: int,
    head_nodes: List[Dict[str, Any]],
    tail_nodes: List[Dict[str, Any]],
    apply_sigmoid: bool = True
):
    """
    Final score = sum_i sum_j w_hi * w_tj * pair_score(h_i, r, t_j)
    """
    if not head_nodes or not tail_nodes:
        return None, []

    rel_emb = model.decoder.rel[relation_id]

    pair_details = []
    final_score = 0.0

    for h in head_nodes:
        h_id = h["node_id"]
        h_w = h["weight"]
        h_emb = z[h_id]

        for t in tail_nodes:
            t_id = t["node_id"]
            t_w = t["weight"]
            t_emb = z[t_id]

            raw_score = complex_score_from_embeddings(h_emb, rel_emb, t_emb)
            pair_score = torch.sigmoid(raw_score).item() if apply_sigmoid else raw_score.item()

            contrib = h_w * t_w * pair_score
            final_score += contrib

            pair_details.append({
                "head_kg_id": h["kg_id"],
                "tail_kg_id": t["kg_id"],
                "head_weight": h_w,
                "tail_weight": t_w,
                "head_sim": h["raw_sim"],
                "tail_sim": t["raw_sim"],
                "pair_score": pair_score,
                "contribution": contrib
            })

    pair_details = sorted(pair_details, key=lambda x: x["contribution"], reverse=True)
    return final_score, pair_details


@torch.no_grad()
def infer_query_cause_to_query_modes_weighted(
    model, z, node2id, rel2id,
    mapped_cause_nodes,
    mapped_mode_items,
    top_k=5,
    weighting="linear"
):
    if "CAUSES" not in rel2id:
        raise KeyError("Relation 'CAUSES' not found in rel2id.")

    cause_weighted_nodes = build_weighted_node_list(
        mapped_nodes=mapped_cause_nodes,
        node2id=node2id,
        weighting=weighting
    )

    if not cause_weighted_nodes:
        return [], cause_weighted_nodes

    relation_id = rel2id["CAUSES"]
    results = []

    for mode_item in mapped_mode_items:
        mode_query_text = mode_item["query_text"]

        mode_weighted_nodes = build_weighted_node_list(
            mapped_nodes=mode_item["mapped_nodes"],
            node2id=node2id,
            weighting=weighting
        )

        if not mode_weighted_nodes:
            continue

        score, pair_details = score_weighted_node_sets(
            model=model,
            z=z,
            relation_id=relation_id,
            head_nodes=cause_weighted_nodes,
            tail_nodes=mode_weighted_nodes,
            apply_sigmoid=APPLY_SIGMOID_TO_PAIR_SCORE
        )

        if score is None:
            continue

        if score >= PRED_SCORE_THRESHOLD:
            results.append({
                "mode_query_text": mode_query_text,
                "score": score,
                "mode_used_nodes": mode_weighted_nodes,
                "pair_details": pair_details
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results, cause_weighted_nodes


@torch.no_grad()
def infer_query_mode_to_query_effects_weighted(
    model, z, node2id, rel2id,
    mapped_mode_nodes,
    mapped_effect_items,
    top_k=5,
    weighting="linear"
):
    if "LEADS_TO" not in rel2id:
        raise KeyError("Relation 'LEADS_TO' not found in rel2id.")

    mode_weighted_nodes = build_weighted_node_list(
        mapped_nodes=mapped_mode_nodes,
        node2id=node2id,
        weighting=weighting
    )

    if not mode_weighted_nodes:
        return [], mode_weighted_nodes

    relation_id = rel2id["LEADS_TO"]
    results = []

    for effect_item in mapped_effect_items:
        effect_query_text = effect_item["query_text"]

        effect_weighted_nodes = build_weighted_node_list(
            mapped_nodes=effect_item["mapped_nodes"],
            node2id=node2id,
            weighting=weighting
        )

        if not effect_weighted_nodes:
            continue

        score, pair_details = score_weighted_node_sets(
            model=model,
            z=z,
            relation_id=relation_id,
            head_nodes=mode_weighted_nodes,
            tail_nodes=effect_weighted_nodes,
            apply_sigmoid=APPLY_SIGMOID_TO_PAIR_SCORE
        )

        if score is None:
            continue

        if score >= PRED_SCORE_THRESHOLD:
            results.append({
                "effect_query_text": effect_query_text,
                "score": score,
                "effect_used_nodes": effect_weighted_nodes,
                "pair_details": pair_details
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results, mode_weighted_nodes


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


def print_weighted_cause_to_mode_predictions(
    query_text,
    used_heads,
    results,
    kg_text_lookup
):
    print(f"\n{'-' * 80}")
    print("Weighted Query Cause -> Query Mode Prediction")
    print(f"{'-' * 80}")
    print(f"Query Cause Text : {query_text}")

    print("\nWeighted mapped cause nodes:")
    if not used_heads:
        print("  None")
    # else:
    #     for rank, item in enumerate(used_heads, 1):
    #         kg_id = item["kg_id"]
    #         print(f"  [{rank}] {kg_id}")
    #         print(f"      KG Text : {kg_text_lookup.get(kg_id, kg_id)}")
    #         print(f"      Sim     : {item['raw_sim']:.4f}")
    #         print(f"      Weight  : {item['weight']:.4f}")

    if not results:
        print("\nNo predicted mode queries.")
        return

    print("\nPredicted mode queries:")
    for rank, item in enumerate(results, 1):
        print(f"\n[{rank}]")
        print(f"  Mode Query Text    : {item['mode_query_text']}")
        print(f"  Final Pred Score   : {item['score']:.8f}")

        top_pairs = item.get("pair_details", [])[:SHOW_TOP_PAIR_DETAILS]
        # if top_pairs:
        #     print("  Top pair contributions:")
        #     for p in top_pairs:
        #         print(f"    - {p['head_kg_id']} -> {p['tail_kg_id']}")
        #         print(f"      head_sim      : {p['head_sim']:.4f}")
        #         print(f"      tail_sim      : {p['tail_sim']:.4f}")
        #         print(f"      head_weight   : {p['head_weight']:.4f}")
        #         print(f"      tail_weight   : {p['tail_weight']:.4f}")
        #         print(f"      pair_score    : {p['pair_score']:.8f}")
        #         print(f"      contribution  : {p['contribution']:.8f}")


def print_weighted_mode_to_effect_predictions(
    query_text,
    used_heads,
    results,
    kg_text_lookup
):
    print(f"\n{'-' * 80}")
    print("Weighted Query Mode -> Query Effect Prediction")
    print(f"{'-' * 80}")
    print(f"Query Mode Text  : {query_text}")

    print("\nWeighted mapped mode nodes:")
    if not used_heads:
        print("  None")
    # else:
    #     for rank, item in enumerate(used_heads, 1):
    #         kg_id = item["kg_id"]
    #         print(f"  [{rank}] {kg_id}")
    #         print(f"      KG Text : {kg_text_lookup.get(kg_id, kg_id)}")
    #         print(f"      Sim     : {item['raw_sim']:.4f}")
    #         print(f"      Weight  : {item['weight']:.4f}")

    if not results:
        print("\nNo predicted effect queries.")
        return

    print("\nPredicted effect queries:")
    for rank, item in enumerate(results, 1):
        print(f"\n[{rank}]")
        print(f"  Effect Query Text : {item['effect_query_text']}")
        print(f"  Final Pred Score  : {item['score']:.8f}")

        top_pairs = item.get("pair_details", [])[:SHOW_TOP_PAIR_DETAILS]
        # if top_pairs:
        #     print("  Top pair contributions:")
        #     for p in top_pairs:
        #         print(f"    - {p['head_kg_id']} -> {p['tail_kg_id']}")
        #         print(f"      head_sim      : {p['head_sim']:.4f}")
        #         print(f"      tail_sim      : {p['tail_sim']:.4f}")
        #         print(f"      head_weight   : {p['head_weight']:.4f}")
        #         print(f"      tail_weight   : {p['tail_weight']:.4f}")
        #         print(f"      pair_score    : {p['pair_score']:.8f}")
        #         print(f"      contribution  : {p['contribution']:.8f}")


# =========================================================
# 11. PIPELINE
# =========================================================

def run_structure_mapping_and_inference(
    session,
    structure_input,
    model, x, edge_index, edge_type,
    node2id, id2node, id2type, id2text, rel2id
):
    mapped = map_structure_input_to_kg(session, structure_input)

    print_mapping_results("CAUSE MAPPING", mapped["causes"])
    print_mapping_results("MODE MAPPING", mapped["modes"])
    print_mapping_results("EFFECT MAPPING", mapped["effects"])

    kg_text_lookup = build_kg_text_lookup(mapped, id2text=id2text, node2id=node2id)

    print(f"\nMapped cause query count  : {len(mapped['causes'])}")
    print(f"Mapped mode query count   : {len(mapped['modes'])}")
    print(f"Mapped effect query count : {len(mapped['effects'])}")

    z = encode_graph_once(model, x, edge_index, edge_type)

    # =====================================================
    # 1) QUERY-LEVEL: Cause query -> Mode queries
    # =====================================================
    print(f"\n{'#' * 80}")
    print("RUNNING WEIGHTED QUERY CAUSE -> QUERY MODE INFERENCE")
    print(f"{'#' * 80}")

    for cause_item in mapped["causes"]:
        query_text = cause_item["query_text"]

        if not cause_item["mapped_nodes"]:
            print(f"\nSkip cause query (no mapping): {query_text}")
            continue

        results, used_heads = infer_query_cause_to_query_modes_weighted(
            model=model,
            z=z,
            node2id=node2id,
            rel2id=rel2id,
            mapped_cause_nodes=cause_item["mapped_nodes"],
            mapped_mode_items=mapped["modes"],
            top_k=TOP_K_PRED,
            weighting=WEIGHTING_METHOD
        )

        print_weighted_cause_to_mode_predictions(
            query_text=query_text,
            used_heads=used_heads,
            results=results,
            kg_text_lookup=kg_text_lookup
        )

    # =====================================================
    # 2) QUERY-LEVEL: Mode query -> Effect queries
    # =====================================================
    print(f"\n{'#' * 80}")
    print("RUNNING WEIGHTED QUERY MODE -> QUERY EFFECT INFERENCE")
    print(f"{'#' * 80}")

    for mode_item in mapped["modes"]:
        query_text = mode_item["query_text"]

        if not mode_item["mapped_nodes"]:
            print(f"\nSkip mode query (no mapping): {query_text}")
            continue

        results, used_heads = infer_query_mode_to_query_effects_weighted(
            model=model,
            z=z,
            node2id=node2id,
            rel2id=rel2id,
            mapped_mode_nodes=mode_item["mapped_nodes"],
            mapped_effect_items=mapped["effects"],
            top_k=TOP_K_PRED,
            weighting=WEIGHTING_METHOD
        )

        print_weighted_mode_to_effect_predictions(
            query_text=query_text,
            used_heads=used_heads,
            results=results,
            kg_text_lookup=kg_text_lookup
        )


# =========================================================
# 12. MAIN
# =========================================================

def main():
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
                id2text=id2text,
                rel2id=rel2id
            )
    finally:
        kg_client.close()


if __name__ == "__main__":
    main()