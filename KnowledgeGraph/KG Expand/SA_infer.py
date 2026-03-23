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


# =========================================================
# 0. USER INPUT
# =========================================================

structure_input_motorcontrol = {
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

SOFTMAP_TEMPERATURE = 0.10


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
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMult(nn.Module):
    def __init__(self, num_relations, emb_dim):
        super().__init__()
        self.rel = nn.Parameter(torch.randn(num_relations, emb_dim) * 0.1)

    def forward(self, z, triples):
        s = z[triples[:, 0]]
        r = self.rel[triples[:, 1]]
        o = z[triples[:, 2]]
        return (s * r * o).sum(dim=1)


class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, out_dim, num_relations)
        self.decoder = DistMult(num_relations, out_dim)

    def encode(self, x, edge_index, edge_type):
        return self.rgcn(x, edge_index, edge_type)

    def score(self, z, triples):
        return self.decoder(z, triples)


def load_model(path, in_dim, hidden_dim, out_dim, num_relations):
    model = Model(in_dim, hidden_dim, out_dim, num_relations)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
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
# 9. SOFT QUERY REPRESENTATION + PREDICTION
# =========================================================

@torch.no_grad()
def encode_graph_once(model, x, edge_index, edge_type):
    return model.encode(x, edge_index, edge_type)


@torch.no_grad()
def build_soft_query_embedding(
    mapped_nodes: List[Dict[str, Any]],
    node2id: Dict[str, int],
    z: torch.Tensor,
    temperature: float = 0.10
):
    valid_items = []

    for m in mapped_nodes:
        kg_id = m["kg_id"]
        sim = float(m["score"])
        if kg_id in node2id:
            valid_items.append((kg_id, sim))

    if not valid_items:
        return None, []

    kg_ids = [kg_id for kg_id, _ in valid_items]
    sims = torch.tensor([sim for _, sim in valid_items], dtype=torch.float)

    weights = torch.softmax(sims / temperature, dim=0)

    emb_list = []
    used_nodes = []

    for i, kg_id in enumerate(kg_ids):
        nid = node2id[kg_id]
        emb_list.append(z[nid] * weights[i])
        used_nodes.append({
            "kg_id": kg_id,
            "raw_sim": float(sims[i].item()),
            "weight": float(weights[i].item())
        })

    query_repr = torch.stack(emb_list, dim=0).sum(dim=0)
    return query_repr, used_nodes


@torch.no_grad()
def score_query_to_candidates(
    model,
    query_repr: torch.Tensor,
    relation_id: int,
    candidate_ids: List[int],
    z: torch.Tensor,
    top_k: int = 10
):
    if query_repr is None or not candidate_ids:
        return []

    r = model.decoder.rel[relation_id]
    tails = z[candidate_ids]

    scores = (query_repr.unsqueeze(0) * r.unsqueeze(0) * tails).sum(dim=1)
    scores = torch.sigmoid(scores)

    values, indices = torch.topk(scores, k=min(top_k, len(candidate_ids)))
    return [(candidate_ids[i], values[j].item()) for j, i in enumerate(indices)]


@torch.no_grad()
def infer_cause_query_to_mode_soft(
    model, z,
    node2id, id2node, id2type, rel2id,
    mapped_cause_nodes: List[Dict[str, Any]],
    candidate_mode_node_ids: List[str] = None,
    top_k: int = 10,
    pred_score_threshold: float = 0.0,
    temperature: float = 0.10
):
    if "CAUSES" not in rel2id:
        raise KeyError("Relation 'CAUSES' not found in rel2id.")

    query_repr, used_heads = build_soft_query_embedding(
        mapped_nodes=mapped_cause_nodes,
        node2id=node2id,
        z=z,
        temperature=temperature
    )

    if query_repr is None:
        return [], []

    relation_id = rel2id["CAUSES"]

    if candidate_mode_node_ids is None:
        candidate_ids = [nid for nid, t in id2type.items() if t == "Mode"]
    else:
        candidate_ids = [
            node2id[mid]
            for mid in candidate_mode_node_ids
            if mid in node2id and id2type[node2id[mid]] == "Mode"
        ]

    results = score_query_to_candidates(
        model=model,
        query_repr=query_repr,
        relation_id=relation_id,
        candidate_ids=candidate_ids,
        z=z,
        top_k=top_k
    )

    results = [(nid, score) for nid, score in results if score >= pred_score_threshold]
    results = [(id2node[nid], score) for nid, score in results]

    return results, used_heads


@torch.no_grad()
def infer_mode_query_to_effect_soft(
    model, z,
    node2id, id2node, id2type, rel2id,
    mapped_mode_nodes: List[Dict[str, Any]],
    candidate_effect_node_ids: List[str] = None,
    top_k: int = 10,
    pred_score_threshold: float = 0.0,
    temperature: float = 0.10
):
    if "LEADS_TO" not in rel2id:
        raise KeyError("Relation 'LEADS_TO' not found in rel2id.")

    query_repr, used_heads = build_soft_query_embedding(
        mapped_nodes=mapped_mode_nodes,
        node2id=node2id,
        z=z,
        temperature=temperature
    )

    if query_repr is None:
        return [], []

    relation_id = rel2id["LEADS_TO"]

    if candidate_effect_node_ids is None:
        candidate_ids = [nid for nid, t in id2type.items() if t == "Effect"]
    else:
        candidate_ids = [
            node2id[eid]
            for eid in candidate_effect_node_ids
            if eid in node2id and id2type[node2id[eid]] == "Effect"
        ]

    results = score_query_to_candidates(
        model=model,
        query_repr=query_repr,
        relation_id=relation_id,
        candidate_ids=candidate_ids,
        z=z,
        top_k=top_k
    )

    results = [(nid, score) for nid, score in results if score >= pred_score_threshold]
    results = [(id2node[nid], score) for nid, score in results]

    return results, used_heads


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


def print_soft_cause_to_mode_predictions(
    query_text,
    used_heads,
    results,
    kg_text_lookup,
    kg_mode_to_query_texts
):
    print(f"\n{'-' * 80}")
    print("Soft Cause Query -> Mode Prediction")
    print(f"{'-' * 80}")
    print(f"Query Cause Text : {query_text}")

    print("\nSoft mapped cause nodes:")
    if not used_heads:
        print("  None")
    else:
        for rank, item in enumerate(used_heads, 1):
            kg_id = item["kg_id"]
            print(f"  [{rank}] {kg_id}")
            print(f"      KG Text : {kg_text_lookup.get(kg_id, kg_id)}")
            print(f"      Sim     : {item['raw_sim']:.4f}")
            print(f"      Weight  : {item['weight']:.4f}")

    if not results:
        print("\nNo predicted modes.")
        return

    print("\nPredicted modes:")
    for rank, (mode_id, pred_score) in enumerate(results, 1):
        mode_text = kg_text_lookup.get(mode_id, mode_id)
        matched_query_modes = kg_mode_to_query_texts.get(mode_id, [])

        print(f"\n[{rank}]")
        print(f"  Pred Mode ID       : {mode_id}")
        print(f"  Pred Mode Text     : {mode_text}")
        print(f"  Query Mode Match   : {format_query_texts(matched_query_modes)}")
        print(f"  Pred Score         : {pred_score:.4f}")


def print_soft_mode_to_effect_predictions(
    query_text,
    used_heads,
    results,
    kg_text_lookup,
    kg_effect_to_query_texts
):
    print(f"\n{'-' * 80}")
    print("Soft Mode Query -> Effect Prediction")
    print(f"{'-' * 80}")
    print(f"Query Mode Text  : {query_text}")

    print("\nSoft mapped mode nodes:")
    if not used_heads:
        print("  None")
    else:
        for rank, item in enumerate(used_heads, 1):
            kg_id = item["kg_id"]
            print(f"  [{rank}] {kg_id}")
            print(f"      KG Text : {kg_text_lookup.get(kg_id, kg_id)}")
            print(f"      Sim     : {item['raw_sim']:.4f}")
            print(f"      Weight  : {item['weight']:.4f}")

    if not results:
        print("\nNo predicted effects.")
        return

    print("\nPredicted effects:")
    for rank, (effect_id, pred_score) in enumerate(results, 1):
        effect_text = kg_text_lookup.get(effect_id, effect_id)
        matched_query_effects = kg_effect_to_query_texts.get(effect_id, [])

        print(f"\n[{rank}]")
        print(f"  Pred Effect ID     : {effect_id}")
        print(f"  Pred Effect Text   : {effect_text}")
        print(f"  Query Effect Match : {format_query_texts(matched_query_effects)}")
        print(f"  Pred Score         : {pred_score:.4f}")


# =========================================================
# 11. PIPELINE
# =========================================================

def run_structure_mapping_and_inference(
    session,
    structure_input,
    model, x, edge_index, edge_type,
    node2id, id2node, id2type, rel2id
):
    mapped = map_structure_input_to_kg(session, structure_input)

    print_mapping_results("CAUSE MAPPING", mapped["causes"])
    print_mapping_results("MODE MAPPING", mapped["modes"])
    print_mapping_results("EFFECT MAPPING", mapped["effects"])

    kg_text_lookup = build_kg_text_lookup(mapped)
    kg_mode_to_query_texts = build_kg_to_query_texts(mapped["modes"])
    kg_effect_to_query_texts = build_kg_to_query_texts(mapped["effects"])

    mapped_mode_ids = collect_mapped_ids(mapped["modes"])
    mapped_effect_ids = collect_mapped_ids(mapped["effects"])

    print(f"\nMapped mode candidate count   : {len(mapped_mode_ids)}")
    print(f"Mapped effect candidate count : {len(mapped_effect_ids)}")

    # encode graph once
    z = encode_graph_once(model, x, edge_index, edge_type)

    print(f"\n{'#' * 80}")
    print("RUNNING SOFT CAUSE -> MODE INFERENCE")
    print(f"{'#' * 80}")

    for cause_item in mapped["causes"]:
        query_text = cause_item["query_text"]

        if not cause_item["mapped_nodes"]:
            print(f"\nSkip cause query (no mapping): {query_text}")
            continue

        results, used_heads = infer_cause_query_to_mode_soft(
            model=model,
            z=z,
            node2id=node2id,
            id2node=id2node,
            id2type=id2type,
            rel2id=rel2id,
            mapped_cause_nodes=cause_item["mapped_nodes"],
            candidate_mode_node_ids=mapped_mode_ids,
            top_k=TOP_K_PRED,
            pred_score_threshold=PRED_SCORE_THRESHOLD,
            temperature=SOFTMAP_TEMPERATURE
        )

        print_soft_cause_to_mode_predictions(
            query_text=query_text,
            used_heads=used_heads,
            results=results,
            kg_text_lookup=kg_text_lookup,
            kg_mode_to_query_texts=kg_mode_to_query_texts
        )

    print(f"\n{'#' * 80}")
    print("RUNNING SOFT MODE -> EFFECT INFERENCE")
    print(f"{'#' * 80}")

    for mode_item in mapped["modes"]:
        query_text = mode_item["query_text"]

        if not mode_item["mapped_nodes"]:
            print(f"\nSkip mode query (no mapping): {query_text}")
            continue

        results, used_heads = infer_mode_query_to_effect_soft(
            model=model,
            z=z,
            node2id=node2id,
            id2node=id2node,
            id2type=id2type,
            rel2id=rel2id,
            mapped_mode_nodes=mode_item["mapped_nodes"],
            candidate_effect_node_ids=mapped_effect_ids,
            top_k=TOP_K_PRED,
            pred_score_threshold=PRED_SCORE_THRESHOLD,
            temperature=SOFTMAP_TEMPERATURE
        )

        print_soft_mode_to_effect_predictions(
            query_text=query_text,
            used_heads=used_heads,
            results=results,
            kg_text_lookup=kg_text_lookup,
            kg_effect_to_query_texts=kg_effect_to_query_texts
        )


# =========================================================
# 12. MAIN
# =========================================================

def main():
    node2id, id2node, id2type, id2text, x = load_nodes(NODE_FILE)
    triples_raw, rel_list_base = load_triples(TRIPLE_FILE, node2id)
    rel2id, id2rel = build_relations(rel_list_base)
    edge_index, edge_type = build_graph(triples_raw, rel2id)

    model = load_model(
        MODEL_PATH,
        in_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
        out_dim=OUT_DIM,
        num_relations=len(rel2id)
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
                rel2id=rel2id
            )
    finally:
        kg_client.close()


if __name__ == "__main__":
    main()