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

# Use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# 0. USER INPUT (example structured FMEA-like input)
# =========================================================
# The user provides a structured "element" with:
#  - failure_element name
#  - a list of modes
#  - a dict/list of causes
#  - a list of effects
# This will be mapped to existing KG nodes via vector search in Neo4j.
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
# Load environment variables (Neo4j credentials)
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Local files used to reconstruct the same graph used at training time
NODE_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\nodes.tsv"
TRIPLE_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\triples.tsv"
MODEL_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\rgcn_complex_weighted_best.pt"

# Must match training hyperparameters
HIDDEN_DIM = 256
EMB_DIM = 256
DROPOUT = 0.1
NUM_BASES = 4

# Graph edge weight normalization settings (same as training)
GRAPH_WEIGHT_CLAMP_MIN = 1e-3
GRAPH_WEIGHT_CLAMP_MAX = 10.0
GRAPH_WEIGHT_POWER = 1.0

# Mapping config: how many KG candidates to retrieve per query text
TOP_K_MAP = 5
MIN_SIM = 0.70
POOL_K = 100

# Inference config: how many predicted query-to-query links to keep
TOP_K_PRED = 3
PRED_SCORE_THRESHOLD = 0.0

# When multiple KG nodes are mapped for one query text, combine them with weights
WEIGHTING_METHOD = "square"   # ["linear", "square", "uniform"]
APPLY_SIGMOID_TO_PAIR_SCORE = True
SHOW_TOP_PAIR_DETAILS = 5

# =========================================================
# 2. TEXT / STRUCTURE HELPERS
# =========================================================

def stable_id(text: str) -> str:
    """Create a stable hash id for a piece of text (useful for caching)."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def safe_text(value) -> str:
    """Convert any input to a clean string; return empty if None."""
    if value is None:
        return ""
    return str(value).strip()

def flatten_causes(causes):
    """
    Normalize causes to a flat list[str].
    Input can be:
      - list[str]
      - dict[str, list[str]] (e.g., mechanics/hardware/software)
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

def deduplicate_mapped_nodes(mapped_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    A query text might map to the same KG node multiple times.
    Keep only the best (highest score) mapping per kg_id.
    """
    best = {}
    for item in mapped_nodes:
        kg_id = item["kg_id"]
        if kg_id not in best or item["score"] > best[kg_id]["score"]:
            best[kg_id] = item
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)

def extract_structure_queries(structure_input: Dict[str, Any]):
    """Extract all cause/mode/effect text strings from the structured input."""
    all_causes = []
    all_modes = []
    all_effects = []

    for node in structure_input.get("nodes", []):
        all_modes.extend([safe_text(x) for x in node.get("modes", []) if safe_text(x)])
        all_effects.extend([safe_text(x) for x in node.get("effects", []) if safe_text(x)])
        all_causes.extend(flatten_causes(node.get("causes", {})))

    return all_causes, all_modes, all_effects

def build_kg_text_lookup(
    mapped: Dict[str, List[Dict[str, Any]]],
    id2text: Dict[int, str],
    node2id: Dict[str, int]
) -> Dict[str, str]:
    """
    Build a kg_id -> display text lookup:
      - Prefer the text returned from Neo4j mapping
      - Fall back to id2text loaded from nodes.tsv
      - Fall back to kg_id itself
    """
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
# 3. EMBEDDING (text -> vector for Neo4j vector search)
# =========================================================

# SentenceTransformer embeddings for mapping user query text to KG nodes
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Simple in-memory embedding cache to avoid recomputing embeddings for repeated texts
_embedding_cache: Dict[str, list] = {}

def embed(text: str):
    """Embed a text string into a vector (list[float])."""
    text = safe_text(text)
    if not text:
        return None
    if text in _embedding_cache:
        return _embedding_cache[text]
    vec = embedder([text])[0]
    _embedding_cache[text] = vec
    return vec


# =========================================================
# 4. LOAD TSV DATA (same format as training)
# =========================================================
from torch_geometric.nn import MessagePassing  # used by WeightedRGCNConv

def load_nodes(node_file: str):
    """
    Load nodes from nodes.tsv:
      - node_id: KG semantic id (string)
      - node_type: label/type used for schema (Cause/Mode/Effect/etc.)
      - embedding: stored as stringified python list
      - text/canonical_text/name (optional): human-readable text
    Returns:
      node2id, id2node, id2type, id2text, x
    """
    df = pd.read_csv(node_file, sep="\t")

    required_cols = {"node_id", "node_type", "embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in node file: {missing}")

    node_ids = df["node_id"].tolist()
    node_types = df["node_type"].tolist()

    # Choose a text column if present; otherwise fallback to node_id
    if "text" in df.columns:
        node_texts = df["text"].fillna("").tolist()
    elif "canonical_text" in df.columns:
        node_texts = df["canonical_text"].fillna("").tolist()
    elif "name" in df.columns:
        node_texts = df["name"].fillna("").tolist()
    else:
        node_texts = [str(n) for n in node_ids]

    # Parse embeddings from string -> tensor, and L2-normalize
    embeddings = []
    for emb_str in df["embedding"]:
        emb = torch.tensor(ast.literal_eval(emb_str), dtype=torch.float)
        embeddings.append(emb)

    x = torch.stack(embeddings)
    x = F.normalize(x, p=2, dim=1)

    # Build id mappings
    node2id = {nid: i for i, nid in enumerate(node_ids)}
    id2node = {i: nid for nid, i in node2id.items()}
    id2type = {node2id[nid]: t for nid, t in zip(node_ids, node_types)}
    id2text = {node2id[nid]: txt for nid, txt in zip(node_ids, node_texts)}

    return node2id, id2node, id2type, id2text, x

def load_triples(triple_file: str, node2id: Dict[str, int]):
    """
    Load triples from triples.tsv and convert head/tail kg_id -> integer node ids.
    Returns:
      triples_raw: List[(h_id, rel_str, t_id, weight)]
      rel_list_base: sorted unique relation strings (without _REV)
    """
    df = pd.read_csv(triple_file, sep="\t")

    required_cols = {"head", "relation", "tail"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in triple file: {missing}")

    has_weight = "weight" in df.columns

    triples = []
    relations = set()

    for _, row in df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]
        if h not in node2id or t not in node2id:
            continue

        w = float(row["weight"]) if has_weight else 1.0
        triples.append((node2id[h], r, node2id[t], w))
        relations.add(r)

    return triples, sorted(list(relations))


# =========================================================
# 5. RELATION MAPPING (must match training ordering!)
# =========================================================
def build_relations(rel_list_base: List[str]):
    """
    IMPORTANT: relation id ordering must match training:
      rel_list_full = base_relations + reverse_relations
    """
    rel_list_full = rel_list_base + [r + "_REV" for r in rel_list_base]
    rel2id = {r: i for i, r in enumerate(rel_list_full)}
    id2rel = {i: r for r, i in rel2id.items()}
    return rel2id, id2rel


# =========================================================
# 6. GRAPH BUILDING (PyG tensors)
# =========================================================
def build_graph(
    triples_raw: List[Tuple[int, str, int, float]],
    rel2id: Dict[str, int],
    add_reverse_edges: bool = True
):
    """
    Build PyG graph tensors:
      edge_index: [2, E] (source, target)
      edge_type:  [E]    relation id per edge
      edge_weight:[E]    float weight per edge
    Optionally add explicit reverse edges with relation name r+"_REV".
    """
    edge_index = []
    edge_type = []
    edge_weight = []

    for h, r, t, w in triples_raw:
        if r not in rel2id:
            continue

        # Forward edge
        edge_index.append([h, t])
        edge_type.append(rel2id[r])
        edge_weight.append(float(w))

        # Reverse edge (separate relation id)
        if add_reverse_edges:
            rev_r = r + "_REV"
            if rev_r not in rel2id:
                raise KeyError(f"Reverse relation missing in rel2id: {rev_r}")
            edge_index.append([t, h])
            edge_type.append(rel2id[rev_r])
            edge_weight.append(float(w))

    if len(edge_index) == 0:
        raise ValueError("No edges built. Check triples/relations.")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return edge_index, edge_type, edge_weight

def normalize_edge_weight(
    edge_weight: torch.Tensor,
    clamp_min: float = 1e-3,
    clamp_max: float = 10.0,
    power: float = 1.0
):
    """
    Normalize graph edge weights similarly to training:
      - clamp minimum to avoid zeros
      - optional power transform
      - divide by mean => average weight becomes 1
      - clamp max to avoid extreme weights
    """
    w = edge_weight.float().clamp_min(clamp_min)
    if power != 1.0:
        w = w.pow(power)
    w = w / w.mean().clamp_min(1e-8)
    if clamp_max is not None:
        w = w.clamp(max=clamp_max)
    return w


# =========================================================
# 7. WEIGHTED R-GCN CONV (same as training)
# =========================================================
class WeightedRGCNConv(MessagePassing):
    """
    Basis-decomposed R-GCN with edge-weight aware aggregation.

    For each relation r:
      W_r = sum_b att[r,b] * basis[b]

    For each edge e: j -> i (of relation r):
      msg_weight = w_e / sum_{k->i} w_{k->i} * softplus(scale_r)
      message    = msg_weight * (x_j W_r)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: int = 4,
        root_weight: bool = True,
        bias: bool = True
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        # Basis matrices and relation-specific mixing weights
        self.basis = nn.Parameter(torch.empty(self.num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.empty(num_relations, self.num_bases))

        # Optional root/self-loop transformation
        self.root = nn.Parameter(torch.empty(in_channels, out_channels)) if root_weight else None
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Relation-specific edge scaling (kept positive via softplus)
        self.rel_edge_scale = nn.Parameter(torch.ones(num_relations))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.basis)
        nn.init.xavier_uniform_(self.att)
        if self.root is not None:
            nn.init.xavier_uniform_(self.root)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.ones_(self.rel_edge_scale)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        num_nodes = x.size(0)
        out = x.new_zeros(num_nodes, self.out_channels)

        # Default weight=1 for all edges if not provided
        if edge_weight is None:
            edge_weight = x.new_ones(edge_type.size(0))
        edge_weight = edge_weight.to(dtype=x.dtype, device=x.device)

        # Expand basis decomposition into full per-relation weights: [R, in, out]
        weight = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        weight = weight.view(self.num_relations, self.in_channels, self.out_channels)

        # Iterate over relations and aggregate messages for edges of that relation
        for rel in range(self.num_relations):
            mask = (edge_type == rel)
            if not bool(mask.any()):
                continue

            rel_edge_index = edge_index[:, mask]
            rel_edge_weight = edge_weight[mask]

            # Normalize by sum of incoming weights (per destination node)
            dst = rel_edge_index[1]
            denom = x.new_zeros(num_nodes)
            denom.index_add_(0, dst, rel_edge_weight)
            norm = 1.0 / denom[dst].clamp_min(1e-12)

            rel_scale = F.softplus(self.rel_edge_scale[rel])
            msg_weight = rel_edge_weight * norm * rel_scale

            x_rel = x @ weight[rel]
            out = out + self.propagate(
                rel_edge_index,
                x=x_rel,
                edge_weight=msg_weight,
                size=None
            )

        # Add root/self-loop transform
        if self.root is not None:
            out = out + x @ self.root
        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_weight):
        """Scale messages by edge weights."""
        return x_j * edge_weight.view(-1, 1)


# =========================================================
# 8. MODEL (encoder + ComplEx decoder)
# =========================================================
class RGCN(nn.Module):
    """Two-layer WeightedRGCN encoder."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, dropout=0.1, num_bases=4):
        super().__init__()
        self.conv1 = WeightedRGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = WeightedRGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type, edge_weight):
        x = self.conv1(x, edge_index, edge_type, edge_weight=edge_weight)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_weight=edge_weight)
        return x

class ComplExDecoder(nn.Module):
    """
    ComplEx decoder with complex embeddings stored as [Re, Im] concatenation.
    Node embeddings z: [num_nodes, 2*emb_dim]
    Relation embeddings: [num_relations, 2*emb_dim]
    """
    def __init__(self, num_relations, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.rel = nn.Parameter(torch.empty(num_relations, 2 * emb_dim))
        nn.init.xavier_uniform_(self.rel)

    def forward(self, z, triples):
        """Compute logits for triples [B,3] = (head, relation, tail)."""
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
    """
    Full model:
      x -> input_proj -> WeightedRGCN -> node embeddings z
      z + relation embeddings -> ComplEx score
    """
    def __init__(self, in_dim, hidden_dim, emb_dim, num_relations, dropout=0.1, num_bases=4):
        super().__init__()

        # Project raw node features to hidden dimension used by GNN
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)

        # Encoder outputs 2*emb_dim for ComplEx
        self.rgcn = RGCN(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=2 * emb_dim,
            num_relations=num_relations,
            dropout=dropout,
            num_bases=num_bases
        )
        self.decoder = ComplExDecoder(num_relations, emb_dim)

    def encode(self, x, edge_index, edge_type, edge_weight):
        """Compute node embeddings z."""
        x = self.input_proj(x)
        x = F.relu(x)
        z = self.rgcn(x, edge_index, edge_type, edge_weight=edge_weight)
        return z

    def score(self, z, triples):
        """Compute triple logits."""
        return self.decoder(z, triples)

    def forward(self, x, edge_index, edge_type, edge_weight, triples):
        """Encode then score."""
        z = self.encode(x, edge_index, edge_type, edge_weight=edge_weight)
        return self.score(z, triples)


# =========================================================
# 9. LOAD MODEL CHECKPOINT
# =========================================================
def load_model(
    path: str,
    in_dim: int,
    hidden_dim: int,
    emb_dim: int,
    num_relations: int,
    dropout: float = 0.1,
    num_bases: int = 4,
    device=DEVICE
):
    """Recreate model architecture and load trained weights."""
    model = Model(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        num_relations=num_relations,
        dropout=dropout,
        num_bases=num_bases
    ).to(device)

    ckpt = torch.load(path, map_location=device)

    # Training saved a dict with model_state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from: {path}")
        if "epoch" in ckpt:
            print(f"Checkpoint epoch: {ckpt['epoch']}")
        if "best_score" in ckpt:
            print(f"Checkpoint best_score: {ckpt['best_score']}")
    else:
        # Or raw state_dict
        model.load_state_dict(ckpt)
        print(f"Loaded raw state_dict from: {path}")

    model.eval()
    return model


# =========================================================
# 10. NEO4J CLIENT
# =========================================================
class Neo4jKGClient:
    """Minimal Neo4j wrapper for opening/closing sessions."""
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def session(self):
        return self.driver.session(database=self.database)


# =========================================================
# 11. KG RETRIEVAL (vector mapping in Neo4j)
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
      1) embed(query_text)
      2) Neo4j vector search against index_name
      3) return top_k_each nodes with cosine similarity >= min_score

    keep_group_or_single_only:
      - optionally filter to "group" nodes or nodes that do not BELONGS_TO a group
        (domain-specific KG design choice).
    """
    results = []

    for text in query_texts:
        emb = embed(text)
        if emb is None:
            results.append({
                "query_text": text,
                "mapped_nodes": []
            })
            continue

        # Ensure embedding is a Python list for Neo4j parameters
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().tolist()

        # Optional filtering for group/single nodes
        if keep_group_or_single_only:
            filter_clause = f"""
            WHERE
                coalesce(node.is_group, false) = true
                OR NOT (node)-[:BELONGS_TO]->(:{label})
            """
        else:
            filter_clause = ""

        # Neo4j vector search + refined cosine similarity
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
    """
    Map all extracted causes/modes/effects text from the user input
    to KG nodes using the corresponding Neo4j vector indices.
    """
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
# 12. WEIGHTED QUERY REPRESENTATION + PREDICTION
# =========================================================

@torch.no_grad()
def encode_graph_once(model, x, edge_index, edge_type, edge_weight):
    """Encode the KG graph once and reuse node embeddings z for all query scoring."""
    model.eval()
    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_type = edge_type.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)
    return model.encode(x, edge_index, edge_type, edge_weight)


@torch.no_grad()
def build_weighted_node_list(
    mapped_nodes: List[Dict[str, Any]],
    node2id: Dict[str, int],
    weighting: str = "linear",
    min_sim: float = 0.0
):
    """
    Convert mapped KG nodes (with similarity scores) into a weighted set:
      - keep only nodes existing in node2id and sim >= min_sim
      - compute normalized weights from similarities (linear/square/uniform)
    Output items contain:
      kg_id, node_id (int), raw_sim, weight (sums to 1)
    """
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

    # Normalize weights to sum to 1
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
    Standard ComplEx score for a single triple using already-encoded embeddings.
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
    Score two weighted node sets against a relation:
      Final score = sum_i sum_j w_hi * w_tj * score(h_i, r, t_j)

    Returns:
      final_score (float) and pair_details (for debugging/explanations)
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


# =========================================================
# 13. BIDIRECTIONAL (forward + reverse) SCORING ENHANCEMENT
# =========================================================

def _normalize_weighted_nodes(weighted_nodes):
    """
    Accept multiple input formats and normalize to:
      [(node_id, weight), ...]
    Supported:
      1) [(node_id, weight), ...]
      2) [{"node_id": ..., "weight": ...}, ...]
    """
    normed = []
    for item in weighted_nodes:
        if isinstance(item, dict):
            nid = item["node_id"]
            w = float(item.get("weight", 1.0))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            nid = item[0]
            w = float(item[1])
        else:
            nid = item
            w = 1.0
        normed.append((nid, w))
    return normed


@torch.no_grad()
def score_weighted_node_sets_bidirectional(
    model,
    z,
    rel2id,
    relation_name,
    head_nodes,
    tail_nodes,
    forward_alpha=0.5,              # weight for forward direction
    consistency_lambda=0.0,         # penalty for disagreement between forward/reverse logits
    apply_sigmoid=True
):
    """
    Compute a fused score using both directions:
      forward: score(h, relation, t)
      reverse: score(t, relation_REV, h)

    Fused logits:
      fused = alpha * forward + (1-alpha) * reverse - lambda * |forward - reverse|

    Then combine all pair scores with pair weights (head_weight * tail_weight).
    """
    if relation_name not in rel2id:
        raise KeyError(f"Relation '{relation_name}' not found in rel2id.")
    rev_relation_name = relation_name + "_REV"
    if rev_relation_name not in rel2id:
        raise KeyError(f"Reverse relation '{rev_relation_name}' not found in rel2id.")

    relation_id = rel2id[relation_name]
    reverse_relation_id = rel2id[rev_relation_name]

    head_nodes = _normalize_weighted_nodes(head_nodes)
    tail_nodes = _normalize_weighted_nodes(tail_nodes)

    if not head_nodes or not tail_nodes:
        return None, []

    # Build all pair triples for forward and reverse scoring
    triples_fwd = []
    triples_rev = []
    pair_meta = []

    for h_id, h_w in head_nodes:
        for t_id, t_w in tail_nodes:
            pair_weight = h_w * t_w

            triples_fwd.append([h_id, relation_id, t_id])
            triples_rev.append([t_id, reverse_relation_id, h_id])

            pair_meta.append({
                "head_id": h_id,
                "tail_id": t_id,
                "head_weight": h_w,
                "tail_weight": t_w,
                "pair_weight": pair_weight
            })

    device = z.device
    triples_fwd = torch.tensor(triples_fwd, dtype=torch.long, device=device)
    triples_rev = torch.tensor(triples_rev, dtype=torch.long, device=device)

    forward_logits = model.score(z, triples_fwd)
    reverse_logits = model.score(z, triples_rev)

    fused_logits = (
        forward_alpha * forward_logits
        + (1.0 - forward_alpha) * reverse_logits
        - consistency_lambda * torch.abs(forward_logits - reverse_logits)
    )

    pair_scores = torch.sigmoid(fused_logits) if apply_sigmoid else fused_logits

    # Weighted average by pair_weight (not sum; keeps score in a stable range)
    pair_weights = torch.tensor(
        [x["pair_weight"] for x in pair_meta],
        dtype=torch.float,
        device=device
    )
    denom = pair_weights.sum().clamp_min(1e-8)
    final_score = (pair_scores * pair_weights).sum() / denom

    # Return per-pair diagnostics as well
    pair_details = []
    for meta, f_logit, r_logit, fused_logit, p_score in zip(
        pair_meta,
        forward_logits.tolist(),
        reverse_logits.tolist(),
        fused_logits.tolist(),
        pair_scores.tolist()
    ):
        d = dict(meta)
        d.update({
            "forward_logit": f_logit,
            "reverse_logit": r_logit,
            "fused_logit": fused_logit,
            "pair_score": p_score
        })
        pair_details.append(d)

    return float(final_score.item()), pair_details


# =========================================================
# 14. QUERY-LEVEL INFERENCE ROUTINES
# =========================================================
# These functions score query->query links by aggregating over mapped KG node sets.

@torch.no_grad()
def infer_query_cause_to_query_modes_weighted(
    model, z, node2id, rel2id,
    mapped_cause_nodes,
    mapped_mode_items,
    top_k=5,
    weighting="linear"
):
    """
    For one cause query (mapped to multiple KG cause nodes),
    rank all mode queries by score(cause_set, CAUSES, mode_set).
    """
    if "CAUSES" not in rel2id:
        raise KeyError("Relation 'CAUSES' not found in rel2id.")
    if "CAUSES_REV" not in rel2id:
        raise KeyError("Relation 'CAUSES_REV' not found in rel2id.")

    cause_weighted_nodes = build_weighted_node_list(
        mapped_nodes=mapped_cause_nodes,
        node2id=node2id,
        weighting=weighting
    )

    if not cause_weighted_nodes:
        return [], cause_weighted_nodes

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

        # Bidirectional fusion uses both CAUSES and CAUSES_REV
        score, pair_details = score_weighted_node_sets_bidirectional(
            model=model,
            z=z,
            rel2id=rel2id,
            relation_name="CAUSES",
            head_nodes=cause_weighted_nodes,
            tail_nodes=mode_weighted_nodes,
            forward_alpha=0.5,
            consistency_lambda=0.05,
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
def infer_query_mode_to_query_causes_weighted(
    model, z, node2id, rel2id,
    mapped_mode_nodes,
    mapped_cause_items,
    top_k=5,
    weighting="linear"
):
    """
    Mode query -> rank cause queries.
    Here we score: score(candidate_cause, CAUSES, query_mode)
    (mode acts as tail in CAUSES relation).
    """
    if "CAUSES" not in rel2id:
        raise KeyError("Relation 'CAUSES' not found in rel2id.")

    mode_weighted_nodes = build_weighted_node_list(
        mapped_nodes=mapped_mode_nodes,
        node2id=node2id,
        weighting=weighting
    )

    if not mode_weighted_nodes:
        return [], mode_weighted_nodes

    relation_id = rel2id["CAUSES"]
    results = []

    for cause_item in mapped_cause_items:
        cause_query_text = cause_item["query_text"]

        cause_weighted_nodes = build_weighted_node_list(
            mapped_nodes=cause_item["mapped_nodes"],
            node2id=node2id,
            weighting=weighting
        )

        if not cause_weighted_nodes:
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
                "cause_query_text": cause_query_text,
                "score": score,
                "cause_used_nodes": cause_weighted_nodes,
                "pair_details": pair_details
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    return results, mode_weighted_nodes


@torch.no_grad()
def infer_query_mode_to_query_effects_weighted(
    model, z, node2id, rel2id,
    mapped_mode_nodes,
    mapped_effect_items,
    top_k=5,
    weighting="linear"
):
    """
    For one mode query, rank effect queries by score(mode_set, LEADS_TO, effect_set).
    Uses bidirectional fusion with LEADS_TO and LEADS_TO_REV.
    """
    if "LEADS_TO" not in rel2id:
        raise KeyError("Relation 'LEADS_TO' not found in rel2id.")
    if "LEADS_TO_REV" not in rel2id:
        raise KeyError("Relation 'LEADS_TO_REV' not found in rel2id.")

    mode_weighted_nodes = build_weighted_node_list(
        mapped_nodes=mapped_mode_nodes,
        node2id=node2id,
        weighting=weighting
    )

    if not mode_weighted_nodes:
        return [], mode_weighted_nodes

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

        score, pair_details = score_weighted_node_sets_bidirectional(
            model=model,
            z=z,
            rel2id=rel2id,
            relation_name="LEADS_TO",
            head_nodes=mode_weighted_nodes,
            tail_nodes=effect_weighted_nodes,
            forward_alpha=0.5,
            consistency_lambda=0.05,
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
# 15. PRINTING / REPORTING
# =========================================================

def print_mapping_results(title: str, mapping_list: List[Dict[str, Any]]):
    """Print query -> mapped KG nodes (with similarity scores)."""
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
    """Print cause->mode query ranking results."""
    print(f"\n{'-' * 80}")
    print("Weighted Query Cause -> Query Mode Prediction")
    print(f"{'-' * 80}")
    print(f"Query Cause Text : {query_text}")

    print("\nWeighted mapped cause nodes:")
    if not used_heads:
        print("  None")

    if not results:
        print("\nNo predicted mode queries.")
        return

    print("\nPredicted mode queries:")
    for rank, item in enumerate(results, 1):
        print(f"\n[{rank}]")
        print(f"  Mode Query Text    : {item['mode_query_text']}")
        print(f"  Final Pred Score   : {item['score']:.8f}")

        # Pair details are available for explanation/debugging if needed
        top_pairs = item.get("pair_details", [])[:SHOW_TOP_PAIR_DETAILS]


def print_weighted_mode_to_cause_predictions(
    query_text,
    used_tails,
    results,
    kg_text_lookup
):
    """Print mode->cause query ranking results."""
    print(f"\n{'-' * 80}")
    print("Weighted Query Mode -> Query Cause Prediction")
    print(f"{'-' * 80}")
    print(f"Query Mode Text  : {query_text}")

    print("\nWeighted mapped mode nodes:")
    if not used_tails:
        print("  None")

    if not results:
        print("\nNo predicted cause queries.")
        return

    print("\nPredicted cause queries:")
    for rank, item in enumerate(results, 1):
        print(f"\n[{rank}]")
        print(f"  Cause Query Text  : {item['cause_query_text']}")
        print(f"  Final Pred Score  : {item['score']:.8f}")

        top_pairs = item.get("pair_details", [])[:SHOW_TOP_PAIR_DETAILS]


def print_weighted_mode_to_effect_predictions(
    query_text,
    used_heads,
    results,
    kg_text_lookup
):
    """Print mode->effect query ranking results."""
    print(f"\n{'-' * 80}")
    print("Weighted Query Mode -> Query Effect Prediction")
    print(f"{'-' * 80}")
    print(f"Query Mode Text  : {query_text}")

    print("\nWeighted mapped mode nodes:")
    if not used_heads:
        print("  None")

    if not results:
        print("\nNo predicted effect queries.")
        return

    print("\nPredicted effect queries:")
    for rank, item in enumerate(results, 1):
        print(f"\n[{rank}]")
        print(f"  Effect Query Text : {item['effect_query_text']}")
        print(f"  Final Pred Score  : {item['score']:.8f}")

        top_pairs = item.get("pair_details", [])[:SHOW_TOP_PAIR_DETAILS]


# =========================================================
# 16. END-TO-END PIPELINE (mapping + inference)
# =========================================================

def run_structure_mapping_and_inference(
    session,
    structure_input,
    model, x, edge_index, edge_type,
    node2id, id2node, id2type, id2text, rel2id,
    edge_weight
):
    """
    Full pipeline:
      1) Map user input texts -> KG nodes (Neo4j vector search)
      2) Encode KG graph once -> z
      3) Score query-to-query predictions using weighted node sets
    """
    mapped = map_structure_input_to_kg(session, structure_input)

    print_mapping_results("CAUSE MAPPING", mapped["causes"])
    print_mapping_results("MODE MAPPING", mapped["modes"])
    print_mapping_results("EFFECT MAPPING", mapped["effects"])

    kg_text_lookup = build_kg_text_lookup(mapped, id2text=id2text, node2id=node2id)

    print(f"\nMapped cause query count  : {len(mapped['causes'])}")
    print(f"Mapped mode query count   : {len(mapped['modes'])}")
    print(f"Mapped effect query count : {len(mapped['effects'])}")

    # Encode KG node embeddings once; all scoring uses z afterwards
    z = encode_graph_once(model, x, edge_index, edge_type, edge_weight)

    # =====================================================
    # 1) Cause query -> Mode query predictions
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
    # 2) Mode query -> Effect query predictions
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
# 17. MAIN (load everything, connect to Neo4j, run pipeline)
# =========================================================

def main():
    # Load TSV graph artifacts
    node2id, id2node, id2type, id2text, x = load_nodes(NODE_FILE)

    # triples_raw: (h_id, r_str, t_id, w)
    triples_raw, rel_list_base = load_triples(TRIPLE_FILE, node2id)

    # Build relation ids in the same ordering as training
    rel2id, id2rel = build_relations(rel_list_base)

    # Build PyG graph tensors and normalize weights
    edge_index, edge_type, edge_weight = build_graph(
        triples_raw=triples_raw,
        rel2id=rel2id,
        add_reverse_edges=True
    )

    edge_weight = normalize_edge_weight(
        edge_weight,
        clamp_min=GRAPH_WEIGHT_CLAMP_MIN,
        clamp_max=GRAPH_WEIGHT_CLAMP_MAX,
        power=GRAPH_WEIGHT_POWER
    )

    # Move tensors to device
    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_type = edge_type.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

    # Load trained model
    model = load_model(
        path=MODEL_PATH,
        in_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
        emb_dim=EMB_DIM,
        num_relations=len(rel2id),
        dropout=DROPOUT,
        num_bases=NUM_BASES,
        device=DEVICE
    )

    # Create Neo4j client for vector retrieval
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
                rel2id=rel2id,
                edge_weight=edge_weight
            )
    finally:
        kg_client.close()


if __name__ == "__main__":
    main()