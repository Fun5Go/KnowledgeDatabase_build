import os
import ast
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# =========================================================
# 0. CONFIG
# =========================================================
# Paths to your TSV datasets and where to save the best checkpoint
NODES_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\nodes.tsv"
TRIPLES_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\triples.tsv"
SAVE_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\rgcn_complex_weighted_best.pt"

# Reproducibility + device
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model/training hyperparameters
HIDDEN_DIM = 256
EMB_DIM = 256                 # ComplEx uses 2*EMB_DIM (real+imag)
DROPOUT = 0.1
NUM_BASES = 4                 # Basis decomposition for R-GCN relation weights

EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_NEG_PER_POS = 3           # negatives per positive triple
PATIENCE = 15                 # early stopping patience

# Train/valid/test split ratios (per relation bucket)
TRAIN_RATIO = 0.90
VALID_RATIO = 0.05

# Used for early stopping: focus on these relations only
FOCUS_RELATIONS = ("CAUSES", "LEADS_TO")

# Edge weight normalization configuration (graph message passing weights)
GRAPH_WEIGHT_CLAMP_MIN = 1e-3
GRAPH_WEIGHT_CLAMP_MAX = 10.0
GRAPH_WEIGHT_POWER = 1.0
USE_LOSS_WEIGHT = True        # if True, use triple weights as loss weights too


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 1. LOAD DATA
# =========================================================
def load_nodes(node_file: str):
    """
    Load nodes:
      - node_id: external string id: cause/mode/effect + hashid to clarify each text
      - node_type: type label used for typed negative sampling and candidate filtering
      - embedding: list-like string -> torch tensor by "all-MiniLM-L6-v2" 384 DIM
    Returns:
      node2id, id2node, id2type, x (normalized node features)
    """
    df = pd.read_csv(node_file, sep="\t")

    required_cols = {"node_id", "node_type", "embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in node file: {missing}")

    node_ids = df["node_id"].tolist()
    node_types = df["node_type"].tolist()

    # Parse stored embedding strings into tensors
    embeddings = []
    for emb_str in df["embedding"]:
        emb = torch.tensor(ast.literal_eval(emb_str), dtype=torch.float)
        embeddings.append(emb)

    # Stack to matrix [num_nodes, feat_dim] and L2-normalize
    x = torch.stack(embeddings)
    x = F.normalize(x, p=2, dim=1)

    # Mapping between external ids and internal contiguous indices
    node2id = {nid: i for i, nid in enumerate(node_ids)}
    id2node = {i: nid for nid, i in node2id.items()}
    id2type = {node2id[nid]: t for nid, t in zip(node_ids, node_types)}

    return node2id, id2node, id2type, x


def load_triples(triple_file: str, node2id: Dict[str, int]):
    """
    Load triples from TSV with schema:
      head, relation, tail, weight

    Converts head/tail string ids -> integer ids using node2id.
    Returns:
      triples: list[(h_id, rel_str, t_id, weight)]
      rel_list_base: sorted unique relation strings (without REV augmentation)
    """
    df = pd.read_csv(triple_file, sep="\t")

    required_cols = {"head", "relation", "tail", "weight"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in triple file: {missing}")

    triples = []
    relations = set()

    for _, row in df.iterrows():
        h = row["head"]
        r = row["relation"]
        t = row["tail"]
        w = float(row["weight"])

        # Skip triples that reference unknown nodes
        if h not in node2id or t not in node2id:
            continue

        triples.append((node2id[h], r, node2id[t], w))
        relations.add(r)

    return triples, sorted(list(relations))


# =========================================================
# 2. DATA SPLIT
# =========================================================
def split_triples_by_relation(
    triples_raw: List[Tuple[int, str, int, float]],
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
    seed: int = 42
):
    """
    Split triples per relation (stratified by relation).
    This avoids a relation disappearing from valid/test when it is rare.
    """
    random.seed(seed)

    # Bucket triples by relation name
    rel_buckets = defaultdict(list)
    for triple in triples_raw:
        rel_buckets[triple[1]].append(triple)

    train_triples, valid_triples, test_triples = [], [], []

    for r, bucket in rel_buckets.items():
        random.shuffle(bucket)
        n = len(bucket)

        # Ensure at least 1 train sample per relation
        n_train = max(1, int(n * train_ratio))

        # For tiny relations (<10), allow 0 valid samples
        n_valid = max(1, int(n * valid_ratio)) if n >= 10 else max(0, int(n * valid_ratio))

        # Ensure at least 1 test sample if possible
        if n_train + n_valid >= n:
            n_valid = max(0, n - n_train - 1)

        train_triples.extend(bucket[:n_train])
        valid_triples.extend(bucket[n_train:n_train + n_valid])
        test_triples.extend(bucket[n_train + n_valid:])

    return train_triples, valid_triples, test_triples


# =========================================================
# 3. RELATION SCHEMA
# =========================================================
def build_full_rel2type(base_rel2type: Dict[str, Tuple[str, str]]):
    """
    Extend schema to include reverse relations:
      R: (head_type, tail_type)
      R_REV: (tail_type, head_type)
    """
    rel2type_full = {}
    for r, (h_type, t_type) in base_rel2type.items():
        rel2type_full[r] = (h_type, t_type)
        rel2type_full[r + "_REV"] = (t_type, h_type)
    return rel2type_full


def validate_relation_schema(rel_list_base, rel2type_base):
    """Ensure all relations appearing in triples have a type schema entry."""
    missing = [r for r in rel_list_base if r not in rel2type_base]
    if missing:
        raise ValueError(f"Relations found in triples.tsv but missing in schema: {missing}")


def build_relation_mappings(rel_list_base, base_rel2type):
    """
    Create:
      rel2id / id2rel for base + reverse relations
      rel2type for base + reverse relations
    """
    validate_relation_schema(rel_list_base, base_rel2type)

    rel_list_full = rel_list_base + [r + "_REV" for r in rel_list_base]
    rel2id = {r: i for i, r in enumerate(rel_list_full)}
    id2rel = {i: r for r, i in rel2id.items()}
    rel2type = build_full_rel2type(base_rel2type)

    return rel2id, id2rel, rel2type


# =========================================================
# 4. REVERSE RELATION AUGMENTATION
# =========================================================
def add_reverse_triples_indexed(
    triples_indexed: List[Tuple[int, int, int]],
    id2rel: Dict[int, str],
    rel2id: Dict[str, int]
):
    """
    Given indexed triples (h, r_id, t) where r_id corresponds to base relation,
    add reverse triple (t, r_rev_id, h).
    """
    rev_triples = []
    for h, r, t in triples_indexed:
        r_str = id2rel[r]
        r_rev = r_str + "_REV"
        if r_rev not in rel2id:
            raise ValueError(f"Reverse relation not found in rel2id: {r_rev}")
        rev_triples.append((t, rel2id[r_rev], h))

    return list(triples_indexed) + rev_triples


def duplicate_weights_for_reverse(weights: List[float]):
    """Duplicate edge/triple weights so reverse triples keep the same weight."""
    return list(weights) + list(weights)


# =========================================================
# 5. EDGE WEIGHT UTILS
# =========================================================
def normalize_weight_list(
    weights: List[float],
    clamp_min: float = 1e-3,
    clamp_max: float = 10.0,
    power: float = 1.0
):
    """
    Normalize weights:
      - clamp_min to avoid zeros
      - optional power transform
      - divide by mean so average weight becomes 1
      - optional clamp_max to avoid extreme weights
    """
    w = torch.tensor(weights, dtype=torch.float)
    w = w.clamp_min(clamp_min)
    if power != 1.0:
        w = w.pow(power)
    w = w / w.mean().clamp_min(1e-8)
    if clamp_max is not None:
        w = w.clamp(max=clamp_max)
    return w.tolist()


def normalize_weight_tensor(
    weights: torch.Tensor,
    clamp_min: float = 1e-3,
    clamp_max: float = 10.0,
    power: float = 1.0
):
    """Tensor version of weight normalization (same logic as above)."""
    w = weights.float().clamp_min(clamp_min)
    if power != 1.0:
        w = w.pow(power)
    w = w / w.mean().clamp_min(1e-8)
    if clamp_max is not None:
        w = w.clamp(max=clamp_max)
    return w


# =========================================================
# 6. BUILD GRAPH
# =========================================================
def build_graph_from_raw(
    triples_raw: List[Tuple[int, str, int, float]],
    rel2id: Dict[str, int],
    add_reverse_edges: bool = True
):
    """
    Build PyG edge tensors from raw triples:
      edge_index: [2, E]
      edge_type : [E] relation ids
      edge_weight: [E] numeric weights
    Optionally adds reverse edges as separate relation types.
    """
    edge_index = []
    edge_type = []
    edge_weight = []

    for h, r, t, w in triples_raw:
        if r not in rel2id:
            continue

        # forward edge
        edge_index.append([h, t])
        edge_type.append(rel2id[r])
        edge_weight.append(float(w))

        # reverse edge (with its own relation id r+"_REV")
        if add_reverse_edges:
            r_rev = r + "_REV"
            if r_rev not in rel2id:
                raise ValueError(f"Missing reverse relation in rel2id: {r_rev}")
            edge_index.append([t, h])
            edge_type.append(rel2id[r_rev])
            edge_weight.append(float(w))

    if not edge_index:
        raise ValueError("No valid edges were built. Check triples and relation mappings.")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return edge_index, edge_type, edge_weight


# =========================================================
# 7. WEIGHTED RGCN LAYER
# =========================================================
class WeightedRGCNConv(MessagePassing):
    """
    R-GCN with:
      - basis decomposition for relation-specific weights (efficient when many relations)
      - edge weights in message passing
      - learnable relation-wise scaling of edge weights (positive via softplus)

    Message for relation r on edge j->i:
      m = (edge_weight_norm * rel_scale[r]) * (x_j W_r)
    where edge_weight_norm is normalized by incoming-weight sum at node i (per relation).
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

        # Basis matrices: [B, in, out]
        self.basis = nn.Parameter(torch.empty(self.num_bases, in_channels, out_channels))
        # Combination coefficients per relation: [R, B]
        self.att = nn.Parameter(torch.empty(num_relations, self.num_bases))

        # Optional self-loop/root transform
        self.root = nn.Parameter(torch.empty(in_channels, out_channels)) if root_weight else None
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Relation-wise scaling factor for edge weights (kept positive via softplus)
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor = None
    ):
        """
        Compute output node features given typed edges and weights.
        Implementation loops over relations (simple + clear).
        """
        num_nodes = x.size(0)
        out = x.new_zeros(num_nodes, self.out_channels)

        # If no weights provided, treat all edges equally
        if edge_weight is None:
            edge_weight = x.new_ones(edge_type.size(0))
        edge_weight = edge_weight.to(dtype=x.dtype, device=x.device)

        # Build full relation weight tensors W_r from basis decomposition:
        # weight: [R, in, out]
        weight = torch.matmul(
            self.att,
            self.basis.view(self.num_bases, -1)
        ).view(self.num_relations, self.in_channels, self.out_channels)

        # Process each relation separately (R-GCN standard approach)
        for rel in range(self.num_relations):
            rel_mask = (edge_type == rel)
            if not bool(rel_mask.any()):
                continue

            rel_edge_index = edge_index[:, rel_mask]
            rel_edge_weight = edge_weight[rel_mask]

            # Normalize by sum of incoming weights per destination node
            dst = rel_edge_index[1]
            denom = x.new_zeros(num_nodes)
            denom.index_add_(0, dst, rel_edge_weight)
            norm = 1.0 / denom[dst].clamp_min(1e-12)

            # Relation-specific scaling (positive)
            rel_scale = F.softplus(self.rel_edge_scale[rel])

            # Final message weight for each edge
            msg_weight = rel_edge_weight * norm * rel_scale

            # Apply relation-specific linear transform
            x_rel = x @ weight[rel]

            # Aggregate messages along edges of this relation
            out = out + self.propagate(
                rel_edge_index,
                x=x_rel,
                edge_weight=msg_weight,
                size=None
            )

        # Add transformed root/self-loop contribution
        if self.root is not None:
            out = out + x @ self.root

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor):
        """Message sent from source node j to target i, scaled by edge_weight."""
        return x_j * edge_weight.view(-1, 1)


# =========================================================
# 8. MODEL
# =========================================================
class RGCN(nn.Module):
    """2-layer weighted R-GCN encoder."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, dropout=0.2, num_bases=4):
        super().__init__()
        self.conv1 = WeightedRGCNConv(in_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = WeightedRGCNConv(hidden_dim, out_dim, num_relations, num_bases=num_bases)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type, edge_weight):
        x = self.conv1(x, edge_index, edge_type, edge_weight)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type, edge_weight)
        return x


class ComplExDecoder(nn.Module):
    """
    ComplEx link prediction decoder.
    Node embedding z is complex, stored as concatenated [Re, Im] => dim = 2*emb_dim.
    Relation embedding is also complex with same format.
    """
    def __init__(self, num_relations, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.rel = nn.Parameter(torch.empty(num_relations, 2 * emb_dim))
        nn.init.xavier_uniform_(self.rel)

    def forward(self, z, triples):
        """Return raw logits for triples [B,3] with columns (s, r, o)."""
        s = z[triples[:, 0]]
        r = self.rel[triples[:, 1]]
        o = z[triples[:, 2]]

        s_re, s_im = torch.chunk(s, 2, dim=-1)
        r_re, r_im = torch.chunk(r, 2, dim=-1)
        o_re, o_im = torch.chunk(o, 2, dim=-1)

        # Standard ComplEx score (trilinear product)
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
      input node features -> projection -> R-GCN encoder -> ComplEx decoder
    """
    def __init__(self, in_dim, hidden_dim, emb_dim, num_relations, dropout=0.2, num_bases=4):
        super().__init__()

        # Project input node embeddings into hidden size used by the GNN
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)

        # Encoder outputs complex embeddings of size 2*emb_dim
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
        z = self.rgcn(x, edge_index, edge_type, edge_weight)
        return z

    def score(self, z, triples):
        """Compute decoder logits for triples."""
        return self.decoder(z, triples)

    def forward(self, x, edge_index, edge_type, edge_weight, triples):
        """Convenience forward: encode then score."""
        z = self.encode(x, edge_index, edge_type, edge_weight)
        return self.score(z, triples)


# =========================================================
# 9. TYPE INDEX / TRUE TRIPLES
# =========================================================
def build_type_index(id2type):
    """Map node_type -> list[node_id] for typed negative sampling / candidate sets."""
    type_to_nodes = defaultdict(list)
    for nid, t in id2type.items():
        type_to_nodes[t].append(nid)
    return type_to_nodes


def build_true_triple_dict(triples):
    """
    Build lookup structures for filtered evaluation/sampling:
      true_tails[(h,r)] = set(all true tails)
      true_heads[(r,t)] = set(all true heads)
      true_triple_set = set((h,r,t))
    """
    true_tails = defaultdict(set)
    true_heads = defaultdict(set)
    true_triple_set = set()

    for h, r, t in triples:
        true_tails[(h, r)].add(t)
        true_heads[(r, t)].add(h)
        true_triple_set.add((h, r, t))

    return true_tails, true_heads, true_triple_set


# =========================================================
# 10. NEGATIVE SAMPLING
# =========================================================
def negative_sampling_filtered(
    triples,
    rel2type,
    id2rel,
    type_to_nodes,
    true_triple_set,
    num_neg_per_pos=8,
    max_tries=100
):
    """
    Filtered typed negative sampling:
      - For each positive (h,r,t), create negatives by corrupting head or tail
      - Sample replacement from correct node type based on rel2type schema
      - Reject any negative that already exists in true_triple_set
    """
    neg = []

    for h, r, t in triples:
        r_str = id2rel[r]
        head_type, tail_type = rel2type[r_str]

        for _ in range(num_neg_per_pos):
            corrupt_head = (random.random() < 0.5)
            sampled = False

            if corrupt_head:
                candidates = type_to_nodes[head_type]
                for _ in range(max_tries):
                    h_neg = random.choice(candidates)
                    if h_neg != h and (h_neg, r, t) not in true_triple_set:
                        neg.append([h_neg, r, t])
                        sampled = True
                        break

                # fallback: corrupt tail if head corruption fails
                if not sampled:
                    candidates = type_to_nodes[tail_type]
                    for _ in range(max_tries):
                        t_neg = random.choice(candidates)
                        if t_neg != t and (h, r, t_neg) not in true_triple_set:
                            neg.append([h, r, t_neg])
                            sampled = True
                            break

            else:
                candidates = type_to_nodes[tail_type]
                for _ in range(max_tries):
                    t_neg = random.choice(candidates)
                    if t_neg != t and (h, r, t_neg) not in true_triple_set:
                        neg.append([h, r, t_neg])
                        sampled = True
                        break

                # fallback: corrupt head if tail corruption fails
                if not sampled:
                    candidates = type_to_nodes[head_type]
                    for _ in range(max_tries):
                        h_neg = random.choice(candidates)
                        if h_neg != h and (h_neg, r, t) not in true_triple_set:
                            neg.append([h_neg, r, t])
                            sampled = True
                            break

    if len(neg) == 0:
        raise ValueError("No negative samples generated. Please check schema/data.")

    return torch.tensor(neg, dtype=torch.long)


# =========================================================
# 11. METRICS
# =========================================================
@torch.no_grad()
def evaluate_tail_prediction(
    model,
    x,
    edge_index,
    edge_type,
    edge_weight,
    eval_triples,
    all_true_triples,
    id2type,
    rel2type,
    id2rel,
    hits_ks=(1, 3, 10)
):
    """
    Filtered ranking evaluation for tail prediction:
      For each (h,r,t), score all candidate tails of the correct type.
      Remove other true tails for (h,r) (filtered setting).
      Rank is 1 + number of candidates with strictly higher score than the true tail.
    """
    model.eval()
    z = model.encode(x, edge_index, edge_type, edge_weight)

    true_tails, _, _ = build_true_triple_dict(all_true_triples)
    type_to_nodes = build_type_index(id2type)

    ranks = []
    rel_ranks = defaultdict(list)

    eval_triples = [tuple(tri) for tri in eval_triples]

    for h, r, t in eval_triples:
        r_str = id2rel[r]
        if r_str not in rel2type:
            continue

        _, tail_type = rel2type[r_str]
        candidate_tails = type_to_nodes[tail_type]

        triples_tail = []
        filtered_tail_ids = []

        # Build filtered candidate list for this query (h,r,?)
        for cand_t in candidate_tails:
            # Skip any other true tail besides the target
            if cand_t != t and cand_t in true_tails[(h, r)]:
                continue
            triples_tail.append([h, r, cand_t])
            filtered_tail_ids.append(cand_t)

        if len(triples_tail) == 0:
            continue

        triples_tail = torch.tensor(triples_tail, dtype=torch.long, device=z.device)
        tail_scores = model.score(z, triples_tail)

        # Ensure the target tail is still in candidates
        if t not in filtered_tail_ids:
            continue

        target_idx = filtered_tail_ids.index(t)
        target_score = tail_scores[target_idx]
        rank_tail = int((tail_scores > target_score).sum().item()) + 1

        ranks.append(rank_tail)
        rel_ranks[r_str].append(rank_tail)

    def calc_metrics(rank_list):
        """Compute MRR and Hits@K from a list of ranks."""
        if len(rank_list) == 0:
            return {
                "MRR": 0.0,
                "Count": 0,
                "Hits@1": 0.0,
                "Hits@3": 0.0,
                "Hits@10": 0.0,
            }

        rank_tensor = torch.tensor(rank_list, dtype=torch.float)
        metrics = {
            "MRR": torch.mean(1.0 / rank_tensor).item(),
            "Count": len(rank_list)
        }
        for k in hits_ks:
            metrics[f"Hits@{k}"] = torch.mean((rank_tensor <= k).float()).item()
        return metrics

    overall_metrics = calc_metrics(ranks)
    relation_metrics = {rel: calc_metrics(rks) for rel, rks in rel_ranks.items()}
    return overall_metrics, relation_metrics


def focus_mrr(relation_metrics, focus_relations=("CAUSES", "LEADS_TO")):
    """Average MRR over a subset of relations for early stopping / LR scheduling."""
    vals = []
    for r in focus_relations:
        if r in relation_metrics and relation_metrics[r]["Count"] > 0:
            vals.append(relation_metrics[r]["MRR"])
    return sum(vals) / len(vals) if vals else 0.0


# =========================================================
# 12. SAVE / LOAD
# =========================================================
def save_checkpoint(model, optimizer, scheduler, epoch, best_score, path):
    """Save a training checkpoint (model + optimizer + scheduler + metadata)."""
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_score": best_score,
    }
    torch.save(ckpt, path)
    print(f"Best model saved to {path}")


def load_model(path, in_dim, hidden_dim, emb_dim, num_relations, dropout=0.2, num_bases=4):
    """Load a saved model checkpoint for inference/evaluation."""
    model = Model(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        num_relations=num_relations,
        dropout=dropout,
        num_bases=num_bases
    ).to(DEVICE)

    ckpt = torch.load(path, map_location=DEVICE)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"Model loaded from {path}")
    return model


# =========================================================
# 13. TRAIN
# =========================================================
def train(
    model,
    x,
    edge_index,
    edge_type,
    edge_weight,
    train_triples,
    train_weights,
    valid_triples_forward_only,
    train_true_triples,
    all_true_triples_for_eval,
    id2type,
    rel2type,
    id2rel,
    epochs=300,
    lr=0.001,
    weight_decay=1e-4,
    num_neg_per_pos=8,
    patience=15,
    save_path="rgcn_complex_best.pt",
    focus_relations=("CAUSES", "LEADS_TO")
):
    """
    Training loop:
      - Generate filtered typed negatives each epoch
      - Encode once per epoch, score positives and negatives
      - Use adversarial negative weighting (softmax over negative scores)
      - Weighted loss: per-triple weight * per-relation mask/weight
      - Early stopping based on focus-relations MRR on validation
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    type_to_nodes = build_type_index(id2type)
    train_true_set = set(train_true_triples)

    # Cache training positives and their weights as tensors
    train_triples_tensor = torch.tensor(train_triples, dtype=torch.long, device=DEVICE)
    train_loss_weight = torch.tensor(train_weights, dtype=torch.float, device=DEVICE)
    train_loss_weight = train_loss_weight / train_loss_weight.mean().clamp_min(1e-8)

    best_score = -1.0
    best_state = None
    bad_epochs = 0

    # Optional: ignore some relations by setting weight to 0 in the loss
    rel_weights = {
        "CAUSES": 1.0,
        "LEADS_TO": 1.0,
        "HAS_MODE": 0.0,
        "HAS_FUNCTION": 0.0,
        "CAUSES_REV": 1.0,
        "LEADS_TO_REV": 1.0,
        "HAS_MODE_REV": 0.0,
        "HAS_FUNCTION_REV": 0.0,
    }

    for epoch in range(1, epochs + 1):
        model.train()

        # Dynamic negatives each epoch (filtered + typed)
        neg_triples = negative_sampling_filtered(
            triples=train_triples,
            rel2type=rel2type,
            id2rel=id2rel,
            type_to_nodes=type_to_nodes,
            true_triple_set=train_true_set,
            num_neg_per_pos=num_neg_per_pos
        ).to(DEVICE)

        # Encode graph once per epoch
        z = model.encode(x, edge_index, edge_type, edge_weight)

        # Score positives and negatives
        pos_score = model.score(z, train_triples_tensor)
        neg_score = model.score(z, neg_triples)

        # Reshape negatives to [num_pos, num_neg_per_pos]
        pos_count = train_triples_tensor.size(0)
        neg_score = neg_score.view(pos_count, num_neg_per_pos)

        # Self-adversarial negative sampling weights (hard negatives get higher weight)
        adv_temp = 1.0
        neg_weight = F.softmax(neg_score.detach() * adv_temp, dim=1)

        # Relation-level weighting/masking
        train_rel_weight = torch.tensor(
            [rel_weights.get(id2rel[r], 1.0) for (_, r, _) in train_triples],
            dtype=torch.float,
            device=DEVICE
        )

        # Final per-sample weight (relation weight * triple weight)
        if USE_LOSS_WEIGHT:
            final_sample_weight = train_rel_weight * train_loss_weight
        else:
            final_sample_weight = train_rel_weight

        # Logistic loss for link prediction (pos and neg)
        pos_loss_each = -F.logsigmoid(pos_score)
        neg_loss_each = -(neg_weight * F.logsigmoid(-neg_score)).sum(dim=1)

        pos_loss = (pos_loss_each * final_sample_weight).mean()
        neg_loss = (neg_loss_each * final_sample_weight).mean()

        # Regularization to avoid exploding embeddings
        reg_loss = model.decoder.rel.pow(2).mean() + z.pow(2).mean()

        # Regularize relation-wise edge scaling towards 1 (after softplus)
        edge_scale_reg = 0.0
        for module in model.modules():
            if isinstance(module, WeightedRGCNConv):
                edge_scale_reg = edge_scale_reg + (F.softplus(module.rel_edge_scale) - 1.0).pow(2).mean()

        # Total loss
        loss = pos_loss + neg_loss + 1e-4 * reg_loss + 1e-4 * edge_scale_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validate using filtered tail prediction ranking (forward relations only)
        valid_metrics, valid_rel_metrics = evaluate_tail_prediction(
            model=model,
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
            eval_triples=valid_triples_forward_only,
            all_true_triples=all_true_triples_for_eval,
            id2type=id2type,
            rel2type=rel2type,
            id2rel=id2rel,
            hits_ks=(1, 3, 10)
        )

        # Early stopping uses focus relations only
        val_focus_score = focus_mrr(valid_rel_metrics, focus_relations=focus_relations)
        scheduler.step(val_focus_score)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"Loss = {loss.item():.4f} | "
            f"Val Overall MRR = {valid_metrics['MRR']:.4f} | "
            f"Val Focus MRR = {val_focus_score:.4f} | "
            f"Hits@1 = {valid_metrics['Hits@1']:.4f} | "
            f"Hits@3 = {valid_metrics['Hits@3']:.4f} | "
            f"Hits@10 = {valid_metrics['Hits@10']:.4f} | "
            f"LR = {current_lr:.6f}"
        )

        # Track best model by focus MRR
        if val_focus_score > best_score:
            best_score = val_focus_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(model, optimizer, scheduler, epoch, best_score, save_path)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Restore best parameters
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =========================================================
# 14. INFERENCE HELPERS
# =========================================================
@torch.no_grad()
def predict_score(model, x, edge_index, edge_type, edge_weight, triple_tensor):
    """Return sigmoid probability for a batch of triples."""
    model.eval()
    z = model.encode(x, edge_index, edge_type, edge_weight)
    logit = model.score(z, triple_tensor.to(DEVICE))
    prob = torch.sigmoid(logit)
    return prob


@torch.no_grad()
def predict_tail(
    model,
    x,
    edge_index,
    edge_type,
    edge_weight,
    head_id,
    relation_id,
    candidate_tail_ids,
    top_k=10
):
    """
    Rank candidate tails for a fixed (head, relation, ?).
    Returns list of (tail_id, probability) sorted by score descending (top_k).
    """
    model.eval()
    z = model.encode(x, edge_index, edge_type, edge_weight)

    triples = torch.tensor(
        [[head_id, relation_id, tail_id] for tail_id in candidate_tail_ids],
        dtype=torch.long,
        device=DEVICE
    )

    scores = torch.sigmoid(model.score(z, triples))
    top_k = min(top_k, len(candidate_tail_ids))
    values, indices = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        results.append((candidate_tail_ids[idx], score))

    return results


@torch.no_grad()
def predict_head(
    model,
    x,
    edge_index,
    edge_type,
    edge_weight,
    relation_id,
    tail_id,
    candidate_head_ids,
    top_k=10
):
    """
    Rank candidate heads for a fixed (?, relation, tail).
    Returns list of (head_id, probability) sorted by score descending (top_k).
    """
    model.eval()
    z = model.encode(x, edge_index, edge_type, edge_weight)

    triples = torch.tensor(
        [[head_id, relation_id, tail_id] for head_id in candidate_head_ids],
        dtype=torch.long,
        device=DEVICE
    )

    scores = torch.sigmoid(model.score(z, triples))
    top_k = min(top_k, len(candidate_head_ids))
    values, indices = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        results.append((candidate_head_ids[idx], score))

    return results


def print_candidate_stats(eval_triples, id2type, rel2type, id2rel):
    """Print average candidate tail-space size per relation (by required tail type)."""
    type_to_nodes = build_type_index(id2type)
    rel_counts = defaultdict(list)

    for h, r, t in eval_triples:
        r_str = id2rel[r]
        _, tail_type = rel2type[r_str]
        rel_counts[r_str].append(len(type_to_nodes[tail_type]))

    print("\n===== Candidate Space Stats =====")
    for rel, vals in rel_counts.items():
        avg_cands = sum(vals) / len(vals)
        print(f"{rel}: avg tail candidates = {avg_cands:.2f}, samples = {len(vals)}")


# =========================================================
# 15. MAIN
# =========================================================
def main():
    set_seed(SEED)

    # -------------------------
    # 1) Load nodes and raw triples
    # -------------------------
    node2id, id2node, id2type, x = load_nodes(NODES_FILE)
    triples_raw, rel_list_base = load_triples(TRIPLES_FILE, node2id)

    print(f"Loaded {len(node2id)} nodes")
    print(f"Loaded {len(triples_raw)} raw triples")
    print(f"Relations in raw data: {rel_list_base}")

    # -------------------------
    # 2) Define base relation schema (types used for candidate filtering + negative sampling)
    # -------------------------
    base_rel2type = {
        "CAUSES": ("Cause", "Mode"),
        "LEADS_TO": ("Mode", "Effect"),
        "HAS_FUNCTION": ("Element", "Function"),
        "HAS_MODE": ("Function", "Mode"),
    }

    # -------------------------
    # 3) Build relation mappings including reverse relations
    # -------------------------
    rel2id, id2rel, rel2type = build_relation_mappings(rel_list_base, base_rel2type)

    print("\nFull relation set used by model:")
    for r, rid in rel2id.items():
        print(f"  {rid:02d} -> {r}")

    # -------------------------
    # 4) Split triples (base relations only) into train/valid/test
    # -------------------------
    train_raw, valid_raw, test_raw = split_triples_by_relation(
        triples_raw,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        seed=SEED
    )

    print(f"\nTrain raw triples: {len(train_raw)}")
    print(f"Valid raw triples: {len(valid_raw)}")
    print(f"Test raw triples : {len(test_raw)}")

    # -------------------------
    # 5) Convert raw triples to indexed triples (base relations only)
    # -------------------------
    train_triples_base = [(h, rel2id[r], t) for (h, r, t, w) in train_raw]
    valid_triples_base = [(h, rel2id[r], t) for (h, r, t, w) in valid_raw]
    test_triples_base = [(h, rel2id[r], t) for (h, r, t, w) in test_raw]

    train_weights_base = [float(w) for (_, _, _, w) in train_raw]

    # -------------------------
    # 6) Add reverse triples for training and for "all_true" filtering
    # -------------------------
    train_triples_full = add_reverse_triples_indexed(train_triples_base, id2rel, rel2id)
    valid_triples_full = add_reverse_triples_indexed(valid_triples_base, id2rel, rel2id)
    test_triples_full = add_reverse_triples_indexed(test_triples_base, id2rel, rel2id)

    train_weights_full = duplicate_weights_for_reverse(train_weights_base)

    all_true_base = train_triples_base + valid_triples_base + test_triples_base
    all_true_full = add_reverse_triples_indexed(all_true_base, id2rel, rel2id)

    train_true_triples_full = train_triples_full

    print(f"\nTrain triples for loss (base + rev): {len(train_triples_full)}")
    print(f"Valid triples full (base + rev):     {len(valid_triples_full)}")
    print(f"Test triples full (base + rev):      {len(test_triples_full)}")

    # -------------------------
    # 7) Build graph edges for the GNN and normalize edge weights
    #    NOTE: This uses triples_raw (train+valid+test). If you want "train only",
    #    pass train_raw instead.
    # -------------------------
    edge_index, edge_type, edge_weight = build_graph_from_raw(
        triples_raw,
        rel2id,
        add_reverse_edges=True
    )

    edge_weight = normalize_weight_tensor(
        edge_weight,
        clamp_min=GRAPH_WEIGHT_CLAMP_MIN,
        clamp_max=GRAPH_WEIGHT_CLAMP_MAX,
        power=GRAPH_WEIGHT_POWER
    )

    # Move tensors to GPU/CPU device
    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_type = edge_type.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

    print(f"\nGraph edges used by RGCN (with reverse): {edge_index.shape[1]}")
    print(
        f"Graph edge weight stats: min={edge_weight.min().item():.4f}, "
        f"max={edge_weight.max().item():.4f}, mean={edge_weight.mean().item():.4f}"
    )

    # -------------------------
    # 8) Build model
    # -------------------------
    model = Model(
        in_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
        emb_dim=EMB_DIM,
        num_relations=len(rel2id),
        dropout=DROPOUT,
        num_bases=NUM_BASES
    ).to(DEVICE)

    print("\nModel:")
    print(model)

    # -------------------------
    # 9) Train model with early stopping on focus-relations MRR
    # -------------------------
    model = train(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        train_triples=train_triples_full,
        train_weights=train_weights_full,
        valid_triples_forward_only=valid_triples_base,  # evaluate only forward relations
        train_true_triples=train_true_triples_full,
        all_true_triples_for_eval=all_true_full,        # for filtered ranking
        id2type=id2type,
        rel2type=rel2type,
        id2rel=id2rel,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        num_neg_per_pos=NUM_NEG_PER_POS,
        patience=PATIENCE,
        save_path=SAVE_PATH,
        focus_relations=FOCUS_RELATIONS
    )

    print("\nTraining finished.")

    # -------------------------
    # 10) Final evaluation (forward-only)
    # -------------------------
    print("\n===== VALID SET EVALUATION (FORWARD ONLY) =====")
    valid_metrics, valid_rel_metrics = evaluate_tail_prediction(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        eval_triples=valid_triples_base,
        all_true_triples=all_true_full,
        id2type=id2type,
        rel2type=rel2type,
        id2rel=id2rel,
        hits_ks=(1, 3, 10)
    )
    print("Overall Valid Metrics:", valid_metrics)
    print("Per-relation Valid Metrics:")
    for rel, metrics in valid_rel_metrics.items():
        print(rel, metrics)

    print("\n===== TEST SET EVALUATION (FORWARD ONLY) =====")
    test_metrics, test_rel_metrics = evaluate_tail_prediction(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        eval_triples=test_triples_base,
        all_true_triples=all_true_full,
        id2type=id2type,
        rel2type=rel2type,
        id2rel=id2rel,
        hits_ks=(1, 3, 10)
    )
    print("Overall Test Metrics:", test_metrics)
    print("Per-relation Test Metrics:")
    for rel, metrics in test_rel_metrics.items():
        print(rel, metrics)

    # -------------------------
    # 11) Optional evaluation using base + reverse triples as queries
    # -------------------------
    print("\n===== VALID SET EVALUATION (BASE + REV) =====")
    valid_metrics_full, valid_rel_metrics_full = evaluate_tail_prediction(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        eval_triples=valid_triples_full,
        all_true_triples=all_true_full,
        id2type=id2type,
        rel2type=rel2type,
        id2rel=id2rel,
        hits_ks=(1, 3, 10)
    )
    print("Overall Valid Full Metrics:", valid_metrics_full)
    print("Per-relation Valid Full Metrics:")
    for rel, metrics in valid_rel_metrics_full.items():
        print(rel, metrics)

    print("\n===== TEST SET EVALUATION (BASE + REV) =====")
    test_metrics_full, test_rel_metrics_full = evaluate_tail_prediction(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        eval_triples=test_triples_full,
        all_true_triples=all_true_full,
        id2type=id2type,
        rel2type=rel2type,
        id2rel=id2rel,
        hits_ks=(1, 3, 10)
    )
    print("Overall Test Full Metrics:", test_metrics_full)
    print("Per-relation Test Full Metrics:")
    for rel, metrics in test_rel_metrics_full.items():
        print(rel, metrics)

    # Print how large the candidate sets are per relation/type
    print_candidate_stats(test_triples_base, id2type, rel2type, id2rel)


if __name__ == "__main__":
    main()