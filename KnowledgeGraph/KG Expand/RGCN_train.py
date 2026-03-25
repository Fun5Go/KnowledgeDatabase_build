import os
import ast
import json
import random
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


# =========================================================
# 0. CONFIG
# =========================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NODES_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\nodes.tsv"
TRIPLES_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\triples.tsv"
SAVE_PATH = "rgcn_weighted_best_model.pt"


# =========================================================
# 1. LOAD DATA
# =========================================================
def load_nodes(node_file):
    df = pd.read_csv(node_file, sep="\t")

    required_cols = {"node_id", "node_type", "embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in node file: {missing}")

    node_ids = df["node_id"].tolist()
    node_types = df["node_type"].tolist()

    embeddings = []
    for emb_str in df["embedding"]:
        emb = torch.tensor(ast.literal_eval(emb_str), dtype=torch.float)
        embeddings.append(emb)

    x = torch.stack(embeddings)
    x = F.normalize(x, p=2, dim=1)

    node2id = {nid: i for i, nid in enumerate(node_ids)}
    id2node = {i: nid for nid, i in node2id.items()}
    id2type = {node2id[nid]: t for nid, t in zip(node_ids, node_types)}

    return node2id, id2node, id2type, x


def load_triples(triple_file, node2id):
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

        if h not in node2id or t not in node2id:
            continue

        triples.append((node2id[h], r, node2id[t], w))
        relations.add(r)

    return triples, sorted(list(relations))


# =========================================================
# 2. DATA SPLIT
# =========================================================
def split_triples_by_relation(triples_raw, train_ratio=0.9, valid_ratio=0.05, seed=42):
    random.seed(seed)

    rel_buckets = defaultdict(list)
    for triple in triples_raw:
        rel_buckets[triple[1]].append(triple)  # triple[1] = relation string

    train_triples, valid_triples, test_triples = [], [], []

    for r, bucket in rel_buckets.items():
        random.shuffle(bucket)
        n = len(bucket)

        n_train = max(1, int(n * train_ratio))
        n_valid = max(1, int(n * valid_ratio)) if n >= 10 else max(0, int(n * valid_ratio))

        if n_train + n_valid >= n:
            n_valid = max(0, n - n_train - 1)

        train_triples.extend(bucket[:n_train])
        valid_triples.extend(bucket[n_train:n_train + n_valid])
        test_triples.extend(bucket[n_train + n_valid:])

    return train_triples, valid_triples, test_triples


# =========================================================
# 3. BUILD GRAPH
# =========================================================
def build_graph(triples_raw, rel2id):
    edge_index = []
    edge_type = []
    edge_weight = []

    for h, r, t, w in triples_raw:
        if r not in rel2id:
            continue

        r_rev = r + "_REV"
        if r_rev not in rel2id:
            continue

        edge_index.append([h, t])
        edge_type.append(rel2id[r])
        edge_weight.append(float(w))

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
# 4. MODEL
# =========================================================
class RGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, dropout=0.2, num_bases=4):
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
    z: [num_nodes, 2 * emb_dim]
    relation emb: [num_relations, 2 * emb_dim]
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
    def __init__(self, in_dim, hidden_dim, emb_dim, num_relations, dropout=0.2, num_bases=4):
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


# =========================================================
# 5. RELATION SCHEMA
# =========================================================
def build_full_rel2type(base_rel2type):
    rel2type_full = {}
    for r, (h_type, t_type) in base_rel2type.items():
        rel2type_full[r] = (h_type, t_type)
        rel2type_full[r + "_REV"] = (t_type, h_type)
    return rel2type_full


def validate_relation_schema(rel_list_base, rel2type_base):
    missing = [r for r in rel_list_base if r not in rel2type_base]
    if missing:
        raise ValueError(
            f"The following relations exist in triples.tsv but are missing in rel2type schema: {missing}"
        )


# =========================================================
# 6. TYPE INDEX / TRUE TRIPLES
# =========================================================
def build_type_index(id2type):
    type_to_nodes = defaultdict(list)
    for nid, t in id2type.items():
        type_to_nodes[t].append(nid)
    return type_to_nodes


def build_true_triple_dict(triples):
    true_tails = defaultdict(set)
    true_heads = defaultdict(set)
    true_triple_set = set()

    for h, r, t in triples:
        true_tails[(h, r)].add(t)
        true_heads[(r, t)].add(h)
        true_triple_set.add((h, r, t))

    return true_tails, true_heads, true_triple_set


# =========================================================
# 7. NEGATIVE SAMPLING
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
    neg = []

    for h, r, t in triples:
        r_str = id2rel[r]
        head_type, tail_type = rel2type[r_str]

        for _ in range(num_neg_per_pos):
            corrupt_head = (random.random() < 0.5)

            if corrupt_head:
                candidates = type_to_nodes[head_type]
                sampled = False
                for _ in range(max_tries):
                    h_neg = random.choice(candidates)
                    if h_neg != h and (h_neg, r, t) not in true_triple_set:
                        neg.append([h_neg, r, t])
                        sampled = True
                        break

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
                sampled = False
                for _ in range(max_tries):
                    t_neg = random.choice(candidates)
                    if t_neg != t and (h, r, t_neg) not in true_triple_set:
                        neg.append([h, r, t_neg])
                        sampled = True
                        break

                if not sampled:
                    candidates = type_to_nodes[head_type]
                    for _ in range(max_tries):
                        h_neg = random.choice(candidates)
                        if h_neg != h and (h_neg, r, t) not in true_triple_set:
                            neg.append([h_neg, r, t])
                            sampled = True
                            break

    if len(neg) == 0:
        raise ValueError("No negative samples generated. Please check your schema/data.")

    return torch.tensor(neg, dtype=torch.long)


# =========================================================
# 8. METRICS
# =========================================================
@torch.no_grad()
def evaluate_tail_prediction(
    model,
    x,
    edge_index,
    edge_type,
    eval_triples,
    all_true_triples,
    id2type,
    rel2type,
    id2rel,
    hits_ks=(1, 3, 10)
):
    model.eval()
    z = model.encode(x, edge_index, edge_type)

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

        for cand_t in candidate_tails:
            if cand_t != t and cand_t in true_tails[(h, r)]:
                continue
            triples_tail.append([h, r, cand_t])
            filtered_tail_ids.append(cand_t)

        if len(triples_tail) == 0:
            continue

        triples_tail = torch.tensor(triples_tail, dtype=torch.long, device=z.device)
        tail_scores = model.score(z, triples_tail)

        if t not in filtered_tail_ids:
            continue

        target_idx = filtered_tail_ids.index(t)
        target_score = tail_scores[target_idx]
        rank_tail = int((tail_scores > target_score).sum().item()) + 1

        ranks.append(rank_tail)
        rel_ranks[r_str].append(rank_tail)

    def calc_metrics(rank_list):
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


def macro_mrr_of_relations(relation_metrics, focus_relations):
    vals = []
    for r in focus_relations:
        if r in relation_metrics and relation_metrics[r]["Count"] > 0:
            vals.append(relation_metrics[r]["MRR"])
    return sum(vals) / len(vals) if vals else 0.0


# =========================================================
# 9. SAVE / LOAD
# =========================================================
def save_checkpoint(model, optimizer, scheduler, epoch, best_score, path):
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
# 10. TRAIN
# =========================================================
def train(
    model,
    x,
    edge_index,
    edge_type,
    train_triples,
    train_weights,
    valid_triples,
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
    save_path="rgcn_weighted_best_model.pt",
    focus_relations=("CAUSES", "LEADS_TO")
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    type_to_nodes = build_type_index(id2type)
    train_true_set = set(train_true_triples)

    train_triples_tensor = torch.tensor(train_triples, dtype=torch.long, device=DEVICE)
    train_edge_weight = torch.tensor(train_weights, dtype=torch.float, device=DEVICE)

    # normalize around mean=1
    train_edge_weight = train_edge_weight / train_edge_weight.mean().clamp_min(1e-8)

    best_score = -1.0
    best_state = None
    bad_epochs = 0

    rel_weights = {
        "CAUSES": 2.0,
        "LEADS_TO": 1.5,
        "HAS_MODE": 1.0,
        "HAS_FUNCTION": 1.0,
        "CAUSES_REV": 1.0,
        "LEADS_TO_REV": 1.0,
        "HAS_MODE_REV": 1.0,
        "HAS_FUNCTION_REV": 1.0,
    }

    for epoch in range(1, epochs + 1):
        model.train()

        neg_triples = negative_sampling_filtered(
            triples=train_triples,
            rel2type=rel2type,
            id2rel=id2rel,
            type_to_nodes=type_to_nodes,
            true_triple_set=train_true_set,
            num_neg_per_pos=num_neg_per_pos
        ).to(DEVICE)

        z = model.encode(x, edge_index, edge_type)

        pos_score = model.score(z, train_triples_tensor)
        neg_score = model.score(z, neg_triples)

        pos_count = train_triples_tensor.size(0)
        neg_score = neg_score.view(pos_count, num_neg_per_pos)

        adv_temp = 1.0
        neg_weight = F.softmax(neg_score.detach() * adv_temp, dim=1)

        train_rel_weight = torch.tensor(
            [rel_weights[id2rel[r]] for (_, r, _) in train_triples],
            dtype=torch.float,
            device=DEVICE
        )

        final_sample_weight = train_rel_weight * train_edge_weight

        pos_loss_each = -F.logsigmoid(pos_score)
        neg_loss_each = -(neg_weight * F.logsigmoid(-neg_score)).sum(dim=1)

        pos_loss = (pos_loss_each * final_sample_weight).mean()
        neg_loss = (neg_loss_each * final_sample_weight).mean()

        reg_loss = model.decoder.rel.pow(2).mean() + z.pow(2).mean()

        loss = pos_loss + neg_loss + 1e-4 * reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        valid_metrics, valid_rel_metrics = evaluate_tail_prediction(
            model=model,
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            eval_triples=valid_triples,
            all_true_triples=all_true_triples_for_eval,
            id2type=id2type,
            rel2type=rel2type,
            id2rel=id2rel,
            hits_ks=(1, 3, 10)
        )

        val_focus_score = (
            0.5 * valid_rel_metrics.get("CAUSES", {}).get("MRR", 0.0) +
            0.5 * valid_rel_metrics.get("LEADS_TO", {}).get("MRR", 0.0)
        )
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

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =========================================================
# 11. INFERENCE HELPERS
# =========================================================
@torch.no_grad()
def predict_score(model, x, edge_index, edge_type, triple_tensor):
    model.eval()
    z = model.encode(x, edge_index, edge_type)
    logit = model.score(z, triple_tensor.to(DEVICE))
    prob = torch.sigmoid(logit)
    return prob


@torch.no_grad()
def predict_tail(
    model,
    x,
    edge_index,
    edge_type,
    head_id,
    relation_id,
    candidate_tail_ids,
    top_k=10
):
    model.eval()
    z = model.encode(x, edge_index, edge_type)

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
    relation_id,
    tail_id,
    candidate_head_ids,
    top_k=10
):
    model.eval()
    z = model.encode(x, edge_index, edge_type)

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
# 12. MAIN
# =========================================================
def main():
    # -------------------------
    # 1) Load data
    # -------------------------
    node2id, id2node, id2type, x = load_nodes(NODES_FILE)
    triples_raw, rel_list_base = load_triples(TRIPLES_FILE, node2id)

    print(f"Loaded {len(node2id)} nodes")
    print(f"Loaded {len(triples_raw)} triples")
    print(f"Relations in data: {rel_list_base}")

    # -------------------------
    # 2) Define base relation schema
    # -------------------------
    base_rel2type = {
        "CAUSES": ("Cause", "Mode"),
        "LEADS_TO": ("Mode", "Effect"),
        "HAS_FUNCTION": ("Element", "Function"),
        "HAS_MODE": ("Function", "Mode"),
    }

    validate_relation_schema(rel_list_base, base_rel2type)

    # -------------------------
    # 3) Build relation mappings
    # -------------------------
    rel_list_full = rel_list_base + [r + "_REV" for r in rel_list_base]
    rel2id = {r: i for i, r in enumerate(rel_list_full)}
    id2rel = {i: r for r, i in rel2id.items()}
    rel2type = build_full_rel2type(base_rel2type)

    # -------------------------
    # 4) Split raw triples
    # -------------------------
    train_raw, valid_raw, test_raw = split_triples_by_relation(
        triples_raw,
        train_ratio=0.9,
        valid_ratio=0.05,
        seed=SEED
    )

    print(f"Train raw triples: {len(train_raw)}")
    print(f"Valid raw triples: {len(valid_raw)}")
    print(f"Test raw triples : {len(test_raw)}")

    # indexed triples for scoring/eval
    train_triples = [(h, rel2id[r], t) for (h, r, t, w) in train_raw]
    valid_triples = [(h, rel2id[r], t) for (h, r, t, w) in valid_raw]
    test_triples = [(h, rel2id[r], t) for (h, r, t, w) in test_raw]

    # edge weights for training loss
    train_weights = [float(w) for (h, r, t, w) in train_raw]

    # filtered evaluation uses all true triples
    all_true_triples = train_triples + valid_triples + test_triples

    # negative sampling only uses train truths
    train_true_triples = train_triples

    # -------------------------
    # 5) Build graph ONLY from train triples
    # -------------------------
    edge_index, edge_type, edge_weight = build_graph(triples_raw, rel2id)

    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_type = edge_type.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

    print(f"Graph edges (with reverse): {edge_index.shape[1]}")
    print(f"Edge weight stats: min={edge_weight.min().item():.4f}, "
          f"max={edge_weight.max().item():.4f}, mean={edge_weight.mean().item():.4f}")

    # -------------------------
    # 6) Build model
    # -------------------------
    model = Model(
        in_dim=x.shape[1],
        hidden_dim=256,
        emb_dim=256,
        num_relations=len(rel2id),
        dropout=0.1,
        num_bases=4
    ).to(DEVICE)

    print(model)

    # -------------------------
    # 7) Train
    # -------------------------
    model = train(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        train_triples=train_triples,
        train_weights=train_weights,
        valid_triples=valid_triples,
        train_true_triples=train_true_triples,
        all_true_triples_for_eval=all_true_triples,
        id2type=id2type,
        rel2type=rel2type,
        id2rel=id2rel,
        epochs=300,
        lr=0.001,
        weight_decay=1e-4,
        num_neg_per_pos=8,
        patience=15,
        save_path=SAVE_PATH,
        focus_relations=("CAUSES", "LEADS_TO")
    )

    print("Training finished.")

    # -------------------------
    # 8) Final evaluation
    # -------------------------
    print("\n===== VALID SET EVALUATION =====")
    valid_metrics, valid_rel_metrics = evaluate_tail_prediction(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        eval_triples=valid_triples,
        all_true_triples=all_true_triples,
        id2type=id2type,
        rel2type=rel2type,
        id2rel=id2rel,
        hits_ks=(1, 3, 10)
    )
    print("Overall Valid Metrics:", valid_metrics)
    print("Per-relation Valid Metrics:")
    for rel, metrics in valid_rel_metrics.items():
        print(rel, metrics)

    print("\n===== TEST SET EVALUATION =====")
    test_metrics, test_rel_metrics = evaluate_tail_prediction(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        eval_triples=test_triples,
        all_true_triples=all_true_triples,
        id2type=id2type,
        rel2type=rel2type,
        id2rel=id2rel,
        hits_ks=(1, 3, 10)
    )
    print("Overall Test Metrics:", test_metrics)
    print("Per-relation Test Metrics:")
    for rel, metrics in test_rel_metrics.items():
        print(rel, metrics)

    print_candidate_stats(test_triples, id2type, rel2type, id2rel)


if __name__ == "__main__":
    main()