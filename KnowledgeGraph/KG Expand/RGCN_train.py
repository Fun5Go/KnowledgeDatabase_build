import ast
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


# =========================
# 1. LOAD DATA
# =========================

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

    node2id = {nid: i for i, nid in enumerate(node_ids)}
    id2type = {node2id[nid]: t for nid, t in zip(node_ids, node_types)}

    return node2id, id2type, x


def load_triples(triple_file, node2id):
    df = pd.read_csv(triple_file, sep="\t")

    required_cols = {"head", "relation", "tail"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in triple file: {missing}")

    triples = []
    relations = set()

    for _, row in df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]

        if h not in node2id or t not in node2id:
            continue

        triples.append((node2id[h], r, node2id[t]))
        relations.add(r)

    return triples, sorted(list(relations))


# =========================
# 2. BUILD GRAPH
# =========================

def build_graph(triples_raw, rel2id):
    edge_index = []
    edge_type = []

    for h, r, t in triples_raw:
        if r not in rel2id:
            continue
        r_rev = r + "_REV"
        if r_rev not in rel2id:
            continue

        edge_index.append([h, t])
        edge_type.append(rel2id[r])

        edge_index.append([t, h])
        edge_type.append(rel2id[r_rev])

    if not edge_index:
        raise ValueError("No valid edges were built. Check triples and relation mappings.")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    return edge_index, edge_type


# =========================
# 3. MODEL
# =========================

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

    def forward(self, x, edge_index, edge_type, triples):
        z = self.encode(x, edge_index, edge_type)
        return self.score(z, triples)


# =========================
# 4. RELATION SCHEMA
# =========================

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


# =========================
# 5. TYPED NEGATIVE SAMPLING
# =========================

def build_type_index(id2type):
    type_to_nodes = {}
    for nid, t in id2type.items():
        type_to_nodes.setdefault(t, []).append(nid)
    return type_to_nodes


def negative_sampling(triples, id2type, rel2type, id2rel, type_to_nodes):
    neg = []

    for h, r, t in triples:
        r_str = id2rel[r]

        if r_str not in rel2type:
            raise KeyError(f"Relation '{r_str}' not found in rel2type.")

        head_type, tail_type = rel2type[r_str]

        if head_type not in type_to_nodes:
            raise KeyError(f"Head type '{head_type}' not found in node types.")
        if tail_type not in type_to_nodes:
            raise KeyError(f"Tail type '{tail_type}' not found in node types.")

        if random.random() < 0.5:
            candidates = type_to_nodes[head_type]
            if len(candidates) == 0:
                h_neg = h
            else:
                h_neg = random.choice(candidates)
            neg.append([h_neg, r, t])
        else:
            candidates = type_to_nodes[tail_type]
            if len(candidates) == 0:
                t_neg = t
            else:
                t_neg = random.choice(candidates)
            neg.append([h, r, t_neg])

    return torch.tensor(neg, dtype=torch.long)


# =========================
# 6. TRAIN
# =========================

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path, in_dim, hidden_dim, out_dim, num_relations):
    model = Model(in_dim, hidden_dim, out_dim, num_relations)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

def train(model, x, edge_index, edge_type, triples, id2type, rel2type, id2rel, epochs=50, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    type_to_nodes = build_type_index(id2type)
    triples = torch.tensor(triples, dtype=torch.long)

    for epoch in range(epochs):
        model.train()

        neg_triples = negative_sampling(
            triples.tolist(),
            id2type,
            rel2type,
            id2rel,
            type_to_nodes
        )

        z = model.encode(x, edge_index, edge_type)
        pos_score = model.score(z, triples)
        neg_score = model.score(z, neg_triples)

        logits = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ], dim=0)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}: Loss = {loss.item():.4f}")

    return model


# =========================
# 7. INFERENCE HELPERS
# =========================

@torch.no_grad()
def predict_score(model, x, edge_index, edge_type, triple_tensor):
    model.eval()
    z = model.encode(x, edge_index, edge_type)
    logit = model.score(z, triple_tensor)
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
        dtype=torch.long
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
        dtype=torch.long
    )

    scores = torch.sigmoid(model.score(z, triples))
    top_k = min(top_k, len(candidate_head_ids))
    values, indices = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        results.append((candidate_head_ids[idx], score))

    return results


# =========================
# 8. MAIN
# =========================

def main():
    node_file = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\nodes.tsv"
    triple_file = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\triples.tsv"

    # 1) load nodes and triples
    node2id, id2type, x = load_nodes(node_file)
    triples_raw, rel_list_base = load_triples(triple_file, node2id)

    # 2) define base relation schema
    base_rel2type = {
        "CAUSES": ("Cause", "Mode"),
        "LEADS_TO": ("Mode", "Effect"),
        # "CAUSES_EFFECT": ("Cause", "Effect"),
        "HAS_FUNCTION": ("Element", "Function"),
        "HAS_MODE": ("Function", "Mode"),
        # "BELONGS_TO": ("Mode", "Mode"),
    }

    # 3) validate schema against data
    validate_relation_schema(rel_list_base, base_rel2type)

    # 4) add inverse relations
    rel_list_full = rel_list_base + [r + "_REV" for r in rel_list_base]
    rel2id = {r: i for i, r in enumerate(rel_list_full)}
    id2rel = {v: k for k, v in rel2id.items()}
    rel2type = build_full_rel2type(base_rel2type)

    # 5) build training triples with relation ids (only original triples, not reversed)
    triples = [(h, rel2id[r], t) for (h, r, t) in triples_raw]

    # 6) build message-passing graph with both forward and reverse edges
    edge_index, edge_type = build_graph(triples_raw, rel2id)

    # 7) build model
    model = Model(
        in_dim=x.shape[1],
        hidden_dim=128,
        out_dim=128,
        num_relations=len(rel2id)
    )

    # 8) train
    model = train(
        model=model,
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        triples=triples,
        id2type=id2type,
        rel2type=rel2type,
        id2rel=id2rel,
        epochs=100,
        lr=0.005
    )

    print("Training finished.")

    save_model(model, "rgcn_model.pt")

    # =========================
    # Example inference
    # =========================
    # Predict possible Mode for a given Cause using relation CAUSES
    #
    # Example:
    # cause_id = node2id["cause:494a96e2a520"]
    # relation_id = rel2id["CAUSES"]
    # candidate_modes = [nid for nid, t in id2type.items() if t == "Mode"]
    
    # results = predict_tail(
    #     model, x, edge_index, edge_type,
    #     head_id=cause_id,
    #     relation_id=relation_id,
    #     vc=candidate_modes,
    #     top_k=10
    # )
    
    
    # print(results)


if __name__ == "__main__":
    main()