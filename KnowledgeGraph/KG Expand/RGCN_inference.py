import ast
import torch
import pandas as pd
from torch_geometric.nn import RGCNConv
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. LOAD DATA
# =========================

def load_nodes(node_file):
    df = pd.read_csv(node_file, sep="\t")

    node_ids = df["node_id"].tolist()
    node_types = df["node_type"].tolist()

    embeddings = []
    for emb_str in df["embedding"]:
        emb = torch.tensor(ast.literal_eval(emb_str), dtype=torch.float)
        embeddings.append(emb)

    x = torch.stack(embeddings)

    node2id = {nid: i for i, nid in enumerate(node_ids)}
    id2node = {i: nid for nid, i in node2id.items()}
    id2type = {node2id[nid]: t for nid, t in zip(node_ids, node_types)}

    return node2id, id2node, id2type, x


def load_triples(triple_file, node2id):
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


# =========================
# 2. GRAPH
# =========================

def build_graph(triples_raw, rel2id):
    edge_index = []
    edge_type = []

    for h, r, t in triples_raw:
        edge_index.append([h, t])
        edge_type.append(rel2id[r])

        edge_index.append([t, h])
        edge_type.append(rel2id[r + "_REV"])

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


# =========================
# 4. LOAD MODEL
# =========================

def load_model(path, in_dim, hidden_dim, out_dim, num_relations):
    model = Model(in_dim, hidden_dim, out_dim, num_relations)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Loaded model from {path}")
    return model


# =========================
# 5. RELATION SETUP
# =========================

def build_relations(rel_list_base):
    rel_list_full = rel_list_base + [r + "_REV" for r in rel_list_base]
    rel2id = {r: i for i, r in enumerate(rel_list_full)}
    id2rel = {v: k for k, v in rel2id.items()}
    return rel2id, id2rel


# =========================
# 6. PREDICT
# =========================

@torch.no_grad()
def predict_tail(model, x, edge_index, edge_type,
                 head_id, relation_id, candidate_ids, top_k=10):

    z = model.encode(x, edge_index, edge_type)

    triples = torch.tensor(
        [[head_id, relation_id, cid] for cid in candidate_ids],
        dtype=torch.long
    )

    scores = torch.sigmoid(model.score(z, triples))

    values, indices = torch.topk(scores, k=min(top_k, len(candidate_ids)))

    return [(candidate_ids[i], values[j].item()) for j, i in enumerate(indices)]


# =========================
# 7. TASK FUNCTIONS
# =========================

def infer_cause_to_mode(model, x, edge_index, edge_type,
                        node2id, id2node, id2type, rel2id,
                        cause_node_id, top_k=10):

    head_id = node2id[cause_node_id]
    relation_id = rel2id["CAUSES"]

    candidates = [nid for nid, t in id2type.items() if t == "Mode"]

    results = predict_tail(
        model, x, edge_index, edge_type,
        head_id, relation_id, candidates, top_k
    )

    output = []
    for nid, score in results:
        output.append((id2node[nid], score))

    return output


def infer_mode_to_effect(model, x, edge_index, edge_type,
                         node2id, id2node, id2type, rel2id,
                         mode_node_id, top_k=10):

    head_id = node2id[mode_node_id]
    relation_id = rel2id["LEADS_TO"]

    candidates = [nid for nid, t in id2type.items() if t == "Effect"]

    results = predict_tail(
        model, x, edge_index, edge_type,
        head_id, relation_id, candidates, top_k
    )

    print("\n=== Mode → Effect ===")
    for nid, score in results:
        print(f"{id2node[nid]} | {score:.4f}")


# =========================
# 8. MAIN
# =========================

def main():

    node_file = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KnowledgeGraph\KG Expand\nodes.tsv"
    triple_file = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KnowledgeGraph\KG Expand\triples.tsv"
    model_path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\rgcn_model.pt"

    # load data
    node2id, id2node, id2type, x = load_nodes(node_file)
    triples_raw, rel_list_base = load_triples(triple_file, node2id)

    # relations
    rel2id, id2rel = build_relations(rel_list_base)

    # graph
    edge_index, edge_type = build_graph(triples_raw, rel2id)

    # load model
    model = load_model(
        model_path,
        in_dim=x.shape[1],
        hidden_dim=128,
        out_dim=128,
        num_relations=len(rel2id)
    )

    # =====================
    # 🔥 TEST HERE
    # =====================

    # 改成你的真实 node_id
    cause_node = "cause:3e9711da8dc3"
    mode_node = "your_mode_node_id"

    if cause_node in node2id:
        result = infer_cause_to_mode(
            model, x, edge_index, edge_type,
            node2id, id2node, id2type, rel2id,
            cause_node, top_k=10
        )
        print(result)

    if mode_node in node2id:
        infer_mode_to_effect(
            model, x, edge_index, edge_type,
            node2id, id2node, id2type, rel2id,
            mode_node, top_k=10
        )


if __name__ == "__main__":
    main()