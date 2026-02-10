from pathlib import Path
from typing import List, Tuple

import numpy as np
import chromadb
from chromadb.utils import embedding_functions

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter, defaultdict

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import json


# =========================================================
# Config
# =========================================================

ROLES = {
    # "failure_element": "tab:blue",
    # "failure_mode": "tab:orange",
    # "failure_effect": "tab:green",
     "failure_cause": "tab:red",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# =========================================================
# Data loading
# =========================================================

def load_failure_embeddings(
    failure_kb_dir: Path,
) -> Tuple[np.ndarray, List[str], List[str]]:
    client = chromadb.PersistentClient(path=str(failure_kb_dir))

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_collection(
        name="all_failure_kb",
        embedding_function=embedder,
    )

    # "ids" must NOT be in include; ids are always returned
    data = collection.get(include=["embeddings", "metadatas", "documents"])

    X, roles, ids, texts, metadatas = [], [], [], [], []

    for emb, meta, doc, _id in zip(
            data["embeddings"],
            data["metadatas"],
            data["documents"],
            data["ids"],
        ):
            role = meta.get("role")
            if role not in ROLES:
                continue

            X.append(emb)
            roles.append(role)
            ids.append(_id)
            texts.append(doc)
            metadatas.append(meta)

    return np.array(X), roles, ids, texts, metadatas


# =========================================================
# Visualization
# =========================================================

def plot_pca_2d(
    X: np.ndarray,
    roles: List[str],
    out_path: Path,
):
    """
    2D PCA scatter: same axis, color by role.
    """
    role_counts = Counter(roles)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(9, 7))

    for role, color in ROLES.items():
        idx = [i for i, r in enumerate(roles) if r == role]
        if not idx:
            continue

        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            c=color,
            label=f"{role} (n={role_counts[role]})",
            alpha=0.6,
            s=20,
        )

    total = len(roles)
    plt.title(f"Failure Fragment Semantic Distribution (PCA 2D)  |  total={total}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pca_1d(
    X: np.ndarray,
    roles: List[str],
    out_path: Path,
):
    """
    1D PCA distribution (histogram) per role.
    """
    role_counts = Counter(roles)

    pca = PCA(n_components=1, random_state=42)
    x_1d = pca.fit_transform(X).flatten()

    plt.figure(figsize=(9, 5))

    for role, color in ROLES.items():
        vals = [x_1d[i] for i, r in enumerate(roles) if r == role]
        if not vals:
            continue

        plt.hist(
            vals,
            bins=50,
            alpha=0.5,
            label=f"{role} (n={role_counts[role]})",
        )

    total = len(roles)
    plt.title(f"Failure Fragment Semantic Distribution (PCA 1D)  |  total={total}")
    plt.xlabel("PC1")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 8,
    random_state: int = 42,
):
    """
    KMeans clustering in embedding space.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = km.fit_predict(X_scaled)
    return labels


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
):
    """
    Density-based clustering (can output -1 for noise).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = db.fit_predict(X_scaled)
    return labels

def run_gmm(
    X: np.ndarray,
    n_components: int = 6,
    random_state: int = 42,
):
    """
    Gaussian Mixture Model clustering (soft).
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
    )
    labels = gmm.fit_predict(X)
    probs = gmm.predict_proba(X)
    return labels, probs, gmm

def plot_pca_2d_with_clusters(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    out_path: Path,
    title_suffix: str = "",
):
    """
    2D PCA scatter colored by cluster label.
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(9, 7))

    unique_labels = sorted(set(cluster_labels))
    for lab in unique_labels:
        idx = cluster_labels == lab
        label_name = f"cluster {lab}" if lab != -1 else "noise"

        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            label=label_name,
            alpha=0.6,
            s=20,
        )

    plt.title(f"PCA 2D with Clusters {title_suffix}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=1.5)
    plt.grid(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def print_cluster_role_stats(
    cluster_labels: np.ndarray,
    roles: List[str],
):
    """
    Print role distribution per cluster.
    """
    clusters = sorted(set(cluster_labels))
    print("\n[CLUSTER ROLE DISTRIBUTION]")

    for c in clusters:
        idx = [i for i, lab in enumerate(cluster_labels) if lab == c]
        role_counter = Counter(roles[i] for i in idx)

        name = f"cluster {c}" if c != -1 else "noise"
        print(f"\n{name} | size={len(idx)}")
        for role, cnt in role_counter.items():
            print(f"  {role}: {cnt}")

def plot_pca_2d_with_gmm(
    X: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(9, 7))

    for lab in sorted(set(labels)):
        idx = labels == lab
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            label=f"component {lab}",
            alpha=0.6,
            s=20,
        )

    plt.title("PCA 2D with GMM Components")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_gmm_group_json(
    X: np.ndarray,
    ids: List[str],
    texts: List[str],
    gmm_labels: np.ndarray,
    gmm_probs: np.ndarray,
    gmm_model,
    group_type: str,
    embedding_model: str,
    confidence_threshold: float = 0.7,
):
    groups = defaultdict(list)
    unassigned = []

    max_probs = gmm_probs.max(axis=1)
    comp_ids = gmm_probs.argmax(axis=1)

    for i in range(len(ids)):
        item = {
            "id": ids[i],
            "text": texts[i],
            "probability": float(max_probs[i]),
        }

        if max_probs[i] >= confidence_threshold:
            groups[comp_ids[i]].append(item)
        else:
            unassigned.append({
                "id": ids[i],
                "text": texts[i],
                "max_probability": float(max_probs[i]),
            })

    group_list = []
    for comp_id, members in groups.items():
        group_list.append({
            "group_id": f"{group_type}_gmm_{comp_id:02d}",
            "component_id": int(comp_id),
            "size": len(members),
            "center_embedding": gmm_model.means_[comp_id].tolist(),
            "members": sorted(
                members,
                key=lambda x: x["probability"],
                reverse=True,
            ),
        })

    return {
        "group_type": group_type,
        "model": "gmm",
        "embedding_model": embedding_model,
        "n_components": gmm_model.n_components,
        "confidence_threshold": confidence_threshold,
        "groups": sorted(
            group_list,
            key=lambda g: g["size"],
            reverse=True,
        ),
        "unassigned": unassigned,
    }

def save_group_json(data: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =========================================================
# Runner
# =========================================================

def run_failure_semantic_visualization(
       failure_kb_dir: Path,
    output_dir: Path,
):
    X, roles, ids, texts, metadatas= load_failure_embeddings(failure_kb_dir)

    if len(X) == 0:
        raise RuntimeError("No failure embeddings found for visualization.")

    counts = Counter(roles)
    total = len(roles)
    print("[COUNT] total =", total)
    for r in ROLES.keys():
        print(f"[COUNT] {r}: {counts.get(r, 0)}")

    # ---- original role-based PCA ----
    plot_pca_2d(X, roles, output_dir / "failure_role_pca_2d.png")
    plot_pca_1d(X, roles, output_dir / "failure_role_pca_1d.png")

    # ---- KMeans clustering ----
    kmeans_labels = run_kmeans(X, n_clusters=8)
    plot_pca_2d_with_clusters(
        X,
        kmeans_labels,
        output_dir / "failure_kmeans_pca_2d.png",
        title_suffix="(KMeans)",
    )
    print_cluster_role_stats(kmeans_labels, roles)

    # ---- DBSCAN clustering (optional but insightful) ----
    dbscan_labels = run_dbscan(X, eps=0.7, min_samples=15)
    plot_pca_2d_with_clusters(
        X,
        dbscan_labels,
        output_dir / "failure_dbscan_pca_2d.png",
        title_suffix="(DBSCAN)",
    )
    print_cluster_role_stats(dbscan_labels, roles)

    # ---- GMM clustering (soft clustering) ----
    gmm_labels, gmm_probs, gmm_model = run_gmm(X, n_components=8)
    plot_pca_2d_with_gmm(
        X,
        gmm_labels,
        output_dir / "failure_gmm_pca_2d.png",
    )
    # 可选：看每个 component 里各 role 分布
    print_cluster_role_stats(gmm_labels, roles)
    group_json = build_gmm_group_json(
        X=X,
        ids=ids,
        texts=texts,
        gmm_labels=gmm_labels,
        gmm_probs=gmm_probs,
        gmm_model=gmm_model,
        group_type="failure_element",
        embedding_model=EMBEDDING_MODEL,
        confidence_threshold=0.7,
    )

    save_group_json(
        group_json,
        output_dir / "failure_element_gmm_groups.json",
    )

    print(f"\n[OK] Visualization written to: {output_dir}")



# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    # KB_DATA_ROOT = BASE_DIR.parent / "KB_motor_drives"
    # FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"
    FAILURE_KB_DIR = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb")
    OUTPUT_DIR = BASE_DIR / "semantic_viz_output"

    run_failure_semantic_visualization(
        failure_kb_dir=FAILURE_KB_DIR,
        output_dir=OUTPUT_DIR,
    )
