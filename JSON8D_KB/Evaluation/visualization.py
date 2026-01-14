from pathlib import Path
from typing import List, Tuple

import numpy as np
import chromadb
from chromadb.utils import embedding_functions

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =========================================================
# Config
# =========================================================

ROLES = {
    "failure_element": "tab:blue",
    "failure_mode": "tab:orange",
    "failure_effect": "tab:green",
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
        name="failure_kb",
        embedding_function=embedder,
    )

    # "ids" must NOT be in include; ids are always returned
    data = collection.get(include=["embeddings", "metadatas"])

    X = []
    roles = []
    ids = []

    for emb, meta, _id in zip(
        data["embeddings"],
        data["metadatas"],
        data["ids"],
    ):
        role = meta.get("role")
        if role in ROLES:
            X.append(emb)
            roles.append(role)
            ids.append(_id)

    return np.array(X), roles, ids


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
            label=role,
            alpha=0.6,
            s=20,
        )

    plt.title("Failure Fragment Semantic Distribution (PCA 2D)")
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
            label=role,
        )

    plt.title("Failure Fragment Semantic Distribution (PCA 1D)")
    plt.xlabel("PC1")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================================================
# Runner
# =========================================================

def run_failure_semantic_visualization(
    failure_kb_dir: Path,
    output_dir: Path,
):
    """
    Entry point for visualization.
    """
    X, roles, _ = load_failure_embeddings(failure_kb_dir)

    if len(X) == 0:
        raise RuntimeError("No failure embeddings found for visualization.")

    plot_pca_2d(
        X,
        roles,
        output_dir / "failure_role_pca_2d.png",
    )

    plot_pca_1d(
        X,
        roles,
        output_dir / "failure_role_pca_1d.png",
    )

    print(f"[OK] Visualization written to: {output_dir}")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    KB_DATA_ROOT = BASE_DIR.parent / "kb_data"
    FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"

    OUTPUT_DIR = BASE_DIR / "semantic_viz_output"

    run_failure_semantic_visualization(
        failure_kb_dir=FAILURE_KB_DIR,
        output_dir=OUTPUT_DIR,
    )
