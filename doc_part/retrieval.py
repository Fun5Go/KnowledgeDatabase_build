import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================================================
# Config
# =========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PERSIST_DIR = "./DATA/chroma_langchain_db"
COLLECTION_NAME = "fs_requirements"


# =========================================================
# Data structure
# =========================================================

@dataclass
class RetrievalResult:
    doc_id: str
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    embedding_distance: Optional[float] = None
    embedding_similarity: Optional[float] = None
    dense_norm: Optional[float] = None

    bm25_score: Optional[float] = None
    bm25_norm: Optional[float] = None

    hybrid_score: Optional[float] = None
    ce_score: Optional[float] = None


# =========================================================
# Init
# =========================================================

def create_embeddings(model_name: str = EMBEDDING_MODEL):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )
    print("[INFO] Embeddings model initialized.")
    return embeddings


def load_vector_store(
    embeddings,
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME
):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    # print(f"[INFO] Loaded vector store from: {persist_dir}")
    return vector_store


def create_cross_encoder(model_name: str = CROSS_ENCODER_MODEL):
    model = CrossEncoder(model_name)
    print("[INFO] CrossEncoder initialized.")
    return model


# =========================================================
# Utils
# =========================================================

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def distance_to_similarity(distance: float) -> float:
    """
    More stable than 1 - distance.
    distance smaller => similarity larger
    """
    return 1.0 / (1.0 + float(distance))


def minmax_normalize(score_dict: Dict[str, float], flat_value: float = 0.5) -> Dict[str, float]:
    """
    Min-max normalize to [0,1].
    - if only one item => 1.0
    - if all values equal and >1 items => flat_value
    """
    if not score_dict:
        return {}

    if len(score_dict) == 1:
        only_k = next(iter(score_dict.keys()))
        return {only_k: 1.0}

    values = list(score_dict.values())
    min_v = min(values)
    max_v = max(values)

    if math.isclose(min_v, max_v):
        return {k: flat_value for k in score_dict}

    return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}


def apply_metadata_filter(
    metadatas: List[Dict[str, Any]],
    doc_type: Optional[str] = None
) -> List[int]:
    valid_idx = []
    for i, md in enumerate(metadatas):
        md = md or {}
        if doc_type is None or md.get("type") == doc_type:
            valid_idx.append(i)
    return valid_idx


def build_doc_key(page_content: str, metadata: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Use stable fields to align dense-returned docs with Chroma collection docs.
    """
    metadata = metadata or {}
    return (
        page_content.strip(),
        str(metadata.get("source", "")),
        str(metadata.get("pages", "")),
        str(metadata.get("primary_requirement_id", ""))
    )


# =========================================================
# Load all docs from Chroma
# =========================================================

def load_all_documents_from_chroma(
    vector_store,
    doc_type: Optional[str] = None
) -> Tuple[List[RetrievalResult], Dict[Tuple[str, str, str, str], str]]:
    collection = vector_store._collection
    raw = collection.get(include=["documents", "metadatas"])

    ids = raw.get("ids", [])
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    keep_indices = apply_metadata_filter(metadatas, doc_type=doc_type)

    all_docs = []
    key_to_doc_id = {}

    for i in keep_indices:
        doc_id = str(ids[i])
        page_content = documents[i]
        metadata = metadatas[i] or {}

        all_docs.append(
            RetrievalResult(
                doc_id=doc_id,
                page_content=page_content,
                metadata=metadata
            )
        )

        key = build_doc_key(page_content, metadata)
        key_to_doc_id[key] = doc_id

    # print(f"[INFO] Loaded {len(all_docs)} documents from Chroma for BM25/hybrid.")
    return all_docs, key_to_doc_id


# =========================================================
# Dense retrieval
# =========================================================

def dense_retrieve(
    vector_store,
    query_text: str,
    key_to_doc_id: Dict[Tuple[str, str, str, str], str],
    top_k: int = 50,
    doc_type: Optional[str] = None
) -> List[RetrievalResult]:
    metadata_filter = {"type": doc_type} if doc_type else None

    docs_with_scores = vector_store.similarity_search_with_score(
        query_text,
        k=top_k,
        filter=metadata_filter
    )

    results = []
    for rank, (doc, distance) in enumerate(docs_with_scores):
        key = build_doc_key(doc.page_content, doc.metadata)
        real_doc_id = key_to_doc_id.get(key, f"unmatched_dense_{rank}")

        sim = distance_to_similarity(distance)

        results.append(
            RetrievalResult(
                doc_id=str(real_doc_id),
                page_content=doc.page_content,
                metadata=doc.metadata,
                embedding_distance=float(distance),
                embedding_similarity=float(sim)
            )
        )

    return results


# =========================================================
# BM25 retrieval
# =========================================================

def bm25_retrieve(
    all_docs: List[RetrievalResult],
    query_text: str,
    top_k: int = 50
) -> List[RetrievalResult]:
    tokenized_corpus = [tokenize(doc.page_content) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query_text)

    scores = bm25.get_scores(tokenized_query)

    results = []
    for doc, score in zip(all_docs, scores):
        results.append(
            RetrievalResult(
                doc_id=doc.doc_id,
                page_content=doc.page_content,
                metadata=doc.metadata,
                bm25_score=float(score)
            )
        )

    results.sort(key=lambda x: x.bm25_score, reverse=True)
    return results[:top_k]


# =========================================================
# Hybrid retrieval
# =========================================================

def hybrid_retrieve(
    vector_store,
    all_docs: List[RetrievalResult],
    key_to_doc_id: Dict[Tuple[str, str, str, str], str],
    query_text: str,
    dense_top_k: int = 100,
    bm25_top_k: int = 100,
    final_top_k: int = 20,
    doc_type: Optional[str] = None,
    alpha: float = 0.5
) -> List[RetrievalResult]:
    """
    Hybrid only merges candidate union of dense top_k and bm25 top_k.
    This avoids assigning fake 0 dense scores to the whole corpus.
    """

    # 1) dense top-k
    dense_results = dense_retrieve(
        vector_store=vector_store,
        query_text=query_text,
        key_to_doc_id=key_to_doc_id,
        top_k=dense_top_k,
        doc_type=doc_type
    )
    dense_map = {r.doc_id: r for r in dense_results}
    dense_score_map = {
        r.doc_id: r.embedding_similarity for r in dense_results
        if r.embedding_similarity is not None
    }

    # 2) bm25 top-k
    tokenized_corpus = [tokenize(doc.page_content) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query_text)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_full = []
    for doc, score in zip(all_docs, bm25_scores):
        bm25_full.append((doc, float(score)))
    bm25_full.sort(key=lambda x: x[1], reverse=True)
    bm25_full = bm25_full[:bm25_top_k]

    bm25_map = {}
    bm25_score_map = {}
    for doc, score in bm25_full:
        bm25_map[doc.doc_id] = doc
        bm25_score_map[doc.doc_id] = score

    # 3) candidate union
    candidate_ids = set(dense_map.keys()) | set(bm25_map.keys())

    # 4) normalize only within candidate union
    dense_norm_map = minmax_normalize(dense_score_map, flat_value=0.5)
    bm25_norm_map = minmax_normalize(bm25_score_map, flat_value=0.5)

    # 5) merge
    final_results = []
    all_doc_map = {doc.doc_id: doc for doc in all_docs}

    for doc_id in candidate_ids:
        base_doc = all_doc_map[doc_id]

        dense_item = dense_map.get(doc_id)
        emb_dist = dense_item.embedding_distance if dense_item else None
        emb_sim = dense_item.embedding_similarity if dense_item else None

        bm25_score = bm25_score_map.get(doc_id, None)

        dense_norm = dense_norm_map.get(doc_id, 0.0)
        bm25_norm = bm25_norm_map.get(doc_id, 0.0)

        hybrid_score = alpha * dense_norm + (1.0 - alpha) * bm25_norm

        final_results.append(
            RetrievalResult(
                doc_id=doc_id,
                page_content=base_doc.page_content,
                metadata=base_doc.metadata,
                embedding_distance=emb_dist,
                embedding_similarity=emb_sim,
                dense_norm=dense_norm,
                bm25_score=bm25_score,
                bm25_norm=bm25_norm,
                hybrid_score=hybrid_score
            )
        )

    final_results.sort(key=lambda x: x.hybrid_score, reverse=True)
    return final_results[:final_top_k]


# =========================================================
# Rerank
# =========================================================

def rerank_results(
    query: str,
    results: List[RetrievalResult],
    cross_encoder,
    top_k: int = 5
) -> List[RetrievalResult]:
    if not results:
        return []

    pairs = [(query, r.page_content) for r in results]
    ce_scores = cross_encoder.predict(pairs)

    for r, score in zip(results, ce_scores):
        r.ce_score = float(score)

    results.sort(key=lambda x: x.ce_score, reverse=True)
    return results[:top_k]


# =========================================================
# Unified retrieve
# =========================================================

def retrieve(
    vector_store,
    query_text: str,
    retrieval_mode: str = "hybrid",   # dense | bm25 | hybrid
    use_rerank: bool = True,
    doc_type: Optional[str] = None,
    candidate_k: int = 50,
    final_k: int = 5,
    hybrid_alpha: float = 0.5,
    cross_encoder=None
) -> List[RetrievalResult]:

    all_docs, key_to_doc_id = load_all_documents_from_chroma(
        vector_store,
        doc_type=doc_type
    )

    if retrieval_mode == "dense":
        results = dense_retrieve(
            vector_store=vector_store,
            query_text=query_text,
            key_to_doc_id=key_to_doc_id,
            top_k=candidate_k,
            doc_type=doc_type
        )

        dense_score_map = {
            r.doc_id: r.embedding_similarity for r in results
            if r.embedding_similarity is not None
        }
        dense_norm_map = minmax_normalize(dense_score_map, flat_value=0.5)

        for r in results:
            r.dense_norm = dense_norm_map.get(r.doc_id, 0.0)
            r.hybrid_score = r.dense_norm

        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        candidates = results[:candidate_k]

    elif retrieval_mode == "bm25":
        results = bm25_retrieve(
            all_docs=all_docs,
            query_text=query_text,
            top_k=candidate_k
        )

        bm25_score_map = {
            r.doc_id: r.bm25_score for r in results
            if r.bm25_score is not None
        }
        bm25_norm_map = minmax_normalize(bm25_score_map, flat_value=0.5)

        for r in results:
            r.bm25_norm = bm25_norm_map.get(r.doc_id, 0.0)
            r.hybrid_score = r.bm25_norm

        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        candidates = results[:candidate_k]

    elif retrieval_mode == "hybrid":
        candidates = hybrid_retrieve(
            vector_store=vector_store,
            all_docs=all_docs,
            key_to_doc_id=key_to_doc_id,
            query_text=query_text,
            dense_top_k=max(candidate_k, 50),
            bm25_top_k=max(candidate_k, 50),
            final_top_k=candidate_k,
            doc_type=doc_type,
            alpha=hybrid_alpha
        )
    else:
        raise ValueError("retrieval_mode must be 'dense', 'bm25', or 'hybrid'")

    if use_rerank:
        if cross_encoder is None:
            raise ValueError("cross_encoder must be provided when use_rerank=True")
        final_results = rerank_results(query_text, candidates, cross_encoder, top_k=final_k)
    else:
        final_results = candidates[:final_k]

    return final_results


# =========================================================
# Print
# =========================================================

def truncate_text(text: str, max_len: int = 500) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_len else text[:max_len] + "..."


def print_results(results: List[RetrievalResult], show_full_text: bool = False):
    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] doc_id = {r.doc_id}")
        print(f"metadata              : {r.metadata}")
        print(f"embedding_distance    : {None if r.embedding_distance is None else f'{r.embedding_distance:.6f}'}")
        print(f"embedding_similarity  : {None if r.embedding_similarity is None else f'{r.embedding_similarity:.6f}'}")
        print(f"dense_norm            : {None if r.dense_norm is None else f'{r.dense_norm:.6f}'}")
        print(f"bm25_score            : {None if r.bm25_score is None else f'{r.bm25_score:.6f}'}")
        print(f"bm25_norm             : {None if r.bm25_norm is None else f'{r.bm25_norm:.6f}'}")
        print(f"hybrid_score          : {None if r.hybrid_score is None else f'{r.hybrid_score:.6f}'}")
        print(f"cross_encoder_score   : {None if r.ce_score is None else f'{r.ce_score:.6f}'}")
        print("-" * 100)
        print(r.page_content if show_full_text else truncate_text(r.page_content))
        print("-" * 100)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    embeddings = create_embeddings()
    vector_store = load_vector_store(
        embeddings=embeddings,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    cross_encoder = create_cross_encoder()

    query = "Motor starts without soft start"

    retrieval_mode = "hybrid"   # dense / bm25 / hybrid
    use_rerank = True
    doc_type = "FS"             # 你的 metadata["type"] 是 FS，不是 requirement/table_row
    candidate_k = 50
    final_k = 5
    hybrid_alpha = 0.5

    results = retrieve(
        vector_store=vector_store,
        query_text=query,
        retrieval_mode=retrieval_mode,
        use_rerank=use_rerank,
        doc_type=doc_type,
        candidate_k=candidate_k,
        final_k=final_k,
        hybrid_alpha=hybrid_alpha,
        cross_encoder=cross_encoder
    )

    print_results(results, show_full_text=True)