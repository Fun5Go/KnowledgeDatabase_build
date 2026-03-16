import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from Retriever.failure_query_tools import _load_kb,query_semantic_kb
import re





def compute_cross_field_similarity_stats(persist_dir: str | Path):

    persist_dir = Path(persist_dir)

    # -------------------------
    # load KB
    # -------------------------
    kb = _load_kb(persist_dir)

    field_store = getattr(kb, "field_store", {}) or {}
    edge_store = getattr(kb, "edge_store", {}) or {}

    mode_to_cause = edge_store.get("mode_to_cause", {}) or {}
    mode_to_effect = edge_store.get("mode_to_effect", {}) or {}

    # -------------------------
    # load all embeddings
    # -------------------------
    res = kb.collection.get(
        include=["embeddings"]
    )

    ids = res["ids"]
    embeddings = np.array(res["embeddings"], dtype=np.float32)

    embedding_dict = {
        i: e for i, e in zip(ids, embeddings)
    }

    # -------------------------
    # similarity collectors
    # -------------------------
    mc_sims = []
    me_sims = []
    ce_sims = []

    # -------------------------
    # iterate modes
    # -------------------------
    for mode_id, causes_dict in mode_to_cause.items():

        mode_vec = embedding_dict.get(mode_id)
        if mode_vec is None:
            continue

        effects_dict = mode_to_effect.get(mode_id, {})

        # -------- mode-cause --------
        for cause_id in causes_dict.keys():

            cause_vec = embedding_dict.get(cause_id)
            if cause_vec is None:
                continue

            # sim = cosine_similarity(
            #     mode_vec.reshape(1, -1),
            #     cause_vec.reshape(1, -1)
            # )[0][0]
            sim = np.dot(mode_vec, cause_vec) / (
                np.linalg.norm(mode_vec) * np.linalg.norm(cause_vec)
            )
            mc_sims.append(sim)

        # -------- mode-effect --------
        for effect_id in effects_dict.keys():

            effect_vec = embedding_dict.get(effect_id)
            if effect_vec is None:
                continue

            sim = cosine_similarity(
                mode_vec.reshape(1, -1),
                effect_vec.reshape(1, -1)
            )[0][0]

            me_sims.append(sim)

        # -------- cause-effect --------
        for cause_id in causes_dict.keys():

            cause_vec = embedding_dict.get(cause_id)
            if cause_vec is None:
                continue

            for effect_id in effects_dict.keys():

                effect_vec = embedding_dict.get(effect_id)
                if effect_vec is None:
                    continue

                sim = cosine_similarity(
                    cause_vec.reshape(1, -1),
                    effect_vec.reshape(1, -1)
                )[0][0]

                ce_sims.append(sim)

    # -------------------------
    # statistics helper
    # -------------------------
    def describe(arr):

        if len(arr) == 0:
            return {"count": 0}

        arr = np.array(arr)

        return {
            "count": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        }

    stats = {
        "mode_cause": describe(mc_sims),
        "mode_effect": describe(me_sims),
        "cause_effect": describe(ce_sims),
    }

    return stats


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

def _describe(arr):
    if not arr:
        return {"count": 0}
    a = np.asarray(arr, dtype=np.float32)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
    }

def compute_cross_field_tfidf_similarity_stats(
    persist_dir: str | Path,
    *,
    min_token_len: int = 2,
    max_features: int | None = 50000,
    ngram_range=(1, 2),
    stopwords: set[str] | None = None,
    max_edges: int | None = None,  # 可选：截断边数加速
):
    """
    用 TF-IDF cosine（L2 normalize后点积）计算：
      - mode-cause 边分数分布
      - mode-effect 边分数分布
      - cause-effect（同一个mode下组合）分数分布（可选对比）
    """

    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)

    field_store = getattr(kb, "field_store", {}) or {}
    edge_store = getattr(kb, "edge_store", {}) or {}
    mode_to_cause = edge_store.get("mode_to_cause", {}) or {}
    mode_to_effect = edge_store.get("mode_to_effect", {}) or {}

    if stopwords is None:
        # 你可以按语料再扩展；先给个温和的版本，别过度删词
        stopwords = {
            "the","a","an","and","or","to","of","in","on","for","with",
            "is","are","be","by","as","at","from",
            "failure","fail","fails","failed","system","component","device","unit","module"
        }

    def get_text(_id: str) -> str:
        obj = field_store.get(_id) or {}
        return (obj.get("text") or obj.get("name") or "").strip()

    # -------------------------
    # collect edges + ids
    # -------------------------
    mc_edges = []
    me_edges = []
    ce_pairs = []  # (cause, effect) within same mode

    needed_ids = set()

    for mode_id, causes_dict in mode_to_cause.items():
        for cid in (causes_dict or {}).keys():
            mc_edges.append((mode_id, cid))
            needed_ids.add(mode_id); needed_ids.add(cid)

    for mode_id, effects_dict in mode_to_effect.items():
        for eid in (effects_dict or {}).keys():
            me_edges.append((mode_id, eid))
            needed_ids.add(mode_id); needed_ids.add(eid)

    if max_edges is not None:
        mc_edges = mc_edges[:max_edges]
        me_edges = me_edges[:max_edges]

    # build CE pairs aligned with existing edges (same mode)
    for mode_id, causes_dict in mode_to_cause.items():
        effects_dict = mode_to_effect.get(mode_id, {}) or {}
        if not causes_dict or not effects_dict:
            continue
        for cid in causes_dict.keys():
            for eid in effects_dict.keys():
                ce_pairs.append((cid, eid))
                needed_ids.add(cid); needed_ids.add(eid)

    if max_edges is not None:
        ce_pairs = ce_pairs[:max_edges]

    # -------------------------
    # prepare corpus
    # -------------------------
    id_to_text = {i: get_text(i) for i in needed_ids}
    id_to_text = {i: t for i, t in id_to_text.items() if t}

    corpus_ids = list(id_to_text.keys())
    corpus_texts = [id_to_text[i] for i in corpus_ids]
    id_to_row = {cid: idx for idx, cid in enumerate(corpus_ids)}

    # -------------------------
    # TF-IDF with custom tokenizer
    # -------------------------
    def tokenizer(s: str):
        toks = _TOKEN_RE.findall(s.lower())
        out = []
        for w in toks:
            if len(w) < min_token_len:
                continue
            if w in stopwords:
                continue
            out.append(w)
        return out

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        lowercase=False,          # 我们 tokenizer 已 lower
        ngram_range=ngram_range,  # (1,2) 通常对技术短句更好
        max_features=max_features,
        norm="l2",
        min_df=1,
    )
    X = vectorizer.fit_transform(corpus_texts)  # CSR, rows L2-normalized

    def pair_score(a_id: str, b_id: str) -> float | None:
        ra = id_to_row.get(a_id)
        rb = id_to_row.get(b_id)
        if ra is None or rb is None:
            return None
        # L2 norm后点积 = cosine
        return (X[ra] @ X[rb].T).toarray()[0,0]

    # -------------------------
    # compute distributions
    # -------------------------
    mc_scores = []
    for m, c in mc_edges:
        s = pair_score(m, c)
        if s is not None:
            mc_scores.append(s)

    me_scores = []
    for m, e in me_edges:
        s = pair_score(m, e)
        if s is not None:
            me_scores.append(s)

    ce_scores = []
    for c, e in ce_pairs:
        s = pair_score(c, e)
        if s is not None:
            ce_scores.append(s)

    return {
        "mode_cause_tfidf": _describe(mc_scores),
        "mode_effect_tfidf": _describe(me_scores),
        "cause_effect_tfidf": _describe(ce_scores),
        "settings": {
            "ngram_range": ngram_range,
            "min_token_len": min_token_len,
            "max_features": max_features,
            "stopwords_size": len(stopwords),
            "max_edges": max_edges,
        },
    }

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    KB_PATH = Path(
            r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_discipline\failure_kb"
        )
    results = compute_cross_field_similarity_stats(KB_PATH)
    print(results)
    # out = compute_cross_field_tfidf_similarity_stats(KB_PATH, ngram_range=(1,2))
    # print(out)