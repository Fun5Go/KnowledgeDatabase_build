from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import math
import json
from JSON_FMEA_KB.kb_structure import FMEAFailureKB
from Retriever.failure_query_tools import _load_kb,query_semantic_kb
from Retriever.SA_query import build_failure_chains_from_structure

# ------------------------------
# Helpers: load kb (use yours)
# ------------------------------


def _safe_norm_text(x: Optional[str]) -> str:
    if not x:
        return ""
    return " ".join(str(x).strip().split())


def _dist_to_sim(dist: Any) -> float:
    # Chroma distance -> similarity ：sim = 1 - dist
    try:
        return max(0.0, 1.0 - float(dist))
    except Exception:
        return 0.0


def _get_semantic_text(kb: Any, semantic_id: str) -> str:
    node = kb.field_store.get(semantic_id) or {}
    return node.get("text") or ""


def _edge_neighbors(
    kb: Any,
    edge_type: str,
    src_sid: str,
    top_k: int = 10,
    min_count: int = 1,
) -> List[Tuple[str, int]]:
    """
    Return neighbors (tgt_sid, count) sorted by count desc.
    edge_store format: edge_store[edge_type][src_sid][tgt_sid] = count
    """
    store = (kb.edge_store or {}).get(edge_type, {}) or {}
    neigh = store.get(src_sid, {}) or {}
    items = [(tgt, int(cnt)) for tgt, cnt in neigh.items() if int(cnt) >= int(min_count)]
    items.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        items = items[: int(top_k)]
    return items


def _edge_strength(count: int, *, mode: str = "log") -> float:
    """
    Convert co-occurrence count -> edge score.
    log is smoother and robust.
    """
    c = max(0, int(count))
    if mode == "sqrt":
        return math.sqrt(c)
    if mode == "linear":
        return float(c)
    # default log
    return math.log1p(c)


def _semantic_query_hits(
    persist_dir: Path,
    query_text: str,
    field_type: str,
    *,
    n_results: int = 25,
    min_similarity: float = 0.5,
    min_count: Optional[int] = None,
    source_type: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Query Chroma and return [(semantic_id, similarity)] above threshold.
    """
    if not _safe_norm_text(query_text):
        return []

    res = query_semantic_kb(
        persist_dir,
        query_text,
        field_type=field_type,
        n_results=n_results,
        min_count=min_count,
        source_type=source_type,
    )
    ids = (res.get("ids", [[]]) or [[]])[0] or []
    dists = (res.get("distances", [[]]) or [[]])[0] or []

    hits: List[Tuple[str, float]] = []
    for sid, dist in zip(ids, dists):
        sim = _dist_to_sim(dist)
        if sim < float(min_similarity):
            continue
        hits.append((sid, float(sim)))

    # sort by similarity desc
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits


def _cap_keep_best(hits: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
    if top_k is None:
        return hits
    return hits[: int(top_k)]


# ------------------------------
# Evidence: support failure_ids
# ------------------------------
def _support_failure_ids_for_edge(
    kb: Any,
    *,
    edge_type: str,
    src_sid: str,
    tgt_sid: str,
    limit: int = 5,
) -> List[str]:
    """
    Find historical failure_ids that actually contain this edge relation.
    We derive it by scanning entity_store (slow but accurate).
    If you later想加速，可以把 edge->failure_ids 在 upsert 时也存起来。
    """
    out: List[str] = []
    ent_store = kb.entity_store or {}

    # Map edge_type to entity fields
    if edge_type == "element_to_mode":
        src_field, tgt_field = "element_id", "mode_id"
    elif edge_type == "mode_to_cause":
        src_field, tgt_field = "mode_id", "cause_id"
    elif edge_type == "mode_to_effect":
        src_field, tgt_field = "mode_id", "effect_id"
    else:
        return out

    for fid, ent in ent_store.items():
        if (ent.get(src_field) == src_sid) and (ent.get(tgt_field) == tgt_sid):
            out.append(fid)
            if len(out) >= int(limit):
                break
    return out


# ------------------------------
# Core: KG expansion per node
# ------------------------------
@dataclass
class ExpandedChain:
    node_id: str
    element_sid: Optional[str]
    mode_sid: Optional[str]
    cause_sid: Optional[str]
    effect_sid: Optional[str]
    score: float
    # explainability
    evidence: Dict[str, Any]


def _expand_candidates_via_graph(
    kb: Any,
    *,
    element_hits: List[Tuple[str, float]],
    mode_hits: List[Tuple[str, float]],
    cause_hits: List[Tuple[str, float]],
    effect_hits: List[Tuple[str, float]],
    # graph expansion params
    graph_top_k: int = 10,
    graph_min_count: int = 1,
    w_sem: float = 1.0,
    w_edge: float = 0.6,
    edge_strength_mode: str = "log",
) -> Dict[str, List[Tuple[str, float, Dict[str, Any]]]]:
    """
    Build expanded candidate pools for each field:
    returns:
      {
        "element": [(sid, score, meta), ...],
        "mode":    [(sid, score, meta), ...],
        "cause":   [(sid, score, meta), ...],
        "effect":  [(sid, score, meta), ...],
      }

    Each entry score is a *field-local* score (not final chain score).
    meta carries explain: semantic source + edges used.
    """
    pools: Dict[str, Dict[str, Tuple[float, Dict[str, Any]]]] = {
        "element": {},
        "mode": {},
        "cause": {},
        "effect": {},
    }

    def _add(pool: str, sid: str, score: float, meta: Dict[str, Any]):
        cur = pools[pool].get(sid)
        if (cur is None) or (score > cur[0]):
            pools[pool][sid] = (score, meta)

    # ---- seed pools with semantic hits ----
    for sid, sim in element_hits:
        _add("element", sid, w_sem * sim, {"source": "semantic", "sim": sim})
    for sid, sim in mode_hits:
        _add("mode", sid, w_sem * sim, {"source": "semantic", "sim": sim})
    for sid, sim in cause_hits:
        _add("cause", sid, w_sem * sim, {"source": "semantic", "sim": sim})
    for sid, sim in effect_hits:
        _add("effect", sid, w_sem * sim, {"source": "semantic", "sim": sim})

    # ---- KG expand: element -> mode ----
    for e_sid, e_sim in element_hits:
        for m_sid, cnt in _edge_neighbors(
            kb, "element_to_mode", e_sid, top_k=graph_top_k, min_count=graph_min_count
        ):
            es = _edge_strength(cnt, mode=edge_strength_mode)
            score = (w_sem * e_sim) + (w_edge * es)
            _add(
                "mode",
                m_sid,
                score,
                {"source": "kg", "via": "element_to_mode", "src_sid": e_sid, "src_sim": e_sim, "edge_count": cnt, "edge_strength": es},
            )

    # ---- KG expand: mode -> cause/effect ----
    for m_sid, m_sim in mode_hits:
        # mode -> cause
        for c_sid, cnt in _edge_neighbors(
            kb, "mode_to_cause", m_sid, top_k=graph_top_k, min_count=graph_min_count
        ):
            es = _edge_strength(cnt, mode=edge_strength_mode)
            score = (w_sem * m_sim) + (w_edge * es)
            _add(
                "cause",
                c_sid,
                score,
                {"source": "kg", "via": "mode_to_cause", "src_sid": m_sid, "src_sim": m_sim, "edge_count": cnt, "edge_strength": es},
            )
        # mode -> effect
        for ef_sid, cnt in _edge_neighbors(
            kb, "mode_to_effect", m_sid, top_k=graph_top_k, min_count=graph_min_count
        ):
            es = _edge_strength(cnt, mode=edge_strength_mode)
            score = (w_sem * m_sim) + (w_edge * es)
            _add(
                "effect",
                ef_sid,
                score,
                {"source": "kg", "via": "mode_to_effect", "src_sid": m_sid, "src_sim": m_sim, "edge_count": cnt, "edge_strength": es},
            )

    # (可选) 如果你也想 “用 semantic cause/effect 去反推 mode”，需要存反向边；
    # 当前 edge_store 没有 cause->mode/effect->mode，因此先不做。

    # convert to sorted lists
    out: Dict[str, List[Tuple[str, float, Dict[str, Any]]]] = {}
    for k, mp in pools.items():
        lst = [(sid, sc_meta[0], sc_meta[1]) for sid, sc_meta in mp.items()]
        lst.sort(key=lambda x: x[1], reverse=True)
        out[k] = lst

    return out


def _compose_chains_from_pools(
    kb: Any,
    *,
    node_id: str,
    pools: Dict[str, List[Tuple[str, float, Dict[str, Any]]]],
    # composition params
    beam_mode: int = 10,
    beam_element: int = 5,
    beam_cause: int = 10,
    beam_effect: int = 10,
    require_element_mode_edge: bool = True,
    require_mode_cause_edge: bool = True,
    require_mode_effect_edge: bool = False,  # 很多 FMEA cause/effect 不一定同现，是否强制由你定
    w_element: float = 1.0,
    w_mode: float = 1.5,
    w_cause: float = 1.5,
    w_effect: float = 1.5,
    w_edge_bonus: float = 0.8,  # 组合时再加一次 edge bonus（避免仅靠 field-local score）
    edge_strength_mode: str = "log",
    top_n: int = 50,
    evidence_limit_ids: int = 5,
) -> List[ExpandedChain]:
    """
    Combine element/mode/cause/effect pools into candidate chains,
    enforcing KG connectivity by checking edge_store existence + counts.
    """
    element_pool = _cap_keep_best(pools.get("element", []), beam_element)
    mode_pool = _cap_keep_best(pools.get("mode", []), beam_mode)
    cause_pool = _cap_keep_best(pools.get("cause", []), beam_cause)
    effect_pool = _cap_keep_best(pools.get("effect", []), beam_effect)

    # quick maps for edge existence / count
    e2m = (kb.edge_store or {}).get("element_to_mode", {}) or {}
    m2c = (kb.edge_store or {}).get("mode_to_cause", {}) or {}
    m2e = (kb.edge_store or {}).get("mode_to_effect", {}) or {}

    def _get_cnt(store: Dict[str, Dict[str, int]], s: str, t: str) -> int:
        return int((store.get(s, {}) or {}).get(t, 0) or 0)

    out: List[ExpandedChain] = []

    # if element_pool empty, allow None element (some structure nodes might not provide element text)
    if not element_pool:
        element_pool = [(None, 0.0, {"source": "none"})]  # type: ignore

    for e_sid, e_sc, e_meta in element_pool:
        for m_sid, m_sc, m_meta in mode_pool:

            # enforce element->mode edge if required and element exists
            e2m_cnt = 0
            if e_sid and require_element_mode_edge:
                e2m_cnt = _get_cnt(e2m, e_sid, m_sid)
                if e2m_cnt <= 0:
                    continue

            for c_sid, c_sc, c_meta in cause_pool:
                m2c_cnt = _get_cnt(m2c, m_sid, c_sid)
                if require_mode_cause_edge and m2c_cnt <= 0:
                    continue

                # effect optional: if no effect candidates, allow None
                local_effect_pool = effect_pool or [(None, 0.0, {"source": "none"})]  # type: ignore
                for ef_sid, ef_sc, ef_meta in local_effect_pool:
                    m2e_cnt = 0
                    if ef_sid and require_mode_effect_edge:
                        m2e_cnt = _get_cnt(m2e, m_sid, ef_sid)
                        if m2e_cnt <= 0:
                            continue
                    elif ef_sid:
                        m2e_cnt = _get_cnt(m2e, m_sid, ef_sid)

                    # chain score: weighted sum of field scores + edge bonuses (by count)
                    score = (
                        (w_element * float(e_sc))
                        + (w_mode * float(m_sc))
                        + (w_cause * float(c_sc))
                        + (w_effect * float(ef_sc))
                    )

                    # edge bonuses
                    if e_sid and e2m_cnt > 0:
                        score += w_edge_bonus * _edge_strength(e2m_cnt, mode=edge_strength_mode)
                    if m2c_cnt > 0:
                        score += w_edge_bonus * _edge_strength(m2c_cnt, mode=edge_strength_mode)
                    if ef_sid and m2e_cnt > 0:
                        score += w_edge_bonus * _edge_strength(m2e_cnt, mode=edge_strength_mode)

                    evidence: Dict[str, Any] = {
                        "field_meta": {
                            "element": e_meta,
                            "mode": m_meta,
                            "cause": c_meta,
                            "effect": ef_meta,
                        },
                        "edges": {
                            "element_to_mode": {"count": e2m_cnt} if e_sid else None,
                            "mode_to_cause": {"count": m2c_cnt},
                            "mode_to_effect": {"count": m2e_cnt} if ef_sid else None,
                        },
                        "support_failure_ids": {
                            "element_to_mode": _support_failure_ids_for_edge(
                                kb, edge_type="element_to_mode", src_sid=e_sid, tgt_sid=m_sid, limit=evidence_limit_ids
                            ) if (e_sid and e2m_cnt > 0) else [],
                            "mode_to_cause": _support_failure_ids_for_edge(
                                kb, edge_type="mode_to_cause", src_sid=m_sid, tgt_sid=c_sid, limit=evidence_limit_ids
                            ) if (m2c_cnt > 0) else [],
                            "mode_to_effect": _support_failure_ids_for_edge(
                                kb, edge_type="mode_to_effect", src_sid=m_sid, tgt_sid=ef_sid, limit=evidence_limit_ids
                            ) if (ef_sid and m2e_cnt > 0) else [],
                        }
                    }

                    out.append(
                        ExpandedChain(
                            node_id=str(node_id),
                            element_sid=e_sid if e_sid else None,
                            mode_sid=m_sid,
                            cause_sid=c_sid,
                            effect_sid=ef_sid if ef_sid else None,
                            score=float(score),
                            evidence=evidence,
                        )
                    )

    out.sort(key=lambda x: x.score, reverse=True)
    return out[: int(top_n)]


# ------------------------------
# Public API: build with expansion
# ------------------------------
def build_failure_chains_with_kg_expansion(
    persist_dir: Union[str, Path],
    structure_input: Dict[str, Any],
    *,
    # vector recall
    top_k_per_field: int = 25,
    min_similarity: float = 0.5,
    min_count: Optional[int] = None,
    source_type: Optional[str] = None,
    # historical retrieval
    top_n_historical: int = 50,
    # graph expansion
    enable_expansion: bool = True,
    graph_top_k: int = 10,
    graph_min_count: int = 1,
    top_n_expanded: int = 50,
    # scoring weights
    weight_element: float = 1.0,
    weight_mode: float = 1.5,
    weight_cause: float = 1.5,
    weight_effect: float = 1.5,
    w_edge: float = 2.0,
    w_edge_bonus: float = 1.0,
    edge_strength_mode: str = "log",
    # constraints
    require_element_mode_edge: bool = True,
    require_mode_cause_edge: bool = False,
    require_mode_effect_edge: bool = False,
    # output
    dedupe_by_semantic_tuple: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
      {
        "historical": [ ... existing KB failure_id chains ... ],
        "expanded":   [ ... synthesized chains via KG expansion ... ],
      }
    """
    persist_dir = Path(persist_dir)
    kb = _load_kb(persist_dir)

    # 1) historical: reuse your existing function
    historical = build_failure_chains_from_structure(
        persist_dir=persist_dir,
        structure_input=structure_input,
        top_k_per_field=top_k_per_field,
        min_count=min_count,
        weight_element=weight_element,
        weight_mode=weight_mode,
        weight_cause=weight_cause,
        weight_effect=weight_effect,
        top_n=top_n_historical,
        source_type=source_type,
        # graph-constrained retrieval already inside this fn
        min_similarity=min_similarity,
    )

    expanded_out: List[Dict[str, Any]] = []
    if not enable_expansion:
        return {"historical": historical, "expanded": expanded_out}

    nodes = structure_input.get("nodes", []) or []
    for node in nodes:
        node_id = node.get("element_id") or ""

        element_text = _safe_norm_text(node.get("failure_element"))
        modes = [ _safe_norm_text(x) for x in (node.get("modes", []) or []) if _safe_norm_text(x) ]
        causes = [ _safe_norm_text(x) for x in (node.get("causes", []) or []) if _safe_norm_text(x) ]
        effects = [ _safe_norm_text(x) for x in (node.get("effects", []) or []) if _safe_norm_text(x) ]

        # 2) semantic recall: per-field hits
        element_hits = _semantic_query_hits(
            persist_dir, element_text, "element",
            n_results=top_k_per_field, min_similarity=min_similarity,
            min_count=min_count, source_type=source_type
        ) if element_text else []

        mode_hits: List[Tuple[str, float]] = []
        for m in modes:
            mode_hits.extend(_semantic_query_hits(
                persist_dir, m, "mode",
                n_results=top_k_per_field, min_similarity=min_similarity,
                min_count=min_count, source_type=source_type
            ))
        mode_hits.sort(key=lambda x: x[1], reverse=True)
        mode_hits = _cap_keep_best(mode_hits, top_k_per_field)

        cause_hits: List[Tuple[str, float]] = []
        for c in causes:
            cause_hits.extend(_semantic_query_hits(
                persist_dir, c, "cause",
                n_results=top_k_per_field, min_similarity=min_similarity,
                min_count=min_count, source_type=source_type
            ))
        cause_hits.sort(key=lambda x: x[1], reverse=True)
        cause_hits = _cap_keep_best(cause_hits, top_k_per_field)

        effect_hits: List[Tuple[str, float]] = []
        for e in effects:
            effect_hits.extend(_semantic_query_hits(
                persist_dir, e, "effect",
                n_results=top_k_per_field, min_similarity=min_similarity,
                min_count=min_count, source_type=source_type
            ))
        effect_hits.sort(key=lambda x: x[1], reverse=True)
        effect_hits = _cap_keep_best(effect_hits, top_k_per_field)
        # print("element_hits:", element_hits[:3])
        # print("mode_hits:", mode_hits[:3])
        # print("cause_hits:", cause_hits[:3])
        # print("effect_hits:", effect_hits[:3])

        # 3) KG expansion: build pools
        pools = _expand_candidates_via_graph(
            kb,
            element_hits=element_hits,
            mode_hits=mode_hits,
            cause_hits=cause_hits,
            effect_hits=effect_hits,
            graph_top_k=graph_top_k,
            graph_min_count=graph_min_count,
            w_sem=1.0,
            w_edge=w_edge,
            edge_strength_mode=edge_strength_mode,
        )
        print("mode pool size:", len(pools["mode"]))
        print("cause pool size:", len(pools["cause"]))
        print("effect pool size:", len(pools["effect"]))

        # 4) Compose synthesized chains from pools with connectivity constraints
        expanded = _compose_chains_from_pools(
            kb,
            node_id=str(node_id),
            pools=pools,
            require_element_mode_edge=require_element_mode_edge,
            require_mode_cause_edge=require_mode_cause_edge,
            require_mode_effect_edge=require_mode_effect_edge,
            w_element=weight_element,
            w_mode=weight_mode,
            w_cause=weight_cause,
            w_effect=weight_effect,
            w_edge_bonus=w_edge_bonus,
            edge_strength_mode=edge_strength_mode,
            top_n=top_n_expanded,
        )

        # 5) Format output
        for ch in expanded:
            expanded_out.append({
                "node_id": ch.node_id,
                "chain_type": "expanded",
                "failure_id": None,  # synthesized
                "semantic_ids": {
                    "element": ch.element_sid,
                    "mode": ch.mode_sid,
                    "cause": ch.cause_sid,
                    "effect": ch.effect_sid,
                },
                "element": _get_semantic_text(kb, ch.element_sid) if ch.element_sid else None,
                "mode": _get_semantic_text(kb, ch.mode_sid) if ch.mode_sid else None,
                "cause": _get_semantic_text(kb, ch.cause_sid) if ch.cause_sid else None,
                "effect": _get_semantic_text(kb, ch.effect_sid) if ch.effect_sid else None,
                "score": round(float(ch.score), 4),
                "evidence": ch.evidence,
            })

    # 6) Optional dedupe: avoid returning same semantic tuple multiple times
    if dedupe_by_semantic_tuple:
        seen = set()
        deduped = []
        for r in expanded_out:
            sids = r.get("semantic_ids") or {}
            key = (sids.get("element"), sids.get("mode"), sids.get("cause"), sids.get("effect"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        expanded_out = deduped

    # sort expanded globally
    expanded_out.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    return {"historical": historical, "expanded": expanded_out}


# Visulization print
def _short_list(x: List[Any], n: int = 5) -> str:
    if not x:
        return "[]"
    if len(x) <= n:
        return json.dumps(x, ensure_ascii=False)
    return json.dumps(x[:n], ensure_ascii=False)[:-1] + f', "...(+{len(x)-n})"]'


def print_failure_result(
    result: Dict[str, Any],
    *,
    top_n_per_node: int = 5,
    show_evidence: bool = True,
    show_support_ids: bool = True,
    max_support_ids: int = 5,
):
    historical = result.get("historical", []) or []
    expanded = result.get("expanded", []) or []

    def group_by_node(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        g = defaultdict(list)
        for r in items:
            g[str(r.get("node_id", ""))].append(r)
        for k in g:
            g[k].sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return g

    hist_g = group_by_node(historical)
    exp_g = group_by_node(expanded)
    all_nodes = sorted(set(list(hist_g.keys()) + list(exp_g.keys())))

    print("\n" + "=" * 110)
    print(f"RESULT SUMMARY | historical={len(historical)} | expanded={len(expanded)} | nodes={len(all_nodes)}")
    print("=" * 110 + "\n")

    for node_id in all_nodes:
        print(f"\n{'#' * 40}  NODE: {node_id}  {'#' * 40}")

        # -------------------------
        # Historical (from failure_id)
        # -------------------------
        hlist = hist_g.get(node_id, [])
        print(f"\n--- Historical (KB failure_id) | count={len(hlist)} | show_top={min(len(hlist), top_n_per_node)} ---")
        for i, r in enumerate(hlist[:top_n_per_node], 1):
            print(f"\n[H{i}] score={r.get('score')}  failure_id={r.get('failure_id')}")
            print(f"     element: {r.get('element')}")
            print(f"       mode: {r.get('mode')}")
            print(f"      cause: {r.get('cause')}")
            print(f"     effect: {r.get('effect')}")
            print(f" matched_fields: {r.get('matched_fields')}")
            md = r.get("match_detail") or {}
            # 每个field命中了哪些semantic_id
            for ft in ["element", "mode", "cause", "effect"]:
                hits = md.get(ft) or []
                if hits:
                    # hits like: {"semantic_id":..., "similarity":..., "structure_text":...}
                    top_hits = hits[:3]
                    pretty = [
                        f"{h.get('semantic_id')} (sim={h.get('similarity')}, q='{h.get('structure_text')}')"
                        for h in top_hits
                    ]
                    suffix = "" if len(hits) <= 3 else f" ...(+{len(hits)-3})"
                    print(f"   {ft:>7} hits: {pretty}{suffix}")

        # -------------------------
        # Expanded (synthesized via KG)
        # -------------------------
        elist = exp_g.get(node_id, [])
        print(f"\n--- Expanded (KG synthesized) | count={len(elist)} | show_top={min(len(elist), top_n_per_node)} ---")
        for i, r in enumerate(elist[:top_n_per_node], 1):
            print(f"\n[E{i}] score={r.get('score')}  chain_type={r.get('chain_type')}  failure_id={r.get('failure_id')}")
            sids = r.get("semantic_ids") or {}
            print(f" semantic_ids: element={sids.get('element')}  mode={sids.get('mode')}  cause={sids.get('cause')}  effect={sids.get('effect')}")
            print(f"     element: {r.get('element')}")
            print(f"       mode: {r.get('mode')}")
            print(f"      cause: {r.get('cause')}")
            print(f"     effect: {r.get('effect')}")

            if show_evidence:
                ev = r.get("evidence") or {}
                edges = (ev.get("edges") or {})
                print(" edges:")
                # edges: {"element_to_mode": {"count":..} or None, "mode_to_cause": {"count":..}, ...}
                for k in ["element_to_mode", "mode_to_cause", "mode_to_effect"]:
                    v = edges.get(k)
                    if not v:
                        print(f"   - {k}: None")
                    else:
                        print(f"   - {k}: count={v.get('count')}")

                if show_support_ids:
                    supp = (ev.get("support_failure_ids") or {})
                    print(" support_failure_ids:")
                    for k in ["element_to_mode", "mode_to_cause", "mode_to_effect"]:
                        ids = supp.get(k) or []
                        print(f"   - {k}: {_short_list(ids, n=max_support_ids)}")

    print("\n" + "=" * 110)
    print("END")
    print("=" * 110 + "\n")

if  __name__ == "__main__":


    from pprint import pprint

    # -----------------------------------------------------
    # 1) KB Path
    # -----------------------------------------------------
    KB_PATH = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb"
    )

    # -----------------------------------------------------
    # 2) Structure Input
    # -----------------------------------------------------
    structure_input = {
        "product_domain": "motor_drives",
        "nodes": [
            {
                "element_id": "E1",
                "failure_element": "Power train",
                "modes": [
                    "Incorrect",
                    "No pulses seen",
                    "No voltage applied",
                    "Incorrect torque applied",
                    "Not enough torque",
                    "Motor breaks/overheats (e.g. resulting in demagnetisation)",
                    "Unstable regulation",
                    "High loss in torque transfer",
                    "Gear train breaks/wears out",
                    "Transmission ratio drifts",
                    "creates too much noise"
                ],
                "causes": [
                    "Gears loose on motor shaft (slips)",
                    "External force on spline",
                    "Motor can not provide enough torque",
                    "Too much friction in gear train",
                    "Gears material/design choice",
                    "Manufacturing tolerances of gears",
                    "Lubrication choice (e.g. degradation)",
                    "Motor design (temperature spec, actuation length/duty cycle)",
                    "Encoder circuit crosstalk",
                    "HW cannot supply enough power",
                    "ADC measurements incorrect (incl. bandwidth)",
                    "Wrong motor driver dimension (current rating etc.)",
                    "Overcurrent detection incorrect (threshold etc.)",
                    "Incorrect control loop (bandwidth)",
                    "Motor not shorted while device is not powered",
                    "Control parameters incorrect",
                    "Thermal protection fails (e.g. I2T)"
                ],
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
                    "Update takes too much time (>5 minutes)"
                ]
            }
        ]
    }
    result = build_failure_chains_with_kg_expansion(structure_input=structure_input, persist_dir= KB_PATH)
    print_failure_result(result, top_n_per_node=15, show_evidence=True, show_support_ids=True)



