
from typing import Dict, List, Optional, Any, Tuple


def build_semi_chain_query_text_from_graph_results(
    graph_results: Dict[str, Any],
    *,
    target_n_mc: Optional[int] = 20,
    target_n_me: Optional[int] = 20,
    strict_unique: bool = True,
    blank: str = "____",
    # filters (from query_combo_stats)
    min_count: int = 2,
    min_best_score: float = 0.5,
    # show stats line
    show_stats: bool = True,
) -> str:
    """
    基于 generate_query_unique_chains() 的输出 graph_results，构造给 LLM 的 semi chain query text。
    输出两部分：
      - MC: 给 mode+cause（用 query 文本展示），effect 留 blank
      - ME: 给 mode+effect（用 query 文本展示），cause 留 blank

    ✅本版改动（按你例子）：
    - failure_mode 行：直接用 query_mode（不用 matched mode）
    - MC 的 failure_cause 行：直接用 query_cause（不用 matched cause）
    - ME 的 failure_effect 行：直接用 query_effect（不用 matched effect）
    - 不再输出单独 query_text 行
    - stats 保留（可关 show_stats=False）
    """

    if not graph_results:
        return "No graph results."

    partial = (graph_results.get("partial_chains") or [])
    if not partial:
        return "No partial (semi) chains were generated."

    stats = (graph_results.get("query_combo_stats") or {})
    stats_mc = stats.get("MC") or []
    stats_me = stats.get("ME") or []

    def _safe(x) -> str:
        return "" if x is None else str(x).strip()

    def _norm(x) -> str:
        return " ".join(_safe(x).split()).lower()

    def _uniq_key_mc(r: Dict[str, Any]) -> Tuple[str, str, str]:
        # 用 query 文本去重更符合你现在的展示逻辑
        return (_norm(r.get("failure_element")), _norm(r.get("query_mode")), _norm(r.get("query_cause")))

    def _uniq_key_me(r: Dict[str, Any]) -> Tuple[str, str, str]:
        return (_norm(r.get("failure_element")), _norm(r.get("query_mode")), _norm(r.get("query_effect")))

    # build allowed query_combo set from stats
    allowed_mc: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in stats_mc:
        try:
            c = int(r.get("count", 0) or 0)
            s = float(r.get("best_score", 0.0) or 0.0)
        except Exception:
            continue
        if c < min_count or s < min_best_score:
            continue
        combo = r.get("query_combo") or []
        if len(combo) < 2:
            continue
        key = (_safe(combo[0]), _safe(combo[1]))  # (query_mode, query_cause)
        allowed_mc[key] = {"count": c, "best_score": s}

    allowed_me: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in stats_me:
        try:
            c = int(r.get("count", 0) or 0)
            s = float(r.get("best_score", 0.0) or 0.0)
        except Exception:
            continue
        if c < min_count or s < min_best_score:
            continue
        combo = r.get("query_combo") or []
        if len(combo) < 2:
            continue
        key = (_safe(combo[0]), _safe(combo[1]))  # (query_mode, query_effect)
        allowed_me[key] = {"count": c, "best_score": s}

    mc_rows_all = [r for r in partial if _safe(r.get("chain_type")) == "MC"]
    me_rows_all = [r for r in partial if _safe(r.get("chain_type")) == "ME"]

    def _pass_mc(r: Dict[str, Any]) -> bool:
        return (_safe(r.get("query_mode")), _safe(r.get("query_cause"))) in allowed_mc

    def _pass_me(r: Dict[str, Any]) -> bool:
        return (_safe(r.get("query_mode")), _safe(r.get("query_effect"))) in allowed_me

    mc_rows = [r for r in mc_rows_all if _pass_mc(r)]
    me_rows = [r for r in me_rows_all if _pass_me(r)]

    # sort by semi score
    mc_rows.sort(key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)
    me_rows.sort(key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)

    def _select(rows: List[Dict[str, Any]], kind: str, target_n: Optional[int]) -> List[Dict[str, Any]]:
        if target_n is None:
            target_n = len(rows)

        selected: List[Dict[str, Any]] = []
        seen = set()
        duplicates: List[Dict[str, Any]] = []

        for r in rows:
            key = _uniq_key_mc(r) if kind == "MC" else _uniq_key_me(r)
            if key in seen:
                duplicates.append(r)
                continue
            seen.add(key)
            selected.append(r)
            if len(selected) >= target_n:
                break

        if not strict_unique and len(selected) < target_n:
            need = target_n - len(selected)
            selected.extend(duplicates[:need])

        return selected

    mc_sel = _select(mc_rows, "MC", target_n_mc)
    me_sel = _select(me_rows, "ME", target_n_me)

    blocks: List[str] = []

    blocks.append(
        f"## SEMI CHAINS - MC (effect left blank for LLM to fill)\n"
        f"## filter: count>={min_count}, best_score>={min_best_score}\n"
    )
    if not mc_sel:
        blocks.append("(No MC semi chains after filtering)\n")
    else:
        for i, r in enumerate(mc_sel, start=1):
            element = _safe(r.get("failure_element")) or blank

            # ✅展示用 query 文本
            mode_q = _safe(r.get("query_mode")) or _safe(r.get("mode")) or blank
            cause_q = _safe(r.get("query_cause")) or _safe(r.get("cause")) or blank
            effect_blank = blank

            st = allowed_mc.get((_safe(r.get("query_mode")), _safe(r.get("query_cause"))), {})
            cnt = st.get("count")
            bst = st.get("best_score")

            blocks.append(
                f"[MC_SEMI_CHAIN {i}]"
                + "\n"
                f"failure_element: {element}\n"
                f"failure_mode: {mode_q}\n"
                f"failure_cause: {cause_q}\n"
                f"failure_effect: {effect_blank}\n"
                + (
                    f"stats: count={cnt}, best_score={float(bst):.6f}\n"
                    if show_stats and cnt is not None and bst is not None
                    else ""
                )
            )

    blocks.append(
        f"\n## SEMI CHAINS - ME (cause left blank for LLM to fill)\n"
    )
    if not me_sel:
        blocks.append("(No ME semi chains after filtering)\n")
    else:
        for i, r in enumerate(me_sel, start=1):
            element = _safe(r.get("failure_element")) or blank

            # ✅failure_mode 用 query_mode；failure_effect 用 query_effect
            mode_q = _safe(r.get("query_mode")) or _safe(r.get("mode")) or blank
            effect_q = _safe(r.get("query_effect")) or _safe(r.get("effect")) or blank
            cause_blank = blank

            st = allowed_me.get((_safe(r.get("query_mode")), _safe(r.get("query_effect"))), {})
            cnt = st.get("count")
            bst = st.get("best_score")

            blocks.append(
                f"[ME_SEMI_CHAIN {i}]"
                + "\n"
                f"failure_element: {element}\n"
                f"failure_mode: {mode_q}\n"
                f"failure_cause: {cause_blank}\n"
                f"failure_effect: {effect_q}\n"
                + (
                    f"stats: count={cnt}, best_score={float(bst):.6f}\n"
                    if show_stats and cnt is not None and bst is not None
                    else ""
                )
            )

    return "\n".join(blocks)