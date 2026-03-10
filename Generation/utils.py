from typing import Any, Dict, List, Optional, Tuple, Set
from Retriever.PPL_score import rank_causes_with_lm, rank_effects_with_lm,ChainPPLEvaluator_v2

evaluator = ChainPPLEvaluator_v2("gpt2")

def build_semi_chain_query_text_from_graph_results(
    graph_results: Dict[str, Any],
    structure_element,
    target_n_mc: Optional[int] = 20,
    target_n_me: Optional[int] = 20,
    strict_unique: bool = True,
    blank: str = "____",
    # filters (from query_combo_stats)
    min_count: int = 2,
    min_best_score: float = 0.5,
    # show stats line
    show_stats: bool = True,
    # NEW: keep option to join cartesian
    join_cartesian: bool = True,
) -> str:
    """
    基于 query_combo_stats 构造给 LLM 的 semi chain query text。

    join_cartesian:
      - True : 三部分输出
          PART 1) FULL CHAINS: 同一 query_mode 下 MC × ME 拼接（去重）
          PART 2) 剩余 MC: effect 留 blank
          PART 3) 剩余 ME: cause 留 blank
      - False: 两部分输出（不拼接）
          MC semi-chains（effect blank）
          ME semi-chains（cause blank）

    过滤逻辑：严格以 query_combo_stats 的 (count>=min_count & best_score>=min_best_score) 为准。
    """

    if not graph_results:
        return "No graph results."

    stats = (graph_results.get("query_combo_stats") or {})
    if not stats:
        return "No query_combo_stats found in graph results."

    def _safe(x) -> str:
        return "" if x is None else str(x).strip()

    def _norm(x) -> str:
        # collapse whitespace + lower，避免多空格/换行/大小写/不可见空格导致 key 对不上
        return " ".join(_safe(x).split()).lower()

    # ---------------------------
    # 1) 取 strong MC/ME stats（完全按你 strong 筛选逻辑）
    # ---------------------------
    def _get_strong(kind: str) -> List[Dict[str, Any]]:
        rows = stats.get(kind) or []
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                c = int(r.get("count", 0) or 0)
                s = float(r.get("best_score", 0.0) or 0.0)
            except Exception:
                continue
            if c < min_count or s < min_best_score:
                continue
            out.append(
                {
                    "chain_type": kind,
                    "query_combo": r.get("query_combo") or [],
                    "count": c,
                    "best_score": s,
                    "example": r.get("example") or {},
                }
            )
        # 排序：best_score desc, count desc（与你 print 里 _rank_key reverse 一致）
        out.sort(key=lambda x: (float(x["best_score"]), int(x["count"])), reverse=True)
        return out

    mc_rows: List[Dict[str, Any]] = _get_strong("MC")
    me_rows: List[Dict[str, Any]] = _get_strong("ME")

    if not mc_rows and not me_rows:
        return f"No strong semi chains found in query_combo_stats (count>={min_count}, best_score>={min_best_score})."

    def _rank_key(r: Dict[str, Any]) -> Tuple[float, int]:
        return (float(r.get("best_score", 0.0)), int(r.get("count", 0)))

    def _get_qc(r: Dict[str, Any]) -> List[str]:
        return (r.get("query_combo") or [])

    def _get_mc_qmode_qcause(r: Dict[str, Any]) -> Tuple[str, str]:
        qc = _get_qc(r)
        q_mode, q_cause = (qc + ["", ""])[:2]
        return _safe(q_mode), _safe(q_cause)

    def _get_me_qmode_qeffect(r: Dict[str, Any]) -> Tuple[str, str]:
        qc = _get_qc(r)
        q_mode, q_effect = (qc + ["", ""])[:2]
        return _safe(q_mode), _safe(q_effect)

    # ---------------------------
    # 2) 选择（去重 + obey strict_unique + obey target_n）
    # ---------------------------
    def _uniq_key_mc(r: Dict[str, Any]) -> Tuple[str, str]:
        q_mode, q_cause = _get_mc_qmode_qcause(r)
        return (_norm(q_mode), _norm(q_cause))

    def _uniq_key_me(r: Dict[str, Any]) -> Tuple[str, str]:
        q_mode, q_effect = _get_me_qmode_qeffect(r)
        return (_norm(q_mode), _norm(q_effect))

    def _select(rows: List[Dict[str, Any]], kind: str, target_n: Optional[int]) -> List[Dict[str, Any]]:
        if target_n is None:
            target_n = len(rows)

        rows_sorted = sorted(rows, key=_rank_key, reverse=True)

        selected: List[Dict[str, Any]] = []
        seen = set()
        duplicates: List[Dict[str, Any]] = []

        for r in rows_sorted:
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

    # ---------------------------
    # MODE A) 不拼接：直接输出 MC/ME
    # ---------------------------
    if not join_cartesian:
        mc_sel = _select(mc_rows, "MC", target_n_mc)
        me_sel = _select(me_rows, "ME", target_n_me)

        blocks: List[str] = []
        blocks.append(
            "## SEMI CHAINS - MC (effect left blank for LLM to fill)\n"
            # f"## filter: count>={min_count}, best_score>={min_best_score}\n"
        )
        if not mc_sel:
            blocks.append("(No MC semi chains after filtering)\n")
        else:
            for i, r in enumerate(mc_sel, start=1):
                q_mode, q_cause = _get_mc_qmode_qcause(r)
                ex = r.get("example") or {}
                blocks.append(
                    f"[MC_SEMI_CHAIN {i}]\n"
                    f"failure_element: {structure_element}\n"
                    f"failure_mode: {q_mode or blank}\n"
                    f"failure_cause: {q_cause or blank}\n"
                    f"failure_effect: {blank}\n"
                    + (
                        f"stats: count={int(r.get('count',0))}, best_score={float(r.get('best_score',0.0)):.6f}\n"
                        if show_stats
                        else ""
                    )
                    # + (
                    #     "example:\n"
                    #     f"  matched_mode: {_safe(ex.get('matched_mode'))}\n"
                    #     f"  matched_cause: {_safe(ex.get('matched_cause'))}\n"
                    #     f"  edge_count: {_safe(ex.get('edge_count'))}\n"
                    #     if ex
                    #     else ""
                    # )
                )

        blocks.append(
            "\n## SEMI CHAINS - ME (cause left blank for LLM to fill)\n"
            # f"## filter: count>={min_count}, best_score>={min_best_score}\n"
        )
        if not me_sel:
            blocks.append("(No ME semi chains after filtering)\n")
        else:
            for i, r in enumerate(me_sel, start=1):
                q_mode, q_effect = _get_me_qmode_qeffect(r)
                ex = r.get("example") or {}
                blocks.append(
                    f"[ME_SEMI_CHAIN {i}]\n"
                    f"failure_element: {structure_element}\n"
                    f"failure_mode: {q_mode or blank}\n"
                    f"failure_cause: {blank}\n"
                    f"failure_effect: {q_effect or blank}\n"
                    + (
                        f"stats: count={int(r.get('count',0))}, best_score={float(r.get('best_score',0.0)):.6f}\n"
                        if show_stats
                        else ""
                    )
                    # + (
                    #     "example:\n"
                    #     f"  matched_mode: {_safe(ex.get('matched_mode'))}\n"
                    #     f"  matched_effect: {_safe(ex.get('matched_effect'))}\n"
                    #     f"  edge_count: {_safe(ex.get('edge_count'))}\n"
                    #     if ex
                    #     else ""
                    # )
                )

        return "\n".join(blocks)

    # ---------------------------
    # MODE B) 拼接：三段输出（对齐 print_strong_semi_chains_3parts_cartesian）
    # ---------------------------

    # group by query_mode（用 _norm 做 key）
    mc_by_qmode: Dict[str, List[Dict[str, Any]]] = {}
    me_by_qmode: Dict[str, List[Dict[str, Any]]] = {}

    for r in mc_rows:
        q_mode, _ = _get_mc_qmode_qcause(r)
        if q_mode:
            mc_by_qmode.setdefault(_norm(q_mode), []).append(r)

    for r in me_rows:
        q_mode, _ = _get_me_qmode_qeffect(r)
        if q_mode:
            me_by_qmode.setdefault(_norm(q_mode), []).append(r)

    # cartesian join within same query_mode, then dedupe
    full_chains: List[Dict[str, Any]] = []
    full_keys: Set[Tuple[str, str, str, str, str, str]] = set()

    used_mc_ids: Set[int] = set()
    used_me_ids: Set[int] = set()

    common_modes = sorted(set(mc_by_qmode.keys()) & set(me_by_qmode.keys()))
    for q_mode_norm in common_modes:
        mcs = sorted(mc_by_qmode[q_mode_norm], key=_rank_key, reverse=True)
        mes = sorted(me_by_qmode[q_mode_norm], key=_rank_key, reverse=True)

        # display mode：优先 MC 的原文
        q_mode_display = _get_mc_qmode_qcause(mcs[0])[0] if mcs else (
            _get_me_qmode_qeffect(mes[0])[0] if mes else ""
        )

        for mc in mcs:
            _, q_cause = _get_mc_qmode_qcause(mc)
            ex_mc = mc.get("example") or {}

            for me in mes:
                _, q_effect = _get_me_qmode_qeffect(me)
                ex_me = me.get("example") or {}

                matched_mode = _safe(ex_mc.get("matched_mode") or ex_me.get("matched_mode") or "")
                matched_cause = _safe(ex_mc.get("matched_cause") or "")
                matched_effect = _safe(ex_me.get("matched_effect") or "")

                key = (
                    _norm(q_mode_display),
                    _norm(q_cause),
                    _norm(q_effect),
                    _norm(matched_mode),
                    _norm(matched_cause),
                    _norm(matched_effect),
                )
                if key in full_keys:
                    continue
                full_keys.add(key)

                used_mc_ids.add(id(mc))
                used_me_ids.add(id(me))

                full_chains.append(
                    {
                        "query_mode": q_mode_display,
                        "query_cause": q_cause,
                        "query_effect": q_effect,
                        "mc_count": mc.get("count", 0),
                        "mc_best_score": mc.get("best_score", 0.0),
                        "me_count": me.get("count", 0),
                        "me_best_score": me.get("best_score", 0.0),
                        "mc_edge_count": ex_mc.get("edge_count", ""),
                        "me_edge_count": ex_me.get("edge_count", ""),
                        "matched_mode": matched_mode,
                        "matched_cause": matched_cause,
                        "matched_effect": matched_effect,
                    }
                )

    full_chains.sort(
        key=lambda d: (
            _norm(d.get("query_mode", "")),
            -float(d.get("mc_best_score", 0.0)),
            -int(d.get("mc_count", 0)),
            -float(d.get("me_best_score", 0.0)),
            -int(d.get("me_count", 0)),
            _norm(d.get("query_cause", "")),
            _norm(d.get("query_effect", "")),
        )
    )

    remaining_mc = [r for r in mc_rows if id(r) not in used_mc_ids]
    remaining_me = [r for r in me_rows if id(r) not in used_me_ids]

    mc_sel = _select(remaining_mc, "MC", target_n_mc)
    me_sel = _select(remaining_me, "ME", target_n_me)

    # FULL 行数：用一个合理映射（不新增参数）
    if target_n_mc is None and target_n_me is None:
        full_limit = None
    else:
        a = 0 if target_n_mc is None else int(target_n_mc)
        b = 0 if target_n_me is None else int(target_n_me)
        full_limit = max(a, b)

    blocks: List[str] = []
    blocks.append(
        "## PART 1) FULL CHAINS (MC×ME within same query_mode)\n"
        f"## filter: count>={min_count}, best_score>={min_best_score}\n"
    )
    if not full_chains:
        blocks.append("(No FULL chains after cartesian join)\n")
    else:
        view = full_chains if full_limit is None else full_chains[:full_limit]
        for i, fc in enumerate(view, start=1):
            blocks.append(
                f"[FULL_CHAIN {i}]\n"
                f"failure_element: {structure_element}\n"
                f"failure_mode: {fc['query_mode']}\n"
                f"failure_cause: {fc['query_cause']}\n"
                f"failure_effect: {fc['query_effect']}\n"
                + (
                    f"stats_mc: count={int(fc['mc_count'])}, best_score={float(fc['mc_best_score']):.6f}, edge_count={fc['mc_edge_count']}\n"
                    f"stats_me: count={int(fc['me_count'])}, best_score={float(fc['me_best_score']):.6f}, edge_count={fc['me_edge_count']}\n"
                    if show_stats
                    else ""
                )
                + (
                    "example:\n"
                    f"  matched_mode: {fc['matched_mode']}\n"
                    f"  matched_cause: {fc['matched_cause']}\n"
                    f"  matched_effect: {fc['matched_effect']}\n"
                    if (fc.get("matched_mode") or fc.get("matched_cause") or fc.get("matched_effect"))
                    else ""
                )
            )

    blocks.append(
        "\n## PART 2) REMAINING MC SEMI CHAINS (effect left blank for LLM to fill)\n"
        f"## filter: count>={min_count}, best_score>={min_best_score}\n"
    )
    if not mc_sel:
        blocks.append("(No remaining MC semi chains)\n")
    else:
        for i, r in enumerate(mc_sel, start=1):
            q_mode, q_cause = _get_mc_qmode_qcause(r)
            ex = r.get("example") or {}
            blocks.append(
                f"[MC_SEMI_CHAIN {i}]\n"
                f"failure_element: {structure_element}\n"
                f"failure_mode: {q_mode or blank}\n"
                f"failure_cause: {q_cause or blank}\n"
                f"failure_effect: {blank}\n"
                + (
                    f"stats: count={int(r.get('count',0))}, best_score={float(r.get('best_score',0.0)):.6f}\n"
                    if show_stats
                    else ""
                )
                # + (
                #     "example:\n"
                #     f"  matched_mode: {_safe(ex.get('matched_mode'))}\n"
                #     f"  matched_cause: {_safe(ex.get('matched_cause'))}\n"
                #     f"  edge_count: {_safe(ex.get('edge_count'))}\n"
                #     if ex
                #     else ""
                # )
            )

    blocks.append(
        "\n## PART 3) REMAINING ME SEMI CHAINS (cause left blank for LLM to fill)\n"
        f"## filter: count>={min_count}, best_score>={min_best_score}\n"
    )
    if not me_sel:
        blocks.append("(No remaining ME semi chains)\n")
    else:
        for i, r in enumerate(me_sel, start=1):
            q_mode, q_effect = _get_me_qmode_qeffect(r)
            ex = r.get("example") or {}
            blocks.append(
                f"[ME_SEMI_CHAIN {i}]\n"
                f"failure_element: {structure_element}\n"
                f"failure_mode: {q_mode or blank}\n"
                f"failure_cause: {blank}\n"
                f"failure_effect: {q_effect or blank}\n"
                + (
                    f"stats: count={int(r.get('count',0))}, best_score={float(r.get('best_score',0.0)):.6f}\n"
                    if show_stats
                    else ""
                )
                # + (
                #     "example:\n"
                #     f"  matched_mode: {_safe(ex.get('matched_mode'))}\n"
                #     f"  matched_effect: {_safe(ex.get('matched_effect'))}\n"
                #     f"  edge_count: {_safe(ex.get('edge_count'))}\n"
                #     if ex
                #     else ""
                # )
            )

    return "\n".join(blocks)


from typing import Dict, Any, Optional, List, Tuple, Set, Callable

def _build_cause_discipline_map(structure_input, node_index=0):
    nodes = structure_input.get("nodes") or []
    if node_index >= len(nodes):
        return {}

    node = nodes[node_index]
    causes_dict = node.get("causes") or {}

    m = {}
    for discipline, cause_list in causes_dict.items():
        for c in cause_list:
            key = " ".join(str(c).strip().split()).lower()
            m[key] = discipline

    return m

def build_semi_chain_query_text_from_graph_results_PPL(
    graph_results: Dict[str, Any],
    structure_element,
    target_n_mc: Optional[int] = 20,
    target_n_me: Optional[int] = 20,
    strict_unique: bool = True,
    blank: str = "____",
    # filters (from query_combo_stats)
    min_count: int = 2,
    min_best_score: float = 0.5,
    # show stats line
    show_stats: bool = True,
    # keep option to join cartesian
    join_cartesian: bool = True,

    # =========================
    # NEW: PPL candidates injection
    # =========================
    enable_ppl_candidates: bool = True,
    evaluator=evaluator,
    structure_input=None,
    node_index: int = 0,
    ppl_top_k: int = 5,
    # optional: filter or formatting controls
    ppl_max_show: int = 10,
    ppl_min_score: Optional[float] = None,   # if your ranker returns a score (higher better)
    ppl_max_ppl: Optional[float] = None,     # if your ranker returns ppl (lower better)
) -> str:
    """
    基于 query_combo_stats 构造给 LLM 的 semi chain query text。

    输出结构不变：仍旧是 element + mode + cause/effect + blank。
    NEW: 在每个 semi chain 下追加 PPL 过滤后的候选项，帮助 LLM 填 blank。
    """

    if not graph_results:
        return "No graph results."

    stats = (graph_results.get("query_combo_stats") or {})
    if not stats:
        return "No query_combo_stats found in graph results."

    def _safe(x) -> str:
        return "" if x is None else str(x).strip()

    def _norm(x) -> str:
        return " ".join(_safe(x).split()).lower()
    

    cause_disc_map = _build_cause_discipline_map(structure_input, node_index)
    def _cause_with_discipline(cause: str, cause_disc_map):
        key = " ".join(str(cause).strip().split()).lower()
        disc = cause_disc_map.get(key)
        return f"[{disc}] {cause}" if disc else cause

    # ---------------------------
    # 1) strong MC/ME stats
    # ---------------------------
    def _get_strong(kind: str) -> List[Dict[str, Any]]:
        rows = stats.get(kind) or []
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                c = int(r.get("count", 0) or 0)
                s = float(r.get("best_score", 0.0) or 0.0)
            except Exception:
                continue
            if c < min_count or s < min_best_score:
                continue
            out.append(
                {
                    "chain_type": kind,
                    "query_combo": r.get("query_combo") or [],
                    "count": c,
                    "best_score": s,
                    "example": r.get("example") or {},
                }
            )
        out.sort(key=lambda x: (float(x["best_score"]), int(x["count"])), reverse=True)
        return out

    mc_rows: List[Dict[str, Any]] = _get_strong("MC")
    me_rows: List[Dict[str, Any]] = _get_strong("ME")

    if not mc_rows and not me_rows:
        return (
            f"No strong semi chains found in query_combo_stats "
            # f"(count>={min_count}, best_score>={min_best_score})."
        )

    def _rank_key(r: Dict[str, Any]) -> Tuple[float, int]:
        return (float(r.get("best_score", 0.0)), int(r.get("count", 0)))

    def _get_qc(r: Dict[str, Any]) -> List[str]:
        return (r.get("query_combo") or [])

    def _get_mc_qmode_qcause(r: Dict[str, Any]) -> Tuple[str, str]:
        qc = _get_qc(r)
        q_mode, q_cause = (qc + ["", ""])[:2]
        return _safe(q_mode), _safe(q_cause)

    def _get_me_qmode_qeffect(r: Dict[str, Any]) -> Tuple[str, str]:
        qc = _get_qc(r)
        q_mode, q_effect = (qc + ["", ""])[:2]
        return _safe(q_mode), _safe(q_effect)

    # ---------------------------
    # 2) dedupe + select
    # ---------------------------
    def _uniq_key_mc(r: Dict[str, Any]) -> Tuple[str, str]:
        q_mode, q_cause = _get_mc_qmode_qcause(r)
        return (_norm(q_mode), _norm(q_cause))

    def _uniq_key_me(r: Dict[str, Any]) -> Tuple[str, str]:
        q_mode, q_effect = _get_me_qmode_qeffect(r)
        return (_norm(q_mode), _norm(q_effect))

    def _select(rows: List[Dict[str, Any]], kind: str, target_n: Optional[int]) -> List[Dict[str, Any]]:
        if target_n is None:
            target_n = len(rows)

        rows_sorted = sorted(rows, key=_rank_key, reverse=True)

        selected: List[Dict[str, Any]] = []
        seen = set()
        duplicates: List[Dict[str, Any]] = []

        for r in rows_sorted:
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

    # =========================
    # NEW: PPL candidate helpers + cache
    # =========================
    effects_cache: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = {}
    causes_cache: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = {}

    def _format_candidates(
        items: List[Dict[str, Any]],
        label: str,
        max_show: int,
    ) -> str:
        """
        你只需要保证 rank_* 返回的 item 里能取到:
          - text: 候选字符串（effect 或 cause）
          - ppl 或 score（可选）
        如果你的字段不同，把下面 key 改掉即可。
        """
        if not items:
            return ""

        lines = [f"{label}:"]
        shown = 0
        for it in items:
            if shown >= max_show:
                break
            text = _safe(it.get("text") or it.get("candidate") or it.get("value"))
            disc = it.get("discipline")
            prefix = f"[{disc}] " if disc else ""
            if not text:
                continue

            # optional metrics
            coh = it.get("coh", None)
            f_nll = it.get("f_nll", None)

            lines.append(f"  -{prefix}{text}")
            # optional filtering
            # if coh is not None and f_nll is not None:
            #     lines.append(f"  - {text}  (coh={float(coh):.4f}, f_nll={float(f_nll):.4f})")
            # elif coh is not None:
            #     lines.append(f"  - {text}  (coh={float(coh):.4f})")
            # elif f_nll is not None:
            #     lines.append(f"  - {text}  (f_nll={float(f_nll):.4f})")
            # else:
            #     lines.append(f"  - {prefix}{text}")
            shown += 1

        return "\n".join(lines) + "\n"

    def _get_effect_candidates(mode: str, cause: str) -> str:
        if not enable_ppl_candidates:
            return ""
        if evaluator is None or structure_input is None:
            return ""
        key = (_norm(mode), _norm(cause), int(node_index), int(ppl_top_k))
        if key not in effects_cache:
            top = rank_effects_with_lm(
                evaluator,
                structure_input,
                fixed_mode=mode,
                fixed_cause=cause,
                node_index=node_index,
                top_k=ppl_top_k,
            )
            # 统一成 list[dict] 方便格式化；按你真实返回结构改这里
            items: List[Dict[str, Any]] = []
            for e in (top or []):
                items.append({
                    "text": getattr(e, "failure_effect", None),
                    "coh": getattr(e, "coherence_score", None),
                    "f_nll": getattr(e, "forward_nll", None),
                })
            effects_cache[key] = items
        return _format_candidates(effects_cache[key], label="ppl_candidates_effect", max_show=ppl_max_show)

    def _get_cause_candidates(mode: str, effect: str) -> str:
        if not enable_ppl_candidates:
            return ""
        if evaluator is None or structure_input is None:
            return ""
        key = (_norm(mode), _norm(effect), int(node_index), int(ppl_top_k))
        if key not in causes_cache:
            top = rank_causes_with_lm(
                evaluator,
                structure_input,
                fixed_mode=mode,
                fixed_effect=effect,
                node_index=node_index,
                top_k=ppl_top_k,
            )
            items: List[Dict[str, Any]] = []
            for c in (top or []):
                items.append({
                    "text": getattr(c, "failure_cause", None),
                    "discipline": getattr(c, "discipline", None),  
                    "coh": getattr(c, "coherence_score", None),
                    "f_nll": getattr(c, "forward_nll", None),
                })
            causes_cache[key] = items
        return _format_candidates(causes_cache[key], label="ppl_candidates_cause", max_show=ppl_max_show)

    # ---------------------------
    # MODE A) 不拼接：直接输出 MC/ME
    # ---------------------------
    if not join_cartesian:
        mc_sel = _select(mc_rows, "MC", target_n_mc)
        me_sel = _select(me_rows, "ME", target_n_me)

        blocks: List[str] = []
        blocks.append(
            "## SEMI CHAINS - MC (effect left blank for LLM to fill)\n"
            f"## filter: count>={min_count}, best_score>={min_best_score}\n"
        )
        if not mc_sel:
            blocks.append("(No MC semi chains after filtering)\n")
        else:
            for i, r in enumerate(mc_sel, start=1):
                q_mode, q_cause = _get_mc_qmode_qcause(r)
                cause_text = q_cause or blank
                if q_cause:
                    cause_text = _cause_with_discipline(q_cause, cause_disc_map)
                blocks.append(
                    f"[MC_SEMI_CHAIN {i}]\n"
                    f"failure_element: {structure_element}\n"
                    f"failure_mode: {q_mode or blank}\n"
                    f"failure_cause: {cause_text}\n"
                    f"failure_effect: {blank}\n"
                    + (
                        f"stats: count={int(r.get('count',0))}, best_score={float(r.get('best_score',0.0)):.6f}\n"
                        if show_stats else ""
                    )
                    # NEW: candidates for filling effect
                    + _get_effect_candidates(q_mode or "", q_cause or "")
                )

        blocks.append(
            "\n## SEMI CHAINS - ME (cause left blank for LLM to fill)\n"
            f"## filter: count>={min_count}, best_score>={min_best_score}\n"
        )
        if not me_sel:
            blocks.append("(No ME semi chains after filtering)\n")
        else:
            for i, r in enumerate(me_sel, start=1):
                q_mode, q_effect = _get_me_qmode_qeffect(r)
                blocks.append(
                    f"[ME_SEMI_CHAIN {i}]\n"
                    f"failure_element: {structure_element}\n"
                    f"failure_mode: {q_mode or blank}\n"
                    f"failure_cause: {blank}\n"
                    f"failure_effect: {q_effect or blank}\n"
                    + (
                        f"stats: count={int(r.get('count',0))}, best_score={float(r.get('best_score',0.0)):.6f}\n"
                        if show_stats else ""
                    )
                    # NEW: candidates for filling cause
                    + _get_cause_candidates(q_mode or "", q_effect or "")
                )

        return "\n".join(blocks)

    # ---------------------------
    # MODE B) 拼接：三段输出（FULL + remaining MC + remaining ME）
    # ---------------------------
    mc_by_qmode: Dict[str, List[Dict[str, Any]]] = {}
    me_by_qmode: Dict[str, List[Dict[str, Any]]] = {}

    for r in mc_rows:
        q_mode, _ = _get_mc_qmode_qcause(r)
        if q_mode:
            mc_by_qmode.setdefault(_norm(q_mode), []).append(r)

    for r in me_rows:
        q_mode, _ = _get_me_qmode_qeffect(r)
        if q_mode:
            me_by_qmode.setdefault(_norm(q_mode), []).append(r)

    full_chains: List[Dict[str, Any]] = []
    full_keys: Set[Tuple[str, str, str, str, str, str]] = set()

    used_mc_ids: Set[int] = set()
    used_me_ids: Set[int] = set()

    common_modes = sorted(set(mc_by_qmode.keys()) & set(me_by_qmode.keys()))
    for q_mode_norm in common_modes:
        mcs = sorted(mc_by_qmode[q_mode_norm], key=_rank_key, reverse=True)
        mes = sorted(me_by_qmode[q_mode_norm], key=_rank_key, reverse=True)

        q_mode_display = _get_mc_qmode_qcause(mcs[0])[0] if mcs else (
            _get_me_qmode_qeffect(mes[0])[0] if mes else ""
        )

        for mc in mcs:
            _, q_cause = _get_mc_qmode_qcause(mc)
            ex_mc = mc.get("example") or {}

            for me in mes:
                _, q_effect = _get_me_qmode_qeffect(me)
                ex_me = me.get("example") or {}

                matched_mode = _safe(ex_mc.get("matched_mode") or ex_me.get("matched_mode") or "")
                matched_cause = _safe(ex_mc.get("matched_cause") or "")
                matched_effect = _safe(ex_me.get("matched_effect") or "")

                key = (
                    _norm(q_mode_display),
                    _norm(q_cause),
                    _norm(q_effect),
                    _norm(matched_mode),
                    _norm(matched_cause),
                    _norm(matched_effect),
                )
                if key in full_keys:
                    continue
                full_keys.add(key)

                used_mc_ids.add(id(mc))
                used_me_ids.add(id(me))

                full_chains.append(
                    {
                        "query_mode": q_mode_display,
                        "query_cause": q_cause,
                        "query_effect": q_effect,
                        "mc_count": mc.get("count", 0),
                        "mc_best_score": mc.get("best_score", 0.0),
                        "me_count": me.get("count", 0),
                        "me_best_score": me.get("best_score", 0.0),
                        "mc_edge_count": ex_mc.get("edge_count", ""),
                        "me_edge_count": ex_me.get("edge_count", ""),
                        "matched_mode": matched_mode,
                        "matched_cause": matched_cause,
                        "matched_effect": matched_effect,
                    }
                )

    full_chains.sort(
        key=lambda d: (
            _norm(d.get("query_mode", "")),
            -float(d.get("mc_best_score", 0.0)),
            -int(d.get("mc_count", 0)),
            -float(d.get("me_best_score", 0.0)),
            -int(d.get("me_count", 0)),
            _norm(d.get("query_cause", "")),
            _norm(d.get("query_effect", "")),
        )
    )

    remaining_mc = [r for r in mc_rows if id(r) not in used_mc_ids]
    remaining_me = [r for r in me_rows if id(r) not in used_me_ids]

    mc_sel = _select(remaining_mc, "MC", target_n_mc)
    me_sel = _select(remaining_me, "ME", target_n_me)

    if target_n_mc is None and target_n_me is None:
        full_limit = None
    else:
        a = 0 if target_n_mc is None else int(target_n_mc)
        b = 0 if target_n_me is None else int(target_n_me)
        full_limit = max(a, b)

    blocks: List[str] = []
    blocks.append(
        "## PART 1) FULL CHAINS (MC×ME within same query_mode)\n"
        f"## filter: count>={min_count}, best_score>={min_best_score}\n"
    )
    if not full_chains:
        blocks.append("(No FULL chains after cartesian join)\n")
    else:
        view = full_chains if full_limit is None else full_chains[:full_limit]
        for i, fc in enumerate(view, start=1):
            # FULL 链已经给了 cause+effect，不需要 blank；一般也不需要候选项
            # 如果你也想附：可对 (mode,cause) 给 effect 候选、对 (mode,effect) 给 cause 候选
            blocks.append(
                f"[FULL_CHAIN {i}]\n"
                f"failure_element: {structure_element}\n"
                f"failure_mode: {fc['query_mode']}\n"
                f"failure_cause: {fc['query_cause']}\n"
                f"failure_effect: {fc['query_effect']}\n"
                + (
                    f"stats_mc: count={int(fc['mc_count'])}, best_score={float(fc['mc_best_score']):.6f}, edge_count={fc['mc_edge_count']}\n"
                    f"stats_me: count={int(fc['me_count'])}, best_score={float(fc['me_best_score']):.6f}, edge_count={fc['me_edge_count']}\n"
                    if show_stats else ""
                )
                + (
                    "example:\n"
                    f"  matched_mode: {fc['matched_mode']}\n"
                    f"  matched_cause: {fc['matched_cause']}\n"
                    f"  matched_effect: {fc['matched_effect']}\n"
                    if (fc.get("matched_mode") or fc.get("matched_cause") or fc.get("matched_effect"))
                    else ""
                )
            )

    blocks.append(
        "\n## PART 2) REMAINING MC SEMI CHAINS (effect left blank for LLM to fill)\n"
        f"## filter: count>={min_count}, best_score>={min_best_score}\n"
    )
    if not mc_sel:
        blocks.append("(No remaining MC semi chains)\n")
    else:
        for i, r in enumerate(mc_sel, start=1):
            q_mode, q_cause = _get_mc_qmode_qcause(r)
            blocks.append(
                f"[MC_SEMI_CHAIN {i}]\n"
                f"failure_element: {structure_element}\n"
                f"failure_mode: {q_mode or blank}\n"
                f"failure_cause: {q_cause or blank}\n"
                f"failure_effect: {blank}\n"
                + (
                    f"stats: count={int(r.get('count',0))}, best_score={float(r.get('best_score',0.0)):.6f}\n"
                    if show_stats else ""
                )
                + _get_effect_candidates(q_mode or "", q_cause or "")
            )

    blocks.append(
        "\n## PART 3) REMAINING ME SEMI CHAINS (cause left blank for LLM to fill)\n"
        f"## filter: count>={min_count}, best_score>={min_best_score}\n"
    )
    if not me_sel:
        blocks.append("(No remaining ME semi chains)\n")
    else:
        for i, r in enumerate(me_sel, start=1):
            q_mode, q_effect = _get_me_qmode_qeffect(r)
            blocks.append(
                f"[ME_SEMI_CHAIN {i}]\n"
                f"failure_element: {structure_element}\n"
                f"failure_mode: {q_mode or blank}\n"
                f"failure_cause: {blank}\n"
                f"failure_effect: {q_effect or blank}\n"
                + (
                    f"stats: count={int(r.get('count',0))}, best_score={float(r.get('best_score',0.0)):.6f}\n"
                    if show_stats else ""
                )
                + _get_cause_candidates(q_mode or "", q_effect or "")
            )

    return "\n".join(blocks)