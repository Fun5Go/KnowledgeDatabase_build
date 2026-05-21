from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from GraphRAG.Extraction.evaluate_oneprocess_results import (  # noqa: E402
    dedupe_names,
    name_variants,
    names_match,
    normalize_key,
)
from GraphRAG.Extraction.retrieval_evaluation import (  # noqa: E402
    QueryTextBuilder,
    RetrievalConfig,
    chunk_display_name,
    json_safe,
    mean,
    normalize_text,
    parse_bool_list,
    parse_float_list,
    parse_int_list,
    split_csv,
)


EXTRACTION_DIR = Path(__file__).resolve().parent
DEFAULT_REVIEW_DIR = Path(r"C:\Users\FW\Desktop\FMEA_AI\Review Material\extract_by_FW")
DEFAULT_OUTPUT_PATH = EXTRACTION_DIR / "review_retrieval_evaluation_results.json"

FUNCTION_MODE_QUERY_TEMPLATE: str | None = "{query_mode} in {function_text}"
CAUSE_QUERY_TEMPLATE: str | None = "{query_cause}"
EFFECT_QUERY_TEMPLATE: str | None = "{function_text} with {query_effect}"


@dataclass(frozen=True)
class ReviewRetrievalConfig(RetrievalConfig):
    """Retrieval parameter combination for review-based evaluation."""


class ReviewQueryTextBuilder(QueryTextBuilder):
    """Build query text for review files, including effect queries."""

    def __init__(
        self,
        function_mode_template: str | None = None,
        cause_template: str | None = None,
        effect_template: str | None = None,
        use_review_query_text: bool = True,
    ) -> None:
        super().__init__(function_mode_template, cause_template)
        self.effect_template = effect_template
        self.use_review_query_text = use_review_query_text

    def build(self, analysis_item: dict[str, Any]) -> str:
        query_type = normalize_text(analysis_item.get("query_type"))
        if query_type == "function_mode" and self.function_mode_template:
            return render_review_query_template(self.function_mode_template, analysis_item)
        if query_type == "cause" and self.cause_template:
            return render_review_query_template(self.cause_template, analysis_item)
        if query_type == "effect" and self.effect_template:
            return render_review_query_template(self.effect_template, analysis_item)
        if not self.use_review_query_text:
            return super().build(analysis_item)
        return normalize_text(analysis_item.get("query_text")) or super().build(analysis_item)


def render_review_query_template(template: str, analysis_item: dict[str, Any]) -> str:
    values = {
        "analysis_id": normalize_text(analysis_item.get("analysis_id")),
        "query_type": normalize_text(analysis_item.get("query_type")),
        "element_id": normalize_text(analysis_item.get("element_id")),
        "failure_element": normalize_text(analysis_item.get("failure_element")),
        "function_text": normalize_text(analysis_item.get("function_text")),
        "query_mode": normalize_text(analysis_item.get("query_mode")),
        "mode_text": normalize_text(analysis_item.get("query_mode")),
        "query_cause": normalize_text(analysis_item.get("query_cause")),
        "cause_text": normalize_text(analysis_item.get("query_cause")),
        "query_effect": normalize_text(analysis_item.get("query_effect")),
        "effect_text": normalize_text(analysis_item.get("query_effect")),
        "cause_discipline": normalize_text(analysis_item.get("cause_discipline")),
        "discipline": normalize_text(analysis_item.get("cause_discipline")),
    }
    return " ".join(template.format(**values).split())


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        cleaned = value.strip().casefold()
        if cleaned in {"true", "t", "yes", "y", "1"}:
            return True
        if cleaned in {"false", "f", "no", "n", "0"}:
            return False
    return None


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reviewed_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        chunks = payload.get("chunk_aggregates")
        if isinstance(chunks, list):
            return [item for item in chunks if isinstance(item, dict)]
        chunks = payload.get("items")
        if isinstance(chunks, list):
            return [item for item in chunks if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def analysis_item_from_review_payload(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    query = payload.get("query") if isinstance(payload.get("query"), dict) else {}
    return {
        "analysis_id": normalize_text(payload.get("analysis_id")) or path.stem,
        "query_type": normalize_text(query.get("query_type")),
        "element_id": normalize_text(payload.get("element_id")),
        "failure_element": normalize_text(payload.get("failure_element")),
        "function_text": normalize_text(query.get("function_text")),
        "query_mode": normalize_text(query.get("query_mode")),
        "query_cause": normalize_text(query.get("query_cause")),
        "query_effect": normalize_text(query.get("query_effect")),
        "cause_discipline": normalize_text(query.get("discipline_text")),
        "query_text": normalize_text(query.get("query_text")),
        "source_file": path.name,
    }


def chunk_name(item: dict[str, Any], fallback_index: int) -> str:
    return normalize_text(
        first_present(item, ["name", "chunk_id", "node_id"])
        or f"chunk_{fallback_index}"
    )


def gold_names_from_review_items(items: list[dict[str, Any]]) -> tuple[list[str], list[str], int]:
    true_positive_names: list[str] = []
    false_negative_names: list[str] = []
    skipped = 0

    for index, item in enumerate(items, start=1):
        selected = normalize_bool(item.get("selected"))
        is_select_correct = normalize_bool(
            first_present(
                item,
                [
                    "is_select_correct",
                    "is_selected_correct",
                    "selected_correct",
                    "is_selection_correct",
                ],
            )
        )
        if selected is None or is_select_correct is None:
            skipped += 1
            continue

        name = chunk_name(item, index)
        if not name:
            skipped += 1
            continue
        if selected is True and is_select_correct is True:
            true_positive_names.append(name)
        elif selected is False and is_select_correct is False:
            false_negative_names.append(name)

    return dedupe_names(true_positive_names), dedupe_names(false_negative_names), skipped


def load_review_query_items(
    review_dir: Path = DEFAULT_REVIEW_DIR,
    glob_pattern: str = "*.review.json",
    query_builder: ReviewQueryTextBuilder | None = None,
) -> list[dict[str, Any]]:
    if not review_dir.exists():
        raise FileNotFoundError(f"Review directory not found: {review_dir}")

    paths = sorted(review_dir.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No review files matched {review_dir / glob_pattern}")

    query_builder = query_builder or ReviewQueryTextBuilder()
    query_items: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue

        analysis_item = analysis_item_from_review_payload(payload, path)
        tp_names, fn_names, skipped = gold_names_from_review_items(reviewed_items(payload))
        gold_names = dedupe_names(tp_names + fn_names)
        analysis_item["query_text"] = query_builder.build(analysis_item)
        analysis_item["ground_truth_chunk_names"] = gold_names
        analysis_item["review_counts"] = {
            "tp": len(tp_names),
            "fn": len(fn_names),
            "actual_value": len(gold_names),
            "skipped": skipped,
        }
        analysis_item["review_ground_truth"] = {
            "tp_names": tp_names,
            "fn_names": fn_names,
            "actual_value_names": gold_names,
        }
        query_items.append(analysis_item)

    return query_items


def run_retrieval_for_review_item(
    retriever: Any,
    analysis_item: dict[str, Any],
    top_k: int = 15,
    per_label_k: int = 30,
    retrieval_mode: str = "hybrid",
    use_cross_encoder_rerank: bool = False,
    cross_encoder_top_n: int = 30,
    use_section_tag_bonus: bool = True,
    section_bonus_mode: str = "hybrid",
    section_bonus_weight: float = 0.05,
) -> dict[str, Any]:
    from GraphRAG.main_sentence import (
        build_sentence_doc_chunk_query,
        query_doc_chunks_for_sentence,
    )

    query_text = normalize_text(analysis_item.get("query_text"))
    query_spec = build_sentence_doc_chunk_query(sentence=query_text)
    query_spec["query_type"] = analysis_item.get("query_type", "")
    query_spec["function_text"] = analysis_item.get("function_text", "")
    query_spec["query_mode"] = analysis_item.get("query_mode", "")
    query_spec["query_cause"] = analysis_item.get("query_cause", "")
    query_spec["query_effect"] = analysis_item.get("query_effect", "")
    query_spec["cause_discipline"] = analysis_item.get("cause_discipline", "")

    return query_doc_chunks_for_sentence(
        retriever=retriever,
        query_spec=query_spec,
        top_k=top_k,
        per_label_k=per_label_k,
        retrieval_mode=retrieval_mode,
        disciplines=None,
        use_cross_encoder_rerank=use_cross_encoder_rerank,
        cross_encoder_top_n=cross_encoder_top_n,
        use_section_tag_bonus=use_section_tag_bonus,
        section_bonus_mode=section_bonus_mode,
        section_bonus_weight=section_bonus_weight,
    )


def canonical_truth_key(name: str) -> str:
    variants = sorted(name_variants(name))
    return variants[0] if variants else normalize_key(name)


def calculate_review_retrieval_metrics(
    ranked_chunk_names: list[str],
    ground_truth_chunk_names: list[str],
) -> dict[str, Any]:
    true_names = [name for name in ground_truth_chunk_names if normalize_text(name)]
    if not true_names:
        return {
            "mrr": 0.0,
            "recall": 0.0,
            "hit_at_k": 0.0,
            "first_relevant_rank": None,
            "hit_count": 0,
            "ground_truth_count": 0,
            "matched_ground_truth_names": [],
            "missed_ground_truth_names": [],
        }

    matched_truth_keys: set[str] = set()
    matched_truth_names: list[str] = []
    first_relevant_rank: int | None = None

    for rank, chunk_name in enumerate(ranked_chunk_names, start=1):
        for truth_name in true_names:
            truth_key = canonical_truth_key(truth_name)
            if truth_key in matched_truth_keys:
                continue
            if not names_match(chunk_name, truth_name):
                continue
            matched_truth_keys.add(truth_key)
            matched_truth_names.append(truth_name)
            if first_relevant_rank is None:
                first_relevant_rank = rank
            break

    missed_truth_names = [
        truth_name
        for truth_name in true_names
        if canonical_truth_key(truth_name) not in matched_truth_keys
    ]
    return {
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "recall": len(matched_truth_keys) / len({canonical_truth_key(name) for name in true_names}),
        "hit_at_k": 1.0 if first_relevant_rank else 0.0,
        "first_relevant_rank": first_relevant_rank,
        "hit_count": len(matched_truth_keys),
        "ground_truth_count": len({canonical_truth_key(name) for name in true_names}),
        "matched_ground_truth_names": matched_truth_names,
        "missed_ground_truth_names": missed_truth_names,
    }


def evaluate_review_config(
    retriever: Any,
    config: RetrievalConfig,
    query_items: list[dict[str, Any]],
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    for index, analysis_item in enumerate(query_items, start=1):
        query_result = run_retrieval_for_review_item(
            retriever=retriever,
            analysis_item=analysis_item,
            **config.to_kwargs(),
        )
        evidence = query_result.get("evidence", [])
        ranked_names = [chunk_display_name(chunk) for chunk in evidence]
        true_names = analysis_item.get("ground_truth_chunk_names", [])
        metrics = calculate_review_retrieval_metrics(ranked_names, true_names)
        per_query.append(
            {
                "query_number": index,
                "analysis_id": normalize_text(analysis_item.get("analysis_id")),
                "source_file": normalize_text(analysis_item.get("source_file")),
                "query_type": normalize_text(analysis_item.get("query_type")),
                "query_text": normalize_text(analysis_item.get("query_text")),
                "review_counts": analysis_item.get("review_counts", {}),
                "ground_truth_chunk_names": true_names,
                "retrieved_chunk_names": ranked_names,
                **metrics,
            }
        )

    evaluated = [item for item in per_query if item["ground_truth_count"] > 0]
    total_hits = sum(int(item["hit_count"]) for item in evaluated)
    total_truth = sum(int(item["ground_truth_count"]) for item in evaluated)
    return {
        "config_id": config.config_id,
        "parameters": config.__dict__,
        "average_mrr": mean(float(item["mrr"]) for item in evaluated),
        "average_recall": mean(float(item["recall"]) for item in evaluated),
        "hit_rate_at_k": mean(float(item["hit_at_k"]) for item in evaluated),
        "micro_recall": total_hits / total_truth if total_truth else 0.0,
        "evaluated_query_count": len(evaluated),
        "total_query_count": len(per_query),
        "per_query": per_query,
    }


def evaluate_review_retrieval(
    review_dir: Path = DEFAULT_REVIEW_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    glob_pattern: str = "*.review.json",
    configs: list[RetrievalConfig] | None = None,
    query_builder: ReviewQueryTextBuilder | None = None,
) -> dict[str, Any]:
    configs = configs or [ReviewRetrievalConfig()]
    query_builder = query_builder or ReviewQueryTextBuilder()
    query_items = load_review_query_items(
        review_dir=review_dir,
        glob_pattern=glob_pattern,
        query_builder=query_builder,
    )

    from GraphRAG.fmea_retrieverV2 import FMEASentenceRetrieverV2

    retriever = FMEASentenceRetrieverV2()
    try:
        config_results = [
            evaluate_review_config(
                retriever=retriever,
                config=config,
                query_items=query_items,
            )
            for config in configs
        ]
    finally:
        retriever.close()

    output = {
        "review_dir": str(review_dir),
        "glob_pattern": glob_pattern,
        "query_template": {
            "use_review_query_text": query_builder.use_review_query_text,
            "function_mode_template": query_builder.function_mode_template or "review query_text",
            "cause_template": query_builder.cause_template or "review query_text",
            "effect_template": query_builder.effect_template or "review query_text",
        },
        "review_ground_truth_source": "TP + FN from selected/is_selected_correct review labels",
        "results": config_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(output), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def build_review_config_grid(args: argparse.Namespace) -> list[ReviewRetrievalConfig]:
    return [
        ReviewRetrievalConfig(
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            per_label_k=per_label_k,
            use_cross_encoder_rerank=use_cross_encoder_rerank,
            cross_encoder_top_n=args.cross_encoder_top_n,
            use_section_tag_bonus=use_section_tag_bonus,
            section_bonus_mode=args.section_bonus_mode,
            section_bonus_weight=section_bonus_weight,
        )
        for (
            retrieval_mode,
            top_k,
            per_label_k,
            use_cross_encoder_rerank,
            use_section_tag_bonus,
            section_bonus_weight,
        ) in itertools.product(
            split_csv(args.retrieval_modes),
            parse_int_list(args.top_k),
            parse_int_list(args.per_label_k),
            parse_bool_list(args.cross_encoder),
            parse_bool_list(args.section_tag_bonus),
            parse_float_list(args.section_bonus_weight),
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GraphRAG retrieval against TP/FN chunks from reviewed extraction files."
    )
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--glob", default="*.review.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--retrieval-modes",
        default="dense",
        help="Comma-separated retrieval modes, e.g. hybrid,dense,sparse.",
    )
    parser.add_argument("--top-k", default="30", help="Comma-separated top_k values.")
    parser.add_argument("--per-label-k", default="60", help="Comma-separated per-label candidate sizes.")
    parser.add_argument("--cross-encoder", default="false", help="Comma-separated booleans.")
    parser.add_argument("--cross-encoder-top-n", type=int, default=30)
    parser.add_argument("--section-tag-bonus", default="false", help="Comma-separated booleans.")
    parser.add_argument("--section-bonus-mode", default="hybrid")
    parser.add_argument("--section-bonus-weight", default="0.01", help="Comma-separated section bonus weights.")
    parser.add_argument(
        "--use-review-query-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use query.query_text from review files when no template is supplied.",
    )
    parser.add_argument(
        "--function-mode-template",
        default=FUNCTION_MODE_QUERY_TEMPLATE,
        help="Optional template for function_mode queries.",
    )
    parser.add_argument("--cause-template", default=None, help="Optional template for cause queries.")
    parser.add_argument("--effect-template", default=EFFECT_QUERY_TEMPLATE, help="Optional template for effect queries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_review_config_grid(args)
    query_builder = ReviewQueryTextBuilder(
        function_mode_template=args.function_mode_template,
        cause_template=args.cause_template,
        effect_template=args.effect_template,
        use_review_query_text=args.use_review_query_text,
    )
    output = evaluate_review_retrieval(
        review_dir=args.review_dir,
        output_path=args.output,
        glob_pattern=args.glob,
        configs=configs,
        query_builder=query_builder,
    )
    print(f"[INFO] Evaluated {len(output['results'])} retrieval config(s).")
    print(f"[INFO] Wrote evaluation results to: {args.output}")


if __name__ == "__main__":
    main()
