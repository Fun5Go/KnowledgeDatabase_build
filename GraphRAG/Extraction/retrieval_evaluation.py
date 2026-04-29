from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from .main import (
        build_query_text as build_main_query_text,
        iter_structure_analysis_items,
        json_safe,
        normalize_text,
        run_retrieval_for_analysis_item,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from main import (
        build_query_text as build_main_query_text,
        iter_structure_analysis_items,
        json_safe,
        normalize_text,
        run_retrieval_for_analysis_item,
    )


EXTRACTION_DIR = Path(__file__).resolve().parent
DEFAULT_GROUND_TRUTH_PATH = EXTRACTION_DIR / "retrieval_ground_truth.json"
DEFAULT_EVALUATION_OUTPUT_PATH = EXTRACTION_DIR / "retrieval_evaluation_results.json"
DEFAULT_SELECTION_GLOB = "chunk_selection_results_query_*.json"

# Edit these templates when you want to change the query text used by evaluation.
# Set a template to None to use the current GraphRAG/Extraction/main.py default.
#
# Available placeholders:
# {failure_element}, {function_text}, {query_mode}, {mode_text},
# {query_cause}, {cause_text}, {cause_discipline}, {discipline},
# {element_id}, {analysis_id}, {query_type}
FUNCTION_MODE_QUERY_TEMPLATE = "{query_mode}"
CAUSE_QUERY_TEMPLATE: str | None = None

# Examples:
# FUNCTION_MODE_QUERY_TEMPLATE = "{function_text} failure mode: {query_mode}"
# CAUSE_QUERY_TEMPLATE = "{cause_discipline} cause: {query_cause}"


@dataclass(frozen=True)
class RetrievalConfig:
    """One retrieval parameter combination to evaluate."""

    retrieval_mode: str = "hybrid"
    top_k: int = 15
    per_label_k: int = 30
    use_cross_encoder_rerank: bool = False
    cross_encoder_top_n: int = 30
    use_section_tag_bonus: bool = True
    section_bonus_mode: str = "hybrid"
    section_bonus_weight: float = 0.01

    @property
    def config_id(self) -> str:
        rerank = "ce" if self.use_cross_encoder_rerank else "no-ce"
        bonus = (
            f"section-{self.section_bonus_mode}-{self.section_bonus_weight:g}"
            if self.use_section_tag_bonus
            else "no-section"
        )
        return (
            f"mode={self.retrieval_mode}|top_k={self.top_k}|"
            f"per_label_k={self.per_label_k}|{rerank}|{bonus}"
        )

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "per_label_k": self.per_label_k,
            "retrieval_mode": self.retrieval_mode,
            "use_cross_encoder_rerank": self.use_cross_encoder_rerank,
            "cross_encoder_top_n": self.cross_encoder_top_n,
            "use_section_tag_bonus": self.use_section_tag_bonus,
            "section_bonus_mode": self.section_bonus_mode,
            "section_bonus_weight": self.section_bonus_weight,
        }


class QueryTextBuilder:
    """
    Build evaluation query text.

    By default this mirrors GraphRAG/Extraction/main.py. Pass templates to test
    alternative query wording without changing the extraction pipeline.
    """

    def __init__(
        self,
        function_mode_template: str | None = None,
        cause_template: str | None = None,
    ) -> None:
        self.function_mode_template = function_mode_template
        self.cause_template = cause_template

    def build(self, analysis_item: dict[str, Any]) -> str:
        query_type = normalize_text(analysis_item.get("query_type"))
        if query_type == "function_mode" and self.function_mode_template:
            return render_query_template(self.function_mode_template, analysis_item)
        if query_type == "cause" and self.cause_template:
            return render_query_template(self.cause_template, analysis_item)
        return build_main_query_text(analysis_item)


def render_query_template(template: str, analysis_item: dict[str, Any]) -> str:
    """Render a query template against common mode/cause fields."""

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
        "cause_discipline": normalize_text(analysis_item.get("cause_discipline")),
        "discipline": normalize_text(analysis_item.get("cause_discipline")),
    }
    return " ".join(template.format(**values).split())


def apply_query_builder(
    analysis_item: dict[str, Any],
    query_builder: QueryTextBuilder,
) -> dict[str, Any]:
    """Return a copy of an analysis item with the evaluation query text set."""

    item = dict(analysis_item)
    item["query_text"] = query_builder.build(item)
    return item


def chunk_key(chunk: dict[str, Any]) -> str:
    """Return the stable id used for retrieval matching."""

    node_id = normalize_text(chunk.get("node_id"))
    if node_id:
        return node_id
    label = normalize_text(chunk.get("label"))
    name = normalize_text(chunk.get("name"))
    return f"{label}::{name}" if label or name else ""


def chunk_display_name(chunk: dict[str, Any]) -> str:
    """Return the human-readable chunk name used in evaluation output."""

    name = normalize_text(chunk.get("name"))
    if name:
        return name
    label = normalize_text(chunk.get("label"))
    node_id = normalize_text(chunk.get("node_id"))
    return f"{label}::{node_id}" if label or node_id else ""


def selected_chunk_key(
    selected_chunk: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> str:
    """Map an LLM-selected chunk back to the retrieved evidence chunk id."""

    selected_label = normalize_text(selected_chunk.get("label"))
    selected_name = normalize_text(selected_chunk.get("name"))
    for candidate in evidence:
        if (
            normalize_text(candidate.get("label")) == selected_label
            and normalize_text(candidate.get("name")) == selected_name
        ):
            return chunk_key(candidate)
    return f"{selected_label}::{selected_name}" if selected_label or selected_name else ""


def build_ground_truth_from_selection_results(
    selection_glob: str = DEFAULT_SELECTION_GLOB,
) -> list[dict[str, Any]]:
    """
    Build editable ground-truth entries from strong-confidence selections.

    Each chunk_selection_results_query_n.json is expected to contain the output
    from main.py: analysis_item, query_result.evidence, and selection.top_chunks.
    """

    entries: list[dict[str, Any]] = []
    for path in sorted(EXTRACTION_DIR.glob(selection_glob), key=query_result_sort_key):
        for result in load_result_list(path):
            analysis_item = result.get("analysis_item", {})
            evidence = result.get("query_result", {}).get("evidence", [])
            top_chunks = result.get("selection", {}).get("top_chunks", [])
            strong_chunks = [
                chunk
                for chunk in top_chunks
                if normalize_text(chunk.get("support_capability")).lower() == "strong"
            ]

            ground_truth_chunks: list[dict[str, Any]] = []
            for chunk in strong_chunks:
                key = selected_chunk_key(chunk, evidence)
                if not key:
                    continue
                ground_truth_chunks.append(
                    {
                        "chunk_id": key,
                        "label": normalize_text(chunk.get("label")),
                        "name": normalize_text(chunk.get("name")),
                    }
                )

            entries.append(
                {
                    "query_number": extract_query_number(path),
                    "analysis_id": normalize_text(analysis_item.get("analysis_id")),
                    "query_type": normalize_text(analysis_item.get("query_type")),
                    "function_text": normalize_text(analysis_item.get("function_text")),
                    "mode_text": normalize_text(analysis_item.get("query_mode")),
                    "cause_text": normalize_text(analysis_item.get("query_cause")),
                    "cause_discipline": normalize_text(analysis_item.get("cause_discipline")),
                    "current_query_text": normalize_text(analysis_item.get("query_text")),
                    "ground_truth_chunk_ids": [
                        chunk["chunk_id"] for chunk in ground_truth_chunks
                    ],
                    "ground_truth_chunks": ground_truth_chunks,
                    "source_file": path.name,
                }
            )
    return entries


def save_ground_truth(path: Path = DEFAULT_GROUND_TRUTH_PATH) -> list[dict[str, Any]]:
    """Write the editable ground-truth JSON file."""

    entries = build_ground_truth_from_selection_results()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(entries), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return entries


def evaluate_retrieval(
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
    output_path: Path = DEFAULT_EVALUATION_OUTPUT_PATH,
    configs: list[RetrievalConfig] | None = None,
    query_builder: QueryTextBuilder | None = None,
) -> dict[str, Any]:
    """Run retrieval with each config and calculate per-query and average metrics."""

    configs = configs or [RetrievalConfig()]
    query_builder = query_builder or QueryTextBuilder()
    ground_truth_by_id = load_ground_truth_by_analysis_id(ground_truth_path)

    from GraphRAG.fmea_retrieverV2 import FMEASentenceRetrieverV2

    retriever = FMEASentenceRetrieverV2()
    try:
        query_items = [
            apply_query_builder(item, query_builder)
            for item in iter_structure_analysis_items()
        ]
        config_results = [
            evaluate_config(
                retriever=retriever,
                config=config,
                query_items=query_items,
                ground_truth_by_id=ground_truth_by_id,
            )
            for config in configs
        ]
    finally:
        retriever.close()

    output = {
        "ground_truth_path": str(ground_truth_path),
        "query_template": {
            "function_mode_template": query_builder.function_mode_template or "main.py default",
            "cause_template": query_builder.cause_template or "main.py default",
        },
        "results": config_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(output), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def evaluate_config(
    retriever: Any,
    config: RetrievalConfig,
    query_items: list[dict[str, Any]],
    ground_truth_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one retrieval parameter combination."""

    per_query: list[dict[str, Any]] = []
    for index, analysis_item in enumerate(query_items, start=1):
        analysis_id = normalize_text(analysis_item.get("analysis_id"))
        truth = ground_truth_by_id.get(analysis_id, {})
        true_ids = [
            normalize_text(value)
            for value in truth.get("ground_truth_chunk_ids", [])
            if normalize_text(value)
        ]
        true_names = ground_truth_display_names(truth)
        query_result = run_retrieval_for_analysis_item(
            retriever=retriever,
            analysis_item=analysis_item,
            **config.to_kwargs(),
        )
        evidence = query_result.get("evidence", [])
        ranked_ids = [chunk_key(chunk) for chunk in evidence]
        ranked_names = [chunk_display_name(chunk) for chunk in evidence]
        metrics = calculate_retrieval_metrics(ranked_ids, true_ids)
        per_query.append(
            {
                "query_number": truth.get("query_number", index),
                "analysis_id": analysis_id,
                "query_type": normalize_text(analysis_item.get("query_type")),
                "query_text": normalize_text(analysis_item.get("query_text")),
                "ground_truth_chunk_names": true_names,
                "retrieved_chunk_names": ranked_names,
                **metrics,
            }
        )

    evaluated = [item for item in per_query if item["ground_truth_count"] > 0]
    return {
        "config_id": config.config_id,
        "parameters": config.__dict__,
        "average_mrr": mean(item["mrr"] for item in evaluated),
        "average_recall": mean(item["recall"] for item in evaluated),
        "evaluated_query_count": len(evaluated),
        "total_query_count": len(per_query),
        "per_query": per_query,
    }


def ground_truth_display_names(truth: dict[str, Any]) -> list[str]:
    """Return readable ground-truth chunk names for result output."""

    chunks = truth.get("ground_truth_chunks", [])
    if isinstance(chunks, list):
        names = [
            normalize_text(chunk.get("name"))
            for chunk in chunks
            if isinstance(chunk, dict) and normalize_text(chunk.get("name"))
        ]
        if names:
            return names

    return [
        normalize_text(value)
        for value in truth.get("ground_truth_chunk_ids", [])
        if normalize_text(value)
    ]


def calculate_retrieval_metrics(
    ranked_chunk_ids: list[str],
    ground_truth_chunk_ids: list[str],
) -> dict[str, Any]:
    """Calculate MRR and recall for one query."""

    true_ids = [item for item in ground_truth_chunk_ids if item]
    true_set = set(true_ids)
    if not true_set:
        return {
            "mrr": 0.0,
            "recall": 0.0,
            "first_relevant_rank": None,
            "hit_count": 0,
            "ground_truth_count": 0,
        }

    hit_count = 0
    first_relevant_rank: int | None = None
    seen_hits: set[str] = set()
    for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id not in true_set or chunk_id in seen_hits:
            continue
        seen_hits.add(chunk_id)
        hit_count += 1
        if first_relevant_rank is None:
            first_relevant_rank = rank

    return {
        "mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "recall": hit_count / len(true_set),
        "first_relevant_rank": first_relevant_rank,
        "hit_count": hit_count,
        "ground_truth_count": len(true_set),
    }


def load_ground_truth_by_analysis_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {path}. Run with --init-ground-truth first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Ground-truth file must contain a JSON list: {path}")
    return {
        normalize_text(item.get("analysis_id")): item
        for item in data
        if isinstance(item, dict) and normalize_text(item.get("analysis_id"))
    }


def load_result_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def query_result_sort_key(path: Path) -> tuple[int, str]:
    query_number = extract_query_number(path)
    return (query_number if query_number is not None else 10**9, path.name)


def extract_query_number(path: Path) -> int | None:
    stem = path.stem
    marker = "_query_"
    if marker not in stem:
        return None
    value = stem.rsplit(marker, maxsplit=1)[-1]
    return int(value) if value.isdigit() else None


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def parse_bool_list(value: str) -> list[bool]:
    mapping = {
        "1": True,
        "true": True,
        "yes": True,
        "y": True,
        "0": False,
        "false": False,
        "no": False,
        "n": False,
    }
    parsed = []
    for item in split_csv(value):
        key = item.lower()
        if key not in mapping:
            raise argparse.ArgumentTypeError(f"Invalid boolean value: {item}")
        parsed.append(mapping[key])
    return parsed


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in split_csv(value)]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in split_csv(value)]


def build_config_grid(args: argparse.Namespace) -> list[RetrievalConfig]:
    return [
        RetrievalConfig(
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
        description="Evaluate GraphRAG retrieval results with MRR and recall."
    )
    parser.add_argument(
        "--init-ground-truth",
        action="store_true",
        help=(
            "Build retrieval_ground_truth.json from strong-confidence chunks in "
            "chunk_selection_results_query_*.json."
        ),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help="Editable ground-truth JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EVALUATION_OUTPUT_PATH,
        help="Evaluation result JSON path.",
    )
    parser.add_argument(
        "--retrieval-modes",
        default="hybrid",
        help="Comma-separated retrieval modes, e.g. hybrid,dense,sparse.",
    )
    parser.add_argument(
        "--top-k",
        default="15",
        help="Comma-separated top_k values, e.g. 5,10,15.",
    )
    parser.add_argument(
        "--per-label-k",
        default="30",
        help="Comma-separated per-label candidate sizes.",
    )
    parser.add_argument(
        "--cross-encoder",
        default="false",
        help="Comma-separated booleans for cross encoder reranking.",
    )
    parser.add_argument(
        "--cross-encoder-top-n",
        type=int,
        default=30,
        help="Candidate count used when cross encoder reranking is enabled.",
    )
    parser.add_argument(
        "--section-tag-bonus",
        default="true",
        help="Comma-separated booleans for section tag bonus.",
    )
    parser.add_argument(
        "--section-bonus-mode",
        default="hybrid",
        help="Section tag bonus retrieval mode.",
    )
    parser.add_argument(
        "--section-bonus-weight",
        default="0.05",
        help="Comma-separated section bonus weights.",
    )
    parser.add_argument(
        "--function-mode-template",
        default=None,
        help=(
            "Optional query template for mode queries. Placeholders include "
            "{function_text}, {query_mode}, {failure_element}."
        ),
    )
    parser.add_argument(
        "--cause-template",
        default=None,
        help=(
            "Optional query template for cause queries. Placeholders include "
            "{query_cause}, {cause_discipline}, {failure_element}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_ground_truth:
        entries = save_ground_truth(args.ground_truth)
        print(f"[INFO] Wrote {len(entries)} ground-truth entries to: {args.ground_truth}")
        return

    configs = build_config_grid(args)
    query_builder = QueryTextBuilder(
        function_mode_template=args.function_mode_template or FUNCTION_MODE_QUERY_TEMPLATE,
        cause_template=args.cause_template or CAUSE_QUERY_TEMPLATE,
    )
    output = evaluate_retrieval(
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        configs=configs,
        query_builder=query_builder,
    )
    print(f"[INFO] Evaluated {len(output['results'])} retrieval config(s).")
    print(f"[INFO] Wrote evaluation results to: {args.output}")


if __name__ == "__main__":
    main()
