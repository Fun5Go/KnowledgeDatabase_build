from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))


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
FUNCTION_MODE_QUERY_TEMPLATE = "{function_text} with {query_mode}"
CAUSE_QUERY_TEMPLATE: str | None = None

# Examples:
# FUNCTION_MODE_QUERY_TEMPLATE = "{function_text} failure mode: {query_mode}"
# CAUSE_QUERY_TEMPLATE = "{cause_discipline} cause: {query_cause}"

structure_input_motorcontrol = {
    "product_domain": "motor_drives",
    "nodes": [
        {
            "element_id": "E1",
            "failure_element": "Motor control",
            "modes": {
                "Soft starter": [
                    "Component break-down",
                    "Unbalanced motor currents",
                ],
                "Zero-crossing detection": [
                    "Incorrect interpretation zero-crossing",
                    "Soft start too long",
                    "No detection",
                ],
                "Relay switching": [
                    "Welded relay",
                    "Relay cannot close",
                    "False turn-on / turn-off",
                ],
            },
            "causes": {
                "mechanics": [
                    "Cooling insufficient",
                    "Compressor vibrations",
                ],
                "hardware": [
                    "(Starting) Motor current too high for chosen components",
                    "Overvoltage due to motor disconnect",
                    "Under Voltage due to incorrect triggering",
                    "Live switching of relays",
                ],
                "software": [
                    "Priority zero-crossing interrupt too low",
                    "Open loop control",
                ],
                "other": [
                    "No (correctly designed) snubber design",
                    "Too high dT junction as a result of power cycling of component",
                ],
            },
            "effects": [
                "Motor cannot start",
                "Overcurrent towards motor",
                "Motor starts without soft start",
                "Short-circuit",
            ],
        }
    ],
}


def normalize_text(value: Any) -> str:
    """Normalize values to safe strings without importing Extraction/main.py."""

    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def json_safe(value: Any) -> Any:
    """Convert values such as NumPy scalars into JSON-serializable objects."""

    if isinstance(value, dict):
        return {normalize_text(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return normalize_text(value)


def load_structure_input() -> dict[str, Any]:
    """Return the structure analysis input used by this evaluation pass."""

    return structure_input_motorcontrol


def build_function_mode_query_text(function_text: str, mode_text: str) -> str:
    """Default main.py-compatible query text for Function + failure mode."""

    return f"{function_text} has {mode_text}"


def build_cause_query_text(cause_text: str, discipline: str) -> str:
    """Default main.py-compatible query text for cause queries."""

    return f"{cause_text}"


def build_analysis_id(*parts: str) -> str:
    """Build a stable readable id from structure query parts."""

    cleaned = [normalize_text(part).lower().replace(" ", "-") for part in parts if normalize_text(part)]
    return ":".join(cleaned)


def map_cause_discipline_to_retrieval_labels(discipline: str) -> list[str] | None:
    """Map structure cause discipline to GraphRAG retrieval labels."""

    discipline_key = normalize_text(discipline).lower()
    mapping = {
        "hardware": ["HW"],
        "hw": ["HW"],
        "software": ["ESW"],
        "sw": ["ESW"],
        "esw": ["ESW"],
        "fs": ["FS"],
        "functional safety": ["FS"],
        "mechanics": ["MCH"],
    }
    return mapping.get(discipline_key)


def build_function_mode_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Function text + failure mode text query items."""

    items: list[dict[str, Any]] = []
    for node in structure_input.get("nodes", []):
        element_id = normalize_text(node.get("element_id"))
        failure_element = normalize_text(node.get("failure_element"))
        modes = node.get("modes", {})
        if not isinstance(modes, dict):
            continue

        for function_text, mode_texts in modes.items():
            function_text = normalize_text(function_text)
            if not isinstance(mode_texts, list):
                continue

            for mode_text in mode_texts:
                mode_text = normalize_text(mode_text)
                if not function_text or not mode_text:
                    continue

                items.append(
                    {
                        "analysis_id": build_analysis_id(
                            element_id,
                            "function_mode",
                            function_text,
                            mode_text,
                        ),
                        "query_type": "function_mode",
                        "element_id": element_id,
                        "failure_element": failure_element,
                        "function_text": function_text,
                        "query_mode": mode_text,
                        "query_cause": "",
                        "cause_discipline": "",
                        "query_text": build_function_mode_query_text(function_text, mode_text),
                        "disciplines": None,
                    }
                )

    return items


def build_cause_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Build cause text query items with their discipline/category."""

    items: list[dict[str, Any]] = []
    for node in structure_input.get("nodes", []):
        element_id = normalize_text(node.get("element_id"))
        failure_element = normalize_text(node.get("failure_element"))
        causes = node.get("causes", {})
        if not isinstance(causes, dict):
            continue

        for discipline, cause_texts in causes.items():
            discipline = normalize_text(discipline)
            if not isinstance(cause_texts, list):
                continue

            for cause_text in cause_texts:
                cause_text = normalize_text(cause_text)
                if not cause_text:
                    continue

                items.append(
                    {
                        "analysis_id": build_analysis_id(
                            element_id,
                            "cause",
                            discipline,
                            cause_text,
                        ),
                        "query_type": "cause",
                        "element_id": element_id,
                        "failure_element": failure_element,
                        "function_text": "",
                        "query_mode": "",
                        "query_cause": cause_text,
                        "cause_discipline": discipline,
                        "query_text": build_cause_query_text(cause_text, discipline),
                        "disciplines": map_cause_discipline_to_retrieval_labels(discipline),
                    }
                )

    return items


def build_structure_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Split structure input into function-mode and cause query items."""

    return build_function_mode_query_items(structure_input) + build_cause_query_items(structure_input)


def iter_structure_analysis_items() -> Iterable[dict[str, Any]]:
    """Loop the evaluation structure query items without importing main.py."""

    yield from build_structure_query_items(load_structure_input())


def build_query_text(analysis_item: dict[str, Any]) -> str:
    """Return the default main.py-compatible retrieval query text."""

    query_text = normalize_text(analysis_item.get("query_text"))
    if query_text:
        return query_text

    if analysis_item.get("query_type") == "function_mode":
        return build_function_mode_query_text(
            normalize_text(analysis_item.get("function_text")),
            normalize_text(analysis_item.get("query_mode")),
        )

    if analysis_item.get("query_type") == "cause":
        return build_cause_query_text(
            normalize_text(analysis_item.get("query_cause")),
            normalize_text(analysis_item.get("cause_discipline")),
        )

    return ""


def run_retrieval_for_analysis_item(
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
    """Run retrieval without LangSmith tracing from Extraction/main.py."""

    from GraphRAG.main_sentence import (
        build_sentence_doc_chunk_query,
        query_doc_chunks_for_sentence,
    )

    query_text = build_query_text(analysis_item)
    query_spec = build_sentence_doc_chunk_query(sentence=query_text)
    query_spec["query_type"] = analysis_item.get("query_type", "")
    query_spec["function_text"] = analysis_item.get("function_text", "")
    query_spec["query_mode"] = analysis_item.get("query_mode", "")
    query_spec["query_cause"] = analysis_item.get("query_cause", "")
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
        return build_query_text(analysis_item)


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
            top_chunks = result.get("selection", {}).get("top_chunks", [])
            strong_chunks = [
                chunk
                for chunk in top_chunks
                if normalize_text(chunk.get("support_capability")).lower() == "strong"
            ]

            ground_truth_chunks: list[dict[str, Any]] = []
            for chunk in strong_chunks:
                name = normalize_text(chunk.get("name"))
                if not name:
                    continue
                ground_truth_chunks.append(
                    {
                        "label": normalize_text(chunk.get("label")),
                        "name": name,
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
                    "ground_truth_chunk_names": [
                        chunk["name"] for chunk in ground_truth_chunks if chunk["name"]
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
        true_names = ground_truth_names(truth)
        query_result = run_retrieval_for_analysis_item(
            retriever=retriever,
            analysis_item=analysis_item,
            **config.to_kwargs(),
        )
        evidence = query_result.get("evidence", [])
        ranked_names = [chunk_display_name(chunk) for chunk in evidence]
        metrics = calculate_retrieval_metrics(ranked_names, true_names)
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


def ground_truth_names(truth: dict[str, Any]) -> list[str]:
    """Return ground-truth chunk names used for matching and result output."""

    explicit_names = [
        normalize_text(value)
        for value in truth.get("ground_truth_chunk_names", [])
        if normalize_text(value)
    ]
    if explicit_names:
        return explicit_names

    chunks = truth.get("ground_truth_chunks", [])
    if isinstance(chunks, list):
        names = [
            normalize_text(chunk.get("name"))
            for chunk in chunks
            if isinstance(chunk, dict) and normalize_text(chunk.get("name"))
        ]
        if names:
            return names

    # Backward compatibility for older ground-truth files. New files should use
    # ground_truth_chunk_names, because evaluation now matches by chunk name.
    return [
        normalize_text(value)
        for value in truth.get("ground_truth_chunk_ids", [])
        if normalize_text(value)
    ]


def calculate_retrieval_metrics(
    ranked_chunk_names: list[str],
    ground_truth_chunk_names: list[str],
) -> dict[str, Any]:
    """Calculate MRR and recall for one query."""

    true_names = [item for item in ground_truth_chunk_names if item]
    true_set = set(true_names)
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
    for rank, chunk_name in enumerate(ranked_chunk_names, start=1):
        if chunk_name not in true_set or chunk_name in seen_hits:
            continue
        seen_hits.add(chunk_name)
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
        default="20",
        help="Comma-separated top_k values, e.g. 5,10,15.",
    )
    parser.add_argument(
        "--per-label-k",
        default="30",
        help="Comma-separated per-label candidate sizes.",
    )
    parser.add_argument(
        "--cross-encoder",
        default="true",
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
        default="0.01",
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
