from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from DocLLM.LLMs.llm_init import configure_langsmith

try:
    from .main import (
        build_analysis_id,
        build_cause_query_text,
        build_function_mode_query_text,
        get_query_item_by_number,
        load_structure_input,
        normalize_text,
        save_results,
    )
    from .qd_detection_control_agent import QDDetectionControlAgent, json_safe
except ImportError:  # pragma: no cover - supports direct script execution
    from main import (
        build_analysis_id,
        build_cause_query_text,
        build_function_mode_query_text,
        get_query_item_by_number,
        load_structure_input,
        normalize_text,
        save_results,
    )
    from qd_detection_control_agent import QDDetectionControlAgent, json_safe

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


load_dotenv()
LANGSMITH_PROJECT_NAME = configure_langsmith(
    os.getenv("LANGSMITH_PROJECT", "GraphRAGQDDetectionControl")
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "qd_results"


def redact_runtime_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep LangSmith trace inputs focused on serializable query data."""

    cleaned = dict(inputs)
    cleaned.pop("retriever", None)
    cleaned.pop("agent", None)
    return json_safe(cleaned)


def build_qd_structure_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Split structure input into QD queries for every mode, cause, and effect."""

    return (
        build_qd_mode_query_items(structure_input)
        + build_qd_cause_query_items(structure_input)
        + build_qd_effect_query_items(structure_input)
    )


def build_qd_mode_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one QD query item for each failure mode."""

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
                query_text = build_function_mode_query_text(function_text, mode_text)
                items.append(
                    {
                        "analysis_id": build_analysis_id(
                            element_id,
                            "mode",
                            function_text,
                            mode_text,
                        ),
                        "query_type": "function_mode",
                        "item_type": "mode",
                        "element_id": element_id,
                        "failure_element": failure_element,
                        "function_text": function_text,
                        "query_mode": mode_text,
                        "query_cause": "",
                        "query_effect": "",
                        "cause_discipline": "",
                        "query_text": query_text,
                        "disciplines": None,
                    }
                )

    return items


def build_qd_cause_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one QD query item for each failure cause."""

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
                query_text = build_cause_query_text(cause_text, discipline)
                items.append(
                    {
                        "analysis_id": build_analysis_id(
                            element_id,
                            "cause",
                            discipline,
                            cause_text,
                        ),
                        "query_type": "cause",
                        "item_type": "cause",
                        "element_id": element_id,
                        "failure_element": failure_element,
                        "function_text": "",
                        "query_mode": "",
                        "query_cause": cause_text,
                        "query_effect": "",
                        "cause_discipline": discipline,
                        "query_text": query_text,
                        "disciplines": None,
                    }
                )

    return items


def build_qd_effect_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one QD query item for each failure effect."""

    items: list[dict[str, Any]] = []
    for node in structure_input.get("nodes", []):
        element_id = normalize_text(node.get("element_id"))
        failure_element = normalize_text(node.get("failure_element"))
        effects = node.get("effects", {})
        if not isinstance(effects, dict):
            continue

        for function_text, effect_texts in effects.items():
            function_text = normalize_text(function_text)
            if not isinstance(effect_texts, list):
                continue

            for effect_text in effect_texts:
                effect_text = normalize_text(effect_text)
                if not effect_text:
                    continue
                query_text = build_effect_query_text(function_text, effect_text)
                items.append(
                    {
                        "analysis_id": build_analysis_id(
                            element_id,
                            "effect",
                            function_text,
                            effect_text,
                        ),
                        "query_type": "effect",
                        "item_type": "effect",
                        "element_id": element_id,
                        "failure_element": failure_element,
                        "function_text": function_text,
                        "query_mode": "",
                        "query_cause": "",
                        "query_effect": effect_text,
                        "cause_discipline": "",
                        "query_text": query_text,
                        "disciplines": None,
                    }
                )

    return items


def build_effect_query_text(function_text: str, effect_text: str) -> str:
    """Format the retrieval query for Function text + failure effect text."""

    function_text = normalize_text(function_text)
    effect_text = normalize_text(effect_text)
    if function_text:
        return f"{function_text} with {effect_text}"
    return effect_text


def iter_qd_structure_analysis_items() -> Iterable[dict[str, Any]]:
    """Loop all QD analysis items from structure_input_motorcontrol."""

    yield from build_qd_structure_query_items(load_structure_input())


def build_qd_query_text(analysis_item: dict[str, Any]) -> str:
    """Return the retrieval query text for one QD structure item."""

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

    if analysis_item.get("query_type") == "effect":
        return build_effect_query_text(
            normalize_text(analysis_item.get("function_text")),
            normalize_text(analysis_item.get("query_effect")),
        )

    return ""


def get_qd_query_item_text(analysis_item: dict[str, Any]) -> str:
    """Return the user-facing mode/cause/effect text for output naming."""

    for key in ("query_mode", "query_cause", "query_effect", "query_text"):
        text = normalize_text(analysis_item.get(key))
        if text:
            return text
    return normalize_text(analysis_item.get("analysis_id")) or "query"


def safe_output_stem(text: str, max_length: int = 120) -> str:
    """Convert query text into a stable, Windows-safe filename stem."""

    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", normalize_text(text))
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" ._")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "query"
    return cleaned[:max_length].rstrip(" ._") or "query"


def output_path_for_analysis_item(
    output_dir: Path,
    analysis_item: dict[str, Any],
    used_stems: set[str],
) -> Path:
    """Build a per-query qd_connection_<text>.json path."""

    base_stem = f"qd_connection_{safe_output_stem(get_qd_query_item_text(analysis_item))}"
    stem = base_stem
    suffix = 2
    while stem.lower() in used_stems:
        stem = f"{base_stem}_{suffix}"
        suffix += 1
    used_stems.add(stem.lower())
    return output_dir / f"{stem}.json"


def save_single_result(result: dict[str, Any], output_path: Path) -> None:
    """Persist one QD connection result as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_qd_query_items() -> list[dict[str, Any]]:
    """Return QD query items with user-facing query numbers."""

    return [
        {
            "query_number": index,
            "analysis_id": item.get("analysis_id", ""),
            "item_type": item.get("item_type", ""),
            "query_type": item.get("query_type", ""),
            "query_text": item.get("query_text", ""),
            "output_file": f"qd_connection_{safe_output_stem(get_qd_query_item_text(item))}.json",
            "cause_discipline": item.get("cause_discipline", ""),
        }
        for index, item in enumerate(iter_qd_structure_analysis_items(), start=1)
    ]


@traceable(
    run_type="retriever",
    name="graphrag_extraction_retrieve_qd_chunks",
    tags=["graphrag", "extraction", "qd", "retrieval"],
    project_name=LANGSMITH_PROJECT_NAME,
    process_inputs=redact_runtime_trace_inputs,
)
def run_qd_retrieval_for_analysis_item(
    retriever: Any,
    analysis_item: dict[str, Any],
    top_k: int = 20,
    per_label_k: int = 30,
    retrieval_mode: str = "hybrid",
    use_cross_encoder_rerank: bool = False,
    cross_encoder_top_n: int = 30,
    use_section_tag_bonus: bool = True,
    section_bonus_mode: str = "hybrid",
    section_bonus_weight: float = 0.05,
) -> dict[str, Any]:
    """Run main_sentence.py QD-only retrieval for one structure item."""

    from GraphRAG.main_sentence import (
        build_sentence_doc_chunk_query,
        query_doc_chunks_for_sentence,
    )

    query_text = build_qd_query_text(analysis_item)
    query_spec = build_sentence_doc_chunk_query(sentence=query_text)
    query_spec["query_type"] = analysis_item.get("query_type", "")
    query_spec["item_type"] = analysis_item.get("item_type", "")
    query_spec["function_text"] = analysis_item.get("function_text", "")
    query_spec["query_mode"] = analysis_item.get("query_mode", "")
    query_spec["query_cause"] = analysis_item.get("query_cause", "")
    query_spec["query_effect"] = analysis_item.get("query_effect", "")
    query_spec["cause_discipline"] = analysis_item.get("cause_discipline", "")
    query_spec["retrieval_target"] = "qd_detection_control"

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
        is_QD=True,
    )


@traceable(
    run_type="chain",
    name="graphrag_extraction_qd_detection_control_single_query",
    tags=["graphrag", "extraction", "qd", "single-query"],
    project_name=LANGSMITH_PROJECT_NAME,
    process_inputs=redact_runtime_trace_inputs,
)
def run_qd_detection_control_for_item(
    retriever: Any,
    agent: QDDetectionControlAgent,
    analysis_item: dict[str, Any],
    top_k: int = 20,
    per_label_k: int = 30,
    retrieval_mode: str = "hybrid",
) -> dict[str, Any]:
    """Run QD-only retrieval and select up to three detection-control chunks."""

    query_result = run_qd_retrieval_for_analysis_item(
        retriever=retriever,
        analysis_item=analysis_item,
        top_k=top_k,
        per_label_k=per_label_k,
        retrieval_mode=retrieval_mode,
    )
    selection = agent.select_detection_control(
        query_result=query_result,
        analysis_item=analysis_item,
    )
    return {
        "analysis_item": analysis_item,
        "query_result": query_result,
        "selection": selection,
    }


@traceable(
    run_type="chain",
    name="graphrag_extraction_qd_detection_control_pipeline",
    tags=["graphrag", "extraction", "qd", "pipeline"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def run_qd_detection_control_pipeline(
    output_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    use_placeholder_llm: bool | None = None,
    query_number: int | None = None,
    top_k: int = 20,
    per_label_k: int = 30,
    retrieval_mode: str = "hybrid",
) -> list[dict[str, Any]]:
    """Loop all mode/cause/effect items and save each QD result separately."""

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")

    agent_kwargs: dict[str, Any] = {}
    if use_placeholder_llm is not None:
        agent_kwargs["use_placeholder"] = use_placeholder_llm

    agent = QDDetectionControlAgent(**agent_kwargs)
    from GraphRAG.fmea_retrieverV2 import FMEASentenceRetrieverV2

    retriever = FMEASentenceRetrieverV2()
    results: list[dict[str, Any]] = []
    analysis_items = list(iter_qd_structure_analysis_items())
    if query_number is not None:
        analysis_items = [get_query_item_by_number(query_number, analysis_items)]

    used_output_stems: set[str] = set()
    try:
        for index, analysis_item in enumerate(analysis_items, start=1):
            print(
                f"[INFO] Processing QD query {index}/{len(analysis_items)}: "
                f"{analysis_item.get('analysis_id', '')}"
            )
            result = run_qd_detection_control_for_item(
                retriever=retriever,
                agent=agent,
                analysis_item=analysis_item,
                top_k=top_k,
                per_label_k=per_label_k,
                retrieval_mode=retrieval_mode,
            )
            item_output_path = output_path_for_analysis_item(
                output_dir=output_dir,
                analysis_item=analysis_item,
                used_stems=used_output_stems,
            )
            save_single_result(result, item_output_path)
            result["output_path"] = str(item_output_path)
            results.append(result)
            print(f"[INFO] Wrote QD connection result to: {item_output_path}")
    finally:
        retriever.close()

    if output_path is not None:
        save_results(results, output_path)
        print(f"[INFO] Wrote aggregate QD detection-control result(s) to: {output_path}")

    print(f"[INFO] Wrote {len(results)} QD connection result(s) under: {output_dir}")
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the QD detection-control workflow."""

    parser = argparse.ArgumentParser(description="Run GraphRAG QD detection-control selection.")
    parser.add_argument(
        "--query-number",
        type=int,
        default=None,
        help="Run one 1-based query number from structure_input_motorcontrol.",
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Print available mode/cause/effect query numbers and exit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for an aggregate JSON output file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-query qd_connection_<text>.json output files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Final number of retrieved QD/FAT candidates passed to the selector.",
    )
    parser.add_argument(
        "--per-label-k",
        type=int,
        default=30,
        help="Number of candidates to retrieve per QD/FAT label before final ranking.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="QD/FAT retrieval mode.",
    )
    parser.add_argument(
        "--placeholder-llm",
        action="store_true",
        help="Use the local placeholder selector instead of calling an LLM.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the GraphRAG QD detection-control pipeline."""

    args = parse_args()
    if args.list_queries:
        print(json.dumps(list_qd_query_items(), indent=2, ensure_ascii=False))
        return

    run_qd_detection_control_pipeline(
        output_path=args.output,
        output_dir=args.output_dir,
        use_placeholder_llm=args.placeholder_llm if args.placeholder_llm else None,
        query_number=args.query_number,
        top_k=args.top_k,
        per_label_k=args.per_label_k,
        retrieval_mode=args.retrieval_mode,
    )


if __name__ == "__main__":
    main()
