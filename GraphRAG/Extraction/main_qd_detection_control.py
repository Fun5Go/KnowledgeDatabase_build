from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from DocLLM.LLMs.llm_init import configure_langsmith

try:
    from .main import (
        build_query_text,
        get_query_item_by_number,
        iter_structure_analysis_items,
        list_query_items,
        normalize_text,
        output_path_with_query_number,
        save_results,
    )
    from .qd_detection_control_agent import QDDetectionControlAgent, json_safe
except ImportError:  # pragma: no cover - supports direct script execution
    from main import (
        build_query_text,
        get_query_item_by_number,
        iter_structure_analysis_items,
        list_query_items,
        normalize_text,
        output_path_with_query_number,
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

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "qd_detection_control_results.json"


def redact_runtime_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep LangSmith trace inputs focused on serializable query data."""

    cleaned = dict(inputs)
    cleaned.pop("retriever", None)
    cleaned.pop("agent", None)
    return json_safe(cleaned)


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
    top_k: int = 10,
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

    query_text = build_query_text(analysis_item)
    query_spec = build_sentence_doc_chunk_query(sentence=query_text)
    query_spec["query_type"] = analysis_item.get("query_type", "")
    query_spec["function_text"] = analysis_item.get("function_text", "")
    query_spec["query_mode"] = analysis_item.get("query_mode", "")
    query_spec["query_cause"] = analysis_item.get("query_cause", "")
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
) -> dict[str, Any]:
    """Run QD-only retrieval and select up to three detection-control chunks."""

    query_result = run_qd_retrieval_for_analysis_item(
        retriever=retriever,
        analysis_item=analysis_item,
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
    output_path: Path = DEFAULT_OUTPUT_PATH,
    use_placeholder_llm: bool | None = None,
    query_number: int | None = None,
) -> list[dict[str, Any]]:
    """Loop structure items, retrieve QD candidates, and call the QD LLM agent."""

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")

    agent_kwargs: dict[str, Any] = {}
    if use_placeholder_llm is not None:
        agent_kwargs["use_placeholder"] = use_placeholder_llm

    agent = QDDetectionControlAgent(**agent_kwargs)
    from GraphRAG.fmea_retrieverV2 import FMEASentenceRetrieverV2

    retriever = FMEASentenceRetrieverV2()
    results: list[dict[str, Any]] = []
    analysis_items = list(iter_structure_analysis_items())
    if query_number is not None:
        analysis_items = [get_query_item_by_number(query_number, analysis_items)]

    try:
        for index, analysis_item in enumerate(analysis_items, start=1):
            print(
                f"[INFO] Processing QD query {index}/{len(analysis_items)}: "
                f"{analysis_item.get('analysis_id', '')}"
            )
            results.append(
                run_qd_detection_control_for_item(
                    retriever=retriever,
                    agent=agent,
                    analysis_item=analysis_item,
                )
            )
    finally:
        retriever.close()

    save_results(results, output_path)
    print(f"[INFO] Wrote {len(results)} QD detection-control result(s) to: {output_path}")
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
        help="Print available query numbers and exit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the JSON output file.",
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
        print(json.dumps(list_query_items(), indent=2, ensure_ascii=False))
        return

    output_path = args.output or output_path_with_query_number(
        DEFAULT_OUTPUT_PATH,
        args.query_number,
    )

    run_qd_detection_control_pipeline(
        output_path=output_path,
        use_placeholder_llm=args.placeholder_llm if args.placeholder_llm else None,
        query_number=args.query_number,
    )


if __name__ == "__main__":
    main()
