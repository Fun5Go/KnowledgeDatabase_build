from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from DocLLM.LLMs.llm_init import configure_langsmith

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from .chunk_selection_agent import ChunkSelectionAgent, json_safe
except ImportError:  # pragma: no cover - supports direct script execution
    from chunk_selection_agent import ChunkSelectionAgent, json_safe

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


load_dotenv()
LANGSMITH_PROJECT_NAME = configure_langsmith(
    os.getenv("LANGSMITH_PROJECT", "GraphRAGExtraction")
)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "chunk_selection_results.json"

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


def load_structure_input() -> dict[str, Any]:
    """Return the structure analysis input used by this extraction pass."""

    return structure_input_motorcontrol


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


def build_function_mode_query_text(function_text: str, mode_text: str) -> str:
    """Format the retrieval query for Function text + failure mode text."""

    return f"{function_text} has {mode_text}"


def build_cause_query_text(cause_text: str, discipline: str) -> str:
    """Format the retrieval query for cause text with discipline."""


    return f"{cause_text}"


def map_cause_discipline_to_retrieval_labels(discipline: str) -> list[str] | None:
    """
    Map structure cause discipline to GraphRAG retrieval labels.

    `main_sentence.normalize_discipline_labels` currently accepts ESW, HW, and FS.
    Mechanics/other stay unfiltered so the query can still search all chunks.
    """

    discipline_key = normalize_text(discipline).lower()
    mapping = {
        "hardware": ["HW"],
        "hw": ["HW"],
        "software": ["ESW"],
        "sw": ["ESW"],
        "esw": ["ESW"],
        "fs": ["FS"],
        "functional safety": ["FS"],
        "mechanics": ["MCH"]
    }
    return mapping.get(discipline_key)


def iter_structure_analysis_items() -> Iterable[dict[str, Any]]:
    """Loop the structure_input_motorcontrol query items."""

    yield from build_structure_query_items(load_structure_input())


def build_query_text(analysis_item: dict[str, Any]) -> str:
    """Return the retrieval query text for one split structure item."""

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


def redact_runtime_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep LangSmith trace inputs focused on serializable query data."""

    cleaned = dict(inputs)
    cleaned.pop("retriever", None)
    cleaned.pop("agent", None)
    return json_safe(cleaned)


@traceable(
    run_type="retriever",
    name="graphrag_extraction_retrieve_chunks",
    tags=["graphrag", "extraction", "retrieval"],
    project_name=LANGSMITH_PROJECT_NAME,
    process_inputs=redact_runtime_trace_inputs,
)
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
    """Run the existing main_sentence.py query function for one structure item."""

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
        disciplines=analysis_item.get("disciplines"),
        use_cross_encoder_rerank=use_cross_encoder_rerank,
        cross_encoder_top_n=cross_encoder_top_n,
        use_section_tag_bonus=use_section_tag_bonus,
        section_bonus_mode=section_bonus_mode,
        section_bonus_weight=section_bonus_weight,
    )


@traceable(
    run_type="chain",
    name="graphrag_extraction_single_query",
    tags=["graphrag", "extraction", "single-query"],
    project_name=LANGSMITH_PROJECT_NAME,
    process_inputs=redact_runtime_trace_inputs,
)
def run_chunk_selection_for_item(
    retriever: Any,
    agent: ChunkSelectionAgent,
    analysis_item: dict[str, Any],
) -> dict[str, Any]:
    """Run retrieval and LLM chunk selection for one structure query item."""

    query_result = run_retrieval_for_analysis_item(
        retriever=retriever,
        analysis_item=analysis_item,
    )
    selection = agent.select_chunks(
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
    name="graphrag_extraction_pipeline",
    tags=["graphrag", "extraction", "pipeline"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def run_chunk_selection_pipeline(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    use_placeholder_llm: bool | None = None,
    query_number: int | None = None,
) -> list[dict[str, Any]]:
    """Loop structure items, retrieve candidate chunks, and call the LLM agent."""

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")

    agent_kwargs: dict[str, Any] = {}
    if use_placeholder_llm is not None:
        agent_kwargs["use_placeholder"] = use_placeholder_llm

    agent = ChunkSelectionAgent(**agent_kwargs)
    from GraphRAG.fmea_retrieverV2 import FMEASentenceRetrieverV2

    retriever = FMEASentenceRetrieverV2()
    results: list[dict[str, Any]] = []
    analysis_items = list(iter_structure_analysis_items())
    if query_number is not None:
        analysis_items = [get_query_item_by_number(query_number, analysis_items)]

    try:
        for index, analysis_item in enumerate(analysis_items, start=1):
            print(
                f"[INFO] Processing query {index}/{len(analysis_items)}: "
                f"{analysis_item.get('analysis_id', '')}"
            )
            results.append(
                run_chunk_selection_for_item(
                    retriever=retriever,
                    agent=agent,
                    analysis_item=analysis_item,
                )
            )
    finally:
        retriever.close()

    save_results(results, output_path)
    print(f"[INFO] Wrote {len(results)} chunk-selection result(s) to: {output_path}")
    return results


def get_query_item_by_number(
    query_number: int,
    analysis_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a 1-based query item from the generated structure queries."""

    items = analysis_items or list(iter_structure_analysis_items())
    if query_number < 1 or query_number > len(items):
        raise IndexError(f"query_number must be between 1 and {len(items)}, got {query_number}.")
    return items[query_number - 1]


def list_query_items() -> list[dict[str, Any]]:
    """Return query items with user-facing query numbers."""

    return [
        {
            "query_number": index,
            "analysis_id": item.get("analysis_id", ""),
            "query_type": item.get("query_type", ""),
            "query_text": item.get("query_text", ""),
            "cause_discipline": item.get("cause_discipline", ""),
        }
        for index, item in enumerate(iter_structure_analysis_items(), start=1)
    ]


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Persist pipeline output as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def output_path_with_query_number(output_path: Path, query_number: int | None) -> Path:
    """Add a query-number suffix to the default output path."""

    if query_number is None:
        return output_path
    return output_path.with_name(f"{output_path.stem}_query_{query_number}{output_path.suffix}")


def build_analysis_id(*parts: str) -> str:
    """Build a stable readable id from structure query parts."""

    cleaned = [normalize_text(part).lower().replace(" ", "-") for part in parts if normalize_text(part)]
    return ":".join(cleaned)


def normalize_text(value: Any) -> str:
    """Normalize values to safe strings."""

    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the extraction workflow."""

    parser = argparse.ArgumentParser(description="Run GraphRAG chunk-selection extraction.")
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
    """Entry point for the GraphRAG extraction chunk-selection pipeline."""

    args = parse_args()
    if args.list_queries:
        print(json.dumps(list_query_items(), indent=2, ensure_ascii=False))
        return

    output_path = args.output or output_path_with_query_number(
        DEFAULT_OUTPUT_PATH,
        args.query_number,
    )

    run_chunk_selection_pipeline(
        output_path=output_path,
        use_placeholder_llm=args.placeholder_llm if args.placeholder_llm else None,
        query_number=args.query_number,
    )


if __name__ == "__main__":
    main()
