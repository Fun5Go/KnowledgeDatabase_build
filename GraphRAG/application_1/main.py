from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv
from DocLLM.LLMs.llm_init import configure_langsmith

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from .agent import CausalInferenceAgent, json_safe, normalize_text
except ImportError:  # pragma: no cover - supports direct script execution
    from agent import CausalInferenceAgent, json_safe, normalize_text

try:
    from GraphRAG.Extraction.main import (
        build_analysis_id,
        load_structure_input,
    )
except ImportError:  # pragma: no cover - supports unusual direct execution
    from ..Extraction.main import (
        build_analysis_id,
        load_structure_input,
    )

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


load_dotenv()
LANGSMITH_PROJECT_NAME = configure_langsmith(
    os.getenv("LANGSMITH_PROJECT", "GraphRAGApplication1")
)

APP_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = APP_DIR.parent / "Extraction"
DEFAULT_INPUT_GLOB = str(EXTRACTION_DIR / "chunk_selection_results_query*.json")
DEFAULT_OUTPUT_PATH = APP_DIR / "causal_inference_results.json"


def load_selection_results(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load chunk-selection result records from one or more JSON files."""

    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for record in payload:
                if isinstance(record, dict):
                    record = dict(record)
                    record["_source_file"] = str(path)
                    records.append(record)
        elif isinstance(payload, dict):
            payload = dict(payload)
            payload["_source_file"] = str(path)
            records.append(payload)
    return records


def resolve_input_paths(inputs: Sequence[str] | None) -> list[Path]:
    """Resolve CLI input paths/globs."""

    if not inputs:
        inputs = [DEFAULT_INPUT_GLOB]

    paths: list[Path] = []
    for item in inputs:
        candidate = Path(item)
        if any(token in item for token in "*?[]"):
            paths.extend(Path(path) for path in sorted(glob.glob(item)))
        elif candidate.is_dir():
            paths.extend(sorted(candidate.glob("chunk_selection_results_query*.json")))
        else:
            paths.append(candidate)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not resolved.exists():
            raise FileNotFoundError(f"Selection result file not found: {resolved}")
        seen.add(resolved)
        unique_paths.append(resolved)
    return unique_paths


def index_selection_results(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index selection records by analysis_id."""

    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        analysis_item = record.get("analysis_item") if isinstance(record.get("analysis_item"), dict) else {}
        selection = record.get("selection") if isinstance(record.get("selection"), dict) else {}
        analysis_id = normalize_text(
            analysis_item.get("analysis_id") or selection.get("analysis_id")
        )
        if not analysis_id:
            continue
        indexed[analysis_id] = record
    return indexed


def iter_causal_inference_items(
    structure_input: dict[str, Any],
    selection_index: dict[str, dict[str, Any]],
    include_unmatched: bool = False,
) -> Iterable[dict[str, Any]]:
    """Generate one possible-cause inference item per mode selection record."""

    for node in structure_input.get("nodes", []):
        element_id = normalize_text(node.get("element_id"))
        failure_element = normalize_text(node.get("failure_element"))
        mode_items = list(iter_mode_items(node))
        cause_items = list(iter_cause_items(node))

        for mode_item in mode_items:
            mode_record = selection_index.get(mode_item["analysis_id"])
            if mode_record and get_record_query_type(mode_record) != "function_mode":
                continue
            mode_chunks = extract_selected_chunks(mode_record, source="mode")
            if not include_unmatched and not mode_chunks:
                continue

            analysis_id = build_analysis_id(
                element_id,
                "causal_batch",
                mode_item["function_text"],
                mode_item["mode_text"],
            )
            yield {
                "analysis_id": analysis_id,
                "element_id": element_id,
                "failure_element": failure_element,
                "function_text": mode_item["function_text"],
                "mode_text": mode_item["mode_text"],
                "cause_candidates": [cause_item["cause_text"] for cause_item in cause_items],
                "mode_selection_analysis_id": mode_item["analysis_id"],
                "selection_chunks": mode_chunks,
            }


def get_record_query_type(record: dict[str, Any]) -> str:
    """Return the query_type from a chunk-selection result record."""

    analysis_item = record.get("analysis_item") if isinstance(record.get("analysis_item"), dict) else {}
    selection = record.get("selection") if isinstance(record.get("selection"), dict) else {}
    return normalize_text(analysis_item.get("query_type") or selection.get("query_type"))


def iter_mode_items(node: dict[str, Any]) -> Iterable[dict[str, str]]:
    """Yield function/mode entries with their chunk-selection analysis ids."""

    element_id = normalize_text(node.get("element_id"))
    modes = node.get("modes", {})
    if not isinstance(modes, dict):
        return
    for function_text, mode_texts in modes.items():
        function_text = normalize_text(function_text)
        if not isinstance(mode_texts, list):
            continue
        for mode_text in mode_texts:
            mode_text = normalize_text(mode_text)
            if not function_text or not mode_text:
                continue
            yield {
                "function_text": function_text,
                "mode_text": mode_text,
                "analysis_id": build_analysis_id(
                    element_id,
                    "function_mode",
                    function_text,
                    mode_text,
                ),
            }


def iter_cause_items(node: dict[str, Any]) -> Iterable[dict[str, str]]:
    """Yield cause entries with their chunk-selection analysis ids."""

    element_id = normalize_text(node.get("element_id"))
    causes = node.get("causes", {})
    if not isinstance(causes, dict):
        return
    for discipline, cause_texts in causes.items():
        discipline = normalize_text(discipline)
        if not isinstance(cause_texts, list):
            continue
        for cause_text in cause_texts:
            cause_text = normalize_text(cause_text)
            if not cause_text:
                continue
            yield {
                "cause_text": cause_text,
                "analysis_id": build_analysis_id(
                    element_id,
                    "cause",
                    discipline,
                    cause_text,
                ),
            }


def extract_selected_chunks(record: dict[str, Any] | None, source: str) -> list[dict[str, Any]]:
    """Return self/upstream selected chunks with at least moderate support."""

    if not record:
        return []
    selection = record.get("selection") if isinstance(record.get("selection"), dict) else {}
    chunks = selection.get("top_chunks") or selection.get("selected_chunks") or []
    if not isinstance(chunks, list):
        return []

    selected: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        relationship = normalize_text(chunk.get("relationship")).lower()
        support_capability = normalize_text(chunk.get("support_capability")).lower()
        if relationship not in {"self", "upstream"}:
            continue
        if support_capability == "weak":
            continue
        selected_chunk = dict(chunk)
        selected_chunk["source"] = source
        selected_chunk["source_file"] = record.get("_source_file", "")
        selected.append(selected_chunk)
    return selected


def select_item_by_number(items: list[dict[str, Any]], item_number: int | None) -> list[dict[str, Any]]:
    """Return all items or one 1-based item."""

    if item_number is None:
        return items
    if item_number < 1 or item_number > len(items):
        raise IndexError(f"item_number must be between 1 and {len(items)}, got {item_number}.")
    return [items[item_number - 1]]


def list_inference_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact item metadata for CLI listing."""

    return [
        {
            "item_number": index,
            "analysis_id": item.get("analysis_id", ""),
            "mode_text": item.get("mode_text", ""),
            "cause_candidate_count": len(item.get("cause_candidates", [])),
            "selection_chunk_count": len(item.get("selection_chunks", [])),
        }
        for index, item in enumerate(items, start=1)
    ]


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Persist causal-inference results as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@traceable(
    run_type="chain",
    name="graphrag_application_1_causal_inference_single_item",
    tags=["graphrag", "application_1", "causal-inference", "single-item"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def run_causal_inference_for_item(
    agent: CausalInferenceAgent,
    inference_item: dict[str, Any],
) -> dict[str, Any]:
    """Run LLM possible-cause inference for one generated item."""

    return agent.infer_causal_chain(inference_item)


@traceable(
    run_type="chain",
    name="graphrag_application_1_causal_inference_pipeline",
    tags=["graphrag", "application_1", "causal-inference", "pipeline"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def run_causal_inference_pipeline(
    input_paths: Sequence[Path],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    use_placeholder_llm: bool | None = None,
    item_number: int | None = None,
    include_unmatched: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read mode chunk-selection JSON files and infer possible causes."""

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")
    records = load_selection_results(input_paths)
    selection_index = index_selection_results(records)
    items = list(
        iter_causal_inference_items(
            structure_input=load_structure_input(),
            selection_index=selection_index,
            include_unmatched=include_unmatched,
        )
    )
    items = select_item_by_number(items, item_number)
    if limit is not None:
        items = items[:limit]

    agent_kwargs: dict[str, Any] = {}
    if use_placeholder_llm is not None:
        agent_kwargs["use_placeholder"] = use_placeholder_llm
    agent = CausalInferenceAgent(**agent_kwargs)

    results: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        print(
            f"[INFO] Processing causal item {index}/{len(items)}: "
            f"{item.get('analysis_id', '')}"
        )
        results.append(run_causal_inference_for_item(agent=agent, inference_item=item))

    save_results(results, output_path)
    print(f"[INFO] Wrote {len(results)} causal-inference result(s) to: {output_path}")
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for application_1 causal inference."""

    parser = argparse.ArgumentParser(
        description="Run GraphRAG application_1 causal inference over chunk-selection results."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help=(
            "Selection JSON file, directory, or glob. May be repeated. "
            "Default: GraphRAG/Extraction/chunk_selection_results_query*.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the JSON output file.",
    )
    parser.add_argument(
        "--item-number",
        type=int,
        default=None,
        help="Run one 1-based generated possible-cause item.",
    )
    parser.add_argument(
        "--list-items",
        action="store_true",
        help="Print generated inference items and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of generated inference items.",
    )
    parser.add_argument(
        "--include-unmatched",
        action="store_true",
        help="Generate items even when mode/cause selected chunks are missing.",
    )
    parser.add_argument(
        "--placeholder-llm",
        action="store_true",
        help="Use the local placeholder inference instead of calling an LLM.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for application_1 causal inference."""

    args = parse_args()
    input_paths = resolve_input_paths(args.input)
    records = load_selection_results(input_paths)
    selection_index = index_selection_results(records)
    items = list(
        iter_causal_inference_items(
            structure_input=load_structure_input(),
            selection_index=selection_index,
            include_unmatched=args.include_unmatched,
        )
    )

    if args.list_items:
        print(json.dumps(list_inference_items(items), indent=2, ensure_ascii=False))
        return

    selected_paths = input_paths
    run_causal_inference_pipeline(
        input_paths=selected_paths,
        output_path=args.output,
        use_placeholder_llm=args.placeholder_llm if args.placeholder_llm else None,
        item_number=args.item_number,
        include_unmatched=args.include_unmatched,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
