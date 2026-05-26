from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Literal

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from DocLLM.LLMs.llm_init import configure_langsmith

try:
    from .connection_agents import EvidenceRelationExtractionAgent, json_safe
    from .connection_workflow import ConnectionWorkflow
    from .prompts import LOOSE_EVIDENCE_RELATION_EXTRACTION_PROMPT
    from .validators import build_summary
except ImportError:  # pragma: no cover - supports direct script execution
    from connection_agents import EvidenceRelationExtractionAgent, json_safe
    from connection_workflow import ConnectionWorkflow
    from prompts import LOOSE_EVIDENCE_RELATION_EXTRACTION_PROMPT
    from validators import build_summary


load_dotenv()
LANGSMITH_PROJECT_NAME = configure_langsmith(
    os.getenv("LANGSMITH_PROJECT", "GraphRAGConnection")
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "connection_results.json"
DEFAULT_RESULTS_DIR_NAME = "results"

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
            "effects": {
                "Soft starter": [
                    "Motor cannot start",
                    "Overcurrent towards motor",
                    "Motor starts without soft start",
                ],
                "Electrical failure": [
                    "Short-circuit",
                ],
            },
        }
    ],
}


def load_structure_input() -> dict[str, Any]:
    """Return the structure analysis input used by this Connection pass."""

    return structure_input_motorcontrol


def build_function_mode_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Function text + Failure mode query items."""

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
    """Build Failure cause query items."""

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
                        "disciplines": None,
                    }
                )
    return items


def build_effect_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Function text + Failure effect query items."""

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
                if not function_text or not effect_text:
                    continue
                items.append(
                    {
                        "analysis_id": build_analysis_id(
                            element_id,
                            "effect",
                            function_text,
                            effect_text,
                        ),
                        "query_type": "effect",
                        "element_id": element_id,
                        "failure_element": failure_element,
                        "function_text": function_text,
                        "query_mode": "",
                        "query_cause": "",
                        "query_effect": effect_text,
                        "cause_discipline": "",
                        "query_text": build_effect_query_text(function_text, effect_text),
                        "disciplines": None,
                    }
                )
    return items


def build_structure_query_items(structure_input: dict[str, Any]) -> list[dict[str, Any]]:
    return (
        build_function_mode_query_items(structure_input)
        + build_cause_query_items(structure_input)
        + build_effect_query_items(structure_input)
    )


def iter_structure_analysis_items() -> Iterable[dict[str, Any]]:
    yield from build_structure_query_items(load_structure_input())


def build_function_mode_query_text(function_text: str, mode_text: str) -> str:
    return f"{mode_text} in {function_text}  "


def build_cause_query_text(cause_text: str, discipline: str) -> str:
    return cause_text


def build_effect_query_text(function_text: str, effect_text: str) -> str:
    return effect_text


def build_query_text(analysis_item: dict[str, Any]) -> str:
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


def build_connection_payload(
    query_result: dict[str, Any],
    analysis_item: dict[str, Any],
    max_chunk_text_length: int = 1800,
    rerank_input_start: int | None = None,
    rerank_input_end: int | None = None,
) -> dict[str, Any]:
    """Structure retrieval output into the Connection workflow input payload.

    The payload intentionally omits retrieval labels. section_tag is retained because it
    can provide useful document-context hints to the rerank/extraction agents.
    """

    evidence = order_evidence_by_connected_groups(
        query_result.get("evidence", []),
        query_result.get("connected_evidence_groups", []),
    )
    group_id_by_node_id = build_connected_group_id_by_node_id(
        query_result.get("connected_evidence_groups", [])
    )
    candidate_chunks = [
        normalize_candidate_chunk(
            item,
            index,
            max_chunk_text_length,
            connected_group_id=group_id_by_node_id.get(normalize_text(item.get("node_id"))),
        )
        for index, item in enumerate(evidence, start=1)
    ]
    candidate_chunks = filter_candidate_chunks_by_retrieval_rank(
        candidate_chunks,
        start=rerank_input_start,
        end=rerank_input_end,
    )

    return {
        "analysis_id": normalize_text(analysis_item.get("analysis_id")),
        "query_type": normalize_text(analysis_item.get("query_type")),
        "query": build_llm_query_payload(analysis_item),
        "candidate_chunks": candidate_chunks,
        "connected_chunk_groups": normalize_connected_chunk_groups(
            query_result.get("connected_evidence_groups", []),
            evidence,
            candidate_chunks,
        ),
    }


def filter_candidate_chunks_by_retrieval_rank(
    candidate_chunks: list[dict[str, Any]],
    start: int | None = None,
    end: int | None = None,
) -> list[dict[str, Any]]:
    if start is None and end is None:
        return candidate_chunks
    start = start or 1
    return [
        chunk
        for chunk in candidate_chunks
        if rank_is_in_range(normalize_int(chunk.get("retrieval rank")), start, end)
    ]


def rank_is_in_range(rank: int | None, start: int, end: int | None) -> bool:
    if rank is None:
        return False
    if rank < start:
        return False
    if end is not None and rank > end:
        return False
    return True


def build_llm_query_payload(analysis_item: dict[str, Any]) -> dict[str, str]:
    if normalize_text(analysis_item.get("query_type")) == "cause":
        return {
            "Failure cause": normalize_text(analysis_item.get("query_cause")),
            "Discipline": normalize_text(analysis_item.get("cause_discipline")),
        }
    if normalize_text(analysis_item.get("query_type")) == "effect":
        return {
            "Function": normalize_text(analysis_item.get("function_text")),
            "Failure effect": normalize_text(analysis_item.get("query_effect")),
        }
    return {
        "Function": normalize_text(analysis_item.get("function_text")),
        "Failure mode": normalize_text(analysis_item.get("query_mode")),
    }


def normalize_candidate_chunk(
    item: dict[str, Any],
    retrieval_rank: int,
    max_text_length: int,
    connected_group_id: int | None = None,
) -> dict[str, Any]:
    chunk = {
        "retrieval rank": normalize_int(item.get("retrieval_rank")) or retrieval_rank,
        "name": normalize_text(item.get("name")),
        "section_tag": normalize_text(item.get("section_tag")),
        "text": trim_text(item.get("text"), max_text_length),
    }
    if connected_group_id is not None:
        chunk["connected_group_id"] = connected_group_id
    return chunk


def order_evidence_by_connected_groups(evidence: Any, connected_groups: Any) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []
    evidence_items = [item for item in evidence if isinstance(item, dict)]
    evidence_by_id = {
        normalize_text(item.get("node_id")): item
        for item in evidence_items
        if normalize_text(item.get("node_id"))
    }
    if not isinstance(connected_groups, list) or not evidence_by_id:
        return evidence_items

    ordered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for group in connected_groups:
        if not isinstance(group, dict):
            continue
        group_ids = [
            normalize_text(node_id)
            for node_id in group.get("node_ids", [])
            if normalize_text(node_id) in evidence_by_id
        ]
        group_ids.sort(
            key=lambda node_id: normalize_int(evidence_by_id[node_id].get("retrieval_rank")) or 0
        )
        for node_id in group_ids:
            if node_id in seen_ids:
                continue
            ordered.append(evidence_by_id[node_id])
            seen_ids.add(node_id)

    for item in evidence_items:
        node_id = normalize_text(item.get("node_id"))
        if node_id and node_id in seen_ids:
            continue
        ordered.append(item)
    return ordered


def build_connected_group_id_by_node_id(connected_groups: Any) -> dict[str, int]:
    if not isinstance(connected_groups, list):
        return {}
    group_id_by_node_id: dict[str, int] = {}
    for group in connected_groups:
        if not isinstance(group, dict) or not group.get("has_relationships"):
            continue
        relationships = [
            relationship
            for relationship in group.get("relationships", [])
            if isinstance(relationship, dict)
            and normalize_text(relationship.get("relationship")) in {"RELATED", "IMPLEMENT"}
        ]
        if not relationships:
            continue
        group_id = normalize_int(group.get("group_id"))
        if group_id is None:
            continue
        for node_id in group.get("node_ids", []):
            node_id = normalize_text(node_id)
            if node_id:
                group_id_by_node_id[node_id] = group_id
    return group_id_by_node_id


def normalize_connected_chunk_groups(
    connected_groups: Any,
    evidence: list[dict[str, Any]],
    candidate_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(connected_groups, list):
        return {}
    allowed_ranks = None
    if candidate_chunks is not None:
        allowed_ranks = {
            normalize_int(chunk.get("retrieval rank"))
            for chunk in candidate_chunks
            if normalize_int(chunk.get("retrieval rank")) is not None
        }
    chunk_by_node_id = {
        normalize_text(item.get("node_id")): {
            "rank": normalize_int(item.get("retrieval_rank")) or index,
            "name": normalize_text(item.get("name")),
        }
        for index, item in enumerate(evidence, start=1)
        if normalize_text(item.get("node_id"))
    }
    groups: dict[str, dict[str, Any]] = {}
    for group in connected_groups:
        if not isinstance(group, dict) or not group.get("has_relationships"):
            continue
        node_ids = [
            normalize_text(node_id)
            for node_id in group.get("node_ids", [])
            if normalize_text(node_id) in chunk_by_node_id
            and (
                allowed_ranks is None
                or normalize_int(chunk_by_node_id[normalize_text(node_id)].get("rank"))
                in allowed_ranks
            )
        ]
        if len(node_ids) < 2:
            continue
        relationships = []
        for relationship in group.get("relationships", []):
            if not isinstance(relationship, dict):
                continue
            source_id = normalize_text(relationship.get("source_id"))
            target_id = normalize_text(relationship.get("target_id"))
            if source_id not in chunk_by_node_id or target_id not in chunk_by_node_id:
                continue
            if allowed_ranks is not None and (
                normalize_int(chunk_by_node_id[source_id].get("rank")) not in allowed_ranks
                or normalize_int(chunk_by_node_id[target_id].get("rank")) not in allowed_ranks
            ):
                continue
            relationships.append(
                {
                    "source_rank": chunk_by_node_id[source_id]["rank"],
                    "source_name": chunk_by_node_id[source_id]["name"],
                    "relationship": normalize_text(relationship.get("relationship")),
                    "target_rank": chunk_by_node_id[target_id]["rank"],
                    "target_name": chunk_by_node_id[target_id]["name"],
                }
            )
        if relationships:
            group_id = normalize_int(group.get("group_id")) or len(groups) + 1
            groups[str(group_id)] = {
                "chunks": [chunk_by_node_id[node_id] for node_id in node_ids],
                "relationships": relationships,
            }
    return groups


def run_retrieval_for_analysis_item(
    retriever: Any,
    analysis_item: dict[str, Any],
    top_k: int = 90,
    per_label_k: int = 80,
    retrieval_mode: str = "dense",
    use_cross_encoder_rerank: bool = False,
    cross_encoder_top_n: int = 30,
    use_section_tag_bonus: bool = False,
    section_bonus_mode: str = "hybrid",
    section_bonus_weight: float = 0.000,
) -> dict[str, Any]:
    """Run the existing sentence-chunk retriever for one structure query item."""

    from GraphRAG.main_sentence import (
        build_sentence_doc_chunk_query,
        query_doc_chunks_for_sentence,
    )

    query_spec = build_sentence_doc_chunk_query(sentence=build_query_text(analysis_item))
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
        disciplines=analysis_item.get("disciplines"),
        use_cross_encoder_rerank=use_cross_encoder_rerank,
        cross_encoder_top_n=cross_encoder_top_n,
        use_section_tag_bonus=use_section_tag_bonus,
        section_bonus_mode=section_bonus_mode,
        section_bonus_weight=section_bonus_weight,
    )


def run_connection_for_item(
    retriever: Any,
    workflow: ConnectionWorkflow,
    analysis_item: dict[str, Any],
    stage: Literal["rerank", "extract", "auto"] = "auto",
    batch_size: int | None = None,
    retrieval_top_k: int = 60,
    rerank_input_start: int | None = None,
    rerank_input_end: int | None = None,
) -> dict[str, Any]:
    query_result = run_retrieval_for_analysis_item(
        retriever=retriever,
        analysis_item=analysis_item,
        top_k=retrieval_top_k,
    )
    payload = build_connection_payload(
        query_result=query_result,
        analysis_item=analysis_item,
        rerank_input_start=rerank_input_start,
        rerank_input_end=rerank_input_end,
    )
    result = workflow.run(payload, stage=stage, batch_size=batch_size)
    return {
        "analysis_item": analysis_item,
        "query_result": query_result,
        "connection_payload": payload,
        "connection": result,
    }


def run_connection_pipeline(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    output_dir: Path | None = None,
    stage: Literal["rerank", "extract", "auto"] = "auto",
    batch_size: int | None = None,
    use_placeholder_llm: bool | None = None,
    query_number: int | None = None,
    retrieval_top_k: int = 60,
    rerank_input_start: int | None = None,
    rerank_input_end: int | None = None,
) -> list[dict[str, Any]]:
    """Loop structure items, retrieve chunks, build payloads, and run Connection."""

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")
    print(f"[INFO] Retrieval top_k: {retrieval_top_k}")
    if rerank_input_start is not None or rerank_input_end is not None:
        range_label = f"{rerank_input_start or 1}:{rerank_input_end or retrieval_top_k}"
        print(f"[INFO] Connection rerank input retrieval-rank range: {range_label}")
    workflow = ConnectionWorkflow(use_placeholder=use_placeholder_llm)

    from GraphRAG.fmea_retrieverV2 import FMEASentenceRetrieverV2

    retriever = FMEASentenceRetrieverV2()
    results: list[dict[str, Any]] = []
    analysis_items = list(iter_structure_analysis_items())
    if query_number is not None:
        analysis_items = [get_query_item_by_number(query_number, analysis_items)]
    auto_results_dir = output_path.parent / DEFAULT_RESULTS_DIR_NAME if stage == "auto" else None
    if auto_results_dir is not None:
        auto_results_dir.mkdir(parents=True, exist_ok=True)
    stage_output_dir = output_dir if stage in {"rerank", "extract"} else None
    stage_output_names: set[str] = set()
    if stage_output_dir is not None:
        stage_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for index, analysis_item in enumerate(analysis_items, start=1):
            print(
                f"[INFO] Processing query {index}/{len(analysis_items)}: "
                f"{analysis_item.get('analysis_id', '')}"
            )
            result = run_connection_for_item(
                retriever=retriever,
                workflow=workflow,
                analysis_item=analysis_item,
                stage=stage,
                batch_size=batch_size,
                retrieval_top_k=retrieval_top_k,
                rerank_input_start=rerank_input_start,
                rerank_input_end=rerank_input_end,
            )
            results.append(result)
            if auto_results_dir is not None:
                saved_paths = save_auto_stage_results([result], auto_results_dir)
                print(
                    "[INFO] Wrote auto stage result file(s): "
                    + ", ".join(str(path) for path in saved_paths)
                )
            if stage_output_dir is not None:
                saved_path = save_single_stage_result(
                    result,
                    stage_output_dir,
                    stage,
                    stage_output_names,
                )
                print(f"[INFO] Wrote {stage} stage result file: {saved_path}")
    finally:
        retriever.close()

    if stage == "auto":
        print(f"[INFO] Wrote auto stage result files to: {auto_results_dir}")
        return results

    if output_dir is not None:
        print(
            f"[INFO] Wrote {len(results)} {stage} stage result file(s) to: {output_dir}"
        )
        return results

    save_result(results, output_path)
    print(f"[INFO] Wrote {len(results)} Connection result(s) to: {output_path}")
    return results


def run_connection_payload_file(
    input_path: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    stage: Literal["rerank", "extract", "auto"] = "auto",
    batch_size: int | None = None,
    use_placeholder_llm: bool | None = None,
) -> Any:
    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")
    loaded = load_payload(input_path)
    workflow = ConnectionWorkflow(use_placeholder=use_placeholder_llm)
    payloads = build_payloads_from_loaded_input(loaded, stage=stage)
    results = [
        workflow.run(payload, stage=stage, batch_size=batch_size)
        for payload in payloads
    ]
    result: Any = results[0] if isinstance(loaded, dict) and len(results) == 1 else results
    save_result(result, output_path)
    print(f"[INFO] Wrote Connection result to: {output_path}")
    return result


def run_loose_extraction_for_rerank_dir(
    input_dir: Path,
    output_dir: Path | None = None,
    batch_size: int | None = None,
    use_placeholder_llm: bool | None = None,
    glob_pattern: str = "rerank_*.json",
) -> list[Path]:
    """Run loose extraction for every saved rerank result JSON in a directory."""

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Rerank results directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--extract-rerank-dir must be a directory: {input_dir}")

    output_dir = output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(input_dir.glob(glob_pattern))
    if not input_paths:
        raise FileNotFoundError(f"No rerank result JSON files matched {glob_pattern!r} in {input_dir}")

    extraction_agent = EvidenceRelationExtractionAgent(
        use_placeholder=use_placeholder_llm if use_placeholder_llm is not None else False,
        prompt_template=LOOSE_EVIDENCE_RELATION_EXTRACTION_PROMPT,
    )
    workflow = ConnectionWorkflow(
        extraction_agent=extraction_agent,
        use_placeholder=use_placeholder_llm,
    )

    saved_paths: list[Path] = []
    for index, input_path in enumerate(input_paths, start=1):
        print(f"[INFO] Loose extracting {index}/{len(input_paths)}: {input_path}")
        loaded = load_payload(input_path)
        payloads = build_payloads_from_loaded_input(loaded, stage="extract")
        extracted_results: list[dict[str, Any]] = []
        for payload in payloads:
            extract_connection = workflow.run(payload, stage="extract", batch_size=batch_size)
            extracted_results.append(
                build_loose_extract_pipeline_result(
                    loaded=loaded,
                    payload=payload,
                    extract_connection=extract_connection,
                )
            )

        result: Any = extracted_results[0] if len(extracted_results) == 1 else extracted_results
        output_path = output_dir / loose_extract_filename_for_rerank(input_path)
        save_result(result, output_path)
        saved_paths.append(output_path)
        print(f"[INFO] Wrote loose extraction result to: {output_path}")

    return saved_paths


def loose_extract_filename_for_rerank(input_path: Path) -> str:
    stem = input_path.stem
    if stem.startswith("rerank_"):
        stem = stem[len("rerank_") :]
    return f"loose_extract_{stem}{input_path.suffix}"


def build_loose_extract_pipeline_result(
    loaded: Any,
    payload: dict[str, Any],
    extract_connection: dict[str, Any],
) -> dict[str, Any]:
    original = loaded if isinstance(loaded, dict) else {}
    connection = dict(extract_connection)
    connection["summary"] = build_summary(
        reranked_chunks=payload.get("reranked_chunks", []),
        evidence_units=connection.get("evidence_units", []),
        chunk_aggregates=connection.get("chunk_aggregates", []),
        candidate_chunks=payload.get("candidate_chunks", []),
    )
    return {
        "analysis_item": original.get("analysis_item", {}),
        "query_result": original.get("query_result", {}),
        "connection_payload": payload,
        "connection": connection,
    }


def load_payload(input_path: Path) -> Any:
    if not input_path.exists():
        raise FileNotFoundError(f"Input payload not found: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def build_payloads_from_loaded_input(
    loaded: Any,
    stage: Literal["rerank", "extract", "auto"],
) -> list[dict[str, Any]]:
    """Accept raw payloads or previous Connection result files as --input."""

    if isinstance(loaded, list):
        return [
            payload
            for item in loaded
            for payload in build_payloads_from_loaded_input(item, stage)
        ]
    if not isinstance(loaded, dict):
        raise TypeError("--input must contain a Connection payload or Connection result object.")

    if "connection_payload" in loaded:
        payload = dict(loaded.get("connection_payload") or {})
        connection = loaded.get("connection") if isinstance(loaded.get("connection"), dict) else {}
        if stage == "extract":
            reranked_chunks = connection.get("reranked_chunks", [])
            if reranked_chunks:
                payload["reranked_chunks"] = reranked_chunks
        return [payload]

    if "candidate_chunks" in loaded:
        return [loaded]

    raise ValueError(
        "--input did not look like a raw Connection payload or a saved "
        "Connection pipeline result with connection_payload."
    )


def save_result(result: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_auto_stage_results(results: list[dict[str, Any]], results_dir: Path) -> list[Path]:
    """Save auto output as one rerank and one extract JSON file per query text."""

    results_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    used_names: set[str] = set()

    for result in results:
        analysis_item = result.get("analysis_item") if isinstance(result, dict) else {}
        if not isinstance(analysis_item, dict):
            analysis_item = {}
        query_text = build_query_text(analysis_item) or normalize_text(
            analysis_item.get("analysis_id")
        )
        filename_text = slugify_filename_part(query_text or "query")

        rerank_path = unique_output_path(results_dir, f"rerank_{filename_text}.json", used_names)
        extract_path = unique_output_path(results_dir, f"extract_{filename_text}.json", used_names)
        save_result(build_auto_stage_result(result, "rerank"), rerank_path)
        save_result(build_auto_stage_result(result, "extract"), extract_path)
        saved_paths.extend([rerank_path, extract_path])

    return saved_paths


def save_stage_results(
    results: list[dict[str, Any]],
    results_dir: Path,
    stage: Literal["rerank", "extract"],
) -> list[Path]:
    """Save one stage output JSON file per query text."""

    results_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    used_names: set[str] = set()

    for result in results:
        output_path = save_single_stage_result(result, results_dir, stage, used_names)
        saved_paths.append(output_path)

    return saved_paths


def save_single_stage_result(
    result: dict[str, Any],
    results_dir: Path,
    stage: Literal["rerank", "extract"],
    used_names: set[str],
) -> Path:
    """Save one stage output JSON file for one query text."""

    analysis_item = result.get("analysis_item") if isinstance(result, dict) else {}
    if not isinstance(analysis_item, dict):
        analysis_item = {}
    query_text = build_query_text(analysis_item) or normalize_text(
        analysis_item.get("analysis_id")
    )
    filename_text = slugify_filename_part(query_text or "query")
    output_path = unique_output_path(
        results_dir,
        f"{stage}_{filename_text}.json",
        used_names,
    )
    save_result(result, output_path)
    return output_path


def build_auto_stage_result(
    result: dict[str, Any],
    stage: Literal["rerank", "extract"],
) -> dict[str, Any]:
    connection = result.get("connection") if isinstance(result.get("connection"), dict) else {}
    stage_connection: dict[str, Any] = {
        "analysis_id": connection.get("analysis_id", ""),
        "query_type": connection.get("query_type", ""),
        "stage": stage,
    }
    if stage == "rerank":
        stage_connection["reranked_chunks"] = connection.get("reranked_chunks", [])
    else:
        stage_connection["evidence_units"] = connection.get("evidence_units", [])
        stage_connection["chunk_aggregates"] = connection.get("chunk_aggregates", [])
        if "summary" in connection:
            stage_connection["summary"] = connection.get("summary", {})
    if "validation_errors" in connection:
        stage_connection["validation_errors"] = connection.get("validation_errors", [])

    return {
        "analysis_item": result.get("analysis_item", {}),
        "query_result": result.get("query_result", {}),
        "connection_payload": result.get("connection_payload", {}),
        "connection": stage_connection,
    }


def slugify_filename_part(value: str, max_length: int = 120) -> str:
    slug_chars: list[str] = []
    previous_was_separator = False
    for char in normalize_text(value).lower():
        if char.isalnum():
            slug_chars.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            slug_chars.append("_")
            previous_was_separator = True
    slug = "".join(slug_chars).strip("_")
    if not slug:
        return "query"
    return slug[:max_length].rstrip("_") or "query"


def unique_output_path(results_dir: Path, filename: str, used_names: set[str]) -> Path:
    path = results_dir / filename
    stem = path.stem
    suffix = path.suffix
    index = 2
    while path.name in used_names:
        path = results_dir / f"{stem}_{index}{suffix}"
        index += 1
    used_names.add(path.name)
    return path


def get_query_item_by_number(
    query_number: int,
    analysis_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    items = analysis_items or list(iter_structure_analysis_items())
    if query_number < 1 or query_number > len(items):
        raise IndexError(f"query_number must be between 1 and {len(items)}, got {query_number}.")
    return items[query_number - 1]


def list_query_items() -> list[dict[str, Any]]:
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


def output_path_with_query_number(output_path: Path, query_number: int | None) -> Path:
    if query_number is None:
        return output_path
    return output_path.with_name(f"{output_path.stem}_query_{query_number}{output_path.suffix}")


def build_analysis_id(*parts: str) -> str:
    cleaned = [normalize_text(part).lower().replace(" ", "-") for part in parts if normalize_text(part)]
    return ":".join(cleaned)


def normalize_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def trim_text(text: Any, max_length: int) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def parse_rank_range(value: str) -> tuple[int | None, int | None]:
    text = normalize_text(value)
    separator = ":" if ":" in text else "-"
    if separator not in text:
        raise argparse.ArgumentTypeError(
            "rank range must use START:END, e.g. 61:80."
        )
    start_text, end_text = [part.strip() for part in text.split(separator, 1)]
    try:
        start = int(start_text) if start_text else None
        end = int(end_text) if end_text else None
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rank range bounds must be integers, e.g. 61:80."
        ) from exc
    validate_rank_range(start, end)
    return start, end


def validate_rank_range(start: int | None, end: int | None) -> None:
    if start is not None and start < 1:
        raise argparse.ArgumentTypeError("rank range start must be >= 1.")
    if end is not None and end < 1:
        raise argparse.ArgumentTypeError("rank range end must be >= 1.")
    if start is not None and end is not None and start > end:
        raise argparse.ArgumentTypeError("rank range start must be <= end.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GraphRAG Connection workflow.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to a prebuilt JSON Connection payload. If omitted, run structure_input_motorcontrol retrieval.",
    )
    parser.add_argument(
        "--query-number",
        type=int,
        default=None,
        help="Run one 1-based query number from structure_input_motorcontrol.",
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Print available structure query numbers and exit.",
    )
    parser.add_argument(
        "--extract-rerank-dir",
        type=Path,
        default=None,
        help="Batch-run loose extraction for rerank_*.json files in this directory.",
    )
    parser.add_argument(
        "--extract-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for --extract-rerank-dir results. Defaults to the input directory.",
    )
    parser.add_argument(
        "--rerank-glob",
        default="rerank_*.json",
        help="Glob used with --extract-rerank-dir. Defaults to rerank_*.json.",
    )
    parser.add_argument(
        "--stage",
        choices=["rerank", "extract", "auto"],
        default="auto",
        help="Workflow stage to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Override Connection workflow batch size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the JSON output file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for one JSON output file per structure query.",
    )
    parser.add_argument(
        "--placeholder-llm",
        action="store_true",
        help="Use local placeholder agents instead of calling an LLM.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=60,
        help="Number of retrieval results to fetch before Connection rerank. Defaults to 60.",
    )
    parser.add_argument(
        "--rerank-input-range",
        type=parse_rank_range,
        default=None,
        metavar="START:END",
        help=(
            "Optional 1-based inclusive retrieval-rank range to send to Connection "
            "rerank/extract, e.g. 61:80 for the last 20 of top-80."
        ),
    )
    args = parser.parse_args()
    if args.retrieval_top_k < 1:
        parser.error("--retrieval-top-k must be >= 1.")
    if args.rerank_input_range is not None:
        range_start, range_end = args.rerank_input_range
        if range_start is not None and range_start > args.retrieval_top_k:
            parser.error("--rerank-input-range start cannot exceed --retrieval-top-k.")
        if range_end is not None and range_end > args.retrieval_top_k:
            parser.error("--rerank-input-range end cannot exceed --retrieval-top-k.")
    return args


def main() -> None:
    args = parse_args()
    rerank_input_start, rerank_input_end = (
        args.rerank_input_range if args.rerank_input_range is not None else (None, None)
    )
    if args.list_queries:
        print(json.dumps(list_query_items(), indent=2, ensure_ascii=False))
        return

    if args.extract_rerank_dir is not None:
        saved_paths = run_loose_extraction_for_rerank_dir(
            input_dir=args.extract_rerank_dir,
            output_dir=args.extract_output_dir,
            batch_size=args.batch_size,
            use_placeholder_llm=args.placeholder_llm if args.placeholder_llm else None,
            glob_pattern=args.rerank_glob,
        )
        print(
            "[INFO] Wrote loose extraction result file(s): "
            + ", ".join(str(path) for path in saved_paths)
        )
        return

    output_path = args.output or output_path_with_query_number(
        DEFAULT_OUTPUT_PATH,
        args.query_number,
    )

    if args.input is not None:
        run_connection_payload_file(
            input_path=args.input,
            output_path=output_path,
            stage=args.stage,
            batch_size=args.batch_size,
            use_placeholder_llm=args.placeholder_llm if args.placeholder_llm else None,
        )
        return

    run_connection_pipeline(
        output_path=output_path,
        output_dir=args.output_dir,
        stage=args.stage,
        batch_size=args.batch_size,
        use_placeholder_llm=args.placeholder_llm if args.placeholder_llm else None,
        query_number=args.query_number,
        retrieval_top_k=args.retrieval_top_k,
        rerank_input_start=rerank_input_start,
        rerank_input_end=rerank_input_end,
    )


if __name__ == "__main__":
    main()
