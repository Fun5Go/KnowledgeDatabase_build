from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CAUSE_FILE = REPO_ROOT / "GraphRAG" / "prepocess" / "output" / "motor_control_ts_cause_matches.json"
DEFAULT_MODE_FILE = REPO_ROOT / "GraphRAG" / "prepocess" / "output" / "motor_control_ts_mode_matches.json"
DEFAULT_GROUPED_OUTPUT = REPO_ROOT / "GraphRAG" / "prepocess" / "output" / "motor_control_cause_grouped_sentences.json"
DEFAULT_INFER_OUTPUT = REPO_ROOT / "GraphRAG" / "prepocess" / "output" / "motor_control_cause_mode_inference.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def unique_preserve_order(items: Iterable[Any]) -> List[Any]:
    ordered: List[Any] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def build_mode_lookup(mode_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    lookup: Dict[str, List[Dict[str, Any]]] = {}

    for record in mode_records:
        node_name = record.get("node_name", "")
        node_modes: List[Dict[str, Any]] = []

        for match in record.get("matches", []):
            function_text = match.get("function", "")
            for mode_text in match.get("candidate_modes", []):
                node_modes.append({
                    "function_text": function_text,
                    "mode_text": mode_text,
                })

        lookup[node_name] = unique_preserve_order(
            (item["function_text"], item["mode_text"]) for item in node_modes
        )

    normalized_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for node_name, pairs in lookup.items():
        normalized_lookup[node_name] = [
            {
                "function_text": function_text,
                "mode_text": mode_text,
            }
            for function_text, mode_text in pairs
        ]

    return normalized_lookup


def build_sentence_group(sentences: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for sentence in sentences:
        parts: List[str] = []
        node_text = (sentence.get("node_text") or "").strip()
        if node_text:
            parts.append(f"Node text: {node_text}")

        rationale_texts = [
            (item.get("text") or "").strip()
            for item in sentence.get("rationales", [])
            if (item.get("text") or "").strip()
        ]
        if rationale_texts:
            parts.append(f"Rationale: {' '.join(rationale_texts)}")

        if parts:
            blocks.append("\n".join(parts))

    return "\n\n".join(blocks)


def group_sentences_by_cause(
    cause_records: List[Dict[str, Any]],
    mode_lookup: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for record in cause_records:
        node_name = record.get("node_name", "")
        shared_mode_candidates = mode_lookup.get(node_name, [])

        for match in record.get("matches", []):
            discipline = match.get("discipline", "")
            for matched_cause in match.get("matched_causes", []):
                cause_text = matched_cause.get("cause", "")
                group_key = (discipline, cause_text)

                if group_key not in grouped:
                    grouped[group_key] = {
                        "cause": cause_text,
                        "discipline": discipline,
                        "failure_element": match.get("failure_element", ""),
                        "sentences": [],
                    }

                sentence_id = len(grouped[group_key]["sentences"]) + 1
                grouped[group_key]["sentences"].append({
                    "sentence_id": sentence_id,
                    "node_text": record.get("node_text", ""),
                    "rationales": record.get("rationales", []),
                    "cause_match_coverage": matched_cause.get("cause_match_coverage", 0.0),
                })
                group_modes = grouped[group_key].setdefault("_mode_pairs", [])
                group_modes.extend(
                    (
                        candidate.get("function_text", ""),
                        candidate.get("mode_text", ""),
                    )
                    for candidate in shared_mode_candidates
                )

    final_rows: List[Dict[str, Any]] = []
    for item in sorted(
        grouped.values(),
        key=lambda entry: (entry.get("discipline", ""), entry.get("cause", "")),
    ):
        top_sentences = sorted(
            item.get("sentences", []),
            key=lambda sentence: (
                -float(sentence.get("cause_match_coverage", 0.0)),
                sentence.get("sentence_id", 0),
            ),
        )[:2]
        for new_index, sentence in enumerate(top_sentences, start=1):
            sentence["sentence_id"] = new_index

        unique_mode_pairs = unique_preserve_order(item.pop("_mode_pairs", []))
        grouped_modes: Dict[str, List[str]] = {}
        for function_text, mode_text in unique_mode_pairs:
            if not function_text or not mode_text:
                continue
            grouped_modes.setdefault(function_text, [])
            if mode_text not in grouped_modes[function_text]:
                grouped_modes[function_text].append(mode_text)

        compact_sentences = [
            {
                "sentence_id": sentence["sentence_id"],
                "node_text": sentence.get("node_text", ""),
                "rationales": sentence.get("rationales", []),
            }
            for sentence in top_sentences
        ]

        item["sentences"] = compact_sentences
        item["evidence"] = {
            "sentence_group": build_sentence_group(top_sentences),
            "function_mode_candidates": [
                {
                    "function_text": function_text,
                    "mode_candidates": mode_texts,
                }
                for function_text, mode_texts in grouped_modes.items()
            ],
        }
        final_rows.append(item)

    return final_rows


@traceable(
    run_type="chain",
    name="graphrag_group_sentences_by_cause",
    tags=["fmea", "graphrag", "grouping", "cause_mode"],
)
def build_grouped_cause_payloads(
    cause_file: Path,
    mode_file: Path,
) -> List[Dict[str, Any]]:
    cause_records = load_json(cause_file)
    mode_records = load_json(mode_file)
    mode_lookup = build_mode_lookup(mode_records)
    return group_sentences_by_cause(cause_records, mode_lookup)


@traceable(
    run_type="chain",
    name="graphrag_run_cause_mode_inference",
    tags=["fmea", "graphrag", "cause_mode", "llm"],
)
def run_cause_mode_inference(
    grouped_payloads: List[Dict[str, Any]],
    backend: str,
    model: str | None,
) -> List[Dict[str, Any]]:
    from Agentic_workflow.LLMs.graphrag_cause_mode_infer_agent import (
        GraphRAGCauseModeInferenceAgent,
    )

    agent = GraphRAGCauseModeInferenceAgent(backend=backend, model=model)
    results: List[Dict[str, Any]] = []

    for payload in grouped_payloads:
        results.append(agent.infer_cause_to_mode(payload))

    return results


def build_final_output(
    grouped_payloads: List[Dict[str, Any]],
    inference_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result_lookup = {
        (item.get("discipline", ""), item.get("cause", "")): item
        for item in inference_results
    }

    final_rows: List[Dict[str, Any]] = []
    for payload in grouped_payloads:
        key = (payload.get("discipline", ""), payload.get("cause", ""))
        inference = result_lookup.get(key, {})
        final_rows.append({
            "failure_element": payload.get("failure_element", ""),
            "discipline": payload.get("discipline", ""),
            "cause_text": payload.get("cause", ""),
            "sentences": payload.get("sentences", []),
            "evidence": payload.get("evidence", {}),
            "inferred_modes": inference.get("selected_modes", []),
            "global_reasoning": inference.get("global_reasoning", ""),
            "effect_inference_status": inference.get("effect_inference_status", "pending"),
        })
    return final_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group GraphRAG sentence evidence by cause and infer cause -> mode with an LLM agent.",
    )
    parser.add_argument(
        "--cause-file",
        type=Path,
        default=DEFAULT_CAUSE_FILE,
        help="Cause matching JSON from GraphRAG preprocess output.",
    )
    parser.add_argument(
        "--mode-file",
        type=Path,
        default=DEFAULT_MODE_FILE,
        help="Mode matching JSON from GraphRAG preprocess output.",
    )
    parser.add_argument(
        "--grouped-output",
        type=Path,
        default=DEFAULT_GROUPED_OUTPUT,
        help="Where to write grouped cause -> sentences -> mode candidates JSON.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_INFER_OUTPUT,
        help="Where to write final cause -> mode inference JSON.",
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("LLM_BACKEND", "openai"),
        help="LLM backend name, for example openai or local.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL"),
        help="Optional LLM model override.",
    )
    parser.add_argument(
        "--pretty-print",
        action="store_true",
        help="Print compact inference results to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grouped_payloads = build_grouped_cause_payloads(
        cause_file=args.cause_file.resolve(),
        mode_file=args.mode_file.resolve(),
    )
    write_json(grouped_payloads, args.grouped_output.resolve())

    inference_results = run_cause_mode_inference(
        grouped_payloads=grouped_payloads,
        backend=args.backend,
        model=args.model,
    )
    final_output = build_final_output(grouped_payloads, inference_results)
    write_json(final_output, args.output_file.resolve())

    if args.pretty_print:
        from Agentic_workflow.LLMs.graphrag_cause_mode_infer_agent import (
            pretty_print_cause_mode_results,
        )

        pretty_print_cause_mode_results(inference_results)

    print(f"Grouped {len(grouped_payloads)} causes into {args.grouped_output.resolve()}")
    print(f"Wrote {len(final_output)} cause -> mode inference rows to {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
