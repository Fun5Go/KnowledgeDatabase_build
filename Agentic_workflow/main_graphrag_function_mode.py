from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT_FILE = (
    REPO_ROOT
    / "GraphRAG"
    / "prepocess"
    / "output"
    / "motor_control_ts_chunk_function_matches.json"
)
DEFAULT_OUTPUT_FILE = (
    REPO_ROOT
    / "GraphRAG"
    / "prepocess"
    / "output"
    / "motor_control_ts_function_mode_inference.json"
)

# Set this in main when you want to group by one exact matched-function set.
# Examples:
# ["Soft Starter", "Relay switching"]
# ["Zero-crossing detection"]
DEFAULT_TARGET_MATCHED_FUNCTIONS: Optional[List[str]] = ["Soft starter"]


MOTOR_CONTROL_STRUCTURE = {
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
                "Hardware": [
                    "(Starting) Motor current too high for chosen components",
                    "Overvoltage due to motor disconnect",
                    "Under Voltage due to incorrect triggering",
                    "Live switching of relays",
                ],
                "Software": [
                    "Priority zero-crossing interrupt too low",
                    "Open loop control",
                ],
            },
        }
    ],
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_function_mode_lookup(structure_input: Dict[str, Any]) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {}

    for node in structure_input.get("nodes", []):
        modes = node.get("modes", {})
        for function_text, mode_list in modes.items():
            normalized_key = (function_text or "").strip().lower()
            lookup[normalized_key] = [
                (mode or "").strip()
                for mode in mode_list
                if isinstance(mode, str) and mode.strip()
            ]

    return lookup


def build_cause_lookup(structure_input: Dict[str, Any]) -> Dict[str, List[str]]:
    lookup: Dict[str, List[str]] = {
        "Hardware": [],
        "Software": [],
    }

    for node in structure_input.get("nodes", []):
        causes = node.get("causes", {})
        for category in ["Hardware", "Software"]:
            for cause_text in causes.get(category, []):
                cleaned = (cause_text or "").strip()
                if cleaned and cleaned not in lookup[category]:
                    lookup[category].append(cleaned)

    return lookup


def normalize_function_list(functions: List[str]) -> Tuple[str, ...]:
    return tuple(sorted({
        (function_text or "").strip().lower()
        for function_text in functions
        if (function_text or "").strip()
    }))


def find_node_record(
    records: List[Dict[str, Any]],
    index: Optional[int] = None,
    chunk_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if chunk_name:
        return [
            record for record in records
            if (record.get("chunk_name") or "").strip() == chunk_name.strip()
        ]

    if index is not None:
        if index < 0 or index >= len(records):
            raise IndexError(f"Index {index} is out of range for {len(records)} records.")
        return [records[index]]

    return records


def filter_records_by_matched_functions(
    records: List[Dict[str, Any]],
    target_functions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not target_functions:
        return records

    target_key = normalize_function_list(target_functions)
    return [
        record for record in records
        if normalize_function_list(record.get("matched_functions", [])) == target_key
    ]


def build_payloads_for_record(
    record: Dict[str, Any],
    function_mode_lookup: Dict[str, List[str]],
    cause_lookup: Dict[str, List[str]],
) -> Dict[str, Any]:
    matched_functions: List[Dict[str, Any]] = []
    seen_functions = set()

    for function_text in record.get("matched_functions", []):
        normalized_function = (function_text or "").strip().lower()
        if not normalized_function or normalized_function in seen_functions:
            continue
        seen_functions.add(normalized_function)

        matched_functions.append({
            "function_text": function_text,
            "Failure Mode": function_mode_lookup.get(normalized_function, []),
        })

    return {
        "chunk_name": record.get("chunk_name", ""),
        "failure_element": record.get("failure_element", ""),
        "discipline": record.get("discipline", ""),
        "section_tag": record.get("section_tag", ""),
        "TechnicalSpecification Choice of Motor control": "Motor control",
        "Choice": record.get("text", ""),
        "Rationale": record.get("rationale_text", ""),
        "cause_candidates": cause_lookup,
        "matched_functions": matched_functions,
    }


def build_group_key(record: Dict[str, Any]) -> Tuple[str, str, Tuple[str, ...]]:
    failure_element = (record.get("failure_element") or "").strip()
    topic = "Motor control"
    normalized_functions = normalize_function_list(record.get("matched_functions", []))
    return failure_element, topic, normalized_functions


def group_records_for_inference(
    records: List[Dict[str, Any]],
    function_mode_lookup: Dict[str, List[str]],
    cause_lookup: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, Any]] = {}

    for record in records:
        payload = build_payloads_for_record(record, function_mode_lookup, cause_lookup)
        group_key = build_group_key(record)

        if group_key not in grouped:
            grouped[group_key] = {
                "failure_element": payload.get("failure_element", ""),
                "discipline": payload.get("discipline", ""),
                "TechnicalSpecification Choice of Motor control": payload.get(
                    "TechnicalSpecification Choice of Motor control", ""
                ),
                "cause_candidates": payload.get("cause_candidates", {}),
                "matched_functions": payload.get("matched_functions", []),
                "sentences": [],
            }

        sentence_id = len(grouped[group_key]["sentences"]) + 1
        grouped[group_key]["sentences"].append({
            "sentence_id": sentence_id,
            "chunk_name": payload.get("chunk_name", ""),
            "section_tag": payload.get("section_tag", ""),
            "Choice": payload.get("Choice", ""),
            "Rationale": payload.get("Rationale", ""),
        })

    return list(grouped.values())


@traceable(
    run_type="chain",
    name="graphrag_build_function_mode_payloads",
    tags=["fmea", "graphrag", "function_mode", "payload"],
)
def build_inference_payloads(
    records: List[Dict[str, Any]],
    structure_input: Dict[str, Any],
) -> List[Dict[str, Any]]:
    function_mode_lookup = build_function_mode_lookup(structure_input)
    cause_lookup = build_cause_lookup(structure_input)
    return group_records_for_inference(records, function_mode_lookup, cause_lookup)


@traceable(
    run_type="chain",
    name="graphrag_run_function_mode_inference",
    tags=["fmea", "graphrag", "function_mode", "llm"],
)
def run_function_mode_inference(
    payloads: List[Dict[str, Any]],
    backend: str,
    model: Optional[str],
) -> List[Dict[str, Any]]:
    from Agentic_workflow.LLMs.graphrag_function_mode_infer_agent import (
        GraphRAGFunctionModeInferenceAgent,
    )

    agent = GraphRAGFunctionModeInferenceAgent(backend=backend, model=model)
    results: List[Dict[str, Any]] = []

    for payload in payloads:
        inferred = agent.infer_function_to_mode(payload)
        inferred["discipline"] = payload.get("discipline", "")
        inferred["TechnicalSpecification Choice of Motor control"] = payload.get(
            "TechnicalSpecification Choice of Motor control", ""
        )
        inferred["sentences"] = payload.get("sentences", [])
        inferred["cause_candidates"] = payload.get("cause_candidates", {})
        inferred["matched_functions"] = payload.get("matched_functions", [])
        results.append(inferred)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer failure modes from GraphRAG technical-spec function matches "
            "using an LLM constrained by function-specific mode candidates."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the GraphRAG function-match JSON file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to write the function -> mode inference JSON output.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Record index to run. Leave empty to avoid index filtering.",
    )
    parser.add_argument(
        "--chunk-name",
        type=str,
        default=None,
        help="Specific chunk_name to run instead of index.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all records in the input file.",
    )
    parser.add_argument(
        "--matched-functions",
        nargs="+",
        default=None,
        help=(
            "Exact matched_functions group to run, for example "
            "--matched-functions 'Soft Starter' 'Relay switching'"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_records = load_json(args.input_file)

    target_matched_functions = (
        args.matched_functions
        if args.matched_functions is not None
        else DEFAULT_TARGET_MATCHED_FUNCTIONS
    )

    if args.all or target_matched_functions:
        selected_records = input_records
    else:
        selected_records = find_node_record(
            records=input_records,
            index=args.index,
            chunk_name=args.chunk_name,
        )

    selected_records = filter_records_by_matched_functions(
        records=selected_records,
        target_functions=target_matched_functions,
    )

    if not selected_records:
        raise ValueError(
            "No records matched the requested matched_functions group: "
            f"{target_matched_functions}"
        )

    payloads = build_inference_payloads(
        records=selected_records,
        structure_input=MOTOR_CONTROL_STRUCTURE,
    )

    results = run_function_mode_inference(
        payloads=payloads,
        backend=os.getenv("LLM_BACKEND", "openai"),
        model=os.getenv("LLM_MODEL", "azure/gpt-4.1"),
    )

    write_json(results, args.output_file)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
