from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ONEPROCESS_DIR = Path(__file__).resolve().parent / "select_2"
DEFAULT_GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parents[1]
    / "Connection"
    / "evaluation"
    / "final_extract_review_evaluation.json"
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "oneprocess_evaluation_results.json"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def normalize_key(value: Any) -> str:
    return " ".join(normalize_text(value).casefold().split())


def name_variants(value: Any) -> set[str]:
    """Return comparable forms for chunk names with optional appended reasons."""

    text = normalize_text(value)
    if not text:
        return set()

    variants = {normalize_key(text)}
    for marker in ("\nReason:", "\r\nReason:", " Reason:"):
        if marker in text:
            variants.add(normalize_key(text.split(marker, 1)[0]))
    return {variant for variant in variants if variant}


def names_match(left: str, right: str) -> bool:
    return bool(name_variants(left) & name_variants(right))


def dedupe_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        variants = name_variants(name)
        key = sorted(variants)[0] if variants else normalize_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(name)
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_oneprocess_result(path: Path) -> dict[str, Any]:
    loaded = load_json(path)
    if isinstance(loaded, list):
        if not loaded:
            raise ValueError(f"Oneprocess file is empty: {path}")
        first = loaded[0]
        if isinstance(first, dict):
            return first
    if isinstance(loaded, dict):
        return loaded
    raise TypeError(f"Unsupported oneprocess JSON shape: {path}")


def query_text_from_result(result: dict[str, Any]) -> str:
    analysis_item = result.get("analysis_item")
    if isinstance(analysis_item, dict):
        query_text = normalize_text(analysis_item.get("query_text"))
        if query_text:
            return query_text
    query_result = result.get("query_result")
    if isinstance(query_result, dict):
        query_spec = query_result.get("query_spec")
        if isinstance(query_spec, dict):
            for key in ("query_text", "target_text", "sentence"):
                query_text = normalize_text(query_spec.get(key))
                if query_text:
                    return query_text
    return ""


def candidate_names_from_result(result: dict[str, Any]) -> list[str]:
    query_result = result.get("query_result")
    evidence = query_result.get("evidence") if isinstance(query_result, dict) else []
    if not isinstance(evidence, list):
        return []
    return dedupe_names(
        [
            normalize_text(item.get("name"))
            for item in evidence
            if isinstance(item, dict) and normalize_text(item.get("name"))
        ]
    )


def selected_names_from_result(result: dict[str, Any]) -> list[str]:
    selection = result.get("selection")
    chunks = selection.get("top_chunks") if isinstance(selection, dict) else []
    if not isinstance(chunks, list):
        return []
    return dedupe_names(
        [
            normalize_text(item.get("name"))
            for item in chunks
            if isinstance(item, dict) and normalize_text(item.get("name"))
        ]
    )


def load_ground_truth_by_query(path: Path) -> dict[str, dict[str, Any]]:
    loaded = load_json(path)
    per_query = loaded.get("per_query") if isinstance(loaded, dict) else loaded
    if not isinstance(per_query, list):
        raise TypeError("Ground truth JSON must contain a per_query list.")

    ground_truth_by_query: dict[str, dict[str, Any]] = {}
    for item in per_query:
        if not isinstance(item, dict):
            continue
        query_text = normalize_text(item.get("query_text"))
        if not query_text:
            continue
        ground_truth = item.get("ground_truth") if isinstance(item.get("ground_truth"), dict) else {}
        actual_value = ground_truth.get("actual_value") if isinstance(ground_truth, dict) else {}
        actual_irrelevant = (
            ground_truth.get("actual_irrelevant") if isinstance(ground_truth, dict) else {}
        )
        value_names = item.get("actual_value_names")
        irrelevant_names = item.get("actual_irrelevant_names")
        if not isinstance(value_names, list) and isinstance(actual_value, dict):
            value_names = actual_value.get("names")
        if not isinstance(irrelevant_names, list) and isinstance(actual_irrelevant, dict):
            irrelevant_names = actual_irrelevant.get("names")
        ground_truth_by_query[normalize_key(query_text)] = {
            "query_text": query_text,
            "actual_value_names": dedupe_names(
                [normalize_text(name) for name in value_names or [] if normalize_text(name)]
            ),
            "actual_irrelevant_names": dedupe_names(
                [normalize_text(name) for name in irrelevant_names or [] if normalize_text(name)]
            ),
        }
    return ground_truth_by_query


def split_by_actual_value(
    names: list[str],
    actual_value_names: list[str],
) -> tuple[list[str], list[str]]:
    value_matches: list[str] = []
    irrelevant_matches: list[str] = []
    for name in names:
        if any(names_match(name, truth_name) for truth_name in actual_value_names):
            value_matches.append(name)
        else:
            irrelevant_matches.append(name)
    return value_matches, irrelevant_matches


def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_oneprocess_file(
    path: Path,
    ground_truth_by_query: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = load_oneprocess_result(path)
    query_text = query_text_from_result(result)
    ground_truth = ground_truth_by_query.get(normalize_key(query_text))
    if ground_truth is None:
        return {
            "query_text": query_text,
            "source_file": path.name,
            "matched_ground_truth": False,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            **calculate_metrics(0, 0, 0, 0),
            "error": "No ground truth entry matched this query_text.",
        }

    candidate_names = candidate_names_from_result(result)
    selected_names = selected_names_from_result(result)
    selected_keys = {variant for name in selected_names for variant in name_variants(name)}
    unselected_names = [
        name
        for name in candidate_names
        if not (name_variants(name) & selected_keys)
    ]

    actual_value_names = ground_truth["actual_value_names"]
    # The review file can under-list actual_irrelevant chunks. Retrieval candidates are
    # identical for this evaluation, so every retrieved candidate that is not in
    # actual_value is treated as actual_irrelevant.
    actual_irrelevant_names = [
        name
        for name in candidate_names
        if not any(names_match(name, value_name) for value_name in actual_value_names)
    ]
    tp_names, fp_names = split_by_actual_value(
        selected_names,
        actual_value_names,
    )
    fn_names, tn_names = split_by_actual_value(
        unselected_names,
        actual_value_names,
    )
    tp = len(tp_names)
    tn = len(tn_names)
    fp = len(fp_names)
    fn = len(fn_names)

    return {
        "query_text": ground_truth["query_text"],
        "source_file": path.name,
        "matched_ground_truth": True,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        **calculate_metrics(tp, tn, fp, fn),
        "counts": {
            "candidate_chunks": len(candidate_names),
            "selected_chunks": len(selected_names),
            "unselected_chunks": len(unselected_names),
            "actual_value": len(actual_value_names),
            "actual_irrelevant": len(actual_irrelevant_names),
            "evaluated": tp + tn + fp + fn,
        },
        "matches": {
            "tp_names": tp_names,
            "tn_names": tn_names,
            "fp_names": fp_names,
            "fn_names": fn_names,
        },
        "ground_truth": {
            "actual_value": {
                "count": len(actual_value_names),
                "names": actual_value_names,
            },
            "actual_irrelevant": {
                "count": len(actual_irrelevant_names),
                "names": actual_irrelevant_names,
                "source": "retrieved candidate names minus actual_value",
            },
        },
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_summary(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [item for item in per_query if item.get("matched_ground_truth")]
    total_tp = sum(int(item.get("tp", 0)) for item in matched)
    total_tn = sum(int(item.get("tn", 0)) for item in matched)
    total_fp = sum(int(item.get("fp", 0)) for item in matched)
    total_fn = sum(int(item.get("fn", 0)) for item in matched)
    return {
        "num_files": len(per_query),
        "matched_queries": len(matched),
        "unmatched_queries": len(per_query) - len(matched),
        "average_per_query": {
            "precision": average([float(item.get("precision", 0.0)) for item in matched]),
            "recall": average([float(item.get("recall", 0.0)) for item in matched]),
            "f1": average([float(item.get("f1", 0.0)) for item in matched]),
            "accuracy": average([float(item.get("accuracy", 0.0)) for item in matched]),
        },
        "total_micro": {
            "tp": total_tp,
            "tn": total_tn,
            "fp": total_fp,
            "fn": total_fn,
            **calculate_metrics(total_tp, total_tn, total_fp, total_fn),
        },
    }


def evaluate(
    oneprocess_dir: Path = DEFAULT_ONEPROCESS_DIR,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    glob_pattern: str = "oneprocess*.json",
) -> dict[str, Any]:
    if not oneprocess_dir.exists():
        raise FileNotFoundError(f"Oneprocess directory not found: {oneprocess_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    ground_truth_by_query = load_ground_truth_by_query(ground_truth_path)
    input_paths = sorted(oneprocess_dir.glob(glob_pattern))
    if not input_paths:
        raise FileNotFoundError(f"No files matched {glob_pattern!r} in {oneprocess_dir}")

    per_query = [
        evaluate_oneprocess_file(path, ground_truth_by_query)
        for path in input_paths
    ]
    output = {
        "oneprocess_dir": str(oneprocess_dir),
        "ground_truth_path": str(ground_truth_path),
        "glob_pattern": glob_pattern,
        "summary": build_summary(per_query),
        "per_query": per_query,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate oneprocess chunk-selection JSON files against review ground truth."
    )
    parser.add_argument(
        "--oneprocess-dir",
        type=Path,
        default=DEFAULT_ONEPROCESS_DIR,
        help="Directory containing oneprocess*.json files.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help="Path to final_extract_review_evaluation.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--glob",
        default="oneprocess*.json",
        help="Glob pattern for oneprocess files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = evaluate(
        oneprocess_dir=args.oneprocess_dir,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        glob_pattern=args.glob,
    )
    summary = output["summary"]
    print(f"[INFO] Evaluated {summary['matched_queries']}/{summary['num_files']} matched queries.")
    print(f"[INFO] Wrote evaluation JSON to: {args.output}")


if __name__ == "__main__":
    main()
