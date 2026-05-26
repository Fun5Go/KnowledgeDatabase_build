"""Evaluate one-step chunk selection against integrated human-review labels.

Ground truth comes from Connection/evaluation/human_review_by_query_top60_integrated:
- selected_correct == true is an actual positive sample.
- selected_correct == null is an actual negative sample.

Only queries with at least one selected_correct == true are evaluated. For each
reviewed chunk in those queries, the one-step LLM prediction is positive if the
chunk appears in ``chunk_selection_results.json`` selection.top_chunks;
otherwise it is negative.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .evaluate_oneprocess_results import dedupe_names, names_match, normalize_key, normalize_text
except ImportError:  # pragma: no cover - supports direct script execution
    from evaluate_oneprocess_results import dedupe_names, names_match, normalize_key, normalize_text


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ONE_STEP_INPUT = SCRIPT_DIR / "chunk_selection_results.json"
DEFAULT_REVIEW_DIR = (
    SCRIPT_DIR.parent
    / "Connection"
    / "evaluation"
    / "human_review_by_query_top60_integrated"
)
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "one_step_evaluation_results.json"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "one_step_evaluation_results.csv"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        cleaned = value.strip().casefold()
        if cleaned in {"true", "t", "yes", "y", "1"}:
            return True
        if cleaned in {"false", "f", "no", "n", "0"}:
            return False
    return None


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def chunk_name(chunk: dict[str, Any], fallback_index: int) -> str:
    value = first_present(chunk, ["name", "chunk_id", "node_id"])
    return normalize_text(value) or f"chunk_{fallback_index}"


def query_text_from_one_step_result(result: dict[str, Any]) -> str:
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


def selected_names_from_one_step_result(result: dict[str, Any]) -> list[str]:
    selection = result.get("selection")
    chunks = selection.get("top_chunks") if isinstance(selection, dict) else []
    if not isinstance(chunks, list):
        return []
    return dedupe_names(
        [
            chunk_name(chunk, index)
            for index, chunk in enumerate(chunks, start=1)
            if isinstance(chunk, dict)
        ]
    )


def load_one_step_results(path: Path) -> dict[str, dict[str, Any]]:
    loaded = load_json(path)
    if isinstance(loaded, dict):
        items = [loaded]
    elif isinstance(loaded, list):
        items = [item for item in loaded if isinstance(item, dict)]
    else:
        raise TypeError(f"Unsupported one-step JSON shape: {path}")

    by_query: dict[str, dict[str, Any]] = {}
    for item in items:
        query_text = query_text_from_one_step_result(item)
        if not query_text:
            continue
        by_query[normalize_key(query_text)] = {
            "query_text": query_text,
            "selected_names": selected_names_from_one_step_result(item),
        }
    return by_query


def review_chunks(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    chunks = payload.get("chunks")
    if isinstance(chunks, list):
        return [chunk for chunk in chunks if isinstance(chunk, dict)]
    return []


def load_review_ground_truth(review_dir: Path) -> dict[str, dict[str, Any]]:
    if not review_dir.exists():
        raise FileNotFoundError(f"Review directory not found: {review_dir}")

    ground_truth: dict[str, dict[str, Any]] = {}
    for path in sorted(review_dir.glob("*.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue

        query_text = normalize_text(payload.get("query_text")) or path.stem
        chunks = review_chunks(payload)
        positive_names: list[str] = []
        negative_names: list[str] = []
        skipped_names: list[str] = []

        for index, chunk in enumerate(chunks, start=1):
            label = normalize_bool(chunk.get("selected_correct"))
            name = chunk_name(chunk, index)
            if label is True:
                positive_names.append(name)
            elif label is None:
                negative_names.append(name)
            else:
                skipped_names.append(name)

        positive_names = dedupe_names(positive_names)
        negative_names = dedupe_names(negative_names)
        if not positive_names:
            continue

        ground_truth[normalize_key(query_text)] = {
            "query_text": query_text,
            "source_file": path.name,
            "positive_names": positive_names,
            "negative_names": negative_names,
            "skipped_names": dedupe_names(skipped_names),
        }
    return ground_truth


def contains_name(names: list[str], target: str) -> bool:
    return any(names_match(name, target) for name in names)


def lookup_one_step_result(
    query_text: str,
    one_step_by_query: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    query_key = normalize_key(query_text)
    if query_key in one_step_by_query:
        return one_step_by_query[query_key]

    matches = [
        item
        for key, item in one_step_by_query.items()
        if query_key and (query_key in key or key.endswith(query_key))
    ]
    return matches[0] if len(matches) == 1 else None


def selected_not_in_ground_truth(
    selected_names: list[str],
    positive_names: list[str],
    negative_names: list[str],
) -> list[str]:
    result: list[str] = []
    for selected_name in selected_names:
        if contains_name(positive_names, selected_name) or contains_name(negative_names, selected_name):
            continue
        result.append(selected_name)
    return dedupe_names(result)


def evaluate_query(
    ground_truth: dict[str, Any],
    one_step: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_names = one_step.get("selected_names", []) if isinstance(one_step, dict) else []
    positive_names = ground_truth["positive_names"]
    negative_names = ground_truth["negative_names"]

    tp_names = [name for name in positive_names if contains_name(selected_names, name)]
    fn_names = [name for name in positive_names if not contains_name(selected_names, name)]
    fp_names = [name for name in negative_names if contains_name(selected_names, name)]
    tn_names = [name for name in negative_names if not contains_name(selected_names, name)]
    unmatched_selected_names = selected_not_in_ground_truth(
        selected_names,
        positive_names,
        negative_names,
    )

    tp = len(tp_names)
    tn = len(tn_names)
    fp = len(fp_names)
    fn = len(fn_names)

    return {
        "query_text": ground_truth["query_text"],
        "matched_one_step_query_text": one_step.get("query_text", "") if isinstance(one_step, dict) else "",
        "review_source_file": ground_truth["source_file"],
        "matched_one_step_result": one_step is not None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        **calculate_metrics(tp, tn, fp, fn),
        "counts": {
            "ground_truth_positive": len(positive_names),
            "ground_truth_negative": len(negative_names),
            "ground_truth_skipped": len(ground_truth.get("skipped_names", [])),
            "llm_selected": len(selected_names),
            "llm_selected_not_in_review_ground_truth": len(unmatched_selected_names),
            "evaluated": tp + tn + fp + fn,
        },
        "matches": {
            "tp_names": tp_names,
            "tn_names": tn_names,
            "fp_names": fp_names,
            "fn_names": fn_names,
            "llm_selected_not_in_review_ground_truth": unmatched_selected_names,
        },
    }


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_summary(per_query: list[dict[str, Any]], one_step_by_query: dict[str, dict[str, Any]]) -> dict[str, Any]:
    matched = [item for item in per_query if item.get("matched_one_step_result")]
    total_tp = sum(int(item["tp"]) for item in per_query)
    total_tn = sum(int(item["tn"]) for item in per_query)
    total_fp = sum(int(item["fp"]) for item in per_query)
    total_fn = sum(int(item["fn"]) for item in per_query)
    evaluated_query_keys = {normalize_key(item["query_text"]) for item in per_query}
    evaluated_query_keys.update(
        normalize_key(item.get("matched_one_step_query_text"))
        for item in per_query
        if normalize_text(item.get("matched_one_step_query_text"))
    )
    unmatched_one_step_queries = [
        item["query_text"]
        for key, item in one_step_by_query.items()
        if key not in evaluated_query_keys
    ]

    return {
        "evaluated_queries": len(per_query),
        "matched_one_step_queries": len(matched),
        "missing_one_step_queries": len(per_query) - len(matched),
        "one_step_queries_not_evaluated": len(unmatched_one_step_queries),
        "one_step_queries_not_evaluated_names": unmatched_one_step_queries,
        "macro": {
            "precision": average([float(item["precision"]) for item in per_query]),
            "recall": average([float(item["recall"]) for item in per_query]),
            "f1": average([float(item["f1"]) for item in per_query]),
            "accuracy": average([float(item["accuracy"]) for item in per_query]),
        },
        "micro": {
            "tp": total_tp,
            "tn": total_tn,
            "fp": total_fp,
            "fn": total_fn,
            **calculate_metrics(total_tp, total_tn, total_fp, total_fn),
        },
    }


def evaluate(
    one_step_input: Path,
    review_dir: Path,
) -> dict[str, Any]:
    if not one_step_input.exists():
        raise FileNotFoundError(f"One-step input not found: {one_step_input}")

    one_step_by_query = load_one_step_results(one_step_input)
    ground_truth_by_query = load_review_ground_truth(review_dir)
    per_query = [
        evaluate_query(ground_truth, lookup_one_step_result(ground_truth["query_text"], one_step_by_query))
        for query_key, ground_truth in sorted(
            ground_truth_by_query.items(),
            key=lambda item: normalize_key(item[1]["query_text"]),
        )
    ]

    return {
        "one_step_input": str(one_step_input),
        "review_dir": str(review_dir),
        "ground_truth_definition": {
            "positive": "human_review chunk.selected_correct == true",
            "negative": "human_review chunk.selected_correct is null",
            "evaluated_queries": "only failure attributes with at least one selected_correct == true",
        },
        "prediction_definition": {
            "llm_positive": "chunk name appears in chunk_selection_results selection.top_chunks",
            "llm_negative": "reviewed ground-truth chunk name does not appear in selection.top_chunks",
        },
        "summary": build_summary(per_query, one_step_by_query),
        "per_query": per_query,
    }


def write_csv(path: Path, per_query: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_text",
        "matched_one_step_query_text",
        "review_source_file",
        "matched_one_step_result",
        "tp",
        "tn",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "ground_truth_positive",
        "ground_truth_negative",
        "llm_selected",
        "llm_selected_not_in_review_ground_truth",
        "evaluated",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_query:
            counts = item["counts"]
            writer.writerow(
                {
                    "query_text": item["query_text"],
                    "matched_one_step_query_text": item["matched_one_step_query_text"],
                    "review_source_file": item["review_source_file"],
                    "matched_one_step_result": item["matched_one_step_result"],
                    "tp": item["tp"],
                    "tn": item["tn"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "precision": round(float(item["precision"]), 4),
                    "recall": round(float(item["recall"]), 4),
                    "f1": round(float(item["f1"]), 4),
                    "accuracy": round(float(item["accuracy"]), 4),
                    "ground_truth_positive": counts["ground_truth_positive"],
                    "ground_truth_negative": counts["ground_truth_negative"],
                    "llm_selected": counts["llm_selected"],
                    "llm_selected_not_in_review_ground_truth": counts[
                        "llm_selected_not_in_review_ground_truth"
                    ],
                    "evaluated": counts["evaluated"],
                }
            )


def print_summary(results: dict[str, Any]) -> None:
    summary = results["summary"]
    micro = summary["micro"]
    macro = summary["macro"]
    print(f"[INFO] Evaluated queries: {summary['evaluated_queries']}")
    print(f"[INFO] Matched one-step queries: {summary['matched_one_step_queries']}")
    print(f"[INFO] Missing one-step queries: {summary['missing_one_step_queries']}")
    print(
        "[INFO] Micro: "
        f"TP={micro['tp']} TN={micro['tn']} FP={micro['fp']} FN={micro['fn']} "
        f"P={micro['precision']:.4f} R={micro['recall']:.4f} "
        f"F1={micro['f1']:.4f} Acc={micro['accuracy']:.4f}"
    )
    print(
        "[INFO] Macro: "
        f"P={macro['precision']:.4f} R={macro['recall']:.4f} "
        f"F1={macro['f1']:.4f} Acc={macro['accuracy']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one-step LLM chunk selection against integrated human-review labels."
    )
    parser.add_argument("--one-step-input", type=Path, default=DEFAULT_ONE_STEP_INPUT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args.one_step_input, args.review_dir)
    write_json(args.json_output, results)
    write_csv(args.csv_output, results["per_query"])
    print_summary(results)
    print(f"\n[INFO] Wrote JSON: {args.json_output}")
    print(f"[INFO] Wrote CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
