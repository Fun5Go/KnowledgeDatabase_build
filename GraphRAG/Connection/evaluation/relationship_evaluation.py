"""Evaluate relationship labels on true-positive selected chunks.

The script reads the integrated human-review Top-60 JSON files and evaluates
only chunks where ``selected == true`` and ``selected_correct == true``.

For those true-positive chunks, each item in ``chunk_aggregate.relations`` is
counted as one relationship check:
- correct: relationship_correct == true, multiplied by relation count
- incorrect: relationship_correct is anything else, multiplied by relation count
- accuracy: correct relationship checks / total relationship checks

If the human comment contains a partial label such as ``1/3 false``, the chunk
is expanded into 3 relationship checks: 1 incorrect and 2 correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REVIEW_DIR = SCRIPT_DIR / "human_review_by_query_top60_integrated"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "relationship_evaluation_top60.json"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "relationship_evaluation_top60.csv"


@dataclass
class QueryMetrics:
    query_text: str
    source_file: str
    tp_chunks: int = 0
    relationship_checks: int = 0
    relationship_correct: int = 0
    relationship_incorrect: int = 0
    skipped_non_tp: int = 0
    accuracy: float = 0.0
    correct_names: list[str] = field(default_factory=list)
    incorrect_names: list[str] = field(default_factory=list)
    partial_comment_names: list[str] = field(default_factory=list)


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


def first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def query_text(payload: dict[str, Any], path: Path) -> str:
    query = payload.get("query")
    if isinstance(query, dict):
        value = first_present(query, ["query_text", "question", "text", "content"])
        if value not in (None, ""):
            return str(value).strip()

    value = first_present(payload, ["query_text", "question", "text", "content"])
    if value not in (None, ""):
        return str(value).strip()
    return path.stem


def reviewed_chunks(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    chunks = payload.get("chunks")
    if isinstance(chunks, list):
        return [chunk for chunk in chunks if isinstance(chunk, dict)]
    return []


def chunk_selected(chunk: dict[str, Any]) -> bool | None:
    aggregate = chunk.get("chunk_aggregate")
    if isinstance(aggregate, dict):
        return normalize_bool(aggregate.get("selected"))
    return normalize_bool(chunk.get("selected"))


def chunk_name(chunk: dict[str, Any], fallback_index: int) -> str:
    value = first_present(chunk, ["name", "chunk_id", "node_id"])
    return str(value or f"chunk_{fallback_index}").strip()


def is_true_positive_chunk(chunk: dict[str, Any]) -> bool:
    return chunk_selected(chunk) is True and normalize_bool(chunk.get("selected_correct")) is True


def chunk_relation_count(chunk: dict[str, Any]) -> int:
    aggregate = chunk.get("chunk_aggregate")
    if not isinstance(aggregate, dict):
        return 0
    relations = aggregate.get("relations")
    if not isinstance(relations, list):
        return 0
    return len([relation for relation in relations if relation not in (None, "")])


def partial_false_counts(comment: Any) -> tuple[int, int] | None:
    if not isinstance(comment, str):
        return None
    match = re.search(r"\b(\d+)\s*/\s*(\d+)\s*false\b", comment, flags=re.IGNORECASE)
    if not match:
        return None

    false_count = int(match.group(1))
    total_count = int(match.group(2))
    if total_count <= 0 or false_count < 0 or false_count > total_count:
        return None
    return total_count - false_count, false_count


def finalize_metrics(metrics: QueryMetrics) -> QueryMetrics:
    metrics.relationship_checks = metrics.relationship_correct + metrics.relationship_incorrect
    metrics.accuracy = safe_divide(metrics.relationship_correct, metrics.relationship_checks)
    return metrics


def evaluate_file(path: Path) -> QueryMetrics | None:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None

    chunks = reviewed_chunks(payload)
    if not any(normalize_bool(chunk.get("selected_correct")) is True for chunk in chunks):
        return None

    metrics = QueryMetrics(query_text=query_text(payload, path), source_file=path.name)
    for index, chunk in enumerate(chunks, start=1):
        if not is_true_positive_chunk(chunk):
            metrics.skipped_non_tp += 1
            continue

        name = chunk_name(chunk, index)
        metrics.tp_chunks += 1
        relation_count = chunk_relation_count(chunk)

        partial_counts = partial_false_counts(chunk.get("comments"))
        if partial_counts is not None:
            correct_count, incorrect_count = partial_counts
            metrics.relationship_correct += correct_count
            metrics.relationship_incorrect += incorrect_count
            metrics.partial_comment_names.append(name)
            if correct_count:
                metrics.correct_names.append(f"{name} (+{correct_count})")
            if incorrect_count:
                metrics.incorrect_names.append(f"{name} (+{incorrect_count})")
        elif normalize_bool(chunk.get("relationship_correct")) is True:
            metrics.relationship_correct += relation_count
            metrics.correct_names.append(f"{name} (+{relation_count})")
        else:
            metrics.relationship_incorrect += relation_count
            metrics.incorrect_names.append(f"{name} (+{relation_count})")

    metrics = finalize_metrics(metrics)
    return metrics


def average_query_scores(results: list[QueryMetrics]) -> dict[str, Any]:
    if not results:
        return {"accuracy": 0.0, "num_queries": 0}
    return {
        "accuracy": sum(result.accuracy for result in results) / len(results),
        "num_queries": len(results),
    }


def total_scores(results: list[QueryMetrics]) -> dict[str, Any]:
    totals = {
        "tp_chunks": sum(result.tp_chunks for result in results),
        "relationship_checks": sum(result.relationship_checks for result in results),
        "relationship_correct": sum(result.relationship_correct for result in results),
        "relationship_incorrect": sum(result.relationship_incorrect for result in results),
        "skipped_non_tp": sum(result.skipped_non_tp for result in results),
    }
    totals["accuracy"] = safe_divide(totals["relationship_correct"], totals["relationship_checks"])
    return totals


def evaluate_review_dir(review_dir: Path) -> dict[str, Any]:
    if not review_dir.exists():
        raise FileNotFoundError(f"Review directory not found: {review_dir}")

    results = [
        result
        for path in sorted(review_dir.glob("*.json"))
        if (result := evaluate_file(path)) is not None
    ]
    return {
        "review_dir": str(review_dir),
        "positive_sample_definition": "chunk selected == true and selected_correct == true",
        "metric_definition": {
            "relationship_correct": "Relation count from TP chunks where relationship_correct == true",
            "relationship_incorrect": "Relation count from TP chunks where relationship_correct is not true",
            "partial_comment_rule": "A comment like '1/3 false' counts as 2 correct and 1 incorrect.",
            "accuracy": "relationship_correct / relationship checks",
        },
        "totals": total_scores(results),
        "macro_average": average_query_scores(results),
        "per_query": [asdict(result) for result in results],
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "query_text",
        "tp_chunks",
        "relationship_checks",
        "relationship_correct",
        "relationship_incorrect",
        "skipped_non_tp",
        "accuracy",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "source_file": item["source_file"],
                    "query_text": item["query_text"],
                    "tp_chunks": item["tp_chunks"],
                    "relationship_checks": item["relationship_checks"],
                    "relationship_correct": item["relationship_correct"],
                    "relationship_incorrect": item["relationship_incorrect"],
                    "skipped_non_tp": item["skipped_non_tp"],
                    "accuracy": round(float(item["accuracy"]), 4),
                }
            )


def print_summary(results: dict[str, Any]) -> None:
    totals = results["totals"]
    macro = results["macro_average"]
    print(f"[INFO] Evaluated reviewed queries: {macro['num_queries']}")
    print(
        "[INFO] Totals: "
        f"TP chunks={totals['tp_chunks']} "
        f"relationship_checks={totals['relationship_checks']} "
        f"relationship_correct={totals['relationship_correct']} "
        f"relationship_incorrect={totals['relationship_incorrect']}"
    )
    print(f"[INFO] Micro accuracy: {totals['accuracy']:.4f}")
    print(f"[INFO] Macro accuracy: {macro['accuracy']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate relationship_correct accuracy on true-positive selected chunks."
    )
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_review_dir(args.review_dir)
    write_json(args.output_json, results)
    write_csv(args.output_csv, results["per_query"])
    print_summary(results)
    print(f"\n[INFO] Wrote JSON: {args.output_json}")
    print(f"[INFO] Wrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
