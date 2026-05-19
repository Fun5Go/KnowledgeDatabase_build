"""Evaluate reviewed loose-extraction JSON files.

The script reads ``*.review.json`` files, counts selection outcomes per query,
and reports precision, recall, and F1. By default it evaluates the reviewed
files in ``C:/Users/FW/Desktop/FMEA_AI/Review Material/extract_by_FW``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REVIEW_DIR = Path(r"C:\Users\FW\Desktop\FMEA_AI\Review Material\extract_by_FW")
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "final_extract_review_evaluation.json"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "final_extract_review_evaluation.csv"


@dataclass
class QueryMetrics:
    query_text: str
    source_file: str
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    skipped: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    actual_value_names: list[str] = field(default_factory=list)
    actual_irrelevant_names: list[str] = field(default_factory=list)

    @property
    def evaluated(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def actual_value(self) -> int:
        return self.tp + self.fn

    @property
    def actual_irrelevant(self) -> int:
        return self.tn + self.fp


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
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
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


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


def reviewed_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        chunks = payload.get("chunk_aggregates")
        if isinstance(chunks, list):
            return [item for item in chunks if isinstance(item, dict)]

        chunks = payload.get("items")
        if isinstance(chunks, list):
            return [item for item in chunks if isinstance(item, dict)]

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    return []


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def finalize_metrics(metrics: QueryMetrics) -> QueryMetrics:
    metrics.precision = safe_divide(metrics.tp, metrics.tp + metrics.fp)
    metrics.recall = safe_divide(metrics.tp, metrics.tp + metrics.fn)
    metrics.f1 = safe_divide(2 * metrics.precision * metrics.recall, metrics.precision + metrics.recall)
    return metrics


def evaluate_file(path: Path) -> QueryMetrics:
    payload = load_json(path)
    metrics = QueryMetrics(query_text=query_text(payload, path), source_file=path.name)

    for index, item in enumerate(reviewed_items(payload), start=1):
        selected = normalize_bool(item.get("selected"))
        is_select_correct = normalize_bool(
            first_present(
                item,
                [
                    "is_select_correct",
                    "is_selected_correct",
                    "selected_correct",
                    "is_selection_correct",
                ],
            )
        )

        if selected is None or is_select_correct is None:
            metrics.skipped += 1
            continue

        chunk_name = str(first_present(item, ["name", "chunk_id", "node_id"]) or f"chunk_{index}").strip()

        if selected is True and is_select_correct is True:
            metrics.tp += 1
            metrics.actual_value_names.append(chunk_name)
        elif selected is False and is_select_correct is True:
            metrics.tn += 1
            metrics.actual_irrelevant_names.append(chunk_name)
        elif selected is True and is_select_correct is False:
            metrics.fp += 1
            metrics.actual_irrelevant_names.append(chunk_name)
        elif selected is False and is_select_correct is False:
            metrics.fn += 1
            metrics.actual_value_names.append(chunk_name)

    return finalize_metrics(metrics)


def average_query_scores(results: list[QueryMetrics]) -> dict[str, Any]:
    evaluated_results = [result for result in results if result.evaluated > 0]
    count = len(evaluated_results)
    if count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_queries": 0}

    return {
        "precision": sum(result.precision for result in evaluated_results) / count,
        "recall": sum(result.recall for result in evaluated_results) / count,
        "f1": sum(result.f1 for result in evaluated_results) / count,
        "num_queries": count,
    }


def total_scores(results: list[QueryMetrics]) -> dict[str, Any]:
    totals = {
        "tp": sum(result.tp for result in results),
        "tn": sum(result.tn for result in results),
        "fp": sum(result.fp for result in results),
        "fn": sum(result.fn for result in results),
        "skipped": sum(result.skipped for result in results),
    }
    totals["evaluated"] = totals["tp"] + totals["tn"] + totals["fp"] + totals["fn"]
    totals["precision"] = safe_divide(totals["tp"], totals["tp"] + totals["fp"])
    totals["recall"] = safe_divide(totals["tp"], totals["tp"] + totals["fn"])
    totals["f1"] = safe_divide(
        2 * totals["precision"] * totals["recall"],
        totals["precision"] + totals["recall"],
    )
    return totals


def write_csv(path: Path, results: list[QueryMetrics], averages: dict[str, Any], totals: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_text",
        "source_file",
        "tp",
        "tn",
        "fp",
        "fn",
        "evaluated",
        "skipped",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["evaluated"] = result.evaluated
            row.pop("actual_value_names", None)
            row.pop("actual_irrelevant_names", None)
            writer.writerow(row)

        writer.writerow({})
        writer.writerow(
            {
                "query_text": "AVERAGE_PER_QUERY",
                "evaluated": averages["num_queries"],
                "precision": averages["precision"],
                "recall": averages["recall"],
                "f1": averages["f1"],
            }
        )
        writer.writerow(
            {
                "query_text": "TOTAL_MICRO",
                "tp": totals["tp"],
                "tn": totals["tn"],
                "fp": totals["fp"],
                "fn": totals["fn"],
                "evaluated": totals["evaluated"],
                "skipped": totals["skipped"],
                "precision": totals["precision"],
                "recall": totals["recall"],
                "f1": totals["f1"],
            }
        )


def print_summary(results: list[QueryMetrics], averages: dict[str, Any], totals: dict[str, Any]) -> None:
    header = f"{'Query':45} {'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3} {'P':>7} {'R':>7} {'F1':>7}"
    print(header)
    print("-" * len(header))
    for result in results:
        query = result.query_text[:42] + "..." if len(result.query_text) > 45 else result.query_text
        print(
            f"{query:45} {result.tp:>3} {result.tn:>3} {result.fp:>3} {result.fn:>3} "
            f"{result.precision:>7.3f} {result.recall:>7.3f} {result.f1:>7.3f}"
        )

    print("-" * len(header))
    print(
        f"{'AVERAGE_PER_QUERY':45} {'':>3} {'':>3} {'':>3} {'':>3} "
        f"{averages['precision']:>7.3f} {averages['recall']:>7.3f} {averages['f1']:>7.3f}"
    )
    print(
        f"{'TOTAL_MICRO':45} {totals['tp']:>3} {totals['tn']:>3} {totals['fp']:>3} {totals['fn']:>3} "
        f"{totals['precision']:>7.3f} {totals['recall']:>7.3f} {totals['f1']:>7.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--glob", default="*.review.json")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_paths = sorted(args.review_dir.glob(args.glob))
    if not review_paths:
        raise FileNotFoundError(f"No review files matched {args.review_dir / args.glob}")

    results = [evaluate_file(path) for path in review_paths]
    averages = average_query_scores(results)
    totals = total_scores(results)

    report = {
        "review_dir": str(args.review_dir),
        "num_files": len(review_paths),
        "per_query": [
            asdict(result)
            | {
                "evaluated": result.evaluated,
                "ground_truth": {
                    "actual_value": {
                        "count": result.actual_value,
                        "names": result.actual_value_names,
                    },
                    "actual_irrelevant": {
                        "count": result.actual_irrelevant,
                        "names": result.actual_irrelevant_names,
                    },
                },
            }
            for result in results
        ],
        "average_per_query": averages,
        "total_micro": totals,
    }

    write_json(args.output_json, report)
    write_csv(args.output_csv, results, averages, totals)
    print_summary(results, averages, totals)
    print(f"\nWrote JSON: {args.output_json}")
    print(f"Wrote CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
