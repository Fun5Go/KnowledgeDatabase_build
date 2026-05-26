"""Evaluate rerank labels against selected_correct positives.

Positive samples are chunks with ``selected_correct == true``.

Confusion-count rules:
- TP: selected_correct true and rerank_tag is support or suspect.
- TN: selected_correct null and rerank_tag is irrelevant.
- FP: selected_correct null and rerank_tag is support or suspect.
- FN: selected_correct true and rerank_tag is irrelevant.

Chunks outside these rules are skipped, because they are either unreviewed
support/suspect candidates or labels that do not define this evaluation target.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REVIEW_DIR = SCRIPT_DIR / "human_review_by_query_top60_integrated"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "rerank_evaluation_top60.json"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "rerank_evaluation_top60.csv"
POSITIVE_TAGS = {"support", "suspect"}
NEGATIVE_TAG = "irrelevant"


@dataclass
class QueryMetrics:
    query_text: str
    source_file: str
    num_candidates: int = 0
    num_irrelevant: int = 0
    filtering_rate: float = 0.0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    skipped: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    positive_names: list[str] = field(default_factory=list)
    true_negative_names: list[str] = field(default_factory=list)
    false_negative_names: list[str] = field(default_factory=list)

    @property
    def evaluated(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def actual_positive(self) -> int:
        return self.tp + self.fn


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


def normalize_tag(value: Any) -> str:
    return str(value or "").strip().casefold()


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


def chunk_name(chunk: dict[str, Any], fallback_index: int) -> str:
    value = first_present(chunk, ["name", "chunk_id", "node_id"])
    return str(value or f"chunk_{fallback_index}").strip()


def finalize_metrics(metrics: QueryMetrics) -> QueryMetrics:
    metrics.precision = safe_divide(metrics.tp, metrics.tp + metrics.fp)
    metrics.recall = safe_divide(metrics.tp, metrics.tp + metrics.fn)
    metrics.f1 = safe_divide(
        2 * metrics.precision * metrics.recall,
        metrics.precision + metrics.recall,
    )
    metrics.accuracy = safe_divide(metrics.tp + metrics.tn, metrics.evaluated)
    metrics.filtering_rate = safe_divide(metrics.num_irrelevant, metrics.num_candidates)
    return metrics


def evaluate_file(path: Path) -> QueryMetrics | None:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None

    metrics = QueryMetrics(query_text=query_text(payload, path), source_file=path.name)
    chunks = reviewed_chunks(payload)
    metrics.num_candidates = len(chunks)
    metrics.num_irrelevant = sum(
        1 for chunk in chunks if normalize_tag(chunk.get("rerank_tag")) == NEGATIVE_TAG
    )
    if not any(normalize_bool(chunk.get("selected_correct")) is True for chunk in chunks):
        return None

    for index, chunk in enumerate(chunks, start=1):
        selected_correct = normalize_bool(chunk.get("selected_correct"))
        tag = normalize_tag(chunk.get("rerank_tag"))
        name = chunk_name(chunk, index)

        if selected_correct is True and tag in POSITIVE_TAGS:
            metrics.tp += 1
            metrics.positive_names.append(name)
        elif selected_correct is None and tag == NEGATIVE_TAG:
            metrics.tn += 1
            metrics.true_negative_names.append(name)
        elif selected_correct is None and tag in POSITIVE_TAGS:
            metrics.fp += 1
        elif selected_correct is True and tag == NEGATIVE_TAG:
            metrics.fn += 1
            metrics.false_negative_names.append(name)
        else:
            metrics.skipped += 1

    return finalize_metrics(metrics)


def average_query_scores(results: list[QueryMetrics]) -> dict[str, Any]:
    evaluated = [result for result in results if result.evaluated > 0]
    if not evaluated:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "filtering_rate": 0.0,
            "num_queries": 0,
        }
    return {
        "precision": sum(result.precision for result in evaluated) / len(evaluated),
        "recall": sum(result.recall for result in evaluated) / len(evaluated),
        "f1": sum(result.f1 for result in evaluated) / len(evaluated),
        "accuracy": sum(result.accuracy for result in evaluated) / len(evaluated),
        "filtering_rate": sum(result.filtering_rate for result in evaluated) / len(evaluated),
        "num_queries": len(evaluated),
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
    totals["actual_positive"] = totals["tp"] + totals["fn"]
    totals["num_candidates"] = sum(result.num_candidates for result in results)
    totals["num_irrelevant"] = sum(result.num_irrelevant for result in results)
    totals["filtering_rate"] = safe_divide(totals["num_irrelevant"], totals["num_candidates"])
    totals["precision"] = safe_divide(totals["tp"], totals["tp"] + totals["fp"])
    totals["recall"] = safe_divide(totals["tp"], totals["tp"] + totals["fn"])
    totals["f1"] = safe_divide(
        2 * totals["precision"] * totals["recall"],
        totals["precision"] + totals["recall"],
    )
    totals["accuracy"] = safe_divide(totals["tp"] + totals["tn"], totals["evaluated"])
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
        "positive_sample_definition": "chunk.selected_correct == true",
        "confusion_rules": {
            "tp": "selected_correct == true and rerank_tag in {'support', 'suspect'}",
            "tn": "selected_correct is null and rerank_tag == 'irrelevant'",
            "fp": "selected_correct is null and rerank_tag in {'support', 'suspect'}",
            "fn": "selected_correct == true and rerank_tag == 'irrelevant'",
            "skipped": "all other chunks",
        },
        "filtering_rate_definition": {
            "micro": "sum(rerank_tag == 'irrelevant') / sum(top60 candidates)",
            "macro": "average per-query filtering_rate",
        },
        "totals": total_scores(results),
        "macro_average": average_query_scores(results),
        "per_query": [asdict(result) | {"evaluated": result.evaluated} for result in results],
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "query_text",
        "num_candidates",
        "num_irrelevant",
        "filtering_rate",
        "tp",
        "tn",
        "fp",
        "fn",
        "evaluated",
        "skipped",
        "precision",
        "recall",
        "f1",
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
                    "num_candidates": item["num_candidates"],
                    "num_irrelevant": item["num_irrelevant"],
                    "filtering_rate": round(float(item["filtering_rate"]), 4),
                    "tp": item["tp"],
                    "tn": item["tn"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "evaluated": item["evaluated"],
                    "skipped": item["skipped"],
                    "precision": round(float(item["precision"]), 4),
                    "recall": round(float(item["recall"]), 4),
                    "f1": round(float(item["f1"]), 4),
                    "accuracy": round(float(item["accuracy"]), 4),
                }
            )


def print_summary(results: dict[str, Any]) -> None:
    totals = results["totals"]
    macro = results["macro_average"]
    print(f"[INFO] Evaluated reviewed queries: {macro['num_queries']}")
    print(
        "[INFO] Totals: "
        f"TP={totals['tp']} TN={totals['tn']} FP={totals['fp']} "
        f"FN={totals['fn']} skipped={totals['skipped']}"
    )
    print(
        "[INFO] Micro: "
        f"P={totals['precision']:.4f} R={totals['recall']:.4f} "
        f"F1={totals['f1']:.4f} Acc={totals['accuracy']:.4f} "
        f"Filtering={totals['filtering_rate']:.4f} "
        f"({totals['num_irrelevant']}/{totals['num_candidates']})"
    )
    print(
        "[INFO] Macro: "
        f"P={macro['precision']:.4f} R={macro['recall']:.4f} "
        f"F1={macro['f1']:.4f} Acc={macro['accuracy']:.4f} "
        f"Filtering={macro['filtering_rate']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rerank support/suspect/irrelevant labels against selected_correct positives."
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
