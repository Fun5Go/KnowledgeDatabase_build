"""Evaluate reviewed Top-60 retrieval coverage from human review JSON files.

The script treats chunks with ``selected_correct == true`` as relevant positive
samples. A query file is evaluated only when at least one chunk has been marked
as relevant, which filters out files that have not been reviewed yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REVIEW_DIR = SCRIPT_DIR / "human_review_by_query_top60_integrated"
DEFAULT_JSON_OUTPUT = SCRIPT_DIR / "retrieval_evaluation_top60.json"
DEFAULT_CSV_OUTPUT = SCRIPT_DIR / "retrieval_evaluation_top60.csv"
DEFAULT_K_VALUES = (15, 30, 45, 50, 60)


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


def parse_k_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"K values must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one K value is required.")
    return sorted(set(values))


def get_chunks(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    chunks = payload.get("chunks")
    if isinstance(chunks, list):
        return [chunk for chunk in chunks if isinstance(chunk, dict)]
    return []


def chunk_rank(chunk: dict[str, Any], fallback_rank: int) -> int:
    rank = chunk.get("rank")
    if isinstance(rank, int) and rank > 0:
        return rank
    if isinstance(rank, str) and rank.strip().isdigit():
        parsed = int(rank.strip())
        if parsed > 0:
            return parsed
    return fallback_rank


def chunk_name(chunk: dict[str, Any], fallback_rank: int) -> str:
    value = chunk.get("name") or chunk.get("chunk_id") or chunk.get("node_id")
    return str(value or f"rank_{fallback_rank}").strip()


def reviewed_selected_correct_values(chunks: list[dict[str, Any]]) -> list[bool | None]:
    return [normalize_bool(chunk.get("selected_correct")) for chunk in chunks]


def is_reviewed_query(chunks: list[dict[str, Any]], reviewed_mode: str) -> bool:
    labels = reviewed_selected_correct_values(chunks)
    if reviewed_mode == "any_label":
        return any(label is not None for label in labels)
    return any(label is True for label in labels)


def evaluate_query_file(path: Path, k_values: list[int], reviewed_mode: str) -> dict[str, Any] | None:
    payload = load_json(path)
    chunks = get_chunks(payload)
    if not chunks or not is_reviewed_query(chunks, reviewed_mode):
        return None

    ranked_chunks: list[dict[str, Any]] = []
    for fallback_rank, chunk in enumerate(chunks, start=1):
        rank = chunk_rank(chunk, fallback_rank)
        ranked_chunks.append(
            {
                "rank": rank,
                "name": chunk_name(chunk, rank),
                "selected_correct": normalize_bool(chunk.get("selected_correct")),
                "rerank_tag": chunk.get("rerank_tag"),
            }
        )
    ranked_chunks.sort(key=lambda item: int(item["rank"]))

    relevant_chunks = [chunk for chunk in ranked_chunks if chunk["selected_correct"] is True]
    relevant_ranks = [int(chunk["rank"]) for chunk in relevant_chunks]
    total_relevant = len(relevant_chunks)
    if total_relevant == 0:
        return None

    metrics_by_k: dict[str, dict[str, Any]] = {}
    for k in k_values:
        top_k = [chunk for chunk in ranked_chunks if int(chunk["rank"]) <= k]
        relevant_at_k = sum(1 for chunk in top_k if chunk["selected_correct"] is True)
        retrieved_at_k = len(top_k)
        noise_at_k = retrieved_at_k - relevant_at_k
        metrics_by_k[str(k)] = {
            "relevant_count": relevant_at_k,
            "retrieved_count": retrieved_at_k,
            "noise_count": noise_at_k,
            "coverage": relevant_at_k / total_relevant if total_relevant else 0.0,
            "noise_ratio": noise_at_k / retrieved_at_k if retrieved_at_k else 0.0,
        }

    return {
        "source_file": path.name,
        "query_text": str(payload.get("query_text", "")).strip(),
        "num_candidates": len(ranked_chunks),
        "total_relevant": total_relevant,
        "relevant_average_rank": statistics.fmean(relevant_ranks),
        "relevant_median_rank": statistics.median(relevant_ranks),
        "relevant_chunks": relevant_chunks,
        "metrics_by_k": metrics_by_k,
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def aggregate_results(per_query: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    total_relevant = sum(int(item["total_relevant"]) for item in per_query)
    all_relevant_ranks = [
        int(chunk["rank"])
        for item in per_query
        for chunk in item["relevant_chunks"]
    ]

    by_k: dict[str, dict[str, Any]] = {}
    for k in k_values:
        key = str(k)
        relevant_count = sum(int(item["metrics_by_k"][key]["relevant_count"]) for item in per_query)
        retrieved_count = sum(int(item["metrics_by_k"][key]["retrieved_count"]) for item in per_query)
        noise_count = sum(int(item["metrics_by_k"][key]["noise_count"]) for item in per_query)
        by_k[key] = {
            "micro_coverage": relevant_count / total_relevant if total_relevant else 0.0,
            "macro_coverage": mean([float(item["metrics_by_k"][key]["coverage"]) for item in per_query]),
            "micro_noise_ratio": noise_count / retrieved_count if retrieved_count else 0.0,
            "macro_noise_ratio": mean([float(item["metrics_by_k"][key]["noise_ratio"]) for item in per_query]),
            "relevant_count": relevant_count,
            "retrieved_count": retrieved_count,
            "noise_count": noise_count,
        }

    return {
        "evaluated_query_count": len(per_query),
        "total_relevant": total_relevant,
        "relevant_average_rank": mean([float(rank) for rank in all_relevant_ranks]),
        "relevant_median_rank": statistics.median(all_relevant_ranks) if all_relevant_ranks else None,
        "by_k": by_k,
    }


def write_csv(path: Path, per_query: list[dict[str, Any]], k_values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "query_text",
        "num_candidates",
        "total_relevant",
        "relevant_average_rank",
        "relevant_median_rank",
    ]
    for k in k_values:
        fieldnames.extend(
            [
                f"coverage@{k}",
                f"relevant_count@{k}",
                f"noise_ratio@{k}",
                f"noise_count@{k}",
            ]
        )

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_query:
            row: dict[str, Any] = {
                "source_file": item["source_file"],
                "query_text": item["query_text"],
                "num_candidates": item["num_candidates"],
                "total_relevant": item["total_relevant"],
                "relevant_average_rank": round(float(item["relevant_average_rank"]), 4),
                "relevant_median_rank": item["relevant_median_rank"],
            }
            for k in k_values:
                metrics = item["metrics_by_k"][str(k)]
                row[f"coverage@{k}"] = round(float(metrics["coverage"]), 4)
                row[f"relevant_count@{k}"] = metrics["relevant_count"]
                row[f"noise_ratio@{k}"] = round(float(metrics["noise_ratio"]), 4)
                row[f"noise_count@{k}"] = metrics["noise_count"]
            writer.writerow(row)


def evaluate_review_dir(
    review_dir: Path,
    k_values: list[int],
    reviewed_mode: str,
) -> dict[str, Any]:
    if not review_dir.exists():
        raise FileNotFoundError(f"Review directory not found: {review_dir}")

    per_query = [
        result
        for path in sorted(review_dir.glob("*.json"))
        if (result := evaluate_query_file(path, k_values, reviewed_mode)) is not None
    ]
    return {
        "review_dir": str(review_dir),
        "reviewed_mode": reviewed_mode,
        "relevant_definition": "chunk.selected_correct == true",
        "noise_definition": "Top-K chunks where selected_correct is not true",
        "k_values": k_values,
        "summary": aggregate_results(per_query, k_values),
        "per_query": per_query,
    }


def print_summary(results: dict[str, Any]) -> None:
    summary = results["summary"]
    print(f"[INFO] Evaluated reviewed queries: {summary['evaluated_query_count']}")
    print(f"[INFO] Total relevant chunks: {summary['total_relevant']}")
    print(f"[INFO] Relevant average rank: {summary['relevant_average_rank']:.2f}")
    print(f"[INFO] Relevant median rank: {summary['relevant_median_rank']}")
    print("")
    print("K\tMicroCoverage\tMacroCoverage\tMicroNoise\tMacroNoise")
    for k, metrics in summary["by_k"].items():
        print(
            f"{k}\t"
            f"{metrics['micro_coverage']:.4f}\t"
            f"{metrics['macro_coverage']:.4f}\t"
            f"{metrics['micro_noise_ratio']:.4f}\t"
            f"{metrics['macro_noise_ratio']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Coverage@K, relevant rank, and Noise ratio@K from reviewed Top-60 JSON files."
    )
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--k-values", default="15,30,45,50,60")
    parser.add_argument(
        "--reviewed-mode",
        choices=("any_true", "any_label"),
        default="any_true",
        help="any_true evaluates files with at least one selected_correct=true; any_label also includes files with only false labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.k_values)
    results = evaluate_review_dir(args.review_dir, k_values, args.reviewed_mode)
    write_json(args.json_output, results)
    write_csv(args.csv_output, results["per_query"], k_values)
    print_summary(results)
    print(f"\n[INFO] Wrote JSON: {args.json_output}")
    print(f"[INFO] Wrote CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
