"""Evaluate QD retrieval coverage from existing Top-20 QD result files.

The evaluator uses the selected-chunk review file as ground truth:
``selection_chunk.is_selected_correct == true`` means the QD/FAT chunk is
relevant for that query. It then checks whether those relevant chunks appear in
the ranked retrieval evidence at Top-10, Top-15, and Top-20.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from .evaluate_oneprocess_results import names_match, normalize_key, normalize_text
except ImportError:  # pragma: no cover - supports direct script execution
    from evaluate_oneprocess_results import names_match, normalize_key, normalize_text


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "qd_dense_top20_medium"
DEFAULT_REVIEW_PATH = SCRIPT_DIR / "qd_dense_top20_medium_selected_chunks_review.json"
DEFAULT_JSON_OUTPUT = SCRIPT_DIR / "qd_retrieval_evaluation_top20.json"
DEFAULT_CSV_OUTPUT = SCRIPT_DIR / "qd_retrieval_evaluation_top20.csv"
DEFAULT_K_VALUES = (10, 15, 20)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


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


def query_text_from_result(payload: dict[str, Any], path: Path) -> str:
    analysis_item = payload.get("analysis_item")
    if isinstance(analysis_item, dict):
        query_text = normalize_text(analysis_item.get("query_text"))
        if query_text:
            return query_text

    query_result = payload.get("query_result")
    if isinstance(query_result, dict):
        query_spec = query_result.get("query_spec")
        if isinstance(query_spec, dict):
            for key in ("query_text", "target_text", "sentence"):
                query_text = normalize_text(query_spec.get(key))
                if query_text:
                    return query_text
    return path.stem


def chunk_name(chunk: dict[str, Any], fallback_rank: int) -> str:
    value = chunk.get("name") or chunk.get("chunk_id") or chunk.get("node_id")
    return normalize_text(value) or f"rank_{fallback_rank}"


def chunk_rank(chunk: dict[str, Any], fallback_rank: int) -> int:
    for key in ("retrieval_rank", "rank"):
        value = chunk.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.strip().isdigit():
            parsed = int(value.strip())
            if parsed > 0:
                return parsed
    return fallback_rank


def evidence_from_result(payload: dict[str, Any]) -> list[dict[str, Any]]:
    query_result = payload.get("query_result")
    if isinstance(query_result, dict) and isinstance(query_result.get("evidence"), list):
        return [chunk for chunk in query_result["evidence"] if isinstance(chunk, dict)]
    if isinstance(payload.get("evidence"), list):
        return [chunk for chunk in payload["evidence"] if isinstance(chunk, dict)]
    return []


def review_key(source_file: str, query_text: str) -> tuple[str, str]:
    return source_file, normalize_key(query_text)


def load_relevant_names_by_query(review_path: Path) -> dict[tuple[str, str], list[str]]:
    loaded = load_json(review_path)
    if not isinstance(loaded, list):
        raise ValueError(f"Expected review file to contain a list: {review_path}")

    relevant_by_query: dict[tuple[str, str], list[str]] = {}
    for item in loaded:
        if not isinstance(item, dict):
            continue
        selection_chunk = item.get("selection_chunk")
        source_file = normalize_text(item.get("source_file"))
        query_text = normalize_text(item.get("query_text"))
        if not isinstance(selection_chunk, dict) or not source_file or not query_text:
            continue

        key = review_key(source_file, query_text)
        relevant_by_query.setdefault(key, [])
        if normalize_bool(selection_chunk.get("is_selected_correct")) is not True:
            continue

        name = chunk_name(selection_chunk, len(relevant_by_query) + 1)
        if name and not any(names_match(name, existing) for existing in relevant_by_query[key]):
            relevant_by_query[key].append(name)
    return relevant_by_query


def matched_relevant_names(ranked_chunks: list[dict[str, Any]], relevant_names: list[str], k: int) -> list[str]:
    matched: list[str] = []
    for chunk in ranked_chunks:
        if int(chunk["rank"]) > k:
            continue
        name = normalize_text(chunk["name"])
        for relevant_name in relevant_names:
            if any(names_match(relevant_name, existing) for existing in matched):
                continue
            if names_match(name, relevant_name):
                matched.append(relevant_name)
                break
    return matched


def evaluate_result_file(
    path: Path,
    relevant_by_query: dict[tuple[str, str], list[str]],
    k_values: list[int],
) -> dict[str, Any] | None:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None

    query_text = query_text_from_result(payload, path)
    key = review_key(path.name, query_text)
    if key not in relevant_by_query:
        return None
    relevant_names = relevant_by_query[key]

    ranked_chunks: list[dict[str, Any]] = []
    for fallback_rank, chunk in enumerate(evidence_from_result(payload), start=1):
        rank = chunk_rank(chunk, fallback_rank)
        ranked_chunks.append(
            {
                "rank": rank,
                "name": chunk_name(chunk, rank),
                "label": normalize_text(chunk.get("label")),
            }
        )
    ranked_chunks.sort(key=lambda item: int(item["rank"]))

    relevant_ranks: list[int] = []
    for relevant_name in relevant_names:
        for chunk in ranked_chunks:
            if names_match(str(chunk["name"]), relevant_name):
                relevant_ranks.append(int(chunk["rank"]))
                break

    metrics_by_k: dict[str, dict[str, Any]] = {}
    for k in k_values:
        top_k = [chunk for chunk in ranked_chunks if int(chunk["rank"]) <= k]
        matched = matched_relevant_names(ranked_chunks, relevant_names, k)
        relevant_count = len(matched)
        retrieved_count = len(top_k)
        noise_count = retrieved_count - relevant_count
        metrics_by_k[str(k)] = {
            "relevant_count": relevant_count,
            "retrieved_count": retrieved_count,
            "noise_count": noise_count,
            "coverage": relevant_count / len(relevant_names) if relevant_names else None,
            "noise_ratio": noise_count / retrieved_count if retrieved_count else 0.0,
            "matched_relevant_names": matched,
            "missed_relevant_names": [
                name for name in relevant_names if not any(names_match(name, matched_name) for matched_name in matched)
            ],
        }

    return {
        "source_file": path.name,
        "query_text": query_text,
        "num_candidates": len(ranked_chunks),
        "total_relevant": len(relevant_names),
        "relevant_average_rank": statistics.fmean(relevant_ranks) if relevant_ranks else None,
        "relevant_median_rank": statistics.median(relevant_ranks) if relevant_ranks else None,
        "relevant_names": relevant_names,
        "retrieved_relevant_ranks": relevant_ranks,
        "metrics_by_k": metrics_by_k,
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def aggregate_results(per_query: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    coverage_queries = [item for item in per_query if int(item["total_relevant"]) > 0]
    total_relevant = sum(int(item["total_relevant"]) for item in per_query)
    all_ranks = [
        int(rank)
        for item in per_query
        for rank in item["retrieved_relevant_ranks"]
    ]

    by_k: dict[str, dict[str, Any]] = {}
    for k in k_values:
        key = str(k)
        relevant_count = sum(int(item["metrics_by_k"][key]["relevant_count"]) for item in per_query)
        retrieved_count = sum(int(item["metrics_by_k"][key]["retrieved_count"]) for item in per_query)
        noise_count = sum(int(item["metrics_by_k"][key]["noise_count"]) for item in per_query)
        by_k[key] = {
            "relevant_count": relevant_count,
            "micro_coverage": relevant_count / total_relevant if total_relevant else 0.0,
            "macro_coverage": mean(
                [
                    float(item["metrics_by_k"][key]["coverage"])
                    for item in coverage_queries
                    if item["metrics_by_k"][key]["coverage"] is not None
                ]
            ),
            "micro_noise_ratio": noise_count / retrieved_count if retrieved_count else 0.0,
            "macro_noise_ratio": mean([float(item["metrics_by_k"][key]["noise_ratio"]) for item in per_query]),
            "retrieved_count": retrieved_count,
            "noise_count": noise_count,
        }

    return {
        "reviewed_query_count": len(per_query),
        "coverage_query_count": len(coverage_queries),
        "zero_relevant_query_count": len(per_query) - len(coverage_queries),
        "total_relevant": total_relevant,
        "retrieved_relevant_count": len(all_ranks),
        "relevant_average_rank": mean([float(rank) for rank in all_ranks]),
        "relevant_median_rank": statistics.median(all_ranks) if all_ranks else None,
        "by_k": by_k,
    }


def evaluate_qd_retrieval(
    results_dir: Path,
    review_path: Path,
    k_values: list[int],
) -> dict[str, Any]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not review_path.exists():
        raise FileNotFoundError(f"Review file not found: {review_path}")

    relevant_by_query = load_relevant_names_by_query(review_path)
    per_query = [
        result
        for path in sorted(results_dir.glob("*.json"))
        if (result := evaluate_result_file(path, relevant_by_query, k_values)) is not None
    ]
    return {
        "results_dir": str(results_dir),
        "review_path": str(review_path),
        "relevant_definition": "selection_chunk.is_selected_correct == true",
        "noise_definition": "Top-K chunks not matched to a relevant reviewed chunk",
        "k_values": k_values,
        "summary": aggregate_results(per_query, k_values),
        "per_query": per_query,
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
                "relevant_average_rank": (
                    round(float(item["relevant_average_rank"]), 4)
                    if item["relevant_average_rank"] is not None
                    else ""
                ),
                "relevant_median_rank": item["relevant_median_rank"] or "",
            }
            for k in k_values:
                metrics = item["metrics_by_k"][str(k)]
                row[f"coverage@{k}"] = (
                    round(float(metrics["coverage"]), 4)
                    if metrics["coverage"] is not None
                    else ""
                )
                row[f"relevant_count@{k}"] = metrics["relevant_count"]
                row[f"noise_ratio@{k}"] = round(float(metrics["noise_ratio"]), 4)
                row[f"noise_count@{k}"] = metrics["noise_count"]
            writer.writerow(row)


def print_summary(results: dict[str, Any]) -> None:
    summary = results["summary"]
    print(f"[INFO] Reviewed QD queries: {summary['reviewed_query_count']}")
    print(f"[INFO] Coverage-evaluable QD queries: {summary['coverage_query_count']}")
    print(f"[INFO] Zero-relevant reviewed QD queries: {summary['zero_relevant_query_count']}")
    print(f"[INFO] Total relevant chunks: {summary['total_relevant']}")
    print(f"[INFO] Retrieved relevant chunks: {summary['retrieved_relevant_count']}")
    print(f"[INFO] Relevant average rank: {summary['relevant_average_rank']:.2f}")
    print(f"[INFO] Relevant median rank: {summary['relevant_median_rank']}")
    print("")
    print("K\tRelevantCount\tMicroCoverage\tMacroCoverage\tMicroNoise\tMacroNoise")
    for k, metrics in summary["by_k"].items():
        print(
            f"{k}\t"
            f"{metrics['relevant_count']}\t"
            f"{metrics['micro_coverage']:.4f}\t"
            f"{metrics['macro_coverage']:.4f}\t"
            f"{metrics['micro_noise_ratio']:.4f}\t"
            f"{metrics['macro_noise_ratio']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate QD retrieval Coverage@10/15/20 from existing Top-20 result JSON files."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--review-path", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--k-values", default="10,15,20")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = parse_k_values(args.k_values)
    results = evaluate_qd_retrieval(args.results_dir, args.review_path, k_values)
    write_json(args.json_output, results)
    write_csv(args.csv_output, results["per_query"], k_values)
    print_summary(results)
    print(f"\n[INFO] Wrote JSON: {args.json_output}")
    print(f"[INFO] Wrote CSV: {args.csv_output}")


if __name__ == "__main__":
    main()
