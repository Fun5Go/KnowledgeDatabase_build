from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .evaluate_oneprocess_results import names_match, normalize_key, normalize_text
except ImportError:  # pragma: no cover - supports direct script execution
    from evaluate_oneprocess_results import names_match, normalize_key, normalize_text


DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "qd_dense_top20_QDFAT"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "qd_dense_top20_medium_selected_chunks_review.json"
DEFAULT_EVALUATION_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "qd_dense_top20_medium_selected_chunks_evaluation.json"
)
DEFAULT_GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parents[1]
    / "Connection"
    / "evaluation"
    / "final_extract_review_evaluation.json"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


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


def selected_chunks_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    selection = result.get("selection")
    if not isinstance(selection, dict):
        return []

    selected_qd_chunks = selection.get("selected_qd_chunks")
    if isinstance(selected_qd_chunks, list):
        return [item for item in selected_qd_chunks if isinstance(item, dict)]

    top_chunks = selection.get("top_chunks")
    if isinstance(top_chunks, list):
        return [item for item in top_chunks if isinstance(item, dict)]

    selected_qd_chunk = selection.get("selected_qd_chunk")
    if isinstance(selected_qd_chunk, dict):
        return [selected_qd_chunk]
    return []


def load_ground_truth_by_query(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}

    loaded = load_json(path)
    per_query = loaded.get("per_query") if isinstance(loaded, dict) else loaded
    if not isinstance(per_query, list):
        return {}

    ground_truth_by_query: dict[str, list[str]] = {}
    for item in per_query:
        if not isinstance(item, dict):
            continue
        query_text = normalize_text(item.get("query_text"))
        if not query_text:
            continue
        actual_value_names = item.get("actual_value_names")
        if not isinstance(actual_value_names, list):
            ground_truth = item.get("ground_truth")
            actual_value = ground_truth.get("actual_value") if isinstance(ground_truth, dict) else {}
            actual_value_names = actual_value.get("names") if isinstance(actual_value, dict) else []
        ground_truth_by_query[normalize_key(query_text)] = [
            normalize_text(name) for name in actual_value_names or [] if normalize_text(name)
        ]
    return ground_truth_by_query


def is_chunk_selected_correct(
    chunk: dict[str, Any],
    actual_value_names: list[str],
    default_is_selected_correct: bool,
) -> bool:
    if not actual_value_names:
        return default_is_selected_correct

    chunk_name = normalize_text(chunk.get("name") or chunk.get("chunk_id") or chunk.get("node_id"))
    return any(names_match(chunk_name, truth_name) for truth_name in actual_value_names)


def merge_selected_chunks_from_folder(
    folder_path: str | Path = DEFAULT_INPUT_DIR,
    ground_truth_path: str | Path | None = DEFAULT_GROUND_TRUTH_PATH,
    glob_pattern: str = "*.json",
    default_is_selected_correct: bool = False,
) -> list[dict[str, Any]]:
    """
    Merge selected chunks from every JSON file in a folder.

    Each returned item contains one selected chunk plus an ``is_selected_correct``
    boolean directly on that chunk. If a matching ground-truth file is supplied,
    the boolean is computed from actual_value_names; otherwise it uses
    default_is_selected_correct.
    """

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    truth_path = Path(ground_truth_path) if ground_truth_path is not None else None
    ground_truth_by_query = load_ground_truth_by_query(truth_path)

    merged: list[dict[str, Any]] = []
    for path in sorted(folder.glob(glob_pattern)):
        if not path.is_file():
            continue

        result = load_json(path)
        if isinstance(result, list):
            result_items = [item for item in result if isinstance(item, dict)]
        elif isinstance(result, dict):
            result_items = [result]
        else:
            continue

        for result_item in result_items:
            query_text = query_text_from_result(result_item)
            selection = result_item.get("selection") if isinstance(result_item.get("selection"), dict) else {}
            actual_value_names = ground_truth_by_query.get(normalize_key(query_text), [])

            for chunk in selected_chunks_from_result(result_item):
                selection_chunk = dict(chunk)
                selection_chunk["is_selected_correct"] = is_chunk_selected_correct(
                    selection_chunk,
                    actual_value_names,
                    default_is_selected_correct,
                )
                merged.append(
                    {
                        "source_file": path.name,
                        "analysis_id": normalize_text(selection.get("analysis_id")),
                        "query_type": normalize_text(selection.get("query_type")),
                        "query_text": query_text,
                        "selection_chunk": selection_chunk,
                    }
                )

    return merged


def write_merged_selected_chunks(
    output_path: str | Path,
    folder_path: str | Path = DEFAULT_INPUT_DIR,
    ground_truth_path: str | Path | None = DEFAULT_GROUND_TRUTH_PATH,
    glob_pattern: str = "*.json",
    default_is_selected_correct: bool = False,
) -> list[dict[str, Any]]:
    merged = merge_selected_chunks_from_folder(
        folder_path=folder_path,
        ground_truth_path=ground_truth_path,
        glob_pattern=glob_pattern,
        default_is_selected_correct=default_is_selected_correct,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(merged), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return merged


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def calculate_scores(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, tp + fp + tn + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def selected_correct_value(item: dict[str, Any]) -> Any:
    selection_chunk = item.get("selection_chunk")
    if isinstance(selection_chunk, dict):
        return selection_chunk.get("is_selected_correct")
    return item.get("is_selected_correct")


def review_item_query_text(item: dict[str, Any]) -> str:
    return normalize_text(item.get("query_text")) or normalize_text(item.get("analysis_id"))


def evaluate_selected_chunks_review(review_path: str | Path) -> dict[str, Any]:
    """
    Evaluate selected chunk review JSON.

    Count ``is_selected_correct == true`` as TP, ``false`` as FP, any other
    value as TN, and FN as 0. Macro averages include only queries represented
    by selected chunks in this review file.
    """

    path = Path(review_path)
    loaded = load_json(path)
    if not isinstance(loaded, list):
        raise TypeError(f"Review JSON must be a list: {path}")

    per_query_by_key: dict[str, dict[str, Any]] = {}
    review_items = [item for item in loaded if isinstance(item, dict)]
    for item in review_items:
        query_text = review_item_query_text(item) or "unknown_query"
        query_key = normalize_key(query_text)
        counts = per_query_by_key.setdefault(
            query_key,
            {
                "query_text": query_text,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "selected_chunks": 0,
            },
        )
        counts["selected_chunks"] += 1

        is_selected_correct = selected_correct_value(item)
        if is_selected_correct is True:
            counts["tp"] += 1
        elif is_selected_correct is False:
            counts["fp"] += 1
        else:
            counts["tn"] += 1

    per_query = [
        {
            **counts,
            **calculate_scores(
                int(counts["tp"]),
                int(counts["fp"]),
                int(counts["tn"]),
                int(counts["fn"]),
            ),
        }
        for counts in per_query_by_key.values()
    ]
    per_query.sort(key=lambda item: normalize_key(item.get("query_text")))

    macro_items = [item for item in per_query if int(item.get("selected_chunks", 0)) > 0]
    macro_recall_items = [
        item for item in macro_items if int(item.get("tp", 0)) + int(item.get("fn", 0)) > 0
    ]
    total_tp = sum(int(item.get("tp", 0)) for item in per_query)
    total_fp = sum(int(item.get("fp", 0)) for item in per_query)
    total_tn = sum(int(item.get("tn", 0)) for item in per_query)
    total_fn = 0

    return {
        "review_path": str(path),
        "rules": {
            "is_selected_correct_true": "TP",
            "is_selected_correct_false": "FP",
            "other": "TN",
            "fn": 0,
            "macro": "Average over queries with selected chunks only.",
            "macro_recall": "Average only over queries where TP + FN > 0; all-FP queries have undefined recall and are excluded.",
        },
        "summary": {
            "num_review_items": len(review_items),
            "num_queries": len(per_query),
            "micro": {
                "tp": total_tp,
                "fp": total_fp,
                "tn": total_tn,
                "fn": total_fn,
                **calculate_scores(total_tp, total_fp, total_tn, total_fn),
            },
            "macro": {
                "precision": safe_divide(
                    sum(float(item.get("precision", 0.0)) for item in macro_items),
                    len(macro_items),
                ),
                "recall": safe_divide(
                    sum(float(item.get("recall", 0.0)) for item in macro_recall_items),
                    len(macro_recall_items),
                ),
                "f1": safe_divide(
                    sum(float(item.get("f1", 0.0)) for item in macro_recall_items),
                    len(macro_recall_items),
                ),
                "accuracy": safe_divide(
                    sum(float(item.get("accuracy", 0.0)) for item in macro_items),
                    len(macro_items),
                ),
            },
        },
        "per_query": per_query,
    }


def write_selected_chunks_review_evaluation(
    review_path: str | Path = DEFAULT_OUTPUT_PATH,
    output_path: str | Path = DEFAULT_EVALUATION_OUTPUT_PATH,
) -> dict[str, Any]:
    evaluation = evaluate_selected_chunks_review(review_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(json_safe(evaluation), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge selected chunks from per-query extraction JSON files."
    )
    parser.add_argument(
        "--evaluate-review",
        type=Path,
        default=None,
        help="Evaluate an existing selected-chunks review JSON instead of merging input files.",
    )
    parser.add_argument("--folder", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--glob", default="*.json")
    parser.add_argument(
        "--default-correct",
        action="store_true",
        help="Use true when no matching ground-truth entry is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluate_review:
        evaluation = write_selected_chunks_review_evaluation(
            review_path=args.evaluate_review,
            output_path=args.output,
        )
        summary = evaluation["summary"]
        print(f"[INFO] Evaluated {summary['num_review_items']} selected chunk(s).")
        print(f"[INFO] Micro: {summary['micro']}")
        print(f"[INFO] Macro: {summary['macro']}")
        print(f"[INFO] Wrote evaluation JSON to: {args.output}")
        return

    merged = merge_selected_chunks_from_folder(
        folder_path=args.folder,
        ground_truth_path=args.ground_truth,
        glob_pattern=args.glob,
        default_is_selected_correct=args.default_correct,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(merged), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Wrote {len(merged)} selected chunk(s) to: {args.output}")


if __name__ == "__main__":
    main()
