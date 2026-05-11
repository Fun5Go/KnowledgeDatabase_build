from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from GraphRAG.Connection.validators import (
    validate_evidence_output,
    validate_rerank_output,
)


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def validate_results_dir(results_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(results_dir.glob("*.json"))
    return [validate_result_file(path) for path in paths]


def validate_result_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return build_report(path, errors=[f"Failed to read JSON: {exc}"])

    if isinstance(data, list):
        return validate_result_list(path, data)
    if not isinstance(data, dict):
        return build_report(path, errors=["Top-level JSON must be an object or list."])

    return validate_result_object(path, data)


def validate_result_list(path: Path, data: list[Any]) -> dict[str, Any]:
    child_reports = [
        validate_result_object(path, item)
        if isinstance(item, dict)
        else build_report(path, errors=[f"List item {index} must be an object."])
        for index, item in enumerate(data)
    ]
    errors = [
        f"item[{index}]: {error}"
        for index, report in enumerate(child_reports)
        for error in report.get("errors", [])
    ]
    return build_report(
        path,
        stage="list",
        count=len(data),
        errors=errors,
    )


def validate_result_object(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    connection = data.get("connection")
    if not isinstance(connection, dict):
        return build_report(path, errors=["Missing object field: connection."])

    stage = normalize_text(connection.get("stage"))
    if stage == "rerank":
        errors = validate_rerank_output(connection, get_connection_payload(data))
    elif stage == "extract":
        errors = validate_evidence_output(connection, get_extract_source_chunks(data))
    elif stage == "auto":
        errors = []
        errors.extend(validate_rerank_output(connection, get_connection_payload(data)))
        errors.extend(validate_evidence_output(connection, get_extract_source_chunks(data)))
    else:
        errors = [f"Unsupported connection.stage: {stage!r}."]

    analysis_item = data.get("analysis_item") if isinstance(data.get("analysis_item"), dict) else {}
    return build_report(
        path,
        stage=stage,
        analysis_id=normalize_text(connection.get("analysis_id")),
        query_type=normalize_text(connection.get("query_type")),
        query_text=normalize_text(analysis_item.get("query_text")),
        errors=errors,
    )


def get_connection_payload(data: dict[str, Any]) -> dict[str, Any]:
    payload = data.get("connection_payload")
    return payload if isinstance(payload, dict) else {}


def get_extract_source_chunks(data: dict[str, Any]) -> list[dict[str, Any]]:
    connection = data.get("connection") if isinstance(data.get("connection"), dict) else {}
    aggregates = connection.get("chunk_aggregates")
    if isinstance(aggregates, list) and aggregates:
        return [item for item in aggregates if isinstance(item, dict)]

    payload = get_connection_payload(data)
    reranked_chunks = payload.get("reranked_chunks")
    if isinstance(reranked_chunks, list) and reranked_chunks:
        return [item for item in reranked_chunks if isinstance(item, dict)]

    return []


def build_report(
    path: Path,
    stage: str = "",
    analysis_id: str = "",
    query_type: str = "",
    query_text: str = "",
    count: int | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    errors = errors or []
    report: dict[str, Any] = {
        "path": str(path),
        "ok": not errors,
        "stage": stage,
        "analysis_id": analysis_id,
        "query_type": query_type,
        "query_text": query_text,
        "errors": errors,
    }
    if count is not None:
        report["count"] = count
    return report


def write_report(reports: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(reports)
    output = {
        "summary": summary,
        "results": reports,
    }
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_summary(reports: list[dict[str, Any]]) -> dict[str, int]:
    failed = sum(1 for report in reports if not report.get("ok"))
    return {
        "total_files": len(reports),
        "passed": len(reports) - failed,
        "failed": failed,
    }


def print_reports(reports: list[dict[str, Any]], quiet: bool = False) -> None:
    if not quiet:
        for report in reports:
            status = "OK" if report.get("ok") else "FAIL"
            print(f"[{status}] {report.get('path')}")
            for error in report.get("errors", []):
                print(f"  - {error}")

    summary = build_summary(reports)
    print(
        "Validated "
        f"{summary['total_files']} file(s): "
        f"{summary['passed']} passed, {summary['failed']} failed."
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate split GraphRAG Connection result JSON files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing rerank_*.json and extract_*.json files.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a JSON validation report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final validation summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    reports = validate_results_dir(args.results_dir)
    print_reports(reports, quiet=args.quiet)
    if args.report is not None:
        write_report(reports, args.report)
        print(f"[INFO] Wrote validation report to: {args.report}")

    if any(not report.get("ok") for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
