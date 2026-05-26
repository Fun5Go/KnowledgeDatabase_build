"""Command-line entry point for the FMEA assistant demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - supports direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from GraphRAG.Demonstration.assistance.workflow import FMEAAssistanceWorkflow
else:
    from .workflow import FMEAAssistanceWorkflow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse assistant inputs for the pre-LLM failure evidence interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a pre-LLM FMEA assistant package from one Failure ID. "
            "The output includes element, function, cause, mode, effect, and "
            "attribute-linked chunks under allowed evidence relationship types."
        )
    )
    parser.add_argument(
        "failure_id",
        help="Failure node id, for example DFMEA6011160042R01__R125.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=20,
        help="Maximum chunks to retrieve per attribute type. Default: 20.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the generated output.",
    )
    parser.add_argument(
        "--interpret",
        action="store_true",
        help="Call the configured LLM to interpret why/what/how and evidence support.",
    )
    parser.add_argument(
        "--placeholder-llm",
        action="store_true",
        help="Use deterministic placeholder interpretation instead of calling an LLM.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Execute the pre-LLM failure evidence interface."""
    _configure_stdout()
    args = parse_args()
    workflow = FMEAAssistanceWorkflow()
    if args.placeholder_llm:
        workflow.interpreter.use_placeholder = True
    try:
        if args.interpret:
            result = workflow.interpret_failure_id(
                failure_id=args.failure_id,
                chunk_limit_per_attribute=args.chunk_limit,
            )
        else:
            result = workflow.build_failure_pre_llm_text_package(
                failure_id=args.failure_id,
                chunk_limit_per_attribute=args.chunk_limit,
            )
        output = format_result(result, args.format)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output, encoding="utf-8")
        print(output)
    finally:
        workflow.evidence_linker.close()


def format_result(result: dict[str, Any], output_format: str) -> str:
    """Format the pre-LLM package as text or JSON."""
    if output_format == "json":
        return json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if result.get("interpretation_report"):
        return result["text_report"] + "\n\n" + "=" * 80 + "\n\n" + result["interpretation_report"]
    return result.get("text_report", "")


def _configure_stdout() -> None:
    """Use UTF-8 stdout on Windows when available."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()
