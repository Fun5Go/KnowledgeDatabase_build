from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .main import DEFAULT_OUTPUT_PATH, list_query_items, run_chunk_selection_pipeline
except ImportError:  # pragma: no cover - supports direct script execution
    from main import DEFAULT_OUTPUT_PATH, list_query_items, run_chunk_selection_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: run one GraphRAG chunk-selection query by number."
    )
    parser.add_argument(
        "query_number",
        type=int,
        nargs="?",
        help="1-based query number from structure_input_motorcontrol.",
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Print available query numbers and exit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH.with_name("demo_chunk_selection_result.json"),
        help="Path for the demo JSON output file.",
    )
    parser.add_argument(
        "--placeholder-llm",
        action="store_true",
        help="Use placeholder selection instead of a real LLM call.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_queries:
        for item in list_query_items():
            print(
                f"{item['query_number']:>2}. "
                f"{item['query_type']} | "
                f"{item['query_text'].replace(chr(10), ' | ')}"
            )
        return

    if args.query_number is None:
        raise SystemExit("Please provide query_number, or use --list-queries.")

    results = run_chunk_selection_pipeline(
        output_path=args.output,
        use_placeholder_llm=args.placeholder_llm if args.placeholder_llm else None,
        query_number=args.query_number,
    )
    print(f"Wrote demo result for query {args.query_number} to {args.output}")
    print(f"Selected chunks: {len(results[0].get('selection', {}).get('top_chunks', []))}")


if __name__ == "__main__":
    main()
