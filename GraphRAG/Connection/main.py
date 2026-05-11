from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from DocLLM.LLMs.llm_init import configure_langsmith

try:
    from .connection_agents import json_safe
    from .connection_workflow import ConnectionWorkflow
except ImportError:  # pragma: no cover - supports direct script execution
    from connection_agents import json_safe
    from connection_workflow import ConnectionWorkflow


load_dotenv()
LANGSMITH_PROJECT_NAME = configure_langsmith(
    os.getenv("LANGSMITH_PROJECT", "GraphRAGConnection")
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "connection_results.json"


def load_payload(input_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input payload not found: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def save_result(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_connection_workflow(
    payload: dict[str, Any],
    stage: Literal["rerank", "extract", "auto"] = "auto",
    batch_size: int | None = None,
    use_placeholder_llm: bool | None = None,
) -> dict[str, Any]:
    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")
    workflow = ConnectionWorkflow(use_placeholder=use_placeholder_llm)
    return workflow.run(payload, stage=stage, batch_size=batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GraphRAG Connection workflow.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a JSON Connection payload.",
    )
    parser.add_argument(
        "--stage",
        choices=["rerank", "extract", "auto"],
        default="auto",
        help="Workflow stage to run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for the selected stage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the JSON output file.",
    )
    parser.add_argument(
        "--placeholder-llm",
        action="store_true",
        help="Use local placeholder agents instead of calling an LLM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_payload(args.input)
    result = run_connection_workflow(
        payload=payload,
        stage=args.stage,
        batch_size=args.batch_size,
        use_placeholder_llm=args.placeholder_llm if args.placeholder_llm else None,
    )
    save_result(result, args.output)
    print(f"[INFO] Wrote Connection result to: {args.output}")


if __name__ == "__main__":
    main()
