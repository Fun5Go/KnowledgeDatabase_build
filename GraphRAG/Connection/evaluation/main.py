from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from DocLLM.LLMs.llm_init import configure_langsmith

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

try:
    from .irrelevant_judge_agent import IrrelevantJudgeAgent
    from .support_suspect_judge_agent import SupportSuspectJudgeAgent
except ImportError:  # pragma: no cover - supports direct script execution
    from irrelevant_judge_agent import IrrelevantJudgeAgent
    from support_suspect_judge_agent import SupportSuspectJudgeAgent


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "human_review_by_query"
OUTPUT_DIR = SCRIPT_DIR / "llm_judge_conflicts"
LANGSMITH_PROJECT_NAME = configure_langsmith("Verification")

SUPPORT_SUSPECT_TAGS = {"support", "suspect"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def normalize_tag(value: Any) -> str:
    return str(value or "").strip().lower()


def chunk_rank(chunk: dict[str, Any]) -> Any:
    return chunk.get("rank")


def judge_error_item(chunks: list[dict[str, Any]], agent_name: str, error: Exception) -> dict[str, Any]:
    """Record group judge failures without leaking raw_text into the output JSON."""
    return {
        "rank": None,
        "ranks": [chunk_rank(chunk) for chunk in chunks],
        "issue_type": "judge_error",
        "agent": agent_name,
        "rerank_tag": agent_name,
        "comments": str(error),
    }


@traceable(
    run_type="chain",
    name="verification_judge_file",
    tags=["verification", "judge", "file"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def judge_file(
    input_path: Path,
    support_suspect_agent: SupportSuspectJudgeAgent,
    irrelevant_agent: IrrelevantJudgeAgent,
) -> dict[str, Any]:
    payload = load_json(input_path)
    query_text = str(payload.get("query_text") or "").strip()
    chunks = payload.get("chunks") or []
    if not isinstance(chunks, list):
        chunks = []

    conflicts: list[dict[str, Any]] = []
    valid_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
    support_suspect_chunks = [
        chunk for chunk in valid_chunks if normalize_tag(chunk.get("rerank_tag")) in SUPPORT_SUSPECT_TAGS
    ]
    irrelevant_chunks = [
        chunk for chunk in valid_chunks if normalize_tag(chunk.get("rerank_tag")) == "irrelevant"
    ]

    if support_suspect_chunks:
        try:
            conflicts.extend(support_suspect_agent.judge_chunks(query_text, support_suspect_chunks))
        except Exception as exc:
            conflicts.append(judge_error_item(support_suspect_chunks, "support_suspect", exc))

    if irrelevant_chunks:
        try:
            conflicts.extend(irrelevant_agent.judge_chunks(query_text, irrelevant_chunks))
        except Exception as exc:
            conflicts.append(judge_error_item(irrelevant_chunks, "irrelevant", exc))

    return {
        "source_file": input_path.name,
        "query_text": query_text,
        "summary": {
            "total_chunks": len(valid_chunks),
            "support_suspect_checked": len(support_suspect_chunks),
            "irrelevant_checked": len(irrelevant_chunks),
            "conflict_count": len(conflicts),
        },
        "conflicts": conflicts,
    }


def iter_input_files(input_path: Path) -> list[Path]:
    """Accept either one JSON file or a directory of JSON files."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Input file must be a .json file: {input_path}")
        return [input_path]
    return sorted(path for path in input_path.glob("*.json") if path.is_file())


def run(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR, max_retries: int = 3) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    support_suspect_agent = SupportSuspectJudgeAgent(max_retries=max_retries)
    irrelevant_agent = IrrelevantJudgeAgent(max_retries=max_retries)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = iter_input_files(input_dir)
    if not input_files:
        print(f"No JSON input files found in {input_dir}")
        return

    print(f"[INFO] LangSmith project: {LANGSMITH_PROJECT_NAME}")
    for input_path in input_files:
        result = judge_file(input_path, support_suspect_agent, irrelevant_agent)
        output_path = output_dir / f"{input_path.stem}.judge_conflicts.json"
        write_json(output_path, result)
        print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM judges over human-review connection chunks.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(input_dir=args.input_dir, output_dir=args.output_dir, max_retries=args.max_retries)


if __name__ == "__main__":
    main()
