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


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def query_fields_from_payload(payload: dict[str, Any], query_text: str) -> dict[str, str]:
    """Build structured query fields for judge prompts.

    Human-review files may only contain query_text, so fall back to the
    GraphRAG query naming convention: "<function> with <mode>" or
    "<function> with effect <effect>".
    """
    function_text = normalize_text(payload.get("function_text"))
    query_mode = normalize_text(payload.get("query_mode"))
    effect_text = normalize_text(payload.get("effect_text"))
    query_type = normalize_text(payload.get("query_type")).lower()

    if not query_type:
        lowered = query_text.lower()
        if " with effect " in lowered:
            query_type = "effect"
        elif " with " in lowered:
            query_type = "function_mode"
        else:
            query_type = "unknown"

    if (not function_text or not query_mode) and query_type == "function_mode" and " with " in query_text:
        function_text, query_mode = [part.strip() for part in query_text.split(" with ", 1)]

    if (not function_text or not effect_text) and query_type == "effect" and " with effect " in query_text.lower():
        marker_index = query_text.lower().find(" with effect ")
        function_text = function_text or query_text[:marker_index].strip()
        effect_text = effect_text or query_text[marker_index + len(" with effect ") :].strip()

    target_text = query_mode if query_type == "function_mode" else effect_text if query_type == "effect" else query_text

    return {
        "query_type": query_type,
        "query_text": query_text,
        "function_text": function_text,
        "query_mode": query_mode,
        "effect_text": effect_text,
        "target_text": target_text,
    }


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


def issue_type_counts(conflicts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for conflict in conflicts:
        issue_type = normalize_text(conflict.get("issue_type")) or "unknown"
        counts[issue_type] = counts.get(issue_type, 0) + 1
    return dict(sorted(counts.items()))


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
    query_fields = query_fields_from_payload(payload, query_text) if isinstance(payload, dict) else {
        "query_type": "unknown",
        "query_text": query_text,
        "function_text": "",
        "query_mode": "",
        "effect_text": "",
        "target_text": query_text,
    }
    source_summary = payload.get("summary")
    if not isinstance(source_summary, dict):
        source_summary = {}
    chunks = payload.get("chunks") or []
    if not isinstance(chunks, list):
        chunks = []

    conflicts: list[dict[str, Any]] = []
    judge_errors: list[dict[str, Any]] = []
    valid_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
    support_suspect_chunks = [
        chunk for chunk in valid_chunks if normalize_tag(chunk.get("rerank_tag")) in SUPPORT_SUSPECT_TAGS
    ]
    irrelevant_chunks = [
        chunk for chunk in valid_chunks if normalize_tag(chunk.get("rerank_tag")) == "irrelevant"
    ]

    if support_suspect_chunks:
        try:
            conflicts.extend(
                support_suspect_agent.judge_chunks(
                    query_text,
                    support_suspect_chunks,
                    query_fields=query_fields,
                )
            )
        except Exception as exc:
            judge_errors.append(judge_error_item(support_suspect_chunks, "support_suspect", exc))

    if irrelevant_chunks:
        try:
            conflicts.extend(
                irrelevant_agent.judge_chunks(
                    query_text,
                    irrelevant_chunks,
                    query_fields=query_fields,
                )
            )
        except Exception as exc:
            judge_errors.append(judge_error_item(irrelevant_chunks, "irrelevant", exc))

    output_summary = {
        **source_summary,
        "total_chunks": len(valid_chunks),
        "support_suspect_checked": len(support_suspect_chunks),
        "irrelevant_checked": len(irrelevant_chunks),
        "conflict_count": len(conflicts),
        "issue_type_counts": issue_type_counts(conflicts),
        "judge_error_count": len(judge_errors),
    }

    return {
        "source_file": input_path.name,
        "query_text": query_text,
        "summary": output_summary,
        "conflicts": conflicts,
        "judge_errors": judge_errors,
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
