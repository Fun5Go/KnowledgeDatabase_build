from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CONNECTION_DIR = SCRIPT_DIR.parents[1]
DEFAULT_RAW_DIR = CONNECTION_DIR / "rerank_top60_rawtext_nohighrecall_batch30"
DEFAULT_INTEGRATE_DIR = CONNECTION_DIR / "rerank_top60_integratedtext_nohighrecall_batch30"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "failure_text_support_groundtruth_template.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def normalize_key(value: Any) -> str:
    return " ".join(normalize_text(value).casefold().split())


def analysis_item(payload: dict[str, Any]) -> dict[str, Any]:
    item = payload.get("analysis_item")
    return item if isinstance(item, dict) else {}


def connection(payload: dict[str, Any]) -> dict[str, Any]:
    item = payload.get("connection")
    return item if isinstance(item, dict) else {}


def query_text(payload: dict[str, Any]) -> str:
    item = analysis_item(payload)
    text = normalize_text(item.get("query_text"))
    if text:
        return text

    query_result = payload.get("query_result")
    query_spec = query_result.get("query_spec") if isinstance(query_result, dict) else {}
    if isinstance(query_spec, dict):
        for key in ("query_text", "target_text", "sentence"):
            text = normalize_text(query_spec.get(key))
            if text:
                return text
    return ""


def failure_text(payload: dict[str, Any]) -> str:
    item = analysis_item(payload)
    for key in ("query_mode", "query_cause", "query_effect", "query_text"):
        text = normalize_text(item.get(key))
        if text:
            return text
    return query_text(payload)


def query_type(payload: dict[str, Any]) -> str:
    return normalize_text(
        analysis_item(payload).get("query_type") or connection(payload).get("query_type")
    )


def chunk_id(chunk: dict[str, Any], fallback_index: int) -> str:
    return normalize_text(
        chunk.get("name") or chunk.get("chunk_id") or chunk.get("node_id") or f"support_{fallback_index}"
    )


def support_chunks(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    chunks = connection(payload).get("reranked_chunks")
    if not isinstance(chunks, list):
        chunks = payload.get("reranked_chunks")
    if not isinstance(chunks, list):
        return {}

    supports: dict[str, dict[str, Any]] = {}
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        if normalize_text(chunk.get("rerank_tag")).casefold() != "support":
            continue

        key = chunk_id(chunk, index)
        supports[key] = {
            "rank": chunk.get("rank"),
            "name": normalize_text(chunk.get("name")),
            "section_tag": normalize_text(chunk.get("section_tag")),
            "raw_text": normalize_text(chunk.get("raw_text") or chunk.get("text")),
            "rerank_tag": normalize_text(chunk.get("rerank_tag")),
            "reason": normalize_text(chunk.get("reason")),
        }
    return supports


def load_support_by_failure(folder: Path) -> dict[str, dict[str, Any]]:
    support_by_failure: dict[str, dict[str, Any]] = {}
    if not folder.exists():
        raise FileNotFoundError(f"Rerank folder not found: {folder}")

    for path in sorted(folder.glob("*.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue

        text = failure_text(payload)
        if not text:
            continue

        key = normalize_key(text)
        entry = support_by_failure.setdefault(
            key,
            {
                "failure_text": text,
                "query_text": query_text(payload),
                "query_type": query_type(payload),
                "analysis_id": normalize_text(
                    analysis_item(payload).get("analysis_id")
                    or connection(payload).get("analysis_id")
                ),
                "support": {},
            },
        )
        entry["support"].update(support_chunks(payload))

    return support_by_failure


def build_ground_truth_template(raw_dir: Path, integrate_dir: Path) -> list[dict[str, Any]]:
    raw_by_failure = load_support_by_failure(raw_dir)
    integrate_by_failure = load_support_by_failure(integrate_dir)
    all_keys = sorted(
        set(raw_by_failure) | set(integrate_by_failure),
        key=lambda key: (
            raw_by_failure.get(key, integrate_by_failure.get(key, {})).get("query_type", ""),
            raw_by_failure.get(key, integrate_by_failure.get(key, {})).get("failure_text", ""),
        ),
    )

    output: list[dict[str, Any]] = []
    for key in all_keys:
        base = raw_by_failure.get(key) or integrate_by_failure.get(key) or {}
        output.append(
            {
                "failure_text": base.get("failure_text", ""),
                "query_text": base.get("query_text", ""),
                "query_type": base.get("query_type", ""),
                "analysis_id": base.get("analysis_id", ""),
                "ground_truth": {
                    "raw": raw_by_failure.get(key, {}).get("support", {}),
                    "integrate": integrate_by_failure.get(key, {}).get("support", {}),
                },
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a failure-text ground-truth template from rerank support chunks."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--integrate-dir", type=Path, default=DEFAULT_INTEGRATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_ground_truth_template(args.raw_dir, args.integrate_dir)
    write_json(args.output, output)
    raw_count = sum(len(item["ground_truth"]["raw"]) for item in output)
    integrate_count = sum(len(item["ground_truth"]["integrate"]) for item in output)
    print(f"[INFO] Wrote {len(output)} failure text item(s) to: {args.output}")
    print(f"[INFO] raw support chunks: {raw_count}")
    print(f"[INFO] integrate support chunks: {integrate_count}")


if __name__ == "__main__":
    main()
