from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = CURRENT_DIR / "output"
DEFAULT_OUTPUT_FILE = CURRENT_DIR / "output" / "ips3_fs_node_function_matches.json"
DEFAULT_TAXONOMY_FILE = CURRENT_DIR / "ips3_effect_taxonomy.json"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "in",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "the",
    "through",
    "to",
    "when",
    "while",
    "with",
}

FUNCTION_ALIAS_MAP = {
    "wi fi": ["wifi", "wi fi", "wireless"],
    "azure": ["cloud", "backend"],
    "ble": ["bluetooth","bluetooth low energy"],
    "user interface": ["ui", "display", "led", "button"],
    "power outputs": ["power output", "output", "outputs"],
    "digital inputs": ["digital input", "input", "inputs"],
    "soft starter": ["soft start", "starter", "motor start"],
    "pressure switch": ["pressure", "switch", "setpoint"],
    "internal logging": ["logging", "logs", "log", "timestamp"],
    "data collection": ["statistics", "data"],
    "pneumatic connection": ["pneumatic", "pressure", "hose", "tube"],
    "panel mounting": ["mount", "mounting", "panel", "bracket"],
    "cable cable connector": ["cable", "plug"],
    "electrical failure": ["electrical", "short", "voltage", "current"],
    "aesthetics": ["housing", "appearance", "cosmetic"],
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\xa0", " ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[()\\[\\],:]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_text(text: str) -> List[str]:
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [token for token in tokens if token and token not in STOPWORDS]


def token_set(text: str) -> Set[str]:
    return set(tokenize_text(text))


def expand_function_tokens(product_function: str) -> Set[str]:
    normalized_function = normalize_text(product_function)
    expanded = set(tokenize_text(product_function))
    for key, aliases in FUNCTION_ALIAS_MAP.items():
        if key in normalized_function:
            for alias in aliases:
                expanded.update(tokenize_text(alias))
    return expanded


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(input_dir: Path) -> List[Dict[str, Any]]:
    path = input_dir / "fs_chunks.json"
    if not path.exists():
        return []

    rows = load_json(path)
    for row in rows:
        row["chunk_type"] = "FSChunk"
    return rows


def build_match_text(node: Dict[str, Any]) -> str:
    parts: List[str] = []

    node_text = (node.get("text") or "").strip()
    if node_text:
        parts.append(node_text)

    for rationale in node.get("rationales", []) or []:
        rationale_text = (rationale.get("text") or "").strip()
        if rationale_text:
            parts.append(rationale_text)

    return "\n".join(parts)


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def compute_overlap(
    sentence_tokens: Set[str],
    candidate_tokens: Set[str],
) -> Dict[str, Any]:
    matched_tokens = sorted(sentence_tokens & candidate_tokens)
    if not candidate_tokens:
        return {
            "matched_tokens": [],
            "match_count": 0,
            "coverage": 0.0,
        }

    coverage = len(matched_tokens) / len(candidate_tokens)
    return {
        "matched_tokens": matched_tokens,
        "match_count": len(matched_tokens),
        "coverage": round(coverage, 4),
    }


def is_function_match(
    sentence: str,
    product_function: str,
    function_tokens: Set[str],
) -> bool:
    if not function_tokens:
        return True

    normalized_sentence = normalize_text(sentence)
    normalized_function = normalize_text(product_function)
    if normalized_function and normalized_function in normalized_sentence:
        return True

    sentence_tokens = token_set(sentence)
    overlap = sentence_tokens & function_tokens
    if not overlap:
        return False

    if len(function_tokens) <= 3:
        return True

    coverage = len(overlap) / len(function_tokens)
    return coverage >= 0.25


def match_text(text: str, taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    text_tokens = token_set(text)

    for function_item in taxonomy.get("functions", []):
        product_function = function_item.get("product_function", "")
        function_tokens = expand_function_tokens(product_function)
        if not is_function_match(
            sentence=text,
            product_function=product_function,
            function_tokens=function_tokens,
        ):
            continue

        function_overlap = compute_overlap(text_tokens, function_tokens)
        matches.append(
            {
                "product": taxonomy.get("product", ""),
                "product_function": product_function,
                "matched_function_terms": function_overlap["matched_tokens"],
                "matched_terms": function_overlap["matched_tokens"],
                "function_match_coverage": function_overlap["coverage"],
                "candidate_effects": function_item.get("effects", []),
            }
        )

    return matches


def build_node_records(
    chunks: List[Dict[str, Any]],
    taxonomy: Dict[str, Any],
    keep_unmatched: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for node in chunks:
        node_text = node.get("text", "")
        match_text_value = build_match_text(node)
        matches = match_text(match_text_value, taxonomy)
        if not matches and not keep_unmatched:
            continue

        records.append(
            {
                "product": taxonomy.get("product", ""),
                "chunk_type": node.get("chunk_type", ""),
                "node_name": node.get("name", ""),
                "node_text": node_text,
                "match_text": match_text_value,
                "rationales": node.get("rationales", []),
                "matches": matches,
            }
        )

    return records


def write_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match product function text against complete FSChunk node text using general text preprocessing."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing fs_chunks.json.",
    )
    parser.add_argument(
        "--taxonomy-file",
        type=Path,
        default=DEFAULT_TAXONOMY_FILE,
        help="JSON file describing product functions and their effect lists.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Where to write FSChunk node-level function matching results.",
    )
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Keep FSChunk nodes without any matched function.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_chunks(args.input_dir.resolve())
    taxonomy = load_json(args.taxonomy_file.resolve())
    records = build_node_records(
        chunks=chunks,
        taxonomy=taxonomy,
        keep_unmatched=args.keep_unmatched,
    )

    write_json(records, args.output_file.resolve())
    print(f"Loaded {len(chunks)} FSChunk nodes")
    print(f"Wrote {len(records)} node records to {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
