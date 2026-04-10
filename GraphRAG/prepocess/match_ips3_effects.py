from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = CURRENT_DIR / "output"
DEFAULT_OUTPUT_FILE = CURRENT_DIR / "output" / "ips3_sentence_effect_matches.json"
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
    "ble": ["bluetooth"],
    "user interface": ["ui", "display", "led", "button"],
    "power outputs": ["power output", "output", "outputs", "relay"],
    "digital inputs": ["digital input", "input", "inputs"],
    "soft starter": ["soft start", "starter", "motor start"],
    "pressure switch": ["pressure", "switch", "setpoint"],
    "internal logging": ["logging", "logs", "log", "timestamp"],
    "data collection": ["statistics", "data", "measurement"],
    "pneumatic connection": ["pneumatic", "pressure", "hose", "tube"],
    "panel mounting": ["mount", "mounting", "panel", "bracket"],
    "cable cable connector": ["cable", "connector", "connection", "plug"],
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


def split_into_sentences(text: str) -> List[str]:
    if not text or not text.strip():
        return []

    parts = re.split(r"[\n\r]+|(?<=[.!?;])\s+", text.strip())
    sentences: List[str] = []
    for part in parts:
        cleaned = part.strip(" -\t")
        if cleaned:
            sentences.append(cleaned)
    return sentences


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(input_dir: Path) -> List[Dict[str, Any]]:
    datasets: List[Tuple[str, Path]] = [
        ("FSChunk", input_dir / "fs_chunks.json"),
        ("TSChunk", input_dir / "ts_chunks.json"),
    ]
    merged: List[Dict[str, Any]] = []

    for chunk_type, path in datasets:
        if not path.exists():
            continue

        rows = load_json(path)
        for row in rows:
            row["chunk_type"] = chunk_type
            merged.append(row)

    return merged


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


def is_effect_match(
    sentence: str,
    effect: str,
    sentence_tokens: Set[str],
    effect_tokens: Set[str],
) -> bool:
    if not effect_tokens:
        return False

    normalized_sentence = normalize_text(sentence)
    normalized_effect = normalize_text(effect)
    if normalized_effect and normalized_effect in normalized_sentence:
        return True

    overlap = sentence_tokens & effect_tokens
    if not overlap:
        return False

    if len(effect_tokens) == 1:
        return True

    coverage = len(overlap) / len(effect_tokens)
    return coverage >= 0.4


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


def collect_text_sources(node: Dict[str, Any]) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []

    if node.get("text"):
        sources.append(
            {
                "source_type": "node_text",
                "source_name": node.get("name", ""),
                "text": node["text"],
            }
        )

    for rationale in node.get("rationales", []) or []:
        if rationale.get("text"):
            sources.append(
                {
                    "source_type": "rationale",
                    "source_name": rationale.get("name", ""),
                    "text": rationale["text"],
                }
            )

    return sources


def match_sentence(sentence: str, taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    sentence_tokens = token_set(sentence)

    for function_item in taxonomy.get("functions", []):
        product_function = function_item.get("product_function", "")
        function_tokens = expand_function_tokens(product_function)
        if not is_function_match(
            sentence=sentence,
            product_function=product_function,
            function_tokens=function_tokens,
        ):
            continue

        function_overlap = compute_overlap(sentence_tokens, function_tokens)

        for effect in function_item.get("effects", []):
            effect_tokens = token_set(effect)
            if not is_effect_match(
                sentence=sentence,
                effect=effect,
                sentence_tokens=sentence_tokens,
                effect_tokens=effect_tokens,
            ):
                continue

            effect_overlap = compute_overlap(sentence_tokens, effect_tokens)
            matches.append(
                {
                    "product": taxonomy.get("product", ""),
                    "product_function": product_function,
                    "matched_function_terms": function_overlap["matched_tokens"],
                    "effect": effect,
                    "matched_effect_terms": effect_overlap["matched_tokens"],
                    "matched_terms": unique_preserve_order(
                        function_overlap["matched_tokens"] + effect_overlap["matched_tokens"]
                    ),
                    "effect_match_coverage": effect_overlap["coverage"],
                }
            )

    return matches


def build_sentence_records(
    chunks: List[Dict[str, Any]],
    taxonomy: Dict[str, Any],
    keep_unmatched: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for node in chunks:
        for source in collect_text_sources(node):
            for sentence in split_into_sentences(source["text"]):
                matches = match_sentence(sentence, taxonomy)
                if not matches and not keep_unmatched:
                    continue

                records.append(
                    {
                        "product": taxonomy.get("product", ""),
                        "chunk_type": node.get("chunk_type", ""),
                        "node_name": node.get("name", ""),
                        "source_type": source["source_type"],
                        "source_name": source["source_name"],
                        "sentence": sentence,
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
        description="Match effect lists against exported FSChunk and TSChunk sentences using general text preprocessing."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing fs_chunks.json and ts_chunks.json.",
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
        help="Where to write sentence-level effect matching results.",
    )
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Keep sentences without any matched effect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_chunks(args.input_dir.resolve())
    taxonomy = load_json(args.taxonomy_file.resolve())
    records = build_sentence_records(
        chunks=chunks,
        taxonomy=taxonomy,
        keep_unmatched=args.keep_unmatched,
    )

    write_json(records, args.output_file.resolve())
    print(f"Loaded {len(chunks)} chunk nodes")
    print(f"Wrote {len(records)} sentence records to {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
