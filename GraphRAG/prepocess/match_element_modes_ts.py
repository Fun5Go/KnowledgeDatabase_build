from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = CURRENT_DIR / "output"
DEFAULT_TAXONOMY_FILE = CURRENT_DIR / "fmea_mode_cause_taxonomy.json"
DEFAULT_ELEMENT_NAME = "Motor control"

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
    "to",
    "too",
    "with",
}

ELEMENT_ALIAS_MAP = {
    "motor control": ["motor", "compressor motor", "starter control"],
    "power supply": ["psu", "power", "voltage supply"],
    "user interface": ["ui", "display", "button", "led"],
    "housing": ["enclosure", "casing", "housing"],
}

FUNCTION_ALIAS_MAP = {
    "soft starter": ["soft start", "starter", "motor start"],
    "zero crossing detection": ["zero crossing", "zero cross", "zcd", "detection"],
    "relay switching": ["relay", "switching", "switch", "solid state relay", "ssr"],
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\xa0", " ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[()\\[\\],:;]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def expand_token_variants(token: str) -> List[str]:
    variants = [token]

    if len(token) > 4 and token.endswith("ies"):
        variants.append(token[:-3] + "y")
    elif len(token) > 4 and token.endswith("es"):
        variants.append(token[:-2])
    elif len(token) > 3 and token.endswith("s"):
        variants.append(token[:-1])

    return unique_preserve_order(variants)


def tokenize_text(text: str) -> List[str]:
    normalized = normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    expanded_tokens: List[str] = []
    for token in raw_tokens:
        if not token or token in STOPWORDS:
            continue
        expanded_tokens.extend(expand_token_variants(token))
    return unique_preserve_order(expanded_tokens)


def token_set(text: str) -> Set[str]:
    return set(tokenize_text(text))


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def expand_tokens(text: str, alias_map: Dict[str, List[str]]) -> Set[str]:
    normalized_text = normalize_text(text)
    expanded = set(tokenize_text(text))
    for key, aliases in alias_map.items():
        if key in normalized_text:
            for alias in aliases:
                expanded.update(tokenize_text(alias))
    return expanded


def compute_overlap(text_tokens: Set[str], candidate_tokens: Set[str]) -> Dict[str, Any]:
    matched_tokens = sorted(text_tokens & candidate_tokens)
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


def is_label_match(text: str, label: str, label_tokens: Set[str], threshold: float) -> bool:
    if not label_tokens:
        return False

    normalized_text = normalize_text(text)
    normalized_label = normalize_text(label)
    if normalized_label and normalized_label in normalized_text:
        return True

    overlap = compute_overlap(token_set(text), label_tokens)
    if overlap["match_count"] == 0:
        return False

    if len(label_tokens) <= 3:
        return True

    return overlap["coverage"] >= threshold


def get_mode_specific_tokens(
    mode: str,
    element_tokens: Set[str],
    function_tokens: Set[str],
) -> Set[str]:
    mode_tokens = token_set(mode)
    specific_tokens = mode_tokens - element_tokens - function_tokens
    if specific_tokens:
        return specific_tokens
    specific_tokens = mode_tokens - function_tokens
    if specific_tokens:
        return specific_tokens
    return mode_tokens


def is_mode_match(
    text: str,
    mode: str,
    element_tokens: Set[str],
    function_tokens: Set[str],
    text_tokens: Set[str],
) -> Dict[str, Any]:
    normalized_text = normalize_text(text)
    normalized_mode = normalize_text(mode)
    specific_tokens = get_mode_specific_tokens(mode, element_tokens, function_tokens)

    if normalized_mode and normalized_mode in normalized_text:
        overlap = compute_overlap(text_tokens, specific_tokens)
        return {
            "matched": True,
            "mode_specific_terms": sorted(specific_tokens),
            "matched_mode_terms": overlap["matched_tokens"],
            "mode_match_coverage": overlap["coverage"],
        }

    overlap = compute_overlap(text_tokens, specific_tokens)
    if overlap["match_count"] == 0:
        return {
            "matched": False,
            "mode_specific_terms": sorted(specific_tokens),
            "matched_mode_terms": [],
            "mode_match_coverage": 0.0,
        }

    if len(specific_tokens) == 1:
        return {
            "matched": True,
            "mode_specific_terms": sorted(specific_tokens),
            "matched_mode_terms": overlap["matched_tokens"],
            "mode_match_coverage": overlap["coverage"],
        }

    return {
        "matched": overlap["coverage"] >= 0.5,
        "mode_specific_terms": sorted(specific_tokens),
        "matched_mode_terms": overlap["matched_tokens"],
        "mode_match_coverage": overlap["coverage"],
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ts_chunks(input_dir: Path) -> List[Dict[str, Any]]:
    path = input_dir / "ts_chunks.json"
    if not path.exists():
        return []

    rows = load_json(path)
    for row in rows:
        row["chunk_type"] = "TSChunk"
    return rows


def load_motor_control_taxonomy(path: Path, element_name: str) -> Dict[str, Any]:
    elements = load_json(path)
    for item in elements:
        if item.get("failure_element") == element_name:
            return item
    raise ValueError(f"Could not find failure element '{element_name}' in taxonomy.")


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


def match_text_to_element(text: str, taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    text_tokens = token_set(text)

    element_name = taxonomy.get("failure_element", "")
    element_tokens = expand_tokens(element_name, ELEMENT_ALIAS_MAP)
    element_overlap = compute_overlap(text_tokens, element_tokens)

    for function_name, modes in taxonomy.get("functions", {}).items():
        function_tokens = expand_tokens(function_name, FUNCTION_ALIAS_MAP)
        combined_function_tokens = unique_preserve_order(list(element_tokens | function_tokens))
        combined_function_token_set = set(combined_function_tokens)

        if not is_label_match(text, function_name, combined_function_token_set, threshold=0.25):
            continue

        function_overlap = compute_overlap(text_tokens, combined_function_token_set)
        matched_modes: List[Dict[str, Any]] = []

        for mode in modes:
            mode_result = is_mode_match(
                text=text,
                mode=mode,
                element_tokens=element_tokens,
                function_tokens=function_tokens,
                text_tokens=text_tokens,
            )
            if not mode_result["matched"]:
                continue

            matched_modes.append(
                {
                    "mode": mode,
                    "mode_specific_terms": mode_result["mode_specific_terms"],
                    "matched_mode_terms": mode_result["matched_mode_terms"],
                    "mode_match_coverage": mode_result["mode_match_coverage"],
                }
            )

        matches.append(
            {
                "failure_element": element_name,
                "matched_element_terms": element_overlap["matched_tokens"],
                "element_match_coverage": element_overlap["coverage"],
                "function": function_name,
                "matched_function_terms": function_overlap["matched_tokens"],
                "function_match_coverage": function_overlap["coverage"],
                "candidate_modes": modes,
                "matched_modes": matched_modes,
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
        match_text = build_match_text(node)
        matches = match_text_to_element(match_text, taxonomy)
        if not matches and not keep_unmatched:
            continue

        records.append(
            {
                "chunk_type": node.get("chunk_type", ""),
                "node_name": node.get("name", ""),
                "node_text": node_text,
                "match_text": match_text,
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
        description="Match failure element functions and modes against TSChunk node text."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing ts_chunks.json.",
    )
    parser.add_argument(
        "--taxonomy-file",
        type=Path,
        default=DEFAULT_TAXONOMY_FILE,
        help="JSON file describing failure elements, functions, modes, and causes.",
    )
    parser.add_argument(
        "--element-name",
        default=DEFAULT_ELEMENT_NAME,
        help="Failure element to match, for example 'Motor control' or 'Power Supply'.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Where to write TSChunk node-level mode matching results. Defaults to <element>_ts_mode_matches.json.",
    )
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Keep TSChunk nodes without any matched function.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_ts_chunks(args.input_dir.resolve())
    taxonomy = load_motor_control_taxonomy(
        path=args.taxonomy_file.resolve(),
        element_name=args.element_name,
    )
    records = build_node_records(
        chunks=chunks,
        taxonomy=taxonomy,
        keep_unmatched=args.keep_unmatched,
    )

    if args.output_file is None:
        safe_element = normalize_text(args.element_name).replace(" ", "_")
        output_file = CURRENT_DIR / "output" / f"{safe_element}_ts_mode_matches.json"
    else:
        output_file = args.output_file.resolve()

    write_json(records, output_file)
    print(f"Loaded {len(chunks)} TSChunk nodes")
    print(f"Wrote {len(records)} node records to {output_file}")


if __name__ == "__main__":
    main()
