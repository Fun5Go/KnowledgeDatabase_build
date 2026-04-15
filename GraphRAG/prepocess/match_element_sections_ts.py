from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = CURRENT_DIR / "output"
DEFAULT_TAXONOMY_FILE = CURRENT_DIR / "fmea_mode_cause_taxonomy.json"
DEFAULT_ELEMENT_NAME = "Power Supply"

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
}

# FUNCTION_TOKEN_EXCLUDE_MAP = {
#     "soft starter": {"start", "motor"},
# }


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("\xa0", " ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[()\[\],:;]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


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


def expand_tokens(text: str, alias_map: Dict[str, List[str]]) -> Set[str]:
    normalized_text = normalize_text(text)
    expanded = set(tokenize_text(text))
    for key, aliases in alias_map.items():
        if key in normalized_text:
            for alias in aliases:
                expanded.update(tokenize_text(alias))
    return expanded


def get_aliases(text: str, alias_map: Dict[str, List[str]]) -> List[str]:
    normalized_text = normalize_text(text)
    aliases = [text]
    for key, values in alias_map.items():
        if key in normalized_text:
            aliases.extend(values)
    return unique_preserve_order([item.strip() for item in aliases if item and item.strip()])


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


def contains_alias_phrase(normalized_text: str, alias: str) -> bool:
    normalized_alias = normalize_text(alias)
    if not normalized_alias:
        return False

    pattern = r"(?<![a-z0-9])" + re.escape(normalized_alias) + r"(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def is_function_text_match(
    text: str,
    function_aliases: List[str],
    function_tokens: Set[str],
) -> Dict[str, Any]:
    if not function_aliases and not function_tokens:
        return {
            "matched": False,
            "coverage": 0.0,
        }

    normalized_text = normalize_text(text)
    text_tokens = token_set(text)

    for alias in function_aliases:
        if contains_alias_phrase(normalized_text, alias):
            alias_tokens = token_set(alias)
            overlap = compute_overlap(text_tokens, alias_tokens or function_tokens)
            return {
                "matched": True,
                "coverage": overlap["coverage"],
            }

    overlap = compute_overlap(text_tokens, function_tokens)
    if overlap["match_count"] == 0:
        return {
            "matched": False,
            "coverage": 0.0,
        }

    if len(function_tokens) <= 3:
        return {
            "matched": True,
            "coverage": overlap["coverage"],
        }

    return {
        "matched": overlap["coverage"] >= 0.25,
        "coverage": overlap["coverage"],
    }


def build_function_definitions(taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
    function_defs: List[Dict[str, Any]] = []
    for function_name in taxonomy.get("functions", {}).keys():
        normalized_function = normalize_text(function_name)
        function_tokens = expand_tokens(function_name, FUNCTION_ALIAS_MAP)
    
        function_defs.append(
            {
                "function": function_name,
                "tokens": function_tokens,
                "aliases": get_aliases(function_name, FUNCTION_ALIAS_MAP),
            }
        )
    return function_defs


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


def load_element_taxonomy(path: Path, element_name: str) -> Dict[str, Any]:
    elements = load_json(path)
    for item in elements:
        if item.get("failure_element") == element_name:
            return item
    raise ValueError(f"Could not find failure element '{element_name}' in taxonomy.")


def build_rationale_text(row: Dict[str, Any]) -> str:
    rationale_parts: List[str] = []
    for rationale in row.get("rationales", []) or []:
        rationale_text = (rationale.get("text") or "").strip()
        if rationale_text:
            rationale_parts.append(rationale_text)
    return "\n".join(unique_preserve_order(rationale_parts))


def build_section_records(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in chunks:
        section_tag = (row.get("section_tag") or "").strip()
        if not section_tag:
            continue

        discipline = (row.get("discipline") or "").strip()
        key = (section_tag, discipline)

        if key not in grouped:
            grouped[key] = {
                "section_tag": section_tag,
                "discipline": discipline,
            }

    records = list(grouped.values())
    records.sort(key=lambda row: (row["discipline"], row["section_tag"]))
    return records


def build_chunk_records(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in chunks:
        section_tag = (row.get("section_tag") or "").strip()
        if not section_tag:
            continue

        text = (row.get("text") or "").strip()
        rationale_text = build_rationale_text(row)
        match_text = "\n".join([part for part in [text, rationale_text] if part.strip()])

        records.append(
            {
                "chunk_name": (row.get("name") or "").strip(),
                "text": text,
                "rationale_text": rationale_text,
                "match_text": match_text,
                "section_tag": section_tag,
                "discipline": (row.get("discipline") or "").strip(),
            }
        )

    records.sort(key=lambda row: (row["discipline"], row["section_tag"], row["chunk_name"]))
    return records


def match_sections_to_element(
    section_records: List[Dict[str, Any]],
    chunk_records: List[Dict[str, Any]],
    taxonomy: Dict[str, Any],
    use_section_match: bool = True,
) -> List[Dict[str, Any]]:
    element_name = taxonomy.get("failure_element", "")
    element_tokens = expand_tokens(element_name, ELEMENT_ALIAS_MAP)
    function_defs = build_function_definitions(taxonomy)
    candidate_sections: Set[Tuple[str, str]] = set()

    if use_section_match:
        for section in section_records:
            section_tag = section.get("section_tag", "")
            discipline = section.get("discipline", "")
            if not section_tag:
                continue

            if is_label_match(
                section_tag,
                element_name,
                element_tokens,
                threshold=0.25,
            ):
                candidate_sections.add((section_tag, discipline))
                continue

            for function_def in function_defs:
                if is_label_match(
                    section_tag,
                    function_def["function"],
                    function_def["tokens"],
                    threshold=0.25,
                ):
                    candidate_sections.add((section_tag, discipline))
                    break
    else:
        candidate_sections = {
            (chunk.get("section_tag", ""), chunk.get("discipline", ""))
            for chunk in chunk_records
            if chunk.get("section_tag", "")
        }

    results: List[Dict[str, Any]] = []

    for chunk in chunk_records:
        section_key = (chunk.get("section_tag", ""), chunk.get("discipline", ""))
        if section_key not in candidate_sections:
            continue

        chunk_matches: List[Dict[str, Any]] = []
        for function_def in function_defs:
            chunk_match = is_function_text_match(
                chunk.get("match_text", ""),
                function_aliases=function_def["aliases"],
                function_tokens=function_def["tokens"],
            )
            if not chunk_match["matched"]:
                continue

            chunk_matches.append(
                {
                    "function": function_def["function"],
                    "function_match_coverage": chunk_match["coverage"],
                }
            )

        matched_functions = [item["function"] for item in chunk_matches]
        if not matched_functions:
            continue

        element_overlap = compute_overlap(token_set(chunk.get("section_tag", "")), element_tokens)
        results.append(
            {
                "failure_element": element_name,
                "section_tag": chunk.get("section_tag", ""),
                "discipline": chunk.get("discipline", ""),
                "chunk_name": chunk.get("chunk_name", ""),
                "text": chunk.get("text", ""),
                "rationale_text": chunk.get("rationale_text", ""),
                "match_text": chunk.get("match_text", ""),
                "matched_functions": matched_functions,
                "function_matches": chunk_matches,
                "element_match_coverage": element_overlap["coverage"],
            }
        )

    return results


def write_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optionally filter TS chunks by section_tag first, then output chunk_name with matched functions based on full chunk text."
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
        help="Failure element to match, for example 'Motor control'.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Where to write chunk matching results. Defaults to <element>_ts_chunk_function_matches.json.",
    )
    parser.add_argument(
        "--disable-section-match",
        action="store_true",
        help="Disable section_tag pre-filtering and traverse all TS nodes instead.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_ts_chunks(args.input_dir.resolve())
    taxonomy = load_element_taxonomy(
        path=args.taxonomy_file.resolve(),
        element_name=args.element_name,
    )
    section_records = build_section_records(chunks)
    chunk_records = build_chunk_records(chunks)
    results = match_sections_to_element(
        section_records,
        chunk_records,
        taxonomy,
        use_section_match=not args.disable_section_match,
    )

    if args.output_file is None:
        safe_element = normalize_text(args.element_name).replace(" ", "_")
        output_file = CURRENT_DIR / "output" / f"{safe_element}_ts_chunk_function_matches.json"
    else:
        output_file = args.output_file.resolve()

    write_json(results, output_file)
    print(f"Loaded {len(chunks)} TSChunk rows")
    print(f"Built {len(section_records)} unique TS sections")
    print(f"Section match enabled: {not args.disable_section_match}")
    print(f"Wrote {len(results)} chunk match records to {output_file}")


if __name__ == "__main__":
    main()
