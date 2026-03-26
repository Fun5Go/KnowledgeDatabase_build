#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FMEA cause text cleaning pipeline
================================

Purpose
-------
Clean short FMEA cause texts from records like:
    {"e.text": "Drive pin gets stuck"}

This script focuses on CLEANING / NORMALIZATION only.
It does NOT do clustering.

Main features
-------------
1. Keep original text
2. Basic normalization
3. Protected technical terms
4. Controlled typo correction
5. Phrase-level normalization
6. Simple multi-cause splitting
7. Canonical engineering-style phrase generation
8. Review flags for ambiguous / generic / low-quality records
9. Export to CSV / JSON

Input
-----
A JSON file containing either:
- a list of dicts: [{"e.text": "..."} , ...]
- or a dict with a list under a key

Output columns
--------------
- record_id
- parent_record_id
- split_index
- raw_text
- valid_text
- clean_text
- spelling_fixed_text
- normalized_text
- canonical_text
- split_flag
- review_flag
- review_reasons
- notes

Usage
-----
python clean_fmea_causes.py --input input.json --output_csv cleaned.csv --output_json cleaned.json

Optional:
python clean_fmea_causes.py --input input.json --text_key e.text

Requirements
------------
- Python 3.9+
- pandas

Install:
pip install pandas
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

INVALID_PLACEHOLDERS = {
    "",
    "-",
    "--",
    "---",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "?",
    "/",
}

# Terms that should generally NOT be altered by typo correction.
PROTECTED_TERMS = {
    "adc",
    "pwm",
    "fet",
    "can",
    "bt",
    "pcb",
    "cpu",
    "emi",
    "emc",
    "eld",
    "ksi",
    "ir",
    "gnd",
    "3v3",
    "+3v3",
    "t600a",
    "t600b",
    "motionsp",
    "dt",
    "hhi",
    "imx6",
}

# Very common typo / spelling fixes.
# Keep these conservative.
TYPO_REPLACEMENTS = {
    r"\bambiant\b": "ambient",
    r"\bstrenght\b": "strength",
    r"\bmaintenace\b": "maintenance",
    r"\bseperator\b": "separator",
    r"\buppon\b": "upon",
    r"\bunsufficient\b": "insufficient",
    r"\bto much\b": "too much",
    r"\batleast\b": "at least",
    r"\bdoesnt\b": "doesn't",
    r"\bdidnt\b": "didn't",
    r"\bmanuf\.\b": "manufacturing",
    r"\btemp\.\b": "temperature",
}

# Abbreviation expansion.
# Keep only high-confidence replacements.
ABBREVIATION_REPLACEMENTS = {
    r"\bsw\b": "software",
    r"\bbt\b": "bluetooth",
}

# Phrase-level normalization rules.
# Applied after typo fixing.
PHRASE_REPLACEMENTS = {
    r"\bgets stuck\b": "stuck",
    r"\bget stuck\b": "stuck",
    r"\bnot reviewed\b": "review missing",
    r"\bnot properly configured\b": "configuration incorrect",
    r"\bdoesn't actuate as expected\b": "actuation incorrect",
    r"\bdoes not actuate as expected\b": "actuation incorrect",
    r"\bnot saved properly\b": "save failure",
    r"\bdue to\b": "caused by",
    r"\bnot accurate enough\b": "accuracy insufficient",
    r"\btoo fast\b": "speed too high",
    r"\btoo slow\b": "speed too low",
    r"\btoo high\b": "high",
    r"\btoo low\b": "low",
    r"\bdidn't install according to manual\b": "installation not according to manual",
    r"\bdid not install according to manual\b": "installation not according to manual",
    r"\bdisplacement of sensors after maintenance\b": "sensor displacement after maintenance",
    r"\bdust in unlock system\b": "dust contamination in unlock system",
    r"\bdirt\b": "dirt contamination",
    r"\bdocument review missing\b": "document review missing",
    r"\bdocument not reviewed\b": "document review missing",
    r"\belectronics failure\b": "electronics failure",
    r"\bacts as idle antenna\b": "acts as antenna",
    r"\black of proper termination\b": "improper termination",
    r"\bemi noise\b": "emi noise",
}

# Patterns that often indicate generic / low-information texts.
GENERIC_PATTERNS = [
    r"^dirt contamination$",
    r"^dirt$",
    r"^dust contamination$",
    r"^emc$",
    r"^emc issue$",
    r"^emi$",
    r"^electronics failure$",
    r"^due to electronics failure$",
    r"^display drive strength$",
    r"^eld not allowed for error$",
]

# Patterns that indicate likely need for manual review.
AMBIGUOUS_PATTERNS = [
    r"\bor\b",
    r"\band\b",
    r"/",
    r",",
    r"\bissue\b",
    r"\berror\b",
]

# Heuristic component and mechanism vocabularies for simple canonical generation.
COMPONENT_KEYWORDS = [
    "motor",
    "drive motor",
    "sensor",
    "sensors",
    "display",
    "document",
    "pwm",
    "drive pwm",
    "drive pin",
    "driver pin",
    "lowering valve",
    "reverse valve",
    "select valve",
    "valve",
    "connector",
    "cable",
    "unlock system",
    "earth trace",
    "trace",
    "pcb",
    "input signal",
    "measurement signals",
    "signal",
    "file",
    "electronics",
]

MECHANISM_KEYWORDS = [
    "stuck high",
    "stuck",
    "configuration incorrect",
    "review missing",
    "save failure",
    "actuation incorrect",
    "damaged",
    "displacement",
    "speed too high",
    "speed too low",
    "accuracy insufficient",
    "failure",
    "noise",
    "emi",
    "emc",
    "acts as antenna",
    "improper termination",
    "routed",
]

# Prevent splitting on some technical patterns
DO_NOT_SPLIT_PATTERNS = [
    r"\bwrite/read\b",
    r"\bread/write\b",
    r"\bopen/short\b",
    r"\+3v3 and gnd pads\b",
]

# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------


@dataclass
class CleanedRecord:
    record_id: int
    parent_record_id: int
    split_index: int
    raw_text: str
    valid_text: bool
    clean_text: str
    spelling_fixed_text: str
    normalized_text: str
    canonical_text: str
    split_flag: bool
    review_flag: bool
    review_reasons: str
    notes: str


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def load_json_records(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_records(data: Any, text_key: str) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        # Direct single record
        if text_key in data:
            return [data]

        # Try first list value
        for _, value in data.items():
            if isinstance(value, list) and all(isinstance(i, dict) for i in value):
                return value

    raise ValueError("Could not find a list of dict records in the input JSON.")


def safe_get_text(record: Dict[str, Any], text_key: str) -> str:
    value = record.get(text_key, "")
    if value is None:
        return ""
    return str(value)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def basic_clean(text: str) -> str:
    text = text.strip()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("_", " ")

    # normalize some separators but keep useful symbols
    text = re.sub(r"[;]+", ", ", text)
    text = re.sub(r"[()]+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*-\s*", " - ", text)

    text = text.lower()

    # unify 3 axis -> 3-axis
    text = re.sub(r"\b(\d+)\s+axis\b", r"\1-axis", text)

    # remove most repeated punctuation, keep commas/slashes/+/'/-
    text = re.sub(r"[^\w\s,\-\/\+'\.\:]", " ", text)

    text = normalize_whitespace(text)
    return text


def is_invalid_text(text: str) -> bool:
    t = text.strip().lower()
    return t in INVALID_PLACEHOLDERS


def apply_regex_map(text: str, replacements: Dict[str, str]) -> str:
    out = text
    for pattern, repl in replacements.items():
        out = re.sub(pattern, repl, out)
    return normalize_whitespace(out)


def fix_typos_and_abbreviations(text: str) -> str:
    out = text

    # typo fixes
    out = apply_regex_map(out, TYPO_REPLACEMENTS)

    # abbreviation replacements with token protection
    # For very simple cases we just apply replacements directly.
    out = apply_regex_map(out, ABBREVIATION_REPLACEMENTS)

    return normalize_whitespace(out)


def normalize_phrases(text: str) -> str:
    out = text
    out = apply_regex_map(out, PHRASE_REPLACEMENTS)

    # additional targeted rewrites
    out = re.sub(
        r"\bdrive motor spec(?:ification)? insufficient to move up\b",
        "drive motor specification insufficient for upward movement",
        out,
    )
    out = re.sub(
        r"\bdrive motor speed too high in inclination\b",
        "drive motor speed too high on inclination",
        out,
    )
    out = re.sub(
        r"\bdrive reverse valve actuation incorrect\b",
        "drive reverse valve actuation incorrect",
        out,
    )
    out = re.sub(
        r"\bdrive select valve actuation incorrect\b",
        "drive select valve actuation incorrect",
        out,
    )
    out = re.sub(
        r"\bearth trace acts as antenna caused by improper termination\b",
        "earth trace acts as antenna due to improper termination",
        out,
    )

    # normalize contractions after lowercasing
    out = out.replace("doesn't", "does not").replace("didn't", "did not")
    out = normalize_whitespace(out)

    # repeated cleanup from rewrites
    out = re.sub(r"\bcaused by\b", "caused by", out)
    return normalize_whitespace(out)


def should_avoid_split(text: str) -> bool:
    for pattern in DO_NOT_SPLIT_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def split_multi_causes(text: str) -> List[str]:
    """
    Conservative splitting:
    - split on commas only if they look like separate causes
    - split on " or " only in clearly separate noun phrases
    - keep single text otherwise
    """
    if should_avoid_split(text):
        return [text]

    # Special case seen in examples:
    # "emi noise, connector or cable damaged"
    if re.fullmatch(r"emi noise,\s*connector or cable damaged", text):
        return ["emi noise", "connector or cable damaged"]

    # Conservative comma splitting:
    # only if comma separates obvious independent phrases and each side is meaningful.
    if "," in text:
        parts = [normalize_whitespace(p) for p in text.split(",")]
        parts = [p for p in parts if p]
        if len(parts) == 2 and all(len(p.split()) >= 2 for p in parts):
            return parts

    return [text]


def detect_review_flags(
    raw_text: str,
    clean_text: str,
    normalized_text: str,
    canonical_text: str,
    split_parts: List[str],
) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = []
    notes: List[str] = []

    if is_invalid_text(raw_text):
        reasons.append("invalid_placeholder")

    if len(normalized_text.split()) <= 1:
        reasons.append("too_short")

    for pattern in GENERIC_PATTERNS:
        if re.search(pattern, normalized_text):
            reasons.append("too_generic")
            break

    for pattern in AMBIGUOUS_PATTERNS:
        if re.search(pattern, raw_text.lower()):
            # only mark ambiguous for some cases, not every comma
            if pattern == "," and len(split_parts) > 1:
                continue
            reasons.append("ambiguous_or_multi_cause")
            break

    if raw_text != clean_text and len(raw_text) > 0:
        change_ratio = abs(len(raw_text) - len(clean_text)) / max(len(raw_text), 1)
        if change_ratio > 0.45:
            notes.append("large_change_after_basic_clean")

    if normalized_text != canonical_text and len(normalized_text.split()) <= 2:
        notes.append("canonicalization_based_on_heuristic")

    if "issue" in canonical_text:
        reasons.append("needs_review_issue_placeholder")

    if "connector or cable" in canonical_text:
        reasons.append("component_not_specific")

    if "eld" in canonical_text:
        reasons.append("domain_term_unclear")

    if "display drive strength" in canonical_text:
        reasons.append("meaning_unclear")

    review_flag = len(set(reasons)) > 0
    return review_flag, sorted(set(reasons)), sorted(set(notes))


def infer_component(text: str) -> Optional[str]:
    # Prefer longer matches first
    for kw in sorted(COMPONENT_KEYWORDS, key=len, reverse=True):
        if kw in text:
            return kw
    return None


def infer_mechanism(text: str) -> Optional[str]:
    for kw in sorted(MECHANISM_KEYWORDS, key=len, reverse=True):
        if kw in text:
            return kw
    return None


def canonicalize_text(text: str) -> str:
    """
    Convert to concise engineering phrase style.
    Conservative; do not over-infer.
    """
    out = text

    # high-confidence direct canonical rewrites
    direct_map = {
        "did not install according to manual": "installation not according to manual",
        "installation not according to manual": "installation not according to manual",
        "document review missing": "document review missing",
        "sensor displacement after maintenance": "sensor displacement after maintenance",
        "drive pin stuck": "drive pin stuck",
        "drive pin for lowering valve stuck": "drive pin for lowering valve stuck",
        "driver pin stuck high": "driver pin stuck high",
        "drive strength configuration incorrect": "drive strength configuration incorrect",
        "drive reverse valve actuation incorrect": "drive reverse valve actuation incorrect",
        "drive select valve actuation incorrect": "drive select valve actuation incorrect",
        "dust contamination in unlock system": "dust contamination in unlock system",
        "emi on measurement signals": "emi on measurement signals",
        "emi on input signal caused by motor drive": "emi on input signal caused by motor drive",
        "emi noise": "emi noise",
        "connector or cable damaged": "connector or cable damaged",
        "earth trace routed under ir pcb across +3v3 and gnd pads": "earth trace routed under ir pcb across +3v3 and gnd pads",
        "earth trace acts as antenna due to improper termination": "earth trace acts as antenna due to improper termination",
        "electronics failure": "electronics failure",
        "dirt contamination": "dirt contamination",
    }
    if out in direct_map:
        return direct_map[out]

    # Common compressions
    out = re.sub(r"\bdrive pin .* stuck\b", "drive pin stuck", out)
    out = re.sub(r"\bdriver pin stuck high\b", "driver pin stuck high", out)
    out = re.sub(r"\bdropping wheel on floor\b", "wheel dropped on floor", out)
    out = re.sub(r"\bdrive motor speed too high on inclination\b", "drive motor speed too high on inclination", out)
    out = re.sub(r"\bdrive motor speed too high in track corner\b", "drive motor speed too high in track corner", out)
    out = re.sub(
        r"\bdrive motor specification insufficient for upward movement\b",
        "drive motor specification insufficient for upward movement",
        out,
    )
    out = re.sub(
        r"\bdrive pwm active with brake lifted while stopped caused by software error\b",
        "drive pwm active while stopped with brake lifted caused by software error",
        out,
    )

    # Remove weak leading filler
    out = re.sub(r"^\bcaused by\b\s*", "", out).strip()

    # Heuristic if no good direct map
    component = infer_component(out)
    mechanism = infer_mechanism(out)

    if component and mechanism:
        # If phrase already concise enough, keep original
        if len(out.split()) <= 10:
            return out

        candidate = f"{component} {mechanism}"
        return normalize_whitespace(candidate)

    return normalize_whitespace(out)


def clean_one_text(raw_text: str, record_id: int) -> List[CleanedRecord]:
    raw_text = "" if raw_text is None else str(raw_text)
    raw_text_stripped = raw_text.strip()

    if is_invalid_text(raw_text_stripped):
        rec = CleanedRecord(
            record_id=record_id,
            parent_record_id=record_id,
            split_index=0,
            raw_text=raw_text,
            valid_text=False,
            clean_text="",
            spelling_fixed_text="",
            normalized_text="",
            canonical_text="",
            split_flag=False,
            review_flag=True,
            review_reasons="invalid_placeholder",
            notes="",
        )
        return [rec]

    clean_text = basic_clean(raw_text_stripped)
    spelling_fixed_text = fix_typos_and_abbreviations(clean_text)
    normalized_text = normalize_phrases(spelling_fixed_text)

    split_parts = split_multi_causes(normalized_text)
    split_flag = len(split_parts) > 1

    output_records: List[CleanedRecord] = []

    for idx, part in enumerate(split_parts):
        part = normalize_whitespace(part)
        canonical_text = canonicalize_text(part)
        review_flag, reasons, notes = detect_review_flags(
            raw_text=raw_text_stripped,
            clean_text=clean_text,
            normalized_text=part,
            canonical_text=canonical_text,
            split_parts=split_parts,
        )

        rec = CleanedRecord(
            record_id=record_id,
            parent_record_id=record_id,
            split_index=idx,
            raw_text=raw_text_stripped,
            valid_text=True,
            clean_text=clean_text,
            spelling_fixed_text=spelling_fixed_text,
            normalized_text=part,
            canonical_text=canonical_text,
            split_flag=split_flag,
            review_flag=review_flag,
            review_reasons=";".join(reasons),
            notes=";".join(notes),
        )
        output_records.append(rec)

    return output_records


def process_records(records: List[Dict[str, Any]], text_key: str) -> pd.DataFrame:
    cleaned: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        raw_text = safe_get_text(record, text_key=text_key)
        sub_records = clean_one_text(raw_text=raw_text, record_id=idx)
        for rec in sub_records:
            cleaned.append(asdict(rec))

    df = pd.DataFrame(cleaned)

    # Optional convenience columns
    df["raw_token_count"] = df["raw_text"].fillna("").map(lambda x: len(str(x).split()))
    df["normalized_token_count"] = df["normalized_text"].fillna("").map(lambda x: len(str(x).split()))
    df["canonical_token_count"] = df["canonical_text"].fillna("").map(lambda x: len(str(x).split()))

    return df


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean FMEA cause texts from JSON.")
    parser.add_argument("--input", required=True, help="Path to input JSON file.")
    parser.add_argument("--output_csv", required=False, default="cleaned_fmea_causes.csv")
    parser.add_argument("--output_json", required=False, default="cleaned_fmea_causes.json")
    parser.add_argument(
        "--text_key",
        required=False,
        default="e.text",
        help="Key in each record containing the text. Default: e.text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = load_json_records(input_path)
    records = extract_records(data, text_key=args.text_key)
    df = process_records(records, text_key=args.text_key)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df.to_json(output_json, orient="records", force_ascii=False, indent=2)

    print(f"Input records:   {len(records)}")
    print(f"Output records:  {len(df)}")
    print(f"CSV written to:  {output_csv}")
    print(f"JSON written to: {output_json}")

    print("\nPreview:")
    preview_cols = [
        "record_id",
        "split_index",
        "raw_text",
        "normalized_text",
        "canonical_text",
        "review_flag",
        "review_reasons",
    ]
    print(df[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()