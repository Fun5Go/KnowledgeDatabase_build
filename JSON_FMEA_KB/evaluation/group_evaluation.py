from collections import defaultdict
from pathlib  import Path
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import spacy
from rapidfuzz import fuzz

_WS_RE = re.compile(r"\s+")
def normalize_text(t: str) -> str:
    return _WS_RE.sub(" ", (t or "").strip().lower())


def _safe_float(num: float, den: int) -> float:
    return float(num) / float(max(1, den))

_nlp = spacy.load("en_core_web_sm")

def normalize_and_lemmatize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower().strip())
    doc = _nlp(text)
    return " ".join(tok.lemma_ for tok in doc)

def _tokenize_lemmas(text: str) -> List[str]:
    doc = _nlp(text)
    return [
        tok.lemma_.lower()
        for tok in doc
        if tok.is_alpha and not tok.is_stop and len(tok) > 2
    ]


def _token_coverage(sent_tokens: List[str], src_tokens: List[str]) -> float:
    if not sent_tokens:
        return 0.0
    return len(set(sent_tokens) & set(src_tokens)) / len(set(sent_tokens))

def group_by_field_with_coverage(
    data: dict,
    field: str,
    *,
    coverage_threshold: float = 0.9,
) -> Dict[str, dict]:
    """
    Returns:
    {
      canonical_text: {
        "members": [cause_id, ...],
        "variants": [original_texts...]
      }
    }
    """
    groups = {}  # canonical_text -> group info

    for cid, record in data.items():
        if not isinstance(record, dict):
            continue

        text = record.get(field)
        if not text:
            continue

        # ---- 1. exact match ----
        if text in groups:
            groups[text]["members"].append(cid)
            continue

        # ---- 2. coverage match ----
        sent_tokens = _tokenize_lemmas(text)
        matched_key = None

        for canonical_text, info in groups.items():
            src_tokens = _tokenize_lemmas(canonical_text)
            coverage = _token_coverage(sent_tokens, src_tokens)

            if coverage >= coverage_threshold:
                matched_key = canonical_text
                break

        if matched_key:
            groups[matched_key]["members"].append(cid)
            groups[matched_key]["variants"].append(text)
        else:
            # ---- 3. new group ----
            groups[text] = {
                "members": [cid],
                "variants": [text],
            }

    return groups


JSON_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb\fmea_cause_store.json")
# ----- read JSON -----
with JSON_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)
# ----- run grouping -----
groups_cause = group_by_field_with_coverage(data, "failure_cause")
# groups_element = group_by_field_with_coverage(data, "failure_element")
# groups_mode = group_by_field_with_coverage(data, "failure_mode")
# groups_effect = group_by_field_with_coverage(data, "failure_effect")

# ----- count groups -----
print("failure_cause groups:", len(groups_cause))
# print("failure_element groups:", len(groups_element))
# print("failure_effect groups:", len(groups_effect))

# print("failure_mode groups:", len(groups_mode))

# # (optional) show group contents
# print("\nGroups by failure_cause:", json.dumps(groups_cause, indent=2))
# print("\nGroups by failure_element:", json.dumps(groups_element, indent=2))