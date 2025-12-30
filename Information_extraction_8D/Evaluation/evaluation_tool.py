import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import spacy
from rapidfuzz import fuzz
from Information_extraction_8D.Schemas.eightD_sentence_schema import SelectedSentence

# ----------------------------
# Text helpers
# ----------------------------

_WS_RE = re.compile(r"\s+")

def normalize_text(t: str) -> str:
    return _WS_RE.sub(" ", (t or "").strip().lower())

def _build_source_text(d2: str, d3: str, d4: str) -> str:
    # Keep simple: concat; normalize later in check_faithfulness
    return f"{d2 or ''}\n{d3 or ''}\n{d4 or ''}"

def _safe_float(num: float, den: int) -> float:
    return float(num) / float(max(1, den))

_nlp = spacy.load("en_core_web_sm")

def normalize_and_lemmatize(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower().strip())
    doc = _nlp(text)
    return " ".join(tok.lemma_ for tok in doc)


def _get_annotation(s: SelectedSentence, key: str):
    ann = getattr(s, "annotations", None)
    if not ann:
        return None
    return getattr(ann, key, None)

# ----------------------------
# Faithfulness
# ----------------------------

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


def check_faithfulness(
    sentence: str,
    source_text: str,
    *,
    fuzzy_threshold: int = 85,   #
    coverage_threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "faithful": bool,
        "type": "exact" | "fuzzy" | "partial" | "hallucinated",
        "score": int
      }
    """
    sent_norm = normalize_text(sentence)
    src_norm = normalize_text(source_text)

    if not sent_norm:
        return {"faithful": False, "type": "hallucinated", "score": 0}

    # 1. Exact containment
    if sent_norm in src_norm:
        return {"faithful": True, "type": "exact", "score": 100}

    # 2. Global fuzzy (token-set is safer than partial)
    fuzzy_score = int(fuzz.token_set_ratio(sent_norm, src_norm))

    # 3. Lemma-level token coverage (anti light-rephrase)
    sent_tokens = _tokenize_lemmas(sentence)
    src_tokens = _tokenize_lemmas(source_text)
    coverage = _token_coverage(sent_tokens, src_tokens)
    coverage_score = int(coverage * 100)

    # 4. Final score = strongest signal
    final_score = max(fuzzy_score, coverage_score)

    if final_score >= fuzzy_threshold:
        return {
            "faithful": True,
            "type": "fuzzy",
            "score": final_score,
        }

    if coverage >= coverage_threshold:
        return {
            "faithful": True,
            "type": "partial",
            "score": final_score,
        }

    return {
        "faithful": False,
        "type": "hallucinated",
        "score": final_score,
    }

# ----------------------------
# Atomicity
# ----------------------------

CONJUNCTIONS = {" and ", " or ", " while ", " as well as ", ";"}

def is_atomic(sentence: str, *, max_commas_if_conjunction: int = 1) -> bool:
    """
    Heuristic:
    - If conjunctions exist, allow only a small number of commas (default <= 1)
    - Otherwise treat as atomic.
    """
    s = normalize_text(sentence)
    if any(c in s for c in CONJUNCTIONS):
        return s.count(",") <= max_commas_if_conjunction
    return True


# ----------------------------
# Entity / assertion rules
# ----------------------------

ENTITY_RULES: Dict[str, Dict[str, List[str]]] = {
    "symptom": {
       # "allow": ["fail", "not", "unable", "intermittent", "no ", "loss","breaks","broken","unfunctional",],
       "allow": [],
        "deny": ["temperature", "voltage", "stress"],
    },
    "condition": {
        #"allow": ["temperature", "voltage", "cycle", "humidity", "load","current","condition"],
        "allow": [],
        "deny": ["fail", "not working"],
    },
    "occurrence": {
        #"allow": ["%", "ratio", "times", "intermittent", "rate"],
        "allow": [],
        "deny": [],
    },
    "investigation": {
       # "allow": ["inspect", "test", "verify", "check","show","investigate", "investigated"],
       "allow": [],
        "deny": ["caused by"],
    },
    "root_cause_evidence": {
       # "allow": ["defect", "crack", "short", "misalignment", "solder"],
       "allow": [],
        "deny": ["may", "suspected"],
    },
}

ASSERTION_MARKERS: Dict[str, List[str]] = {
    "confirmed": ["confirmed", "verified", "reproduced"],
    "observed": ["observed", "found", "visible", "measured"],
    "ruled_out": ["ruled out", "excluded", "not the cause","disregarded", "no evidence", "be not"],
    "suspected": ["suspected", "may", "likely", "possible","might"],
}

def validate_assertion(sentence: SelectedSentence) -> bool:
    """
    - observed: we accept even if no marker is present (because many observed facts are plain statements)
    - other levels: require a marker substring match
    """
    level = _get_annotation(sentence, "assertion_level")
    if not level:
        return False
    if level == "observed":
        return True

    markers = ASSERTION_MARKERS.get(level, [])
    if not markers:
        return False

    text = normalize_and_lemmatize(getattr(sentence, "text", ""))
    return any(m in text for m in markers)

def validate_entity_type(sentence: SelectedSentence) -> bool:
    et = _get_annotation(sentence, "entity_type")

    if not et or et not in ENTITY_RULES:
        return False

    rules = ENTITY_RULES[et]
    text = normalize_text(getattr(sentence, "text", ""))

    allow = rules.get("allow", [])
    deny = rules.get("deny", [])

    if allow and not any(k in text for k in allow):
        return False
    if deny and any(k in text for k in deny):
        return False

    return True


# ----------------------------
# Coverage
# ----------------------------

def coverage_metrics(sentences: List[SelectedSentence]) -> Dict[str, Any]:
    types = [_get_annotation(s, "entity_type") for s in sentences]

    types = [t for t in types if t]

    return {
        "has_symptom": "symptom" in types,
        "has_root_or_investigation": any(t in types for t in ["root_cause_evidence", "investigation"]),
        "sentence_count": len(sentences),
        "type_counts": {t: types.count(t) for t in sorted(set(types))},
    }


# ----------------------------
# Main evaluation
# ----------------------------

def evaluate_iter1(
    sentences: List[SelectedSentence],
    d2: str,
    d3: str,
    d4: str,
    *,
    fuzzy_threshold: int = 90,
    include_per_sentence: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates an Iter-1 extraction output.

    Returns:
      {
        rates...,
        counts...,
        coverage...,
        per_sentence: [...]  # if include_per_sentence
      }
    """
    source_text = _build_source_text(d2, d3, d4)

    total = len(sentences)

    counts = {
        "total": total,
        "strong_faithful": 0,
        "weak_faithful": 0,
        "hallucinated": 0,
        "atomic": 0,
        "entity_valid": 0,
        "assertion_valid": 0,
        "confirmed_count": 0,
        "exact_count": 0,
        "fuzzy_count": 0,
        "partial_count": 0,
    }

    per_sentence: List[Dict[str, Any]] = []

    for idx, s in enumerate(sentences, start=1):
        text = getattr(s, "text", "") or ""
        entity_type = _get_annotation(s, "entity_type")
        assertion_level = _get_annotation(s, "assertion_level")

        assertion_level = getattr(s, "assertion_level", None)
        source_section = getattr(s, "source_section", None)
        sid = getattr(s, "id", None) or f"S{idx}"

        faith = check_faithfulness(text, source_text, fuzzy_threshold=fuzzy_threshold)

        faith_type = faith["type"]

        if faith_type in {"exact", "fuzzy"}:
            counts["strong_faithful"] += 1
            if faith_type == "exact":
                counts["exact_count"] += 1
            else:
                counts["fuzzy_count"] += 1
        elif faith_type == "partial":
            counts["weak_faithful"] += 1
            counts["partial_count"] += 1
        else:
            counts["hallucinated"] += 1

        atomic_ok = is_atomic(text)
        if atomic_ok:
            counts["atomic"] += 1

        entity_ok = validate_entity_type(s)
        if entity_ok:
            counts["entity_valid"] += 1

        assertion_ok = validate_assertion(s)
        if assertion_ok:
            counts["assertion_valid"] += 1

        if assertion_level == "confirmed":
            counts["confirmed_count"] += 1

        if include_per_sentence:
            per_sentence.append(
                {
                    "id": sid,
                    "source_section": source_section,
                    "text": text,
                    "entity_type": entity_type,
                    "assertion_level": assertion_level,
                    "faithfulness": faith,     # includes type + score
                    "atomic": atomic_ok,
                    "entity_rule_pass": entity_ok,
                    "assertion_rule_pass": assertion_ok,
                }
            )

    coverage = coverage_metrics(sentences)

    result = {
        "strong_faithfulness_rate": _safe_float(counts["strong_faithful"], total),
        "weak_faithfulness_rate": _safe_float(counts["weak_faithful"], total),
        "hallucination_rate": _safe_float(counts["hallucinated"], total),
        "atomicity_rate": _safe_float(counts["atomic"], total),
        "entity_rule_pass_rate": _safe_float(counts["entity_valid"], total),
        "assertion_rule_pass_rate": _safe_float(counts["assertion_valid"], total),
        "confirmed_ratio": _safe_float(counts["confirmed_count"], total),
        "exact_ratio": _safe_float(counts["exact_count"], total),
        "fuzzy_ratio": _safe_float(counts["fuzzy_count"], total),
        "partial_ratio": _safe_float(counts["partial_count"], total),
        "counts": counts,
        "coverage": coverage,
    }

    if include_per_sentence:
        result["per_sentence"] = per_sentence

    return result


# ----------------------------
# Optional: quick pretty summary
# ----------------------------

def summarize_eval(eval_result: Dict[str, Any]) -> str:
    c = eval_result.get("counts", {})
    cov = eval_result.get("coverage", {})
    lines = [
        f"Total: {c.get('total', 0)}",
        f"Faithful: {c.get('faithful', 0)} (exact={c.get('exact_count', 0)}, fuzzy={c.get('fuzzy_count', 0)})",
        f"Hallucinated: {c.get('hallucinated', 0)}",
        f"Atomic: {c.get('atomic', 0)}",
        f"Entity rule pass: {c.get('entity_valid', 0)}",
        f"Assertion rule pass: {c.get('assertion_valid', 0)}",
        f"Confirmed: {c.get('confirmed_count', 0)}",
        f"Coverage has_symptom={cov.get('has_symptom')}, has_root_or_investigation={cov.get('has_root_or_investigation')}",
        f"Type counts: {cov.get('type_counts')}",
    ]
    return "\n".join(lines)
