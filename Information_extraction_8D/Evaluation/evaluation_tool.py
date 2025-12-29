import re
from rapidfuzz import fuzz
from Information_extraction_8D.Schemas.eightD_sentence_schema import SelectedSentence
from typing import Dict, List, Tuple, Union
def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())

def check_faithfulness(sentence: str, source_text: str) -> Dict:
    sent = normalize_text(sentence)
    src = normalize_text(source_text)

    if sent in src:
        return {"faithful": True, "type": "exact"}

    score = fuzz.partial_ratio(sent, src)
    if score >= 90:
        return {"faithful": True, "type": "fuzzy", "score": score}

    return {"faithful": False, "type": "hallucinated", "score": score}


CONJUNCTIONS = {" and ", " or ", " while ", " as well as ", ";"}

def is_atomic(sentence: str) -> bool:
    s = sentence.lower()
    if any(c in s for c in CONJUNCTIONS):
        # allow short conjunctions
        return s.count(",") <= 1
    return True

ENTITY_RULES = {
    "symptom": {
        "allow": ["fail", "not", "unable", "intermittent", "no ", "loss"],
        "deny": ["temperature", "voltage", "stress"]
    },
    "condition": {
        "allow": ["temperature", "voltage", "cycle", "humidity", "load"],
        "deny": ["fail", "not working"]
    },
    "occurrence": {
        "allow": ["%", "ratio", "times", "intermittent", "rate"],
        "deny": []
    },
    "investigation": {
        "allow": ["inspect", "test", "verify", "check"],
        "deny": ["caused by"]
    },
    "root_cause_evidence": {
        "allow": ["defect", "crack", "short", "misalignment", "solder"],
        "deny": ["may", "suspected"]
    }
}

ASSERTION_MARKERS = {
    "confirmed": ["confirmed", "verified", "reproduced"],
    "observed": ["observed", "found", "visible", "measured"],
    "ruled_out": ["ruled out", "excluded", "not the cause"],
    "suspected": ["suspected", "may", "likely", "possible"]
}

def validate_assertion(sentence: SelectedSentence) -> bool:
    markers = ASSERTION_MARKERS.get(sentence.assertion_level, [])
    if not markers:
        return False

    text = sentence.text.lower()
    return any(m in text for m in markers) or sentence.assertion_level == "observed"

def validate_entity_type(sentence: SelectedSentence) -> bool:
    rules = ENTITY_RULES.get(sentence.entity_type, {})
    text = sentence.text.lower()

    if rules.get("allow"):
        if not any(k in text for k in rules["allow"]):
            return False

    if any(k in text for k in rules.get("deny", [])):
        return False

    return True

def coverage_metrics(sentences: List[SelectedSentence]) -> Dict:
    types = [s.entity_type for s in sentences]
    return {
        "has_symptom": "symptom" in types,
        "has_root_or_investigation": any(
            t in types for t in ["root_cause_evidence", "investigation"]
        ),
        "sentence_count": len(sentences)
    }


def evaluate_iter1(
    sentences: List[SelectedSentence],
    d2: str,
    d3: str,
    d4: str
) -> Dict:

    source_text = f"{d2}\n{d3}\n{d4}"

    stats = {
        "total": len(sentences),
        "faithful": 0,
        "hallucinated": 0,
        "atomic": 0,
        "entity_valid": 0,
        "assertion_valid": 0,
        "confirmed_count": 0
    }

    for s in sentences:
        faith = check_faithfulness(s.text, source_text)
        if faith["faithful"]:
            stats["faithful"] += 1
        else:
            stats["hallucinated"] += 1

        if is_atomic(s.text):
            stats["atomic"] += 1

        if validate_entity_type(s):
            stats["entity_valid"] += 1

        if validate_assertion(s):
            stats["assertion_valid"] += 1

        if s.assertion_level == "confirmed":
            stats["confirmed_count"] += 1

    coverage = coverage_metrics(sentences)

    return {
        "faithfulness_rate": stats["faithful"] / max(1, stats["total"]),
        "hallucination_rate": stats["hallucinated"] / max(1, stats["total"]),
        "atomicity_rate": stats["atomic"] / max(1, stats["total"]),
        "entity_rule_pass_rate": stats["entity_valid"] / max(1, stats["total"]),
        "assertion_rule_pass_rate": stats["assertion_valid"] / max(1, stats["total"]),
        "confirmed_ratio": stats["confirmed_count"] / max(1, stats["total"]),
        "coverage": coverage
    }
