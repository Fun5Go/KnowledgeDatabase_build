from typing import List, Optional
from dataclasses import dataclass, field
from typing import Dict, Any
from KnowledgeBase.JSON_FMEA_KB.kb_structure import Sentence

def _clean(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    text = str(text).strip()
    return text if text else None


def _build_full_chain_sentence(
    element: Optional[str],
    mode: Optional[str],
    cause: Optional[str],
    effect: Optional[str],
) -> Optional[str]:
    """
    Build a natural full-chain FMEA sentence.

    Pattern:
        In {element}, {mode} due to {cause} leading to {effect}.
    Handles missing parts gracefully.
    """

    element = _clean(element)
    mode = _clean(mode)
    cause = _clean(cause)
    effect = _clean(effect)

    if not any([element, mode, cause, effect]):
        return None

    parts = []

    if element:
        parts.append(f"In {element},")

    if mode:
        parts.append(mode)

    if cause:
        parts.append(f"due to {cause}")

    if effect:
        parts.append(f"leading to {effect}")

    sentence = " ".join(parts).strip()

    if not sentence.endswith("."):
        sentence += "."

    return sentence


# =========================================================
# OLD FMEA → full-chain sentence
# =========================================================
def build_sentences_from_old_fmea(record: dict) -> List[Sentence]:
    sentences = []

    case_id = record.get("file_name")
    if not case_id:
        return sentences

    # process_step is treated as element
    element = record.get("process_step")
    mode = record.get("failure_mode")
    cause = record.get("failure_cause")
    effect = record.get("failure_effect")

    text = _build_full_chain_sentence(element, mode, cause, effect)

    if text:
        sentences.append(
            Sentence(
                id=_sid(f"{case_id}|full_chain"),
                text=text,
                sentence_role="full_chain",
                source_type="old_fmea",
                file_name=case_id,
                case_id=case_id,
                metadata={
                    "severity": record.get("severity"),
                    "occurrence": record.get("occurrence"),
                    "detection": record.get("detection"),
                    "rpn": record.get("rpn"),
                },
            )
        )

    return sentences


# =========================================================
# NEW FMEA → full-chain sentence
# =========================================================
def build_sentences_from_new_fmea(record: dict) -> List[Sentence]:
    sentences = []

    case_id = record.get("file_name")
    if not case_id:
        return sentences

    # new FMEA may have explicit element/system
    element = record.get("system_name") or record.get("system")
    mode = record.get("failure_mode")
    cause = record.get("failure_cause")
    effect = record.get("failure_effect")

    text = _build_full_chain_sentence(element, mode, cause, effect)

    if text:
        sentences.append(
            Sentence(
                id=_sid(f"{case_id}|full_chain"),
                text=text,
                sentence_role="full_chain",
                source_type="new_fmea",
                file_name=case_id,
                case_id=case_id,
                metadata={
                    "severity": record.get("severity"),
                    "occurrence": record.get("occurrence"),
                    "detection": record.get("detection"),
                    "rpn": record.get("rpn"),
                    "discipline": record.get("cause_discipline"),
                },
            )
        )

    return sentences
