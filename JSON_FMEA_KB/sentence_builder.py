# fmea_sentence_builder.py
from typing import List
from kb_structure import Sentence
import uuid


def _sid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def build_sentences_from_old_fmea(record: dict) -> List[Sentence]:
    sentences = []
    case_id = record["file_name"]

    if record.get("failure_mode"):
        sentences.append(
            Sentence(
                id=_sid(case_id),
                text=f"Failure mode {record['failure_mode']}.",
                sentence_role="failure",
                source_type="old_fmea",
                file_name=record["file_name"],
                case_id=case_id,
                metadata={
                    "severity": int(record.get("severity", 0)),
                    "occurrence": int(record.get("occurrence", 0)),
                    "detection": int(record.get("detection", 0)),
                    "rpn": int(record.get("rpn", 0)),
                },
            )
        )

    if record.get("failure_cause"):
        sentences.append(
            Sentence(
                id=_sid(case_id),
                text=f"{record['failure_cause']} causes {record['failure_mode']}.",
                sentence_role="cause",
                source_type="old_fmea",
                file_name=record["file_name"],
                case_id=case_id,
                metadata={},
            )
        )

    if record.get("failure_effect"):
        sentences.append(
            Sentence(
                id=_sid(case_id),
                text=f"{record['failure_mode']} leads to {record['failure_effect']}.",
                sentence_role="effect",
                source_type="old_fmea",
                file_name=record["file_name"],
                case_id=case_id,
                metadata={},
            )
        )

    return sentences

def build_sentences_from_new_fmea(record: dict) -> List[Sentence]:
    sentences = []
    case_id = record["file_name"]

    if record.get("failure_mode"):
        sentences.append(
            Sentence(
                id=_sid(case_id),
                text=f"{record['failure_mode']} in {record.get('function','system')}.",
                sentence_role="failure",
                source_type="new_fmea",
                file_name=record["file_name"],
                case_id=case_id,
                metadata={
                    "system": record.get("system_name"),
                    "function": record.get("function"),
                    "severity": record.get("severity"),
                    "occurrence": record.get("occurrence"),
                    "detection": record.get("detection"),
                    "rpn": record.get("rpn"),
                },
            )
        )

    if record.get("failure_cause"):
        sentences.append(
            Sentence(
                id=_sid(case_id),
                text=f"{record['failure_cause']} causes {record['failure_mode']}.",
                sentence_role="cause",
                source_type="new_fmea",
                file_name=record["file_name"],
                case_id=case_id,
                metadata={
                    "discipline": record.get("cause_discipline")
                },
            )
        )

    if record.get("failure_effect"):
        sentences.append(
            Sentence(
                id=_sid(case_id),
                text=f"{record['failure_mode']} leads to {record['failure_effect']}.",
                sentence_role="effect",
                source_type="new_fmea",
                file_name=record["file_name"],
                case_id=case_id,
                metadata={},
            )
        )

    return sentences
