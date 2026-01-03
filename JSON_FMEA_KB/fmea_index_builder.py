# fmea_index_builder.py
from kb_structure import FMEAFailure, FMEACause
import uuid


def build_fmea_index(record: dict, sentences: list):
    fmea_id = f"{record['file_name']}_F1"

    failure_sent_ids = [s.id for s in sentences if s.sentence_role == "failure"]
    cause_sent_ids = [s.id for s in sentences if s.sentence_role == "cause"]
    effect_sent_ids = [s.id for s in sentences if s.sentence_role == "effect"]

    cause_indices = []
    cause_ids = []

    for i, cid in enumerate(cause_sent_ids):
        cause_id = f"{fmea_id}_C{i+1}"
        cause_ids.append(cause_id)
        cause_indices.append(
            FMEACause(
                cause_id=cause_id,
                failure_cause=record.get("failure_cause"),
                discipline=record.get("cause_discipline"),
                supporting_sentence_ids=[cid],
            )
        )

    fmea_index = FMEAFailure(
        fmea_id=fmea_id,
        failure_mode=record.get("failure_mode"),
        failure_effect=record.get("failure_effect"),
        system=record.get("system_name"),
        function=record.get("function"),
        severity=record.get("severity"),
        occurrence=record.get("occurrence"),
        detection=record.get("detection"),
        rpn=record.get("rpn"),
        supporting_sentence_ids=failure_sent_ids + effect_sent_ids,
        cause_ids=cause_ids,
    )

    return fmea_index, cause_indices
