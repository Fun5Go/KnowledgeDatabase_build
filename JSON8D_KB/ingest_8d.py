import json
from pathlib import Path
from typing import List

from kb_structure import (
    Failure, FailureKB,
    Cause, CauseKB,
    Sentence, SentenceKB,
    MaintenanceTag,
    evaluate_failure
)


def parse_maintenance_tag(raw: dict | None) -> MaintenanceTag:
    raw = raw or {}
    return MaintenanceTag(
        review_status=raw.get("review_status", "pending"),
        version=raw.get("Version", "V0"),
        last_updated=raw.get("last_updated", ""),
        supersedes=raw.get("supersedes"),
    )





def ingest_8d_json(
    json_path: Path,
    failure_kb: FailureKB,
    cause_kb: CauseKB,
    sentence_kb: SentenceKB,
):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("documents", [])
    if not docs:
        raise ValueError("Missing documents")

    doc0 = docs[0]
    case_id = doc0.get("file_name")
    product = doc0.get("product_name")

    failure = data["failure"]

    # =====================================================
    #  Sentence KB（failure + cause sentence）
    # =====================================================
    failure_sentence_ids: List[str] = []

    for ent in failure.get("supporting_entities", []):
        s = Sentence(
            id=ent["sentence_id"],
            text=ent["text"],
            source_section=ent.get("source_section", ""),
            case_id=case_id,
            annotations=ent.get("annotations", {}),
        )
        sentence_kb.add(
            sentence=s,
            failure_id=failure["failure_ID"],
            sentence_role="failure_sentence",
        )
        failure_sentence_ids.append(s.id)

    used_sentence_ids = set()

    for ent in failure.get("supporting_entities", []):
        used_sentence_ids.add(ent["sentence_id"])

    for cause in failure.get("root_causes", []):
        for ent in cause.get("supporting_entities", []):
            used_sentence_ids.add(ent["sentence_id"])

    for ent in data.get("selected_sentences", []):
        sid = ent["sentence_id"]

        if sid in used_sentence_ids:
            continue  # Exclude failure / cause support sentence

        s = Sentence(
            id=sid,
            text=ent["text"],
            source_section=ent.get("source_section", ""),
            case_id=case_id,
            annotations=ent.get("annotations", {}),
        )

        sentence_kb.add(
            sentence=s,
            failure_id=failure["failure_ID"],
            sentence_role="other",
            cause_id=None,
        )

    # =====================================================
    #  Failure KB（入口）
    # =====================================================
    status = evaluate_failure(sentence_kb.get_by_ids(failure_sentence_ids))
    failure_maintenance = parse_maintenance_tag(
        failure.get("maintenance_tag")
    )
    

    cause_ids: List[str] = []

    failure_obj = Failure(
        failure_id=failure["failure_ID"],
        failure_mode=failure.get("failure_mode", ""),
        failure_element=failure.get("failure_element", ""),
        failure_effect=failure.get("failure_effect"),
        product=product,
        status=status,
        supporting_sentence_ids=failure_sentence_ids,
        cause_ids=[],
        maintenance=failure_maintenance,
    )

    failure_kb.add(failure_obj)

    # =====================================================
    #  Cause KB
    # =====================================================
    for cause in failure.get("root_causes", []):
        cause_id = cause["cause_ID"]
        cause_sentence_ids: List[str] = []

        for ent in cause.get("supporting_entities", []):
            s = Sentence(
                id=ent["sentence_id"],
                text=ent["text"],
                source_section=ent.get("source_section", ""),
                case_id=case_id,
                annotations=ent.get("annotations", {}),
            )
            sentence_kb.add(
                sentence=s,
                failure_id=failure["failure_ID"],
                sentence_role="cause_sentence",
                cause_id=cause_id,
            )
            cause_sentence_ids.append(s.id)

        cause_maintenance = parse_maintenance_tag(
            cause.get("maintenance_tag")
        )

        cause_obj = Cause(
            cause_id=cause_id,
            failure_id=failure["failure_ID"],
            failure_mode=failure.get("failure_mode", ""),
            failure_element=failure.get("failure_element", ""),
            failure_effect=failure.get("failure_effect"),
            root_cause=cause.get("failure_cause", ""),
            cause_level=cause.get("cause_level", ""),
            discipline=cause.get("discipline_type", ""),
            confidence=cause.get("confidence", ""),
            supporting_sentence_ids=cause_sentence_ids,
            maintenance=cause_maintenance
        )

        cause_kb.add(cause_obj)
        cause_ids.append(cause_id)

    # cause_ids
    failure_kb.store[failure["failure_ID"]]["cause_ids"] = cause_ids