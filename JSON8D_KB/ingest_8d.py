import json
from pathlib import Path
from typing import List

from kb_structure import (
    Failure, FailureKB,
    Cause, CauseKB,
    Sentence, SentenceKB,
    evaluate_failure
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
    # 1️⃣ Sentence KB（failure + cause sentence）
    # =====================================================
    failure_sentence_ids: List[str] = []

    for ent in failure.get("supporting_entities", []):
        s = Sentence(
            id=ent["id"],
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

    # =====================================================
    # 2️⃣ Failure KB（入口）
    # =====================================================
    status = evaluate_failure(sentence_kb.get_by_ids(failure_sentence_ids))

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
    )

    failure_kb.add(failure_obj)

    # =====================================================
    # 3️⃣ Cause KB（必须挂在 failure 下）
    # =====================================================
    for cause in failure.get("root_causes", []):
        cause_id = cause["cause_ID"]
        cause_sentence_ids: List[str] = []

        for ent in cause.get("supporting_entities", []):
            s = Sentence(
                id=ent["id"],
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
        )

        cause_kb.add(cause_obj)
        cause_ids.append(cause_id)

    # 🔑 回写 cause_ids
    failure_kb.store[failure["failure_ID"]]["cause_ids"] = cause_ids