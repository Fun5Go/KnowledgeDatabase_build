from kb_structure import FMEAFailureKB, FMEACauseKB, FMEAFailure, FMEACause

import json
from pathlib import Path
from collections import defaultdict
import hashlib


def normalize(x: str | None) -> str:
    return (x or "").strip().lower()

def build_failure_signature(row: dict) -> tuple:
    """
    For regroup
    """
    if row.get("source_type") == "new_fmea":
        return (
            normalize(row.get("system_name")),
            normalize(row.get("system_element")),
            normalize(row.get("function")),
            normalize(row.get("failure_mode")),
        )
    else:  # old_fmea
        return (
            normalize(row.get("failure_type")),
            normalize(row.get("failure_mode")),
        )


def make_failure_id(signature: tuple, file_name: str) -> str:
    sig_str = "|".join(signature)
    h = hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:8]
    return f"{file_name}__F_{h}"


def ingest_fmea_json(
    json_path: Path,
    failure_kb,
    cause_kb,
):
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]

    # -------------------------------------------------
    # 1️⃣ 按 failure signature regroup
    # -------------------------------------------------
    grouped: dict[tuple, list[dict]] = defaultdict(list)

    for row in rows:
        sig = build_failure_signature(row)
        grouped[sig].append(row)

    # -------------------------------------------------
    # 2️⃣ 每个 failure group → 1 Failure + N Causes
    # -------------------------------------------------
    for sig, group in grouped.items():
        first = group[0]
        source_type = first.get("source_type")
        file_name = first.get("file_name")

        failure_id = make_failure_id(sig, file_name)

        # ---------- Failure fields ----------
        if source_type == "new_fmea":
            system = first.get("system_name")
            element = first.get("system_element")
            function = first.get("function")
        else:
            system = None
            element = first.get("failure_type")
            function = None

        failure_mode = first.get("failure_mode")

        # effect：可能不同，合并
        effects = list({
            r.get("failure_effect")
            for r in group
            if r.get("failure_effect")
        })
        failure_effect = "; ".join(effects)

        severity = max(
            [r.get("severity") for r in group if isinstance(r.get("severity"), (int, float))],
            default=None,
        )

        rpn = max(
            [r.get("rpn") for r in group if isinstance(r.get("rpn"), (int, float))],
            default=None,
        )

        failure_obj = FMEAFailure(
            failure_id=failure_id,
            failure_mode=failure_mode,
            failure_element=element,
            failure_effect=failure_effect,
            system=system,
            function=function,
            severity=severity,
            rpn=rpn,
            cause_ids=[],
        )

        # ---------- 写入 Failure KB（只一次） ----------
        if failure_id not in failure_kb.store:
            failure_kb.add(failure_obj)

        # -------------------------------------------------
        # 3️⃣ 为每条 row 建 Cause
        # -------------------------------------------------
        for idx, row in enumerate(group, start=1):
            cause_text = row.get("failure_cause")
            if not cause_text:
                continue

            cause_id = f"{failure_id}_C{idx}"

            if source_type == "new_fmea":
                discipline = row.get("cause_discipline")
                confidence = "high"
            else:
                discipline = None
                confidence = "low"

            cause_obj = FMEACause(
                cause_id=cause_id,
                failure_id=failure_id,
                failure_cause=cause_text,
                discipline=discipline,
            )

            cause_kb.add(cause_obj)
            failure_obj.cause_ids.append(cause_id)

        # ---------- 回写 failure → cause 关系 ----------
        failure_kb.store[failure_id]["cause_ids"] = failure_obj.cause_ids

    failure_kb.store_path.write_text(
        json.dumps(failure_kb.store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


