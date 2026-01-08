from kb_structure import FMEAFailureKB, FMEACauseKB, FMEAFailure, FMEACause

import json
from pathlib import Path
from collections import defaultdict




# =========================================================
# Helpers
# =========================================================

def normalize(x: str | None) -> str:
    return (x or "").strip().lower()


def is_failure_element_term(failure_type: str | None) -> bool:
    ft = normalize(failure_type)
    if not ft:
        return False

    specific_tokens = [
        "power",
        "controller",
        "control",
        "board",
        "pcb",
        "pcba",
        "interface",
        "algorithm",
        "logic",
        "sensor",
        "actuator",
        "driver",
        "converter",
        "regulator",
        "transceiver",
        "communication",
    ]

    return any(token in ft for token in specific_tokens)

# ---------------------------------------------------------
# Discipline inference (lower priority)
# ---------------------------------------------------------

def infer_discipline_from_failure_type(failure_type: str | None) -> str | None:
    ft = normalize(failure_type)
    if not ft:
        return None

    electronics_tokens = {
        "electronics", "electronic", "electrical",
        "hw", "hardware", "esw", 
    }
    mechanics_tokens = {
        "mechanics", "mechanical", "mech", "mch"
    }
    software_tokens = {
        "software", "sw", "firmware", "embedded software","esw"
    }
    process_tokens = {
        "process", "manufacturing", "assembly",
        "soldering", "welding", "installation", "calibration", "coating"
    }
    design_tokens = {
        "design", "requirement", "requirements",
        "spec", "specification", "architecture",
        "dimensioning", "tolerance"
    }
    generic_tokens = {
        "system", "subsystem", "overall", "general"
    }

    if ft in electronics_tokens:
        return "HW"
    if ft in mechanics_tokens:
        return "MCH"
    if ft in software_tokens:
        return "ESW"
    if ft in process_tokens:
        return "process"
    if ft in design_tokens:
        return "design"
    if ft in generic_tokens:
        return "other"

    return None

def build_failure_signature(row: dict) -> tuple:
    """
    Regroup rule:
    failure_effect is now part of the signature
    """
    if row.get("source_type") == "new_fmea":
        return (
            normalize(row.get("system_name")),
            normalize(row.get("system_element")),
            normalize(row.get("function")),
            normalize(row.get("failure_mode")),
            normalize(row.get("failure_effect")),   
        )
    else:  # old_fmea
        return (
            normalize(row.get("failure_type")),
            normalize(row.get("failure_mode")),
            normalize(row.get("failure_effect")),   
        )


# =========================================================
# Ingest
# =========================================================

def ingest_fmea_json(
    json_path: Path,
    failure_kb: FMEAFailureKB,
    cause_kb: FMEACauseKB,
):
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]

    file_name = rows[0].get("file_name", json_path.stem)

    # -------------------------------------------------
    # Regroup rows by failure signature
    # -------------------------------------------------
    grouped: dict[tuple, list[dict]] = defaultdict(list)

    for row in rows:
        sig = build_failure_signature(row)
        grouped[sig].append(row)

    # -------------------------------------------------
    # Sequential Failure IDs: F1, F2, ...
    # -------------------------------------------------
    failure_counter = 1

    for sig, group in grouped.items():
        first = group[0]
        source_type = first.get("source_type")

        failure_id = f"{file_name}__F{failure_counter}"
        failure_counter += 1

 # ---------- Failure fields ----------
        if source_type == "new_fmea":
            system = first.get("system_name")
            element = first.get("system_element")
            function = first.get("function")
            inferred_discipline = None

        else:  # old_fmea
            system = None
            ft = first.get("failure_type")

            # Priority 1: specific element
            if is_failure_element_term(ft):
                element = ft
                inferred_discipline = None

            # Priority 2: discipline inference
            else:
                inferred_discipline = infer_discipline_from_failure_type(ft)
                element = None if inferred_discipline else ft

            function = None

        failure_mode = first.get("failure_mode")
        failure_effect = first.get("failure_effect")

        severity = max(
            [
                r.get("severity")
                for r in group
                if isinstance(r.get("severity"), (int, float))
            ],
            default=None,
        )

        rpn = max(
            [
                r.get("rpn")
                for r in group
                if isinstance(r.get("rpn"), (int, float))
            ],
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
            source_type=source_type,
        )

        # ---------- Write Failure KB ----------
        if failure_id not in failure_kb.store:
            failure_kb.add(failure_obj)

        # -------------------------------------------------
        # Causes under this failure: C1, C2, ...
        # -------------------------------------------------
        cause_counter = 1

        for row in group:
            cause_text = row.get("failure_cause")
            if not cause_text:
                continue

            cause_id = f"{failure_id}_C{cause_counter}"
            cause_counter += 1

            if source_type == "new_fmea":
                discipline = row.get("cause_discipline")
                cause_obj = FMEACause(
                    cause_id=cause_id,
                    failure_id=failure_id,
                    failure_cause=row.get("failure_cause"),
                    discipline=row.get("cause_discipline"),

                    prevention=row.get("controls_prevention"),
                    detection=row.get("current_detection"),
                    detection_value=row.get("detection"),
                    occurrence=row.get("occurrence"),
                    recommended_action=row.get("recommended_action"),
                )
            else:
                discipline = None

            if source_type == "old_fmea":
                cause_obj = FMEACause(
                    cause_id=cause_id,
                    failure_id=failure_id,
                    failure_cause=row.get("failure_cause"),
                    discipline=None,

                    prevention=None,  
                    detection=row.get("current_detection") or row.get("detection"),
                    detection_value=row.get("detection"),
                    occurrence=row.get("occurrence"),
                    recommended_action=row.get("recommended_action"),

                )


            cause_kb.add(cause_obj)
            failure_obj.cause_ids.append(cause_id)

        # ---------- Back-write failure → causes ----------
        failure_kb.store[failure_id]["cause_ids"] = failure_obj.cause_ids

    # -------------------------------------------------
    # Persist failure store
    # -------------------------------------------------
    failure_kb.store_path.write_text(
        json.dumps(failure_kb.store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

