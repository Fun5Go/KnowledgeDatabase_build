from kb_structure import FMEAFailureKB, FMEACauseKB, FMEAFailure, FMEACause

import json
from pathlib import Path
from collections import defaultdict
import re


GENERIC_RIGHT_TOKENS = {
    "general",
    "specification",
    "specifications",
    "approval",
    "approbation",
}

ELEMENT_RIGHT_TOKENS = {
    "filter",
    "switch",
    "switches",
    "capacitor",
    "capacitors",
    "inductor",
    "inductors",
    "sensor",
    "sensors",
    "connector",
    "connectors",
    "isolation",
    "backcover",
    "input",
    "output",
    "limiter",
    "protection",
}

# =========================================================
# Helpers
# =========================================================

def normalize(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def parse_number(value):
    """
    Convert value to float or int if possible.
    Return None if invalid.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        v = value.strip()
        if v == "" or v.lower() in {"n/a", "na", "null", "none", "-"}:
            return None
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return None

    return None

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
        "support electronics",
        "startup",
        "buck",
        "DC",
        "cooling",

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
        "hw", "hardware",  
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
    if any(token in ft for token in process_tokens):
        return "process"
    if ft in design_tokens:
        return "design"
    if ft in generic_tokens:
        return "other"

    return None

def parse_failure_type_semantics(failure_type: str | None):
    """
    Returns:
        discipline: str | None
        element: str | None
    """
    if not failure_type:
        return None, None

    # ft = normalize(failure_type)
    ft = failure_type
    print(ft)

    if "/" not in ft:
        return None, None

    left_raw, right_raw = [p.strip() for p in ft.split("/", 1)]

    left = normalize(left_raw)
    right = normalize(right_raw)

    # debug
    print("DEBUG left:", left, "right:", right)

    # discipline-only
    if any(tok in right for tok in GENERIC_RIGHT_TOKENS):
        return left, None

    # concrete element
    if any(tok in right for tok in ELEMENT_RIGHT_TOKENS):
        return None, f"{left} / {right}"

    # fallback
    return left, None


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
    

#====================================
#===== Duplicate checking ===========
def is_duplicate_failure(
    failure_kb: FMEAFailureKB,
    system,
    element,
    function,
    failure_mode,
    failure_effect,
) -> str | None:
    """
    If duplicate found, return existing failure_id
    Else return None
    """
    nm = normalize(failure_mode)
    ne = normalize(failure_effect)
    nel = normalize(element)
    ns = normalize(system)

    for fid, f in failure_kb.store.items():
        if (
            normalize(f.get("failure_mode")) == nm
            and normalize(f.get("failure_effect")) == ne
            and normalize(f.get("failure_element")) == nel
            and normalize(f.get("system")) == ns
        ):
            return fid

    return None

def is_duplicate_cause(
    cause_kb: FMEACauseKB,
    failure_id: str,
    cause_text: str,
) -> str | None:
    """
    Only deduplicate causes under the same failure
    """
    nc = normalize(cause_text)

    for cid, c in cause_kb.store.items():
        if c.get("failure_id") != failure_id:
            continue
        if normalize(c.get("failure_cause")) == nc:
            return cid

    return None




# =========================================================
# Ingest
# =========================================================

# =========================================================
# Ingest
# =========================================================

def ingest_fmea_json(
    json_path: Path,
    failure_kb,
    cause_kb,
):
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]

    file_name = rows[0].get("file_name", json_path.stem)

    print(f"[INGEST] {json_path.name}")

    # -------------------------------------------------
    # Group by file-internal failure signature
    # -------------------------------------------------
    grouped = defaultdict(list)
    for row in rows:
        sig = build_failure_signature(row)
        grouped[sig].append(row)

    failure_counter = 1

    for _, group in grouped.items():
        first = group[0]
        source_type = first.get("source_type")

        # -------------------------------------------------
        # Build failure semantic fields FIRST (important)
        # -------------------------------------------------
        if source_type == "new_fmea":
            system = first.get("system_name")
            element = first.get("system_element")
            function = first.get("function")
        else:
            system = None
            ft = first.get("failure_type")
            print(ft)

            discipline, element = parse_failure_type_semantics(ft)

            print("discipline:", discipline)
            print("element:", element)


            function = None

        failure_mode = first.get("failure_mode")
        failure_effect = first.get("failure_effect")

        severity_vals = [
            parse_number(r.get("severity"))
            for r in group
            if parse_number(r.get("severity")) is not None
        ]
        severity = max(severity_vals) if severity_vals else None

        rpn_vals = [
            parse_number(r.get("rpn"))
            for r in group
            if parse_number(r.get("rpn")) is not None
        ]
        rpn = max(rpn_vals) if rpn_vals else None

        # -------------------------------------------------
        # FAILURE DEDUPLICATION (KB-level)
        # -------------------------------------------------
        existing_failure_id = is_duplicate_failure(
            failure_kb,
            system=system,
            element=element,
            function=function,
            failure_mode=failure_mode,
            failure_effect=failure_effect,
        )

        if existing_failure_id:
            failure_id = existing_failure_id
            failure_obj = None
        else:
            failure_id = f"{file_name}__F{failure_counter}"
            failure_counter += 1

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
            failure_kb.add(failure_obj)

        # If reused, load existing failure object
        if failure_obj is None:
            failure_obj = FMEAFailure(**failure_kb.store[failure_id])

        # -------------------------------------------------
        # Causes under this failure
        # -------------------------------------------------
        cause_counter = len(failure_obj.cause_ids) + 1

        for row in group:
            cause_text = row.get("failure_cause")
            if not cause_text:
                continue

            existing_cause_id = is_duplicate_cause(
                cause_kb,
                failure_id=failure_id,
                cause_text=cause_text,
            )

            if existing_cause_id:
                if existing_cause_id not in failure_obj.cause_ids:
                    failure_obj.cause_ids.append(existing_cause_id)
                continue

            cause_id = f"{failure_id}_C{cause_counter}"
            cause_counter += 1

            if source_type == "new_fmea":
                cause_obj = FMEACause(
                    cause_id=cause_id,
                    failure_id=failure_id,
                    failure_mode=failure_mode,
                    failure_element=element,
                    failure_effect=failure_effect,
                    failure_cause=cause_text,
                    discipline=row.get("cause_discipline"),
                    prevention=row.get("controls_prevention"),
                    detection=row.get("current_detection"),
                    detection_value=parse_number(row.get("detection")),
                    occurrence=parse_number(row.get("occurrence")),
                    recommended_action=row.get("recommended_action"),
                )
            else:
                cause_obj = FMEACause(
                    cause_id=cause_id,
                    failure_id=failure_id,
                    failure_mode=failure_mode,
                    failure_element=element,
                    failure_effect=failure_effect,
                    failure_cause=cause_text,
                    discipline=discipline,
                    prevention=None,
                    detection=row.get("current_detection") or row.get("detection"),
                    detection_value=parse_number(row.get("detection")),
                    occurrence=parse_number(row.get("occurrence")),
                    recommended_action=row.get("recommended_action"),
                )

            cause_kb.add(cause_obj)
            failure_obj.cause_ids.append(cause_id)

        # -------------------------------------------------
        # Back-write failure → causes
        # -------------------------------------------------
        failure_kb.store[failure_id]["cause_ids"] = failure_obj.cause_ids

    # -------------------------------------------------
    # Persist failure store
    # -------------------------------------------------
    failure_kb.store_path.write_text(
        json.dumps(failure_kb.store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] {json_path.name} ingested")

