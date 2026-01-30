from kb_structure import FMEAFailureKB, FMEAFailure, FMEACause, FileMeta, FileMetaStore

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

def normalize_excel_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("\u00A0", " ").replace("\u3000", " ")
    s = s.strip()
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
    tokens = set(re.findall(r"[a-z0-9]+", ft))
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
        "system", "subsystem", "overall", "general","emission", 
    }

    if tokens & electronics_tokens:
        return "HW"

    if tokens & mechanics_tokens:
        return "MCH"

    if tokens & software_tokens:
        return "ESW"

    if tokens & process_tokens:
        return "process"

    if tokens & design_tokens:
        return "design"

    if tokens & generic_tokens:
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

    ft_raw = failure_type.strip()
    if "/" not in ft_raw:
        # No structure → try infer discipline only
        discipline = infer_discipline_from_failure_type(ft_raw)
        return discipline, None

    left_raw, right_raw = ft_raw.split("/", 1)
    left = normalize(left_raw.strip())
    right = normalize(right_raw.strip())

    # print("DEBUG left:", repr(left), "right:", repr(right))

    # -----------------------------
    # Tokenize right safely (word level)
    # -----------------------------
    right_tokens = set(re.findall(r"[a-z0-9]+", right))

    # -----------------------------
    # 1. Generic bucket → discipline
    # -----------------------------
    if right in GENERIC_RIGHT_TOKENS:
        return infer_discipline_from_failure_type(left), None

    # -----------------------------
    # 2. Explicit element tokens
    # -----------------------------
    if right_tokens & ELEMENT_RIGHT_TOKENS:
        return None, f"{left} / {right}"

    # -----------------------------
    # 3. Heuristic element detection
    # -----------------------------
    if is_failure_element_term(right):
        return None, f"{left} / {right}"

    # -----------------------------
    # 4. Discipline inference fallback
    # -----------------------------
    inferred = infer_discipline_from_failure_type(left)
    if inferred:
        return inferred, None

    # -----------------------------
    # 5. Default: treat as element
    # -----------------------------
    return None, f"{left} / {right}"

def map_discipline_to_fmea_type(
    discipline: str | None,
) -> str | None:
    if not discipline:
        return None

    discipline = discipline.lower()

    if discipline == "process":
        return "process"

    if discipline in {"hw", "mch", "esw", "design"}:
        return "design"

    if discipline in {"other", "system"}:
        return "system"

    return None


def build_failure_signature(row: dict) -> tuple:
    content = row.get("content", {})

    if row.get("source_type") == "new_fmea":
        return (
            normalize(content.get("system_name")),
            normalize(content.get("system_element")),
            normalize(content.get("function")),
            normalize(content.get("failure_mode")),
            normalize(content.get("failure_effect")),
        )
    else:  # old_fmea
        return (
            normalize(content.get("process_step")),
            normalize(content.get("failure_mode")),
            normalize(content.get("failure_effect")),
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
    failure_kb: FMEAFailureKB,
    failure_id: str,
    cause_text: str,
) -> str | None:
    """
    Deduplicate causes under the same failure
    """
    if not cause_text:
        return None

    nc = normalize(cause_text)

    for cid, c in failure_kb.cause_store.items():
        if c.get("failure_id") != failure_id:
            continue

        existing_text = c.get("failure_cause")
        if not existing_text:
            continue

        if normalize(existing_text) == nc:
            return cid

    return None



# =========================================================
# Ingest
# =========================================================

def ingest_fmea_json(
    json_path: Path,
    failure_kb,
    cause_kb,
    meta_kb,
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
            discipline = first.get("cause_discipline") 
            process_step = None 
            fmea_type = "design"
        else:
            system = None
            process_step = first.get("process_step")

            discipline, element = parse_failure_type_semantics(process_step)

            function = None
            # fmea_type =  map_discipline_to_fmea_type(discipline)
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
                fmea_type=fmea_type,
                failure_effect=failure_effect,
                process_step=process_step,
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
                    # fmea_type=fmea_type,
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

            failure_kb.add_cause(cause_obj)
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

def ingest_fmea_jsonl(
    jsonl_path: Path,
    failure_kb,
    # cause_kb,
    meta_kb,
):
    # =================================================
    # 0) Load JSONL / JSON
    # =================================================
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

    print(f"[INGEST] {jsonl_path.name}")

    # =================================================
    # 1) PRIMARY GROUP: by file_name in each row
    # =================================================
    rows_by_file = defaultdict(list)
    for row in rows:
        file_name = row.get("file_name")
        if not file_name:
            raise ValueError("JSONL row missing file_name; cannot determine FMEA boundary")
        rows_by_file[file_name].append(row)

    # =================================================
    # 2) FILE-LEVEL INGEST
    # =================================================
    for file_name, rows in rows_by_file.items():

        # -------------------------------------------------
        # META (PER FILE)
        # -------------------------------------------------
        first = rows[0]
        metadata = first.get("metadata", {})
        labels = first.get("labels", {})

        file_meta = FileMeta(
            source_type=first.get("source_type"),
            released=metadata.get("released"),
            productName=metadata.get("productName"),
            productPnID= metadata.get("productPnId"),
            # project_description=metadata.get("project_description"),
            file_name=file_name,
            product_domain=labels.get("domain"),
        )
        meta_kb.add(file_meta)
        fmea_type = labels.get("fmea_type") or "system"

        # -------------------------------------------------
        # SECONDARY GROUP: file-internal failure signature
        # -------------------------------------------------
        grouped = defaultdict(list)
        for row in rows:
            sig = build_failure_signature(row)
            grouped[sig].append(row)

        failure_counter = 1

        # =================================================
        # 3) FAILURE INGEST (GROUP LEVEL)
        # =================================================
        for _, group in grouped.items():
            first = group[0]
            source_type = first.get("source_type")
            content = first.get("content", {})

            # ---------------------------------------------
            # Restore new / old FMEA semantic logic
            # ---------------------------------------------
            if source_type == "new_fmea":
                system = content.get("system_name")
                element = content.get("system_element")
                function = content.get("function")
                discipline = content.get("cause_discipline")
                process_step = None
            else:
                system = None
                process_step = content.get("process_step")
                discipline, element = parse_failure_type_semantics(process_step)
                function = None
                # fmea_type = map_discipline_to_fmea_type(discipline)

            failure_mode = content.get("failure_mode")
            if isinstance(failure_mode, str) and failure_mode.strip() and failure_mode.strip().replace(".", "", 1).isdigit():
                print(f"Format error: failure_mode is numeric-string in {file_name}, row_index={row.get('row_index')}")
                continue
            failure_effect = content.get("failure_effect")

            # ---------------------------------------------
            # severity / rpn = MAX within failure group
            # ---------------------------------------------
            severity_vals = [
                parse_number(r.get("RPN", {}).get("severity"))
                for r in group
                if parse_number(r.get("RPN", {}).get("severity")) is not None
            ]
            severity = max(severity_vals) if severity_vals else None

            rpn_vals = [
                parse_number(r.get("RPN", {}).get("RPN"))
                for r in group
                if parse_number(r.get("RPN", {}).get("RPN")) is not None
            ]
            rpn = max(rpn_vals) if rpn_vals else None

            # ---------------------------------------------
            # FAILURE DEDUP (KB-level)
            # ---------------------------------------------
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
                failure_obj = FMEAFailure(**failure_kb.store[failure_id])
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

                    # =====  file-level context =====
                    productPnID=file_meta.productPnID,
                    product_domain=file_meta.product_domain,

                    fmea_type=fmea_type,
                    process_step=process_step,
                )
                failure_kb.add(failure_obj)

            # =================================================
            # 4) CAUSE INGEST (ROW LEVEL UNDER FAILURE)
            # =================================================
            cause_counter = len(failure_obj.cause_ids) + 1

            for row in group:
                content = row.get("content", {})
                rpn_block = row.get("RPN", {})

                cause_text = normalize_excel_text(content.get("failure_cause"))


                if not content.get("failure_mode") and not content.get("failure_cause"):
                    print(f"[SKIP] Missing failure_mode/cause in {file_name}, row_index={row.get('row_index')}")
                    continue

                if isinstance(cause_text, str) and cause_text.strip() and cause_text.strip().replace(".", "", 1).isdigit():
                    print(f"Format error: failure_cause is numeric-string in {file_name}, row_index={row.get('row_index')}")
                    continue
                if not cause_text:
                    continue


                existing_cause_id = is_duplicate_cause(
                    failure_kb,
                    failure_id=failure_id,
                    cause_text=cause_text,
                )

                if existing_cause_id:
                    
                    existing_cause = failure_kb.cause_store.get(existing_cause_id)

                    if existing_cause:
                        cause_ref = {
                            "cause_id": existing_cause_id,
                            "cause_text": existing_cause.get("failure_cause"),
                        }

                        if not any(
                            c.get("cause_id") == existing_cause_id
                            for c in failure_obj.cause_ids
                        ):
                            failure_obj.cause_ids.append(cause_ref)

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
                        discipline=content.get("cause_discipline"),
                        prevention=content.get("controls_prevention"),
                        detection=content.get("current_detection"),
                        detection_value=parse_number(rpn_block.get("detection")),
                        occurrence=parse_number(rpn_block.get("occurrence")),
                        recommended_action=content.get("recommended_action"),

                        # =====  file-level context =====
                        productPnID=file_meta.productPnID,
                        product_domain=file_meta.product_domain,
                        source_type=source_type,
                        fmea_type=fmea_type,
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
                        detection=content.get("current_detection") or rpn_block.get("detection"),
                        detection_value=parse_number(rpn_block.get("detection")),
                        occurrence=parse_number(rpn_block.get("occurrence")),
                        recommended_action=content.get("recommended_action"),

                        # =====  file-level context =====
                        productPnID=file_meta.productPnID,
                        product_domain=file_meta.product_domain,
                        source_type=source_type,
                        fmea_type=fmea_type,

  
                    )

                failure_kb.add_cause(cause_obj)
                cause_ref = {
                    "cause_id": cause_id,
                    "cause_text": cause_text,
                }
                failure_obj.cause_ids.append(cause_ref)

            # ---------------------------------------------
            # Back-write failure → causes
            # ---------------------------------------------
            failure_kb.store[failure_id]["cause_ids"] = failure_obj.cause_ids

    # =================================================
    # 5) Persist failure store
    # =================================================
    failure_kb.store_path.write_text(
        json.dumps(failure_kb.store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] {jsonl_path.name} ingested")