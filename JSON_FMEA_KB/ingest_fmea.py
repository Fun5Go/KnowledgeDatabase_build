from kb_structure import FMEAFailureKB, FileMeta, FileMetaStore, FailureEntity, FailureSemanticNode, Sentence, SentenceKB
from sentence_builder import _build_full_chain_sentence
import json
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime
import hashlib


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

def to_year(released) -> int | None:
    if not released:
        return None

    s = str(released).strip()
    if not s or s.lower() in {"unknown", "n/a", "na", "none", "null", "-"}:
        return None

    # normalize common ISO variants
    s = s.replace("Z", "+00:00")

    # if it's just a year like "2019"
    if len(s) == 4 and s.isdigit():
        return int(s)

    try:
        return datetime.fromisoformat(s).year
    except ValueError:
        # fallback: extract leading year if present, e.g. "2019-01-07 ..."
        if len(s) >= 4 and s[:4].isdigit():
            return int(s[:4])
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
            normalize(content.get("system_element")),
            normalize(content.get("function")),
            normalize(content.get("failure_mode")),
            normalize(content.get("failure_effect")),
            normalize_excel_text(content.get("failure_cause")),  
        )
    else:  # old_fmea
        return (
            normalize(content.get("process_step")),
            normalize(content.get("failure_mode")),
            normalize(content.get("failure_effect")),
            normalize_excel_text(content.get("failure_cause")),  
        )

def make_semantic_id(field_type: str, text: str) -> str:
    norm = normalize_excel_text(text).lower()
    h = hashlib.md5(norm.encode("utf-8")).hexdigest()[:12]
    return f"{field_type}:{h}"

def collect_semantic(
    semantic_map: dict,
    *,
    semantic_id: str,
    field_type: str,
    text: str,
    failure_id: str,
    source_type: str,
):
    node = semantic_map.setdefault(
        semantic_id,
        {
            "semantic_id": semantic_id,
            "field_type": field_type,
            "text": text,
            "failure_ids": [],
            "source_type": source_type,
        },
    )
    if failure_id not in node["failure_ids"]:
        node["failure_ids"].append(failure_id)




# =========================================================
# Ingest
# =========================================================
def ingest_fmea_jsonl(
    jsonl_path: Path,
    failure_kb,
    meta_kb,
    sentence_kb
):
    # =================================================
    # 0) Load JSONL
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
    # 1) PRIMARY GROUP: by file_name
    # =================================================
    rows_by_file = defaultdict(list)
    for row in rows:
        file_name = row.get("file_name")
        if not file_name:
            raise ValueError("JSONL row missing file_name")
        rows_by_file[file_name].append(row)

    # =================================================
    # 2) FILE-LEVEL INGEST
    # =================================================
    for file_name, rows in rows_by_file.items():

        # semantic accumulator (PER FILE!)
        semantic_nodes: dict[str, dict] = {}

        # -------------------------------------------------
        # META
        # -------------------------------------------------
        first = rows[0]
        metadata = first.get("metadata", {})
        labels = first.get("labels", {})

        file_meta = FileMeta(
            source_type=first.get("source_type"),
            released=metadata.get("released"),
            productName=metadata.get("productName"),
            productPnID=metadata.get("productPnId"),
            file_name=file_name,
            product_domain=labels.get("domain"),
        )
        meta_kb.add(file_meta)

        fmea_type = labels.get("fmea_type") or "system"
        released_year = to_year(file_meta.released)

        # -------------------------------------------------
        # SECONDARY GROUP: failure signature
        # -------------------------------------------------
        grouped = defaultdict(list)
        for row in rows:
            sig = build_failure_signature(row)
            grouped[sig].append(row)

        # =================================================
        # 3) FAILURE INGEST
        # =================================================
        for group in grouped.values():
            first = group[0]

            row_index = first.get("row_index")
            if row_index is None:
                raise ValueError(f"Missing row_index in {file_name}")

            failure_id = f"{file_name}__R{row_index}"
            content = first.get("content", {})
            source_type = first.get("source_type")

            if source_type == "new_fmea":
                system = content.get("system_name")
                element = content.get("system_element")
                function = content.get("function")
                discipline = content.get("cause_discipline")
                process_step = None
            else:
                system = None
                element = content.get("process_step")
                function = None
                discipline = None
                process_step = None

            failure_mode = content.get("failure_mode")
            failure_effect = content.get("failure_effect")

            # ---------- severity / rpn ----------
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

            # ---------- semantic IDs ----------
            mode_id = make_semantic_id("mode", failure_mode) if failure_mode else None
            element_id = make_semantic_id("element", element) if element else None
            effect_id = make_semantic_id("effect", failure_effect) if failure_effect else None

            if mode_id:
                collect_semantic(semantic_nodes,
                                 semantic_id=mode_id,
                                 field_type="mode",
                                 text=failure_mode,
                                 failure_id=failure_id,
                                 source_type=source_type)

            if element_id:
                collect_semantic(semantic_nodes,
                                 semantic_id=element_id,
                                 field_type="element",
                                 text=element,
                                 failure_id=failure_id,
                                 source_type=source_type)

            if effect_id:
                collect_semantic(semantic_nodes,
                                 semantic_id=effect_id,
                                 field_type="effect",
                                 text=failure_effect,
                                 failure_id=failure_id,
                                 source_type=source_type)

            # ---------- entity ----------
            cause_text = content.get("failure_cause")

            cause_semantic_id = (
                make_semantic_id("cause", cause_text)
                if cause_text else None
            )

            if cause_semantic_id:
                collect_semantic(
                    semantic_nodes,
                    semantic_id=cause_semantic_id,
                    field_type="cause",
                    text=cause_text,
                    failure_id=failure_id,
                    source_type=source_type
                )


            failure_entity = FailureEntity(
                failure_id=failure_id,
                mode_id=mode_id,
                element_id=element_id,
                effect_id=effect_id,
                cause_id=cause_semantic_id,

                failure_mode_text=failure_mode,
                failure_element_text=element,
                failure_effect_text=failure_effect,
                failure_cause_text=cause_text,

                system=system,
                function=function,
                process_step=process_step,
                discipline=discipline,

                severity=severity,
                rpn=rpn,

                source_type=source_type,
                fmea_type=fmea_type,
                productPnID=file_meta.productPnID,
                product_domain=file_meta.product_domain,
                released_year=released_year,
            )
            failure_kb.upsert_failure_entity(failure_entity)
            full_chain_text = _build_full_chain_sentence(
                element,
                failure_mode,
                cause_text,
                failure_effect,
            )

            if full_chain_text:

                sentence_id = failure_id

                sentence_obj = Sentence(
                    failure_id=failure_id,
                    text=full_chain_text,
                    source_type=source_type,
                    product_domain=file_meta.product_domain,
                )

                sentence_kb.add_sentence(
                    sentence_id,
                    sentence_obj,
                    overwrite=True
                )

        # =================================================
        # 4) FLUSH semantic nodes (PER FILE)
        # =================================================
        for node in semantic_nodes.values():
            failure_kb.upsert_semantic_node(
                semantic_id=node["semantic_id"],
                field_type=node["field_type"],
                text=node["text"],
                failure_ids=node["failure_ids"],
                source_type = node["source_type"],
            )

    print(f"[OK] {jsonl_path.name} ingested")