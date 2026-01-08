from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


# ---------------------------
# Helpers
# ---------------------------

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_blank_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip() == ""


def normalize_null_fields_in_memory(failure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only for validation purpose (does NOT write back):
    - "" => None for known optional text fields
    - missing list => []
    """
    f = dict(failure)

    # Optional text fields: treat "" as None
    for k in ["failure_effect", "product"]:
        if k in f and is_blank_str(f[k]):
            f[k] = None

    # Lists: ensure list type
    for k in ["supporting_sentence_ids", "cause_ids"]:
        if k not in f or f[k] is None:
            f[k] = []
        elif not isinstance(f[k], list):
            # keep as-is; validator will report type error
            pass

    return f


# ---------------------------
# K1: Schema completeness (Hard)
# ---------------------------

def check_failure_schema(failure_id: str, f: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    # required
    if not f.get("failure_id") and not failure_id:
        errors.append("missing failure_id")
    if not f.get("failure_element") or is_blank_str(f.get("failure_element")):
        errors.append("missing failure_element")
    if not f.get("failure_mode") or is_blank_str(f.get("failure_mode")):
        errors.append("missing failure_mode")
    if not f.get("failure_effect") or is_blank_str(f.get("failure_effect")):
        errors.append("missing failure_effect")

    # type expectations (strict)
    for lk in ["supporting_sentence_ids", "cause_ids"]:
        if lk in f and not isinstance(f[lk], list):
            errors.append(f"{lk} should be a list (got {type(f[lk]).__name__})")

    # # maintenance required
    # maint = f.get("maintenance")
    # if not isinstance(maint, dict):
    #     errors.append("missing maintenance dict")
    # else:
    #     if not maint.get("review_status"):
    #         errors.append("missing maintenance.review_status")
    #     if not maint.get("version"):
    #         errors.append("missing maintenance.version")
    #     if not maint.get("last_updated"):
    #         errors.append("missing maintenance.last_updated")

    return errors


def check_cause_schema(cause_id: str, c: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if not c.get("cause_id") and not cause_id:
        errors.append("missing cause_id")
    if not c.get("failure_id") or is_blank_str(c.get("failure_id")):
        errors.append("missing failure_id (in cause)")
    if not c.get("failure_cause") or is_blank_str(c.get("root_cause")):
        errors.append("missing root_cause")

    # Optional but typed fields (if present)
    for k in ["discipline", "cause_level", "confidence"]:
        if k in c and c[k] is not None and not isinstance(c[k], str):
            errors.append(f"{k} should be str or null")

    # maint = c.get("maintenance")
    # if not isinstance(maint, dict):
    #     errors.append("missing maintenance dict")
    # else:
    #     if not maint.get("review_status"):
    #         errors.append("missing maintenance.review_status")
    #     if not maint.get("version"):
    #         errors.append("missing maintenance.version")

    return errors


# ---------------------------
# K2: Null semantics (Hard)
# ---------------------------

def check_null_semantics_failure(failure_id: str, f_raw: Dict[str, Any]) -> List[str]:
    """
    Rules (recommended):
    - optional text fields should be None (not "")
    - lists should be [] (not None / "")
    """
    issues: List[str] = []

    # optional text fields should not be ""
    for k in ["failure_effect", "product"]:
        if k in f_raw and is_blank_str(f_raw[k]):
            issues.append(f'{k} is "" (prefer null/None)')

    # lists should not be None / ""
    for lk in ["supporting_sentence_ids", "cause_ids"]:
        if lk in f_raw and f_raw[lk] is None:
            issues.append(f"{lk} is null (prefer [])")
        if lk in f_raw and is_blank_str(f_raw[lk]):
            issues.append(f'{lk} is "" (prefer [])')

    return issues


def check_null_semantics_cause(cause_id: str, c_raw: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    # root_cause must not be ""
    if "root_cause" in c_raw and is_blank_str(c_raw["root_cause"]):
        issues.append('root_cause is "" (must be non-empty)')
    return issues


# ---------------------------
# K3: ID uniqueness (Hard)
# ---------------------------

def check_unique_ids(store: Dict[str, Any], id_field: str) -> List[str]:
    """
    store is dict keyed by id. But still check:
    - value's id_field matches key (common corruption)
    """
    errors: List[str] = []
    for key_id, obj in store.items():
        if isinstance(obj, dict):
            inner = obj.get(id_field)
            if inner and inner != key_id:
                errors.append(f"id mismatch: key={key_id} but {id_field}={inner}")
    return errors


# ---------------------------
# K4: Referential integrity (Hard)
# ---------------------------

def check_referential_integrity(
    failure_store: Dict[str, Dict[str, Any]],
    cause_store: Dict[str, Dict[str, Any]],
) -> List[str]:
    errors: List[str] = []

    cause_ids_all = set(cause_store.keys())
    failure_ids_all = set(failure_store.keys())

    # Failure.cause_ids must exist in CauseKB
    for fid, f in failure_store.items():
        f_norm = normalize_null_fields_in_memory(f)
        cause_ids = f_norm.get("cause_ids", [])
        if isinstance(cause_ids, list):
            for cid in cause_ids:
                if cid not in cause_ids_all:
                    errors.append(f"failure {fid} references missing cause_id: {cid}")
        else:
            errors.append(f"failure {fid} cause_ids is not list")

    # Cause.failure_id must exist in FailureKB
    for cid, c in cause_store.items():
        fid = c.get("failure_id")
        if not fid or not isinstance(fid, str):
            errors.append(f"cause {cid} has invalid failure_id: {fid}")
        elif fid not in failure_ids_all:
            errors.append(f"cause {cid} references missing failure_id: {fid}")

    return errors


# ---------------------------
# Runner
# ---------------------------

def validate_k1_k4(
    failure_kb_dir: Path,
    cause_kb_dir: Path,
    max_print: int = 30,
) -> Dict[str, Any]:
    failure_store_path = failure_kb_dir / "fmea_failure_store.json"
    cause_store_path = cause_kb_dir / "fmea_cause_store.json"

    failure_store = load_json(failure_store_path)
    cause_store = load_json(cause_store_path)

    # Collect issues
    k1_errors: List[str] = []
    k2_issues: List[str] = []
    k3_errors: List[str] = []
    k4_errors: List[str] = []

    # K1 + K2 for failures
    for fid, f in failure_store.items():
        if not isinstance(f, dict):
            k1_errors.append(f"failure {fid} value is not dict")
            continue
        k1_errors.extend([f"failure {fid}: {e}" for e in check_failure_schema(fid, f)])
        k2_issues.extend([f"failure {fid}: {e}" for e in check_null_semantics_failure(fid, f)])

    # K1 + K2 for causes
    for cid, c in cause_store.items():
        if not isinstance(c, dict):
            k1_errors.append(f"cause {cid} value is not dict")
            continue
        k1_errors.extend([f"cause {cid}: {e}" for e in check_cause_schema(cid, c)])
        k2_issues.extend([f"cause {cid}: {e}" for e in check_null_semantics_cause(cid, c)])

    # K3
    k3_errors.extend([f"FailureKB: {e}" for e in check_unique_ids(failure_store, "failure_id")])
    k3_errors.extend([f"CauseKB: {e}" for e in check_unique_ids(cause_store, "cause_id")])

    # K4
    k4_errors.extend(check_referential_integrity(failure_store, cause_store))

    report = {
        "counts": {
            "failures": len(failure_store),
            "causes": len(cause_store),
            "K1_schema_errors": len(k1_errors),
            "K2_null_semantics_issues": len(k2_issues),
            "K3_id_errors": len(k3_errors),
            "K4_ref_integrity_errors": len(k4_errors),
        },
        "samples": {
            "K1_schema_errors": k1_errors[:max_print],
            "K2_null_semantics_issues": k2_issues[:max_print],
            "K3_id_errors": k3_errors[:max_print],
            "K4_ref_integrity_errors": k4_errors[:max_print],
        },
    }
    return report




if __name__ == "__main__":
    # Example wiring: adapt BASE_DIR in your project entrypoint
    BASE_DIR = Path(__file__).resolve().parent

    KB_DATA_ROOT = BASE_DIR.parent / "kb_data"
    FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"
    CAUSE_KB_DIR = KB_DATA_ROOT / "cause_kb"

    
    report = validate_k1_k4(FAILURE_KB_DIR, CAUSE_KB_DIR, max_print=50)

    print("\n========== KB VALIDATION (K1–K4) ==========")
    for k, v in report["counts"].items():
        print(f"{k:28s}: {v}")

    def dump(title: str, items: List[str]):
        if not items:
            return
        print(f"\n--- {title} (showing up to {len(items)}) ---")
        for s in items:
            print(" -", s)

    dump("K1 schema errors", report["samples"]["K1_schema_errors"])
    dump("K2 null semantics issues", report["samples"]["K2_null_semantics_issues"])
    dump("K3 id errors", report["samples"]["K3_id_errors"])
    dump("K4 referential integrity errors", report["samples"]["K4_ref_integrity_errors"])

    
