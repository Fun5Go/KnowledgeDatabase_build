import pandas as pd
import json
import numpy as np
import os
import math
import re
from fmea_to_json.common_utils import extract_metadata_from_file


###############################################################################
# Helper functions
###############################################################################

def to_scalar(x):
    """Convert pandas / numpy types into native Python scalar."""
    if isinstance(x, pd.Series):
        return to_scalar(x.iloc[0])
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and math.isnan(x):
        return ""
    return x


def strip_prefix(text):
    """
    Remove leading prefixes like:
    [A01], [S01F01], [S01 - HW], etc.
    """
    if not text:
        return ""
    return re.sub(r"^\[[^\]]+\]\s*", "", str(text)).strip()


def extract_discipline(cause_raw):
    """
    Extract discipline from cause text.

    Example:
    "[S01 - HW] Encoder noise"
    → cause_discipline="hardware", cause="Encoder noise"
    """
    if not cause_raw:
        return "", ""

    s = str(cause_raw)

    m = re.match(r"^\[[^\]-]*-\s*([A-Za-z]+)\]\s*(.*)", s)
    if m:
        tag = m.group(1).strip().upper()
        cause = m.group(2).strip()

        tag_map = {
            "ESW": "software",
            "HW": "hardware",
            "MCH": "mechanics",
        }
        return tag_map.get(tag, ""), cause

    return "", strip_prefix(cause_raw)



###############################################################################
# Step 1: Extract system name
###############################################################################

def extract_system_name(path, sheet_index=1):
    """
    Read system name from the header area of the sheet.
    """
    df = pd.read_excel(
        path,
        sheet_name=sheet_index,
        header=None,
        nrows=20,
        engine="openpyxl"
    )

    for r in range(len(df)):
        col0 = str(df.iloc[r, 0]).replace("：", ":").strip().lower()
        if "system" in col0:
            if df.shape[1] > 2:
                val = df.iloc[r, 2]
                if not (isinstance(val, float) and math.isnan(val)):
                    return strip_prefix(val)
            if ":" in col0:
                return strip_prefix(col0.split(":", 1)[1])

    return ""


###############################################################################
# Step 2: Load DFMEA table
###############################################################################

def load_dfmea_table(path, sheet_index=1):
    df = pd.read_excel(
        path,
        sheet_name=sheet_index,
        header=6,
        engine="openpyxl"
    )

    df = df.dropna(how="all")
    df["excel_row"] = df.index + 7  # Excel 行号

    col_map = {
        "Potential Effect(s) of Failure\n(Activity)": "failure_effect",
        "Severity": "severity",
        "Potential Failure Mode\n(Process step)": "failure_mode",
        "Potential Cause(s) of Failure\n(per discipline)": "failure_cause",
        "Controls prevention": "controls_prevention",
        "Occurrence": "occurrence",
        "Current Detection": "current_detection",
        "Detection": "detection",
        "RPN": "rpn",
        "Recommended Actions": "recommended_action"
    }

    df = df.rename(columns=col_map)

    df = df[
        df[["failure_mode", "failure_effect", "failure_cause"]]
        .astype(str)
        .apply(lambda x: x.str.strip() != "")
        .any(axis=1)
    ]
    df = df[df["severity"].astype(str).str.strip() != "-"]

    return df


def extract_structure_context(path, sheet_index=1):
    df = pd.read_excel(path, sheet_name=sheet_index, header=None, engine="openpyxl")

    context = []
    current_element = ""
    current_function = ""

    for r in range(len(df)):
        col0 = str(df.iloc[r, 0]).strip().lower()
        col2 = ""

        if df.shape[1] > 2:
            v2 = df.iloc[r, 2]
            if not (isinstance(v2, float) and math.isnan(v2)):
                col2 = str(v2).strip()

        if "system element" in col0:
            current_element = strip_prefix(col2)

        elif "function" in col0:
            current_function = strip_prefix(col2)

        context.append({
            "row": r,
            "system_element": current_element,
            "function": current_function
        })

    return context

def find_context_for_row(context, excel_row):
    candidates = [c for c in context if c["row"] < excel_row]
    if not candidates:
        return "", ""
    last = candidates[-1]
    return last["system_element"], last["function"]






###############################################################################
# Step 3: Build flat FMEA blocks (NO prefix matching)
###############################################################################

def build_flat_failures(
    system_name,
    project_description,
    dfmea,
    context,
    file_name,
    metadata
):

    records = []

    # ===== last cache =====
    last_failure_mode = ""
    last_failure_effect = ""
    last_controls_prevention = ""
    last_current_detection = ""
    last_recommended_action = ""
    last_function = ""

    # ===== clean function（关键修复 NaN）=====
    import pandas as pd
    def clean(x):
        if x is None or pd.isna(x):
            return ""
        s = str(x).strip()
        if s.lower() == "nan":
            return ""
        return s

    for _, row in dfmea.iterrows():

        system_element, function = find_context_for_row(
            context,
            row["excel_row"]
        )

        # ===== function 切换 → reset block =====
        if function != last_function:
            last_failure_mode = ""
            last_failure_effect = ""
            last_controls_prevention = ""
            last_current_detection = ""
            last_recommended_action = ""
            last_function = function

        # ===== raw =====
        failure_mode_raw = clean(row.get("failure_mode", ""))
        failure_effect_raw = clean(row.get("failure_effect", ""))
        failure_cause_raw = clean(row.get("failure_cause", ""))

        controls_prevention_raw = clean(row.get("controls_prevention", ""))
        current_detection_raw = clean(row.get("current_detection", ""))
        recommended_action_raw = clean(row.get("recommended_action", ""))

        # ===== 判断是否有效行（不依赖 severity）=====
        has_text = any([
            failure_mode_raw,
            failure_effect_raw,
            failure_cause_raw
        ])

        if not has_text:
            continue

        # ===== mode 继承 =====
        if failure_mode_raw:
            failure_mode = strip_prefix(failure_mode_raw)
            last_failure_mode = failure_mode
        else:
            failure_mode = last_failure_mode

        # ===== effect 继承 =====
        if failure_effect_raw:
            failure_effect = strip_prefix(failure_effect_raw)
            last_failure_effect = failure_effect
        else:
            failure_effect = last_failure_effect

        # ===== cause（不继承）=====
        discipline, failure_cause = extract_discipline(failure_cause_raw)

        # ===== controls 继承 =====
        if controls_prevention_raw:
            controls_prevention = controls_prevention_raw
            last_controls_prevention = controls_prevention
        else:
            controls_prevention = last_controls_prevention

        if current_detection_raw:
            current_detection = current_detection_raw
            last_current_detection = current_detection
        else:
            current_detection = last_current_detection

        if recommended_action_raw:
            recommended_action = recommended_action_raw
            last_recommended_action = recommended_action
        else:
            recommended_action = last_recommended_action

        # ===== text =====
        def safe_str(x):
            return str(x).strip() if x is not None else ""

        text = (
            f"Product: {safe_str(metadata.get('productName'))}. "
            f"System name: {safe_str(system_name)}. "
            f"System element: {safe_str(system_element)}. "
            f"Function: {safe_str(function)}. "
            f"Failure mode: {safe_str(failure_mode)}. "
            f"Failure cause: {safe_str(failure_cause)}. "
            f"Failure effect: {safe_str(failure_effect)}. "
            f"Cause discipline: {safe_str(discipline)}. "
            f"Controls (prevention): {safe_str(controls_prevention)}. "
            f"Controls (detection): {safe_str(current_detection)}. "
            f"Recommended action: {safe_str(recommended_action)}."
        )

        record = {
            "source_type": "new_fmea",

            "file_name": file_name,
            "project_description": project_description,
            "released": metadata.get("released"),
            "productId": metadata.get("productId"),
            "productPnId": metadata.get("productPnId"),
            "productName": metadata.get("productName"),

            "system_name": system_name,
            "system_element": system_element,
            "function": function,

            "failure_mode": failure_mode,
            "failure_effect": failure_effect,
            "failure_cause": failure_cause,
            "cause_discipline": discipline,

            "severity": to_scalar(row.get("severity", "")),
            "occurrence": to_scalar(row.get("occurrence", "")),
            "detection": to_scalar(row.get("detection", "")),
            "rpn": to_scalar(row.get("rpn", "")),

            "controls_prevention": controls_prevention,
            "current_detection": current_detection,
            "recommended_action": recommended_action,

            "text": text
        }

        records.append(record)

    return records


###############################################################################
# Step 4: Main entry
###############################################################################

def process_dfmea_xlsm(path, output_json, sheet_index=1,fmea_index=None):
    file_name = os.path.splitext(os.path.basename(path))[0]
    system_name = extract_system_name(path, sheet_index)

    meta = extract_metadata_from_file(
        path,
        sheet_index=sheet_index,
        project_cell="I2",
        date_cell="T4",
        date_fallback_cell=None
    )
    fmea_date = meta.get("fmea_date", "")

    # ===== 从 fmea_index 补 metadata =====
    idx = {}
    if fmea_index and file_name in fmea_index:
        idx = fmea_index.get(file_name, {})


    meta.update({
        "released": idx.get("released") or fmea_date or None,
        "productId": idx.get("productId"),
        "productPnId": idx.get("productPnId"),
        "productName": idx.get("productName"),
    })

    project_description = meta.get("project_description", "")

    context = extract_structure_context(path, sheet_index)
    dfmea = load_dfmea_table(path, sheet_index)

    flat_records = build_flat_failures(
        system_name=system_name,
        project_description=project_description,
        dfmea=dfmea,
        context=context,
        file_name=file_name,
        metadata=meta
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(flat_records, f, ensure_ascii=False, indent=2)

    print("JSON saved to:", output_json)

