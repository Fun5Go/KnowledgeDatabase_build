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
    → discipline="HW", cause="Encoder noise"
    """
    if not cause_raw:
        return "", ""

    m = re.match(r"^\[[^\]-]*-\s*([A-Za-z]+)\]\s*(.*)", str(cause_raw))
    if m:
        return m.group(1).strip(), m.group(2).strip()

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

    df = df[df["failure_cause"].notna()]
    df = df[df["failure_cause"].astype(str).str.strip() != ""]
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
    fmea_date,
    project_description,
    dfmea,
    context,
    file_name
):

    records = []

    for _, row in dfmea.iterrows():

        system_element, function = find_context_for_row(
            context,
            row["excel_row"]
        )

        failure_mode = strip_prefix(to_scalar(row.get("failure_mode", "")))
        failure_effect = strip_prefix(to_scalar(row.get("failure_effect", "")))
        discipline, failure_cause = extract_discipline(
            to_scalar(row.get("failure_cause", ""))
        )

        record = {
            "source_type": "new_fmea",
            "fmea_date": fmea_date,
            "project_description": project_description,

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

            "controls_prevention": str(
                to_scalar(row.get("controls_prevention", ""))
            ).strip(),

            "current_detection": str(
                to_scalar(row.get("current_detection", ""))
            ).strip(),

            "recommended_action": str(
                to_scalar(row.get("recommended_action", ""))
            ).strip(),

            "file_name": file_name
        }

        records.append(record)

    return records



###############################################################################
# Step 4: Main entry
###############################################################################

def dfmea_to_json_xlsm(path, output_json, sheet_index=1):

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
    project_description = meta.get("project_description", "")

    context = extract_structure_context(path, sheet_index)
    dfmea = load_dfmea_table(path, sheet_index)

    flat_records = build_flat_failures(
        system_name=system_name,
        fmea_date=fmea_date,
        project_description=project_description,
        dfmea=dfmea,
        context=context,
        file_name=file_name
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(flat_records, f, ensure_ascii=False, indent=2)

    print("JSON saved to:", output_json)


def process_dfmea_xlsm(path, output_json, sheet_index=1):
    dfmea_to_json_xlsm(path, output_json, sheet_index)
