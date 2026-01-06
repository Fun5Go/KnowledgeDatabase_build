import pandas as pd
import json
import numpy as np
import os
import math
import re


###############################################################################
# Helper functions
###############################################################################

def strip_prefix(text):
    """Remove leading prefixes like [S01F01], [S01 - HW], etc."""
    if not text:
        return ""
    return re.sub(r"^\[[^\]]+\]\s*", "", str(text)).strip()


def extract_discipline(cause_raw):
    """
    Extract discipline from cause text.
    Example:
    "[S01 - HW] Encoder noise" → ("HW", "Encoder noise")
    """
    if not cause_raw:
        return "", ""
    m = re.match(r"^\[[^\]-]*-\s*([A-Za-z]+)\]\s*(.*)", str(cause_raw))
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", strip_prefix(cause_raw)


def to_scalar(x):
    """Convert pandas / numpy values to Python scalars."""
    if isinstance(x, pd.Series):
        return to_scalar(x.iloc[0])
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and math.isnan(x):
        return ""
    return x


###############################################################################
# 1. Extract system name
###############################################################################

def extract_system_name(path, sheet_index=1):
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
# 2. Extract structure context (system element / function with row index)
###############################################################################

def extract_structure_context(path, sheet_index=1):
    """
    Read the whole sheet sequentially and track
    current system element and function with row index.
    """
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


###############################################################################
# 3. Load DFMEA table (keep Excel row index)
###############################################################################

def load_dfmea_table(path, sheet_index=1):
    df = pd.read_excel(path, sheet_name=sheet_index, header=6, engine="openpyxl")

    df = df.dropna(how="all")
    df["excel_row"] = df.index + 7  # header=6 → Excel rows start at 7

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


###############################################################################
# 4. Context lookup
###############################################################################

def find_context_for_row(context, excel_row):
    """Find the nearest context above the DFMEA row."""
    candidates = [c for c in context if c["row"] < excel_row]
    if not candidates:
        return "", ""
    last = candidates[-1]
    return last["system_element"], last["function"]


###############################################################################
# 5. Build flat FMEA records
###############################################################################

def build_flat_failures(system_name, dfmea, context, file_name):
    records = []

    for _, row in dfmea.iterrows():

        element, function = find_context_for_row(context, row["excel_row"])

        failure_effect = strip_prefix(row.get("failure_effect", ""))
        failure_mode = strip_prefix(row.get("failure_mode", ""))

        discipline, failure_cause = extract_discipline(
            row.get("failure_cause", "")
        )

        record = {
            "source_type": "new_fmea",

            "system_name": system_name,
            "system_element": element,
            "function": function,

            "failure_effect": failure_effect,
            "severity": to_scalar(row.get("severity", "")),
            "failure_mode": failure_mode,
            "failure_cause": failure_cause,
            "cause_discipline": discipline,

            "controls_prevention": str(to_scalar(row.get("controls_prevention", ""))).strip(),
            "occurrence": to_scalar(row.get("occurrence", "")),
            "current_detection": str(to_scalar(row.get("current_detection", ""))).strip(),
            "detection": to_scalar(row.get("detection", "")),
            "rpn": to_scalar(row.get("rpn", "")),
            "recommended_action": str(to_scalar(row.get("recommended_action", ""))).strip(),

            "file_name": file_name
        }

        records.append(record)

    return records


###############################################################################
# 6. Main function
###############################################################################

def dfmea_to_json(path, output_json, sheet_index=1):

    file_name = os.path.splitext(os.path.basename(path))[0]

    system_name = extract_system_name(path, sheet_index)
    print("System Name:", system_name)

    context = extract_structure_context(path, sheet_index)

    dfmea = load_dfmea_table(path, sheet_index)
    print("DFMEA Rows:", len(dfmea))

    final_json = build_flat_failures(
        system_name,
        dfmea,
        context,
        file_name
    )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print("JSON saved to:", output_json)


###############################################################################
# 7. Batch run
###############################################################################

if __name__ == "__main__":

    input_folder = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\FMEA\Motor"
    output_folder = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\database\fema_json_raw"

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(".xlsm"):
            full_path = os.path.join(input_folder, file)
            output_json = os.path.join(
                output_folder,
                os.path.splitext(file)[0] + ".json"
            )

            print("\n===================================")
            print("Processing:", file)
            print("===================================")

            try:
                dfmea_to_json(full_path, output_json)
            except Exception as e:
                print("ERROR:", e)
