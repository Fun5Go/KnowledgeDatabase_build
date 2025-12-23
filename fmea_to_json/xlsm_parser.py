import pandas as pd
import json
import numpy as np
import os
import pandas as pd
import json
import math
import re
from fmea_to_json.common_utils import to_scalar, strip_prefix, extract_discipline
from fmea_to_json.common_utils import extract_metadata_from_file


###############################################################################
# Helper: convert numpy types to Python native types
###############################################################################
def convert(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

# Trim the prefix [A01]
def strip_prefix(text):
    """
    Removes a prefix like: [A01F01], [S01 - HW], [S03], etc.
    Returns the remaining text.
    """
    return re.sub(r"^\[[^\]]+\]\s*", "", text).strip()

# Extract the discipline type from the cause [HW/ESW/MCH/Other]
def extract_discipline(cause_raw):
    """
    Example:
    "[S01 - HW] Encoder circuit crosstalk" 
      → discipline="HW", cause="Encoder circuit crosstalk"
    """
    m = re.match(r"^\[[^\]-]*-\s*([A-Za-z]+)\]\s*(.*)", cause_raw)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    else:
        # No discipline found → return None + stripped text
        return None, strip_prefix(cause_raw)
    
def to_scalar(x):
    """Convert Pandas Series, numpy types, or weird objects into scalars."""
    if isinstance(x, pd.Series):
        # Take the first non-null value
        return to_scalar(x.iloc[0])
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and math.isnan(x):
        return ""
    return x

###############################################################################
# 1. Extract System Name
###############################################################################
def extract_system_name(path, sheet_index=1):
    df = pd.read_excel(path, sheet_name=sheet_index, header=None, nrows=20, engine="openpyxl")
    for r in range(len(df)):
        col0 = str(df.iloc[r, 0]).replace("：", ":").strip().lower()
        if "system" in col0:
            if df.shape[1] > 2:
                val2 = df.iloc[r, 2]
                if not (isinstance(val2, float) and math.isnan(val2)):
                    return str(val2).strip()
            # fallback
            raw = str(df.iloc[r, 0])
            if ":" in raw:
                return raw.split(":", 1)[1].strip()
    return ""


###############################################################################
# 2. Extract structure blocks (system element + function)
###############################################################################
def extract_structure_blocks(path, sheet_index=1):
    df = pd.read_excel(path, sheet_name=sheet_index, header=None, engine="openpyxl")

    blocks = []
    current_element = None

    for r in range(len(df)): # Entity name is in A column, value is in C column
        col0 = str(df.iloc[r, 0]).strip().lower()
        col2 = ""
        if df.shape[1] > 2:
            v2 = df.iloc[r, 2]
            if not (isinstance(v2, float) and math.isnan(v2)):
                col2 = str(v2).strip()

        # Detect System Element
        if "system element" in col0:
            current_element = col2

        # Detect Function
        elif "function" in col0:
            if col2:
                func_full = col2
                func_key = func_full.split()[0]  # "[S01F01]"
                blocks.append({
                    "system_element": current_element,
                    "function": func_full,
                    "function_key": func_key
                })

    return blocks


###############################################################################
# 3. Load DFMEA Table
###############################################################################
def load_dfmea_table(path, sheet_index=1):
    """
    Output: DataFrame
    """
    df = pd.read_excel(path, sheet_name=sheet_index, header=6, engine="openpyxl")

    df = df.dropna(how="all")
    df = df.ffill()

    col_map = {
        # Failure information
        "Potential Effect(s) of Failure\n(Activity)": "failure_effect",
        "Severity": "severity",
        "Potential Failure Mode\n(Process step)": "failure_mode",
        "Potential Cause(s) of Failure\n(per discipline)": "failure_cause",

        # Current controls
        "Controls prevention": "controls_prevention",  
        "Ref. FS": "ref_fs",                           
        "Ref. TS": "ref_ts",                          

        "Occurrence": "occurrence",                    

        "Current Detection": "current_detection",      
        "Ref. FAT": "ref_fat",                         
        "Ref. QD": "ref_qd",                           

        "Detection": "detection",                     
        "RPN": "rpn",                               

        "Recommended Actions": "recommended_action"    
    }
    df = df.rename(columns=col_map)

    needed = [
        "failure_effect", "failure_mode", "failure_cause",
        "controls_prevention", "ref_fs", "ref_ts",
        "occurrence",
        "current_detection", "ref_fat", "ref_qd",
        "severity","occurrence", "detection", "rpn",
        "recommended_action"
    ]
    df = df[[c for c in needed if c in df.columns]]

    # Filter out garbage rows (no cause = not a valid DFMEA entry)
    df = df[df["failure_cause"].notna()]
    df = df[df["failure_cause"].astype(str).str.strip() != ""]

    # Remove fake rows created by merged cells (severity = "-")
    df = df[df["severity"].astype(str).str.strip() != "-"]

    return df


###############################################################################
# 4. Split DFMEA rows by function (using function prefix key)
###############################################################################
def split_dfmea_by_function(dfmea, structure_blocks):
    df_blocks = []
    for block in structure_blocks:
        key = block["function_key"]
        mask = dfmea["failure_mode"].astype(str).str.startswith(key)
        df_blocks.append(dfmea[mask].copy())
    return df_blocks



###############################################################################
# 5.B Build FINAL nested flatten schema for human and machine readability
###############################################################################
def build_flat_failures_with_text(system_name, fmea_date, project_description, structure_blocks, df_blocks, file_name):
    """
    Docstring for build_flat_failures_with_text
    
    :param system_name: string
    :param structure_blocks: 
    :param df_blocks: DataFrame
    :param file_name: string

    :return: JSON schema with flatten structure
    """
    records = []

    for sb, df_block in zip(structure_blocks, df_blocks):
        element = sb["system_element"]
        function = sb["function"]

        element_clean = strip_prefix(element)
        function_clean = strip_prefix(function)

        for _, row in df_block.iterrows():

            failure_effect = strip_prefix(str(row["failure_effect"]))
            failure_mode = strip_prefix(str(row["failure_mode"]))

            # Cause + discipline
            discipline, failure_cause_clean = extract_discipline(str(row["failure_cause"]))

            severity = to_scalar(row["severity"])
            occurrence = to_scalar(row.get("occurrence", ""))
            detection = to_scalar(row.get("detection", ""))
            rpn = to_scalar(row.get("rpn", ""))

            controls_prevention = str(to_scalar(row.get("controls_prevention", ""))).strip()
            ref_fs = str(to_scalar(row.get("ref_fs", ""))).strip()
            ref_ts = str(to_scalar(row.get("ref_ts", ""))).strip()

            current_detection = str(to_scalar(row.get("current_detection", ""))).strip()
            ref_fat = str(to_scalar(row.get("ref_fat", ""))).strip()
            ref_qd = str(to_scalar(row.get("ref_qd", ""))).strip()

            recommended_action = str(to_scalar(row.get("recommended_action", ""))).strip()


            text = (
                f"The system is {system_name}. "
                f"The system element is {element_clean} and the function is {function_clean}. "
                f"The failure cause is {failure_cause_clean}, which causes the failure mode {failure_mode}. "
                f"The failure mode {failure_mode} leads to the failure effect {failure_effect}. "
                f"The discipline is {discipline}. "
                f"The severity is {severity}, the occurrence is {occurrence}, and the detection is {detection}, "
                f"resulting in an RPN of {rpn}. "
                f"The controls for prevention are {controls_prevention}. "
                f"The current detection controls are {current_detection}. "
                f"The recommended action is {recommended_action}."
            )

            record = {
                "source_type": "new_fmea",
                "fmea_date": fmea_date,
                "project_description": project_description,


                
                "system_name": system_name,
                "system_element": element_clean,
                "function": function_clean,

                "failure_effect": failure_effect,
                "severity": severity,
                "failure_mode": failure_mode,
                "failure_cause": failure_cause_clean,
                "cause_discipline": discipline,

                # NEW FIELDS
                "controls_prevention": controls_prevention,
                "ref_fs": ref_fs,
                "ref_ts": ref_ts,

                "occurrence": occurrence,

                "current_detection": current_detection,
                "ref_fat": ref_fat,
                "ref_qd": ref_qd,

                "detection": detection,
                "rpn": rpn,

                "recommended_action": recommended_action,

                "text": text,
                "file_name": file_name
            }

            records.append(record)

    return records



###############################################################################
# 6. MAIN FUNCTION
###############################################################################
def dfmea_to_json_xlsm(path, output_json, sheet_index=1):

    print("\n=== STEP 0: Extract file name ===")
    file_name = os.path.splitext(os.path.basename(path))[0]

    print("\n=== STEP 1: Reading System Name ===")
    system_name_raw = extract_system_name(path, sheet_index)
    system_name = strip_prefix(system_name_raw)
    print("System Name:", system_name)

    meta = extract_metadata_from_file(
    path,
    sheet_index=sheet_index,   
    project_cell="I2",         
    date_cell="T4",            
    date_fallback_cell=None
)
    fmea_date = meta["fmea_date"]
    project_description = meta["project_description"]

    print("\n=== STEP 2: Reading Structure Blocks ===")
    structure_blocks = extract_structure_blocks(path, sheet_index)
    print("Structure Blocks:", len(structure_blocks))

    print("\n=== STEP 3: Reading DFMEA Table ===")
    dfmea = load_dfmea_table(path, sheet_index)
    print("DFMEA Rows:", len(dfmea))

    print("\n=== STEP 4: Splitting DFMEA by Function ===")
    df_blocks = split_dfmea_by_function(dfmea, structure_blocks)

    print("\n=== STEP 5: Building Hierarchical Schema ===")
    final_json = build_flat_failures_with_text(
    system_name,
    fmea_date,  
    project_description,        # ✅ NEW
    structure_blocks,
    df_blocks,
    file_name
)

    # Write file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print("\n JSON saved to:", output_json)


def process_dfmea_xlsm(path, output_json, sheet_index=1):
    dfmea_to_json_xlsm(path, output_json, sheet_index)

