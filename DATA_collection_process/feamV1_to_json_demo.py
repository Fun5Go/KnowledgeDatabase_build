import pandas as pd
import json
import numpy as np

import pandas as pd
import json
import math


###############################################################################
# Helper: convert numpy types to Python native types
###############################################################################
def convert(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


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

    for r in range(len(df)):
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
    df = pd.read_excel(path, sheet_name=sheet_index, header=6, engine="openpyxl")

    df = df.dropna(how="all")
    df = df.ffill()

    col_map = {
        "Potential Effect(s) of Failure\n(Activity)": "failure_effect",
        "Severity": "severity",
        "Potential Failure Mode\n(Process step)": "failure_mode",
        "Potential Cause(s) of Failure\n(per discipline)": "failure_cause"
    }
    df = df.rename(columns=col_map)

    needed = ["failure_effect", "severity", "failure_mode", "failure_cause"]
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
# 5. Build FINAL nested hierarchical schema
###############################################################################
# def build_hierarchical_schema(system_name, structure_blocks, df_blocks):
#     """
#     Final structure:
#     {
#       "system_name": "...",
#       "elements": [
#         {
#           "system_element": "...",
#           "functions": [
#             {
#               "function": "...",
#               "failures": [
#                 {
#                   "failure_effect": "...",
#                   "severity": ...,
#                   "failure_mode": "...",
#                   "failure_cause": "..."
#                 }
#               ]
#             }
#           ]
#         }
#       ]
#     }
#   {link} to documents, other products ,etc.
#   {category} to classify the product type
#     """

#     final = {"system_name": system_name, "elements": []}

#     # Build unique element groups
#     element_dict = {}

#     for sb, df_block in zip(structure_blocks, df_blocks):

#         element = sb["system_element"]
#         function = sb["function"]

#         if element not in element_dict:
#             element_dict[element] = {"system_element": element, "functions": []}

#         function_node = {"function": function, "failures": []}

#         # Each DFMEA row = one failure record (1 effect → 1 mode → 1 cause)
#         for _, row in df_block.iterrows():
#             failure = {
#                 "failure_effect": str(row["failure_effect"]),
#                 "severity": convert(row["severity"]),
#                 "failure_mode": str(row["failure_mode"]),
#                 "failure_cause": str(row["failure_cause"])
#             }
#             function_node["failures"].append(failure)

#         element_dict[element]["functions"].append(function_node)

#     # Move into final structure
#     final["elements"] = list(element_dict.values())

#     return final

###############################################################################
# 5.B Build FINAL nested flatten schema for human and machine readability
###############################################################################
def build_flat_failures_with_text(system_name, structure_blocks, df_blocks):
    records = []

    for sb, df_block in zip(structure_blocks, df_blocks):
        element = sb["system_element"]
        function = sb["function"]

        for _, row in df_block.iterrows():
            failure_effect = str(row["failure_effect"])
            severity = convert(row["severity"])
            failure_mode = str(row["failure_mode"])
            failure_cause = str(row["failure_cause"])

            text = (
                f"System: {system_name}; "
                f"Element: {element}; "
                f"Function: {function}; "
                f"Failure mode: {failure_mode}; "
                f"Cause: {failure_cause}; "
                f"Effect: {failure_effect}; "
                f"Severity: {severity}."
            )

            record = {
                "system_name": system_name,
                "system_element": element,
                "function": function,
                "failure_effect": failure_effect,
                "severity": severity,
                "failure_mode": failure_mode,
                "failure_cause": failure_cause,
                "text": text,   # Preparation for semantic search
            }
            records.append(record)

    return records



###############################################################################
# 6. MAIN FUNCTION
###############################################################################
def dfmea_to_json(path, output_json, sheet_index=1):

    print("\n=== STEP 1: Reading System Name ===")
    system_name = extract_system_name(path, sheet_index)
    print("System Name:", system_name)

    print("\n=== STEP 2: Reading Structure Blocks ===")
    structure_blocks = extract_structure_blocks(path, sheet_index)
    print("Structure Blocks:", len(structure_blocks))

    print("\n=== STEP 3: Reading DFMEA Table ===")
    dfmea = load_dfmea_table(path, sheet_index)
    print("DFMEA Rows:", len(dfmea))

    print("\n=== STEP 4: Splitting DFMEA by Function ===")
    df_blocks = split_dfmea_by_function(dfmea, structure_blocks)

    print("\n=== STEP 5: Building Hierarchical Schema ===")
    final_json = build_flat_failures_with_text(system_name, structure_blocks, df_blocks)

    # Write file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print("\n JSON saved to:", output_json)



###############################################################################
# 7. Run example
###############################################################################
if __name__ == "__main__":
    file_path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\FMEA\FMEA6367240034R02.xlsm"  # your FMEA file
    output_json = r"dfmea_effect_flat.json"

    dfmea_to_json(file_path, output_json)
