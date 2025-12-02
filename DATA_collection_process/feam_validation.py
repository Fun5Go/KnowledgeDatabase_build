import pandas as pd
import json
import numpy as np

import pandas as pd
import json
import math


###############################################################################
# 1. Read System Name
###############################################################################
def extract_system_name(path, sheet_index=1):
    """
    System name is usually stored like:
    Row: "System:" in column 0
    Value in column 2 = "[A01] Charged 5k"
    """
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
# 2. Extract Structure Block: System Element + Function (Sequential Order)
###############################################################################
def extract_structure_blocks(path, sheet_index=1):
    df = pd.read_excel(path, sheet_name=sheet_index, header=None, engine="openpyxl")

    blocks = []
    current_system_element = None

    for r in range(len(df)):
        col0 = str(df.iloc[r, 0]).strip().lower()
        col2 = ""
        if df.shape[1] > 2:
            v2 = df.iloc[r, 2]
            if not (isinstance(v2, float) and math.isnan(v2)):
                col2 = str(v2).strip()

        if "system element" in col0:
            current_system_element = col2

        elif "function" in col0:
            func_full = col2  # e.g. "[S01F01] EMC filter"
            if func_full:
                func_key = func_full.split()[0]  # "[S01F01]"
                blocks.append({
                    "system_element": current_system_element,
                    "function": func_full,
                    "function_key": func_key
                })

    return blocks


###############################################################################
# 3. Load DFMEA Table (header row = 6)
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

    # ❗ 关键修复：删除 fake rows（没有 cause 的行不作为有效 FMEA 条目）
    df = df[df["failure_cause"].notna()]
    df = df[df["failure_cause"].astype(str).str.strip() != ""]

    # ❗ 额外修复：将 Severity "-" 去掉（这些是 merged cell 下的垃圾行）
    df = df[df["severity"].astype(str).str.strip() != "-"]

    return df


###############################################################################
# 4. Split DFMEA by Function Name (key)
#    → key = first token of function name, e.g. "[S01F02]"
###############################################################################
def split_dfmea_by_function(dfmea, structure_blocks):
    df_blocks = []

    for block in structure_blocks:
        key = block["function_key"]  # e.g. "[S01F02]"
        mask = dfmea["failure_mode"].astype(str).str.startswith(key)
        df_blocks.append(dfmea[mask].copy())

    return df_blocks


###############################################################################
# 5. Build Effect-centered JSON
###############################################################################
def convert_numpy(value):
    """Convert numpy types (int64, float64, etc.) to Python built-ins."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_json(system_name, structure_blocks, df_blocks):
    results = []

    for sb, df_block in zip(structure_blocks, df_blocks):

        if df_block.empty:
            continue

        se = sb["system_element"]
        func = sb["function"]

        # 每一行是一个独立的 Effect-Mode-Cause 关系，不合并！
        for _, row in df_block.iterrows():

            effect = str(row["failure_effect"])
            severity = row["severity"] if not isinstance(row["severity"], float) else int(row["severity"])
            mode = str(row["failure_mode"])
            cause = str(row["failure_cause"])

            item = {
                "system_name": system_name,
                "system_element": se,
                "function": func,
                "failure_effect": effect,
                "severity": severity,
                "failure_mode": mode,
                "cause": cause
            }

            results.append(item)

    return results


###############################################################################
# 6. Main Function
###############################################################################
def dfmea_to_json(path, output_json, sheet_index=1):
    print("=== STEP 1: Extract System Name ===")
    system_name = extract_system_name(path, sheet_index)
    print("System Name:", system_name)

    print("\n=== STEP 2: Extract Structure Blocks ===")
    structure_blocks = extract_structure_blocks(path, sheet_index)
    print("Found Structure Blocks:", len(structure_blocks))

    print("\n=== STEP 3: Load DFMEA Table ===")
    dfmea = load_dfmea_table(path, sheet_index)
    print("DFMEA Rows:", len(dfmea))

    print("\n=== STEP 4: Split DFMEA by Function Name (block order) ===")
    df_blocks = split_dfmea_by_function(dfmea, structure_blocks)

    print("\n=== STEP 5: Build JSON ===")
    json_data = build_json(system_name, structure_blocks, df_blocks)
    print("JSON Objects Generated:", len(json_data))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print("\n🎉 DONE — JSON saved to:", output_json)



###############################################################################
# 7. Run example
###############################################################################
if __name__ == "__main__":
    file_path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\FMEA\FMEA6371240046R02.xlsm"  # your FMEA file
    output_json = r"dfmea_effect.json"

    dfmea_to_json(file_path, output_json)
