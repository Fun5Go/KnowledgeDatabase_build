import os
import shutil
from pathlib import Path
import json

import pandas as pd


def find_and_copy_fmea_files(source_folder: str, destination_folder: str):
    """
    Search for files in source_folder whose names start with 'FMEA' and end with '.xlsm',
    copy them to destination_folder, and return a list of the copied file paths.
    """
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    copied_files = []

    for filename in os.listdir(source_folder):
        # Check: file name starts with "FMEA" (case-insensitive) and ends with ".xlsm"
        if filename.upper().startswith("FMEA") and filename.lower().endswith(".xlsm"):
            src = os.path.join(source_folder, filename)
            dst = os.path.join(destination_folder, filename)
            shutil.copy2(src, dst)
            copied_files.append(dst)
            print(f"Copied: {filename}")

    return copied_files


#----------Dont use csv -> JSON schema----------#
# def convert_xlsm_to_csv(xlsm_file_path: str, csv_file_path: str, sheet_name=1):
#     """
#     Convert a single .xlsm file to .csv.
#     By default, read the first sheet (sheet_name=0).
#     You can change sheet_name to a sheet index or sheet name if needed.
#     """
#     # Read Excel using pandas; engine 'openpyxl' supports .xlsm files
#     df = pd.read_excel(xlsm_file_path, 
#                        sheet_name=sheet_name,
#                         header= 0,
#                          engine="openpyxl")
    
#         # Remove fully empty rows (common in DFMEA sheet)
#     df = df.dropna(how="all")
#     # Forward-fill merged-cell blank sections (System element, Function, etc.)
#     df = df.ffill()

#     # Save as CSV. Use UTF-8 with BOM to better support non-ASCII characters (e.g. Chinese)
#     df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")
#     print(f"Converted to CSV: {csv_file_path}")


# def process_fmea_files(source_folder: str, destination_folder: str):
#     """
#     Main processing flow:
#     1. Copy FMEA*.xlsm files from source_folder to destination_folder.
#     2. Convert each copied .xlsm file to a .csv file (same name, different extension).
#     """
#     # Step 1: find and copy matching files
#     copied_files = find_and_copy_fmea_files(source_folder, destination_folder)

#     if not copied_files:
#         print("No FMEA .xlsm files found.")
#         return

#     # Step 2: convert each copied .xlsm file to CSV
#     for xlsm_path in copied_files:
#         xlsm_path_obj = Path(xlsm_path)
#         # Replace .xlsm extension with .csv
#         csv_path = xlsm_path_obj.with_suffix(".csv")

#         try:
#             convert_xlsm_to_csv(str(xlsm_path_obj), str(csv_path))
#         except Exception as e:
#             print(f"Failed to convert {xlsm_path_obj.name} to CSV: {e}")

def extract_metadata_rows(path):
    meta = pd.read_excel(path, sheet_name=1, header=None, nrows=12, engine="openpyxl")

    def extract_value(row):
        if isinstance(row, str) and ":" in row:
            return row.split(":", 1)[1].strip()
        return ""

    system_name    = extract_value(meta.iloc[5, 0])
    system_element = extract_value(meta.iloc[7, 0])
    function_name  = extract_value(meta.iloc[8, 0])

    return system_name, system_element, function_name

def dfmea_to_effect_centered_json(path, json_output_path):
    # Extract metadata (system, element, function)
    system_name, system_element, function_name = extract_metadata_rows(path)

    # Load the actual DFMEA table (header likely row 9)
    df = pd.read_excel(path, sheet_name=1, header=9, engine="openpyxl")
    df_debug = pd.read_excel(path, sheet_name=1, header=None, engine="openpyxl")

    pd.set_option('display.max_columns', 200)
    pd.set_option('display.max_rows', 50)
    print(df_debug.iloc[:40, :20])

    df = df.dropna(how="all")
    df = df.ffill()

    # Column mapping: adjust based on actual names
    col_map = {
        "Potential effects of failure": "failure_effect",
        "Severity": "severity",
        "Potential failure mode": "failure_mode",
        "Potential causes of failure": "failure_cause"
    }

    df = df.rename(columns=col_map)

    df = df[["failure_effect", "severity", "failure_mode", "failure_cause"]]

    grouped = []

    # Group by effect
    for (effect, severity), subdf in df.groupby(["failure_effect", "severity"]):
        failure_modes = []

        for failure_mode, fm_group in subdf.groupby("failure_mode"):
            causes = sorted(set(fm_group["failure_cause"].dropna()))
            failure_modes.append({
                "failure_mode": failure_mode,
                "causes": causes
            })

        grouped.append({
            "system_name": system_name,
            "system_element": system_element,
            "function": function_name,
            "failure_effect": effect,
            "severity": severity,
            "failure_modes": failure_modes
        })

    # Save JSON
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(grouped)} JSON objects to {json_output_path}")

def extract_system_name(path, sheet_index=1):
    """
    Extract system name from the DFMEA sheet.
    In your file, the row looks like:
        col0 = "System:"
        col2 = "<system name>"
    So we read the first ~15 rows and look for a cell in column 0 containing 'system',
    then return the value in column 2 of the same row if present.
    """
    # Read first 15 rows without headers
    meta = pd.read_excel(path, sheet_name=sheet_index, header=None, nrows=15, engine="openpyxl")

    for r in range(len(meta)):
        cell0 = meta.iloc[r, 0]
        text0 = str(cell0).replace("：", ":").strip()  # normalize colon variants

        if "system" in text0.lower():  # e.g. "System:"
            # Prefer value from column 2 (your layout)
            system_candidate = None

            # Try col 2 if it exists
            if meta.shape[1] > 2:
                val2 = meta.iloc[r, 2]
                if not (isinstance(val2, float) and math.isnan(val2)) and str(val2).strip() != "NaN":
                    system_candidate = str(val2).strip()

            # Fallback: try to parse from same cell after colon
            if not system_candidate and ":" in text0:
                system_candidate = text0.split(":", 1)[1].strip()

            if system_candidate:
                return system_candidate

    # If nothing found
    return ""


def main():
    # --- Source folder: network path where your FMEA files are stored ---
    source_folder = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\FMEA\FMEA6371240046R02.xlsm"

    # --- Destination folder: local folder where you want to save copies and CSVs ---
    destination_folder = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\FMEA"

    # process_fmea_files(source_folder, destination_folder)
    # dfmea_to_effect_centered_json(source_folder, destination_folder)
    system_name = extract_system_name(source_folder)
    print("Detected system_name:", repr(system_name))


if __name__ == "__main__":
    main()
