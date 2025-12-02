import os
import shutil
from pathlib import Path

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


def main():
    # --- Source folder: network path where your FMEA files are stored ---
    source_folder = r"N:\CHA6371\SSYINV240001\Rxx\DOC\FMEA240046\Rxx"

    # --- Destination folder: local folder where you want to save copies and CSVs ---
    destination_folder = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\FMEA"

    # process_fmea_files(source_folder, destination_folder)


if __name__ == "__main__":
    main()
