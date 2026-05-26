import csv
import os
import shutil
import glob
import re

input_file = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\SQL\test.csv"   
output_8d_dir   = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\RAW\8D"
output_fmea_dir = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\RAW\FMEA"

paths_8d = []
paths_fmea = []

def build_8d_prefix(b_value: str) -> str:
    # keep only digits
    digits = re.sub(r"\D", "", b_value)
    # take first 8 digits
    first8 = digits[:8]
    # build 8D prefix
    return "8D" + first8   # e.g. "8D60011206"

def find_8d_file(folder, prefix):
    """
    1. Try strict match: prefix + *.doc/docx
    2. If not found, fallback to any file starting with 8D
    """
    exts_8d = [".doc", ".docx"]

    # --- Step 1: strict search (e.g. 8D60011206*.docx)
    for ext in exts_8d:
        strict_pattern = os.path.join(folder, prefix + "*" + ext)
        strict_matches = glob.glob(strict_pattern)
        if strict_matches:
            return strict_matches[0]

    # --- Step 2: fallback search (e.g. 8D*.docx)
    for ext in exts_8d:
        loose_pattern = os.path.join(folder, "8D*" + ext)
        loose_matches = glob.glob(loose_pattern)
        if loose_matches:
            print(f"[Fallback] Using generic 8D file: {loose_matches[0]}")
            return loose_matches[0]

    # --- nothing found
    return None

def safe_copy(src, dst_folder):
    if src is None:
        return
    try:
        if os.path.isfile(src):
            shutil.copy(src, dst_folder)
            print(f"Copied: {src}")
        else:
            print(f"[Missing] {src}")
    except Exception as e:
        print(f"[Error] Could not copy {src}: {e}")

with open(input_file, newline='', encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter=';')  # <-- change to correct delimiter

    # If your file has a header row, keep this next line.
    # If there is no header row, comment it out.
    # next(reader, None)  # skip header
    rows = list(reader)[-100:] 
    for row in rows:
        # column indexes (0-based): A=0, B=1, C=2, D=3, E=4, F=5 ...
        num_series = str(row[1]).strip()  # column B: 6001-1206-9301
        file_name = str(row[2])   # column C
        category  = str(row[3])   # column D (8D / FMEA)
        file_path = str(row[5])   # column F

        name_upper = file_name.upper()
        cat_upper  = category.upper()
        path_upper = file_path.upper()
        # skip rows where file name including ProcessFMEA 
        # Exclude if name contains "PROCESS" anywhere
        if "PROCESS" in name_upper:
            continue

        # Exclude if name contains "PFMEA" anywhere
        if "PFMEA" in name_upper:
            continue

        # Exclude if file path does NOT start with N:
        if not path_upper.startswith("N:"):
            continue

        # # separate by category
        # if "8D" in cat_upper:
        #     if not name_upper.startswith("8D"):
        #         continue
        #     paths_8d.append((file_name, file_path))
        #     print(f"8D   | {file_name} | {file_path}")
                # 8D handling: build prefix from column B
        if "8D" in cat_upper:
            prefix = build_8d_prefix(num_series)  # e.g. "8D60011206"
            paths_8d.append((prefix, file_path))

        elif "FMEA" in cat_upper:
            paths_fmea.append((file_name, file_path))
            print(f"FMEA | {file_name} | {file_path}")

# write results to separate CSV files (name + path)
# with open("paths_8D.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["FileName", "FilePath"])
#     writer.writerows(paths_8d)

# with open("paths_FMEA.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["FileName", "FilePath"])
#     writer.writerows(paths_fmea)

# print("Done. Wrote paths_8D.csv and paths_FMEA.csv")

# ---- 8D: only .doc / .docx ----
exts_8d = [".doc", ".docx"]

for prefix, folder in paths_8d:
    src = find_8d_file(folder, prefix)
    if src is None:
        print(f"[Not found 8D] {folder} / {prefix}(.doc/.docx)")
        continue
    safe_copy(src, output_8d_dir)

# # ---- FMEA: .xlsx / .xlsm / .xls ----
# exts_fmea = [".xlsx", ".xlsm", ".xls"]  # add more if needed

# for file_name, file_path in paths_fmea:
#     src = resolve_with_ext(file_path, file_name, exts_fmea)
#     if src is None:
#         print(f"[Not found FMEA] {file_path} / {file_name}(.xlsx/.xlsm/.xls)")
#         continue
#     safe_copy(src, output_fmea_dir)

print("Done copying 8D and FMEA files.")