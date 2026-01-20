import os
import glob
import json
import shutil
from pathlib import Path

# ===============================
# CONFIG
# ===============================
BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "FMEA_selected.json"
OUTPUT_JSON = BASE_DIR / "fmea_with_filename.json"
OUTPUT_FMEA_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\RAW\FMEA"

# ===============================
# UTILS
# ===============================
def normalize(text) -> str:
    return text.lower().strip() if isinstance(text, str) else ""


def mark_is_useful_by_name(item: dict) -> None:
    """
    If name contains 'single fault' -> isUseful = False
    """
    name = normalize(item.get("name"))
    if "single fault" in name:
        item["isUseful"] = False
        item["_usefulMarkedBy"] = "name_keyword"
        item["_usefulKeyword"] = "single fault"


def is_fmea(item: dict) -> bool:
    return item.get("type", "").upper() == "FMEA"


def is_allowed_to_copy(item: dict) -> bool:
    if not is_fmea(item):
        return False
    # missing isUseful => treat as True
    if item.get("isUseful", True) is False:
        return False
    return True


def find_fmea_file(folder: str):
    """
    Find first FMEA Excel file
    """
    if not folder or not os.path.isdir(folder):
        return None

    for ext in (".xlsx", ".xlsm", ".xls"):
        pattern = os.path.join(folder, "*" + ext)
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None


def safe_copy(src: str, dst_folder: str) -> bool:
    if not src:
        return False
    try:
        os.makedirs(dst_folder, exist_ok=True)
        shutil.copy(src, dst_folder)
        print(f"[COPIED] {src}")
        return True
    except Exception as e:
        print(f"[ERROR] Copy failed: {src} | {e}")
        return False


# ===============================
# MAIN
# ===============================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    # ---- Step 1: auto mark isUseful by name ----
    mark_is_useful_by_name(item)

    # ---- Step 2: filter ----
    if not is_allowed_to_copy(item):
        if not is_fmea(item):
            item.setdefault("_copyStatus", "skipped: not FMEA")
        elif item.get("isUseful") is False:
            item.setdefault("_copyStatus", "skipped: isUseful false")
        continue

    folder = item.get("path")
    if not folder:
        item["_copyStatus"] = "failed: missing path"
        continue

    src = find_fmea_file(folder)
    if not src:
        if not os.path.isdir(folder):
            item["_copyStatus"] = "failed: path not found"
        else:
            item["_copyStatus"] = "failed: FMEA file not found"
        print(f"[NOT FOUND] {folder}")
        continue

    # ---- Step 3: copy + write back ----
    if safe_copy(src, OUTPUT_FMEA_DIR):
        item["copiedFileName"] = os.path.basename(src)
        item["_copyStatus"] = "copied"
    else:
        item["_copyStatus"] = "failed: copy error"

# ---- Save updated JSON ----
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("DONE.")
