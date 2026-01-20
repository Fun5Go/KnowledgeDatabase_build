import os
import glob
import json
import shutil
import re
from pathlib import Path

# ===============================
# CONFIG
# ===============================
BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR/"8D_selected.json"
OUTPUT_JSON = BASE_DIR/"8d_with_filename.json"
OUTPUT_8D_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\RAW\8D"

KEYWORDS_MARK_PROCESS = [
    "housing",
    "production",
    "allignment",
    "pick up",
    "bonding",
]

# ===============================
# UTILS
# ===============================
def normalize(text) -> str:
    return text.lower().strip() if isinstance(text, str) else ""


def build_8d_prefix(pn: str) -> str:
    """
    pn example: 6298-1100-3904
    -> 8D62981100
    """
    digits = re.sub(r"\D", "", pn or "")
    first8 = digits[:8]
    return "8D" + first8 if first8 else ""


def mark_is_process_by_name(item: dict) -> None:
    """
    If name contains keywords -> add isProcess=True
    """
    name = normalize(item.get("name"))
    for kw in KEYWORDS_MARK_PROCESS:
        if kw in name:
            item["isProcess"] = True
            item["_processMarkedBy"] = "name_keyword"
            item["_processKeyword"] = kw
            return


def is_8d(item: dict) -> bool:
    return item.get("type", "").upper() == "8D"


def is_allowed_to_copy(item: dict) -> bool:
    # if not is_8d(item):
    #     return False
    if item.get("isProcess", False) is True:
        return False
    return True


def find_8d_file(folder: str, prefix: str):
    """
    1. strict: prefix*.doc / docx
    2. fallback: 8D*.doc / docx
    """
    if not folder or not os.path.isdir(folder):
        return None

    # --- strict search ---
    for ext in (".doc", ".docx"):
        pattern = os.path.join(folder, prefix + "*" + ext)
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    # --- fallback ---
    for ext in (".doc", ".docx"):
        pattern = os.path.join(folder, "8D*" + ext)
        matches = glob.glob(pattern)
        if matches:
            print(f"[Fallback] {folder} -> {matches[0]}")
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
    # ---- Step 1: auto-mark process by name ----
    mark_is_process_by_name(item)

    # ---- Step 2: filter ----
    if not is_allowed_to_copy(item):
        if not is_8d(item):
            item.setdefault("_copyStatus", "skipped: not 8D")
        elif item.get("isProcess"):
            item.setdefault("_copyStatus", "skipped: isProcess true")
        continue

    folder = item.get("path")
    pn = item.get("pn", "")
    prefix = build_8d_prefix(pn)

    if not prefix:
        item["_copyStatus"] = "failed: invalid pn"
        continue

    src = find_8d_file(folder, prefix)

    if not src:
        if not os.path.isdir(folder):
            item["_copyStatus"] = "failed: path not found"
        else:
            item["_copyStatus"] = f"failed: no 8D file for prefix {prefix}"
        print(f"[NOT FOUND] {folder} | {prefix}")
        continue

    # ---- Step 3: copy + write back ----
    if safe_copy(src, OUTPUT_8D_DIR):
        item["copiedFileName"] = os.path.basename(src)
        item["_copyStatus"] = "copied"
        item["_8dPrefixUsed"] = prefix
    else:
        item["_copyStatus"] = "failed: copy error"

# ---- Save updated JSON ----
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("DONE.")
