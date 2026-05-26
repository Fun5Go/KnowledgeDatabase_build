import os
import glob
import json
import shutil
from pathlib import Path

# ===============================
# CONFIG
# ===============================
BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\Orion_list\FS\FS_deduplicated.json"
OUTPUT_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\Orion_list\FS\FS_with_filename.json"
OUTPUT_FS_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\RAW\FS_ALL"

# ===============================
# UTILS
# ===============================
def normalize(text) -> str:
    return text.lower().strip() if isinstance(text, str) else ""


def is_fs(item: dict) -> bool:
    return item.get("type", "").upper() == "FS"


def is_allowed_to_copy(item: dict) -> bool:
    if not is_fs(item):
        return False
    return True


def find_fs_file(folder: str):
    """
    优先找 FS*.pdf
    没有的话再找 FS*.docx / FS*.doc
    """
    if not folder or not os.path.isdir(folder):
        return None

    patterns = [
        os.path.join(folder, "FS*.pdf"),
        os.path.join(folder, "FS*.docx"),
        os.path.join(folder, "FS*.doc"),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # 可选：排序后取第一个，避免结果不稳定
            matches.sort()
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
    # ---- Step 1: filter ----
    if not is_allowed_to_copy(item):
        item.setdefault("_copyStatus", "skipped: not FS")
        continue

    folder = item.get("path")
    if not folder:
        item["_copyStatus"] = "failed: missing path"
        continue

    src = find_fs_file(folder)
    if not src:
        if not os.path.isdir(folder):
            item["_copyStatus"] = "failed: path not found"
        else:
            item["_copyStatus"] = "failed: FS file not found"
        print(f"[NOT FOUND] {folder}")
        continue

    # ---- Step 2: copy + write back ----
    if safe_copy(src, OUTPUT_FS_DIR):
        item["copiedFileName"] = os.path.basename(src)
        item["_copyStatus"] = "copied"
    else:
        item["_copyStatus"] = "failed: copy error"

# ---- Save updated JSON ----
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("DONE.")