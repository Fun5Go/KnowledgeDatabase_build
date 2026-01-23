import json
from pathlib import Path
from openpyxl import load_workbook

# ===============================
# CONFIG
# ===============================
FMEA_DIR = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\RAW\FMEA_ALL")
JSON_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\Orion_list\FMEA\FMEA_with_filename.json")

VALID_SUFFIX = {".xls", ".xlsx", ".xlsm"}
REQUIRED_WORDS = ["failure", "mode", "effect", "analysis"]

# ===============================
# FORMAT CHECK
# ===============================
def get_header_text(sheet):
    """Safely get first non-empty cell text from row 1"""
    for row in sheet.iter_rows(min_row=1, max_row=1):
        for cell in row:
            if cell.value:
                return str(cell.value)
    return None


def check_fmea_format(excel_path: Path) -> tuple[bool, str | None]:
    try:
        wb = load_workbook(excel_path, read_only=True, data_only=True)
    except Exception:
        return False, "cannot open excel"

    if len(wb.worksheets) <= 1:
        return False, "sheet1 not found"

    sheet = wb.worksheets[1]
    header = get_header_text(sheet)

    if not header:
        return False, "header empty"

    text = header.lower()
    if not all(word in text for word in REQUIRED_WORDS):
        return False, "header keywords missing"

    return True, None

# ===============================
# BUILD filename -> item RESULT
# ===============================
def build_result_map_from_folder(fmea_dir: Path) -> dict[str, dict]:
    """
    Scan FMEA_DIR (recursive) and return:
    {
      "xxx.xlsx": {"format": True/False, "_format_reason": "...", "_checkedFile": "xxx.xlsx"}
    }
    """
    result_map: dict[str, dict] = {}

    for p in fmea_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_SUFFIX:
            continue

        is_valid, reason = check_fmea_format(p)
        item = {
            "format": is_valid,
            # "_checkedFile": p.name,
        }
        if not is_valid:
            item["_format_reason"] = reason

        # ⚠️ 如果同名文件存在多个路径，这里后者会覆盖前者
        # 如需更严格，可改成记录冲突列表
        result_map[p.name] = item

    return result_map

# ===============================
# MAIN
# ===============================
def main():
    print(">>> FMEA format check (by folder) started")

    if not FMEA_DIR.exists():
        raise FileNotFoundError(f"FMEA_DIR not found: {FMEA_DIR}")
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"JSON_PATH not found: {JSON_PATH}")

    # 1) build results by scanning folder
    result_map = build_result_map_from_folder(FMEA_DIR)
    print(f">>> scanned excel files: {len(result_map)}")

    # 2) load json and write back by same copiedFileName
    with JSON_PATH.open("r", encoding="utf-8") as f:
        fmea_list = json.load(f)

    if not isinstance(fmea_list, list):
        raise ValueError("JSON must be a list of dicts")

    hit, miss_name, miss_file = 0, 0, 0

    for r in fmea_list:
        r.setdefault("item", {})
        filename = r.get("copiedFileName")

        if not filename:
            r["item"]["format"] = False
            r["item"]["_format_reason"] = "missing filename"
            miss_name += 1
            continue

        item = result_map.get(filename)
        if not item:
            r["item"]["format"] = False
            r["item"]["_format_reason"] = "file not found in FMEA_DIR"
            miss_file += 1
            continue

        r["item"].update(item)
        hit += 1

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(fmea_list, f, ensure_ascii=False, indent=2)

    print(">>> write-back finished")
    print(f">>> matched: {hit}")
    print(f">>> missing copiedFileName: {miss_name}")
    print(f">>> not found in folder: {miss_file}")

if __name__ == "__main__":
    main()
