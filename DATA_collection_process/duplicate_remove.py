from collections import defaultdict
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path


from collections import defaultdict
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path

# ===============================
# CONFIG
# ===============================
# INPUT_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D.json"
INPUT_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\Orion_list\TS\TS.json"
OPERATION = "deduplicate"  # "deduplicate" | "type_classify"

PROCESS_WORDS = [
    # "pfmea",
    # "process", #exclude in 8D
    #====8d====
    "solder", "coating", "assembly", "injection", "molding", "paint", "housing",
    "production",
    "allignment",
    "pick up",
    "bonding",
    "placement",
    "glue",
    "screw",
    "screws",
    "cover",
    "delivery",
    "package",
    "manufacturing",
    "place",
    "adhesive"
]

# ===============================
# UTILS
# ===============================
def normalize(s) -> str:
    return str(s).lower() if s is not None else ""


def load_json_smart(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except UnicodeDecodeError:
        pass

    try:
        with path.open("r", encoding="utf-16") as f:
            return json.load(f)
    except UnicodeDecodeError:
        pass

    with path.open("r", encoding="latin1") as f:
        return json.load(f)
    
def extract_record_list(data):
    """
    兼容两种输入格式：
    1) [ {...}, {...} ]
    2) { "value": [ {...}, {...} ] }
    返回统一的 list[dict]
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("value"), list):
        return data["value"]

    raise ValueError("Input JSON must be a list, or an object with a 'value' list")

# ===============================
# OPERATION 1: DEDUPLICATE
# ===============================
def deduplicate_records(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)

    for r in data:
        key = (r.get("id"), r.get("name"))
        grouped[key].append(r)

    deduplicated = []

    for (_, _), records in grouped.items():
        occurrence_count = len(records)

        valid = [x for x in records if x.get("isCopy") is False]
        if not valid:
            continue

        chosen = max(
            valid,
            key=lambda x: (
                x.get("releaseNo", 0),
                x.get("listId", 0)
            )
        )

        deduplicated.append({
            **chosen,
            "occurrenceCount": occurrence_count
        })

    return deduplicated

# ===============================
# OPERATION 2: TYPE CLASSIFY
# ===============================
def is_process_fmea(name: str) -> bool:
    # Normalize the name to avoid mismatches caused by case, spaces, or symbols
    name = normalize(name)
    # Return True if any process-related keyword appears in the document name
    return any(w in name for w in PROCESS_WORDS)

def split_process_and_sd(data: List[Dict[str, Any]]):
    process = []
    sd_fmea = []
    for r in data:
        name = r.get("name", "")
        if is_process_fmea(name):
            process.append(r)
        else:
            sd_fmea.append(r)
    # Split the process FMEA and System FMEA into two lists
    return process, sd_fmea

# ===============================
# MAIN
# ===============================
def main():
    in_path = Path(INPUT_PATH)
    raw_data = load_json_smart(in_path)

    # 这里兼容 value 结构
    data = extract_record_list(raw_data)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of dicts")

    # -------- deduplicate --------
    if OPERATION == "deduplicate":
        out_path = in_path.with_name(f"{in_path.stem}_deduplicated.json")

        deduplicated = deduplicate_records(data)

        # 直接保存数组，不再包 value
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(deduplicated, f, ensure_ascii=False, indent=2)

        print(f"[OK] Operation: deduplicate")
        print(f"Input: {in_path}")
        print(f"Output: {out_path} (count={len(deduplicated)})")

    # -------- type classify --------
    elif OPERATION == "type_classify":
        out_process = in_path.with_name(f"{in_path.stem}_process.json")
        out_sd = in_path.with_name(f"{in_path.stem}_sd.json")

        process, sd_fmea = split_process_and_sd(data)

        with out_process.open("w", encoding="utf-8") as f:
            json.dump(process, f, ensure_ascii=False, indent=2)

        with out_sd.open("w", encoding="utf-8") as f:
            json.dump(sd_fmea, f, ensure_ascii=False, indent=2)

        print(f"[OK] Operation: type_classify")
        print(f"Input: {in_path}")
        print(f"Process FMEA: {out_process} (count={len(process)})")
        print(f"S/D FMEA: {out_sd} (count={len(sd_fmea)})")

    else:
        raise ValueError(f"Unsupported OPERATION: {OPERATION}")


if __name__ == "__main__":
    main()
