import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

KEYWORDS = {"atpm", "genesis", "converter", "yess"}
PROCESS_WORDS = ("process", "pfmea", "solder", "coating", "assembly")

INPUT_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D.json"


def normalize(s) -> str:
    return str(s).lower() if s is not None else ""


def keyword_hit(parent_name: str, product_name: str) -> bool:
    pn = normalize(parent_name)
    pr = normalize(product_name)
    return any(k in pn or k in pr for k in KEYWORDS)

def load_json_smart(path: Path):
    # 先用 utf-8-sig（兼容 UTF-8 BOM）
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


def process_fmea(fmea_list: List[Dict[str, Any]]):
    rejected = []

    # Step 1 + 2: 基础过滤
    stage12_pass = []
    for r in fmea_list:
        name_raw = r.get("name", "")
        name = normalize(name_raw)

        # Rule 1: exclude the name containing process-related words
        hit = next((w for w in PROCESS_WORDS if w in name), None)
        if hit:
            rejected.append({
                **r,
                "_reject_reason": f'name contains process-related word: "{hit}"'
            })
            continue

        stage12_pass.append(r)

        # Rule 2: select the related product which is motor drives 
        if not keyword_hit(r.get("parentName"), r.get("productName")):
            rejected.append({**r, "_reject_reason": "keyword not found in parentName/productName"})
            continue

        stage12_pass.append(r)

    # Step 3: (id, name) 去重：isCopy==True 且 releaseNo 最大
    grouped = defaultdict(list)
    for r in stage12_pass:
        key = (r.get("id"), r.get("name"))
        grouped[key].append(r)

    selected = []
    for _, records in grouped.items():
        copy_records = [x for x in records if x.get("isCopy") is False]
        if not copy_records:
            for x in records:
                rejected.append({**x, "_reject_reason": "isCopy is not True"})
            continue

        latest = max(copy_records, key=lambda x: x.get("releaseNo", 0))
        selected.append(latest)

        for x in records:
            if x is latest:
                continue
            if x.get("isCopy") is not False:
                reason = "isCopy is not False"
            else:
                reason = f"duplicate (id,name); releaseNo not max (kept {latest.get('releaseNo')})"
            rejected.append({**x, "_reject_reason": reason})

    return selected, rejected


def main():
    in_path = Path(INPUT_PATH)
    out_selected = in_path.with_name("8D_selected.json")
    out_rejected = in_path.with_name("8D_rejected.json")

    # 读取输入 JSON（支持：list 或 {"items":[...]} 两种常见结构）
    with in_path.open("r", encoding="utf-8") as f:
        data = load_json_smart(in_path)
    
    if not isinstance(data, list):
        raise ValueError("FMEA.json is expected to be a list of dicts")
    
    fmea_list = data
    selected, rejected = process_fmea(fmea_list)

    with out_selected.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    with out_rejected.open("w", encoding="utf-8") as f:
        json.dump(rejected, f, ensure_ascii=False, indent=2)

    print(f"Input: {in_path}")
    print(f"Selected: {out_selected}  (count={len(selected)})")
    print(f"Rejected: {out_rejected}  (count={len(rejected)})")


if __name__ == "__main__":
    main()
