from collections import defaultdict
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path


INPUT_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D.json"

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

def deduplicate_process(fmea_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)

    for r in fmea_list:
        key = (r.get("id"), r.get("name"))
        grouped[key].append(r)

    deduplicated = []

    for (fid, name), records in grouped.items():
        occurrence_count = len(records)

        # 只考虑有真实文件的
        valid = [x for x in records if x.get("isCopy") is False]

        if not valid:
            # 原规则保留：没有 isCopy=False 的，整组跳过
            continue

        # releaseNo DESC, listId DESC
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

def main():
    in_path = Path(INPUT_PATH)
    out_path = in_path.with_name("8D_deduplicated.json")

    with in_path.open("r", encoding="utf-8") as f:
        data = load_json_smart(in_path)

    if not isinstance(data, list):
        raise ValueError(".json is expected to be a list of dicts")

    deduplicated = deduplicate_process(data)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(deduplicated, f, ensure_ascii=False, indent=2)

    print(f"Input: {in_path}")
    print(f"Deduplicated: {out_path}  (count={len(deduplicated)})")


if __name__ == "__main__":
    main()
