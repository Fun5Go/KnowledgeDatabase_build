import json
from collections import defaultdict
from pathlib import Path

def group_by_pnid_keep_both(input_json: dict):
    tmp = defaultdict(lambda: {"8D": [], "FMEA": []})

    for _, meta in input_json.items():
        pnid = meta.get("productPnID")
        file_name = meta.get("file_name")
        source_type = meta.get("source_type")

        if pnid is None or file_name is None:
            continue

        pnid_key = str(pnid)

        if source_type == "8D":
            tmp[pnid_key]["8D"].append(file_name)
        else:
            tmp[pnid_key]["FMEA"].append(file_name)

    # only save the ID which contains both 8D and FMEA
    result = {
        pnid: v
        for pnid, v in tmp.items()
        if v["8D"] and v["FMEA"]
    }

    return result

if __name__ == "__main__":

    META_JSON = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\file_meta_store.json")

    with open(META_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = group_by_pnid_keep_both(data)

    # Save in the current folder
    output_path = Path(__file__).parent / "group_file.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(grouped, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")