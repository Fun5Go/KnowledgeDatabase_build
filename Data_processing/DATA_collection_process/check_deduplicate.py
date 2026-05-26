from pathlib import Path
import json

def get_occurrence_count(item: dict) -> int:
    v = item.get("occurrenceCount", 0)
    if isinstance(v, bool):
        return 0
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)
    return 0

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\Orion_list\FS\FS_deduplicated.json")

with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)  # data: list[dict]

total = sum(get_occurrence_count(item) for item in data)

print("Total occurrenceCount:", total)
