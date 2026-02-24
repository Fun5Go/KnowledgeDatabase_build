import json
import random
import math

INPUT_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb\entity_store.json"
OUTPUT_PATH = "sample_10pct.json"
SEED = 42  

KEEP_FIELDS = [
    "failure_id",
    "failure_mode_text",
    "failure_element_text",
    "failure_effect_text",
    "failure_cause_text",
]

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)  # dict: item_key(cause_id) -> record

    keys = list(data.keys())
    n_total = len(keys)
    n_sample = max(1, math.ceil(n_total * 0.10))  # 10%，at least 1 item

    rng = random.Random(SEED)
    sampled_keys = rng.sample(keys, k=min(n_sample, n_total))

    out = {}
    for k in sampled_keys:
        rec = data[k]
        out[k] = {field: rec.get(field) for field in KEEP_FIELDS}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Total items: {n_total}")
    print(f"Sampled items (10%): {len(sampled_keys)}")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()