import json
import random
from pathlib import Path



def main(
    causes_json_path: str,
    n: int = 20,
    seed: int = 42,
    out_path: str = "sample_20_failure_symptom_cause.jsonl",
):
    random.seed(seed)

    data = json.loads(Path(causes_json_path).read_text(encoding="utf-8"))
    items = list(data.items())
    if not items:
        raise RuntimeError("No items found in causes json.")

    sample = random.sample(items, k=min(n, len(items)))

    out_lines = []
    for cause_id, obj in sample:
        cause = obj.get("root_cause", "") or obj.get("failure_cause", "") or ""
        failure_id = obj.get("failure_id", "")

        out_lines.append({
            "failure_id": failure_id,
            "cause_id": cause_id,
            "failure_mode" : obj.get("failure_mode", ""),
            "failure_element" : obj.get("failure_element", ""),
            "failure_effect" : obj.get("failure_effect", ""),
            "cause_text": cause,
        })

    # 
    Path(out_path).write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in out_lines),
        encoding="utf-8",
    )

    print(f"Wrote {len(out_lines)} samples to: {out_path}")
    print("Preview (first 3):")
    for x in out_lines[:3]:
        print(json.dumps(x, ensure_ascii=False))

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    KB_DATA_ROOT = BASE_DIR.parent / "kb_data"
    CAUSE_KB_DIR = KB_DATA_ROOT / "cause_kb"/"fmea_cause_store.json"
    main(causes_json_path=CAUSE_KB_DIR, n=20, seed=42)
