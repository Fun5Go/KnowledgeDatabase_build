import json
from pathlib import Path
from Key_words_list import MOTOR_DRIVE_KEYWORDS, PROCESS_KEYWORDS, PRODUCT_HINTS, NEW_KEYS, OLD_KEYS, ALL_TEXT_KEYS, FAILURE_KEYS
import re

def build_keyword_patterns(keywords):
    """
    Match whole words only.
    Ensures keyword is NOT part of a larger word.
    """
    patterns = {}
    for kw in keywords:
        kw_escaped = re.escape(kw)
        pattern = re.compile(
            rf"(?<![a-zA-Z]){kw_escaped}(?![a-zA-Z])"
        )
        patterns[kw] = pattern
    return patterns

MOTOR_PATTERNS = build_keyword_patterns(MOTOR_DRIVE_KEYWORDS)
PROCESS_PATTERNS = build_keyword_patterns(PROCESS_KEYWORDS)

def normalize(text):
    return text.lower().strip() if isinstance(text, str) else ""


def hit_keywords(text, keywords):
    return [kw for kw in keywords if kw in text]


def extract_text_fields(row):
    texts = {}
    for k in ALL_TEXT_KEYS:
        if k in row and isinstance(row[k], str) and row[k].strip():
            texts[k] = normalize(row[k])
    return texts

def build_labels(row):
    # Collect normalized text only from fields in FAILURE_KEYS
    texts = {
        k: row[k].lower().strip()
        for k in FAILURE_KEYS
        if isinstance(row.get(k), str) and row[k].strip()
    }
    motor_hits = []
    process_hits = []
    # Scan FAILURE_KEYS fields for keyword matches 
    for v in texts.values():
        for kw, pat in MOTOR_PATTERNS.items():
            if pat.search(v):
                motor_hits.append(kw)
        for kw, pat in PROCESS_PATTERNS.items():
            if pat.search(v):
                process_hits.append(kw)
    labels = {}
    # Add product and fmea labels
    if motor_hits:
        labels["domain"] = "motor_drives"
        labels["criterion"] = "high keyword hit"
        labels["motor_drive_hits"] = sorted(set(motor_hits))
    if process_hits:
        labels["fmea_type"] = "process"
        labels["process_hits"] = sorted(set(process_hits))
    # product / parent hint
    product_text = (row.get("productName") or "").lower()
    product_hint = [p for p in PRODUCT_HINTS if p in product_text]
    if product_hint:
        labels["product_hint"] = product_hint
    return labels, texts

def save_failure(row):
    """
    Keep original text for ALL_TEXT_KEYS (no normalize).
    """
    content = {}
    for k in ALL_TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str):
            v = v.strip()
        if v not in (None, "", []): 
            content[k] = v
    return content


def process_fmea_to_jsonl(input_json, output_jsonl):
    input_json = Path(input_json)
    output_jsonl = Path(output_jsonl)

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    with output_jsonl.open("w", encoding="utf-8") as out:
        for row in data:
            labels = build_labels(row)
            if "domain" not in labels:
                continue  

            record = {
                "source": "FMEA",
                "labels": labels,
                "metadata": {
                    "productName": row.get("productName"),
                    "parentName": row.get("parentName"),
                    "released": row.get("released"),
                    "file": row.get("file"),
                },
                "content": {
                    k: row.get(k)
                    for k in ALL_TEXT_KEYS
                    if row.get(k)
                }
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

def batch_process(input_path, output_path):
    INPUT_JSON_DIR = input_path
    OUTPUT_JSONL = output_path

    json_files = list(INPUT_JSON_DIR.glob("*.json"))
    print(f"Found {len(json_files)} json files")

    total_rows = 0
    total_hits = 0

    with OUTPUT_JSONL.open("w", encoding="utf-8") as out:
        for json_file in json_files:
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    rows = json.load(f)
            except Exception as e:
                print(f"❌ Failed to load {json_file.name}: {e}")
                continue

            if not isinstance(rows, list):
                print(f"⚠ Skip {json_file.name}: not a list")
                continue

            for idx, row in enumerate(rows):
                total_rows += 1

                labels, texts = build_labels(row)

                # Primary recall: key words hit
                if "domain" not in labels:
                    continue

                record = {
                    "source_type": row.get("source_type"),
                    "file_name": row.get("file_name", json_file.name),
                    "row_index": idx,
                    "labels": labels,
                    "metadata": {
                        "productName": row.get("productName"),
                        "productId": row.get("productId"),
                        "productPnId": row.get("productPnId"),
                        "released": row.get("released"),
                    },
                    "content": save_failure(row),
                    "RPN": {
                        "severity": row.get("severity"),
                        "occurrence": row.get("occurrence"),
                        "detection": row.get("detection"),
                        "RPN": row.get("RPN"),
                    }
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_hits += 1

    print(" Batch process finished")
    print(f"   Total rows scanned : {total_rows}")
    print(f"   Motor-drive hits   : {total_hits}")
    print(f"   Output jsonl       : {OUTPUT_JSONL}")

if __name__ == "__main__":
    INPUT_JSON_DIR = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\FMEA_JSON_ALL"
    )
    OUTPUT_JSONL = Path(
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\FMEA_motor_drive_recall_notcomplete.jsonl"
    )
    batch_process(INPUT_JSON_DIR,OUTPUT_JSONL)
