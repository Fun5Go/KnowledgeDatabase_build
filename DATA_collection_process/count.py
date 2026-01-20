from collections import Counter
from pathlib import Path
import json

KEYWORDS = {"atpm", "genesis", "yess"}

def normalize(text) -> str:
    return text.lower().strip() if isinstance(text, str) else ""

def classify_product(item: dict, keywords: set[str]) -> set[str]:
    text = normalize(item.get("parentName", "")) + " " + normalize(item.get("productName", ""))
    matched = set()
    for kw in keywords:
        if kw in text:
            matched.add(kw)
    return matched

def count_by_product_only_copied(data: list[dict], keywords: set[str]) -> Counter:
    counter = Counter()
    for item in data:
        #filter: must have copiedFileName
        copied_name = item.get("copiedFileName", "")
        if not isinstance(copied_name, str) or not copied_name.strip():
            continue

        matched = classify_product(item, keywords)
        for kw in matched:
            counter[kw] += 1
    return counter


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = BASE_DIR / "fmea_with_filename.json"
with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

product_counter = count_by_product_only_copied(data, KEYWORDS)

print("Product document count:")
for k in sorted(KEYWORDS):
    print(f"{k.upper():8s}: {product_counter.get(k, 0)}")
