
import json
from pathlib import Path
from langchain_community.document_loaders import JSONLoader


json_path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\eightD_json_raw\8D_test_3.json"

with open(json_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

doc0 = (raw.get("documents") or [{}])[0]
failure = raw.get("failure") or {}


common_meta = {
    "file_name": doc0.get("file_name"),
    "product_name": doc0.get("product_name"),
    "date": doc0.get("date"),
    "system_name": raw.get("system_name"),
    "review_status": (raw.get("maintenance_tag") or {}).get("review_status"),
    "version": (raw.get("maintenance_tag") or {}).get("Version"),
    "failure_ID": failure.get("failure_ID"),
    "failure_level": failure.get("failure_level"),
    "failure_element": failure.get("failure_element"),
    "failure_mode": failure.get("failure_mode"),
}

def metadata_func(record: dict, metadata: dict) -> dict:
    # record 是 signals[] 的每个 dict
    metadata.update(common_meta)
    metadata.update({
        "doc_type": "signal",
        "signal_id": record.get("id"),
        "hint": record.get("hint"),
        "confidence": record.get("confidence"),
        "source_section": record.get("source"),
    })
    return metadata

loader = JSONLoader(
    file_path=str(json_path),
    jq_schema=".selected_sentences.signals[]",
    content_key="text",          # 
    text_content=True,           # 
    metadata_func=metadata_func, # record + common_meta
)

docs = loader.load()

print(len(docs))
print(docs[1].metadata)
print(docs[1].page_content[:200])