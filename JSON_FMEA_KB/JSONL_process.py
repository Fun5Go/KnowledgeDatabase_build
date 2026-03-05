from pathlib import Path
import json

from kb_structure import FMEAFailureKB, FileMetaStore, SentenceKB
from ingest_fmea import ingest_fmea_jsonl  


# =========================================================
# 1) Path setup
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

JSONL_PATH= Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\FMEA_motor_drive_recall_discipline.jsonl")
# Persist KB data folders
KB_DATA_ROOT = BASE_DIR.parent/ "KB_motor_drives_discipline"
FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"
SENTENCE_KB_DIR = KB_DATA_ROOT / "sentence_kb"
# CAUSE_KB_DIR = KB_DATA_ROOT / "cause_kb"

for p in [FAILURE_KB_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) Init KBs
# =========================================================
failure_kb = FMEAFailureKB(persist_dir=FAILURE_KB_DIR)
sentence_kb = SentenceKB(persist_dir=SENTENCE_KB_DIR)
meta_kb = FileMetaStore(persist_dir=KB_DATA_ROOT)


# =========================================================
# 3) Ingest all FMEA JSONL files (row by row)
# =========================================================

ingest_fmea_jsonl(
    jsonl_path=JSONL_PATH,
    failure_kb=failure_kb,
    meta_kb=meta_kb,
    sentence_kb=sentence_kb,
)


print("[INFO] Ingest finished")
field_type = ["mode", "effect", "element", "cause"]

def count_by_where(col, where):
    res = col.get(where=where, include=[])
    return len(res["ids"])
field_type =  ["mode", "effect", "element", "cause"]
for r in field_type:
    n = count_by_where(failure_kb.collection, {"field_type": r})
    print(f"{r} count: {n}")
print(f"Total failure text count: {failure_kb.collection.count()}")
print(f"Total sentence count: {sentence_kb.collection.count()}")

