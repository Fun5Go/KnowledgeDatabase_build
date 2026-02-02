from pathlib import Path
import json

from kb_structure import FMEAFailureKB, FileMetaStore
from ingest_fmea import ingest_fmea_jsonl  


# =========================================================
# 1) Path setup
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

JSONL_PATH= Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\FMEA_motor_drive_recall.jsonl")
# Persist KB data folders
KB_DATA_ROOT = BASE_DIR.parent/ "KB_motor_drives"
FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"
# CAUSE_KB_DIR = KB_DATA_ROOT / "cause_kb"

for p in [FAILURE_KB_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) Init KBs
# =========================================================
failure_kb = FMEAFailureKB(persist_dir=FAILURE_KB_DIR)
# cause_kb = FMEACauseKB(persist_dir=CAUSE_KB_DIR)
meta_kb = FileMetaStore(persist_dir=KB_DATA_ROOT)


# =========================================================
# 3) Ingest all FMEA JSONL files (row by row)
# =========================================================

ingest_fmea_jsonl(
    jsonl_path=JSONL_PATH,
    failure_kb=failure_kb,
    # cause_kb=cause_kb,
    meta_kb=meta_kb,
)


print("[INFO] Ingest finished")
roles = ["failure_mode", "failure_effect", "failure_element", "failure_cause"]

def count_by_where(col, where):
    res = col.get(where=where, include=[])
    return len(res["ids"])
roles = ["failure_mode", "failure_effect", "failure_element", "failure_cause"]
for r in roles:
    n = count_by_where(failure_kb.collection, {"role": r})
    print(f"{r} count: {n}")
print(f"Total count: {failure_kb.collection.count()}")

