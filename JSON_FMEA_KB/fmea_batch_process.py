from pathlib import Path

from kb_structure import FMEAFailureKB, FMEACauseKB
from ingest_fmea import ingest_fmea_json


# =========================================================
# 1) Path setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Root folder containing many FMEA JSON files
JSON_ROOT = BASE_DIR.parent / "fmea_json_raw"
JSON_ROOT = JSON_ROOT.resolve()

if not JSON_ROOT.exists():
    raise FileNotFoundError(f"Cannot find fmea_json_raw folder at: {JSON_ROOT}")

# Persist KB data folders
KB_DATA_ROOT = BASE_DIR / "kb_data"
FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"
CAUSE_KB_DIR = KB_DATA_ROOT / "cause_kb"

for p in [FAILURE_KB_DIR, CAUSE_KB_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) Init KBs
# =========================================================

failure_kb = FMEAFailureKB(persist_dir=FAILURE_KB_DIR)
cause_kb = FMEACauseKB(persist_dir=CAUSE_KB_DIR)


# =========================================================
# 3) Ingest all FMEA JSON files
# =========================================================

json_files = sorted(JSON_ROOT.glob("*.json"))

print(f"[INFO] JSON_ROOT = {JSON_ROOT}")
print(f"[INFO] Found {len(json_files)} FMEA JSON files")

for jp in json_files[:5]:
    print(f"[INGEST] {jp.name}")

    try:
        ingest_fmea_json(
            json_path=jp,
            failure_kb=failure_kb,
            cause_kb=cause_kb,
        )
    except Exception as e:
        print(f"[ERROR] Failed to ingest {jp.name}: {e}")

print("[INFO] Ingest finished")
print(f"Failure KB count : {failure_kb.collection.count()}")
print(f"Cause KB count   : {cause_kb.collection.count()}")
