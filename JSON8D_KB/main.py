from pathlib import Path

from kb_structure import FailureKB, SentenceKB, FileMetaStore
from ingest_8d import ingest_8d_json



# =========================================================
# 1) Resolve paths (robust)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# JSON_ROOT = BASE_DIR.parent / "eightD_json_raw"
JSON_ROOT = Path(r'C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_MD\failure_identification').resolve()
# if not JSON_ROOT.exists():
#     JSON_ROOT = BASE_DIR.parent / "eightD_json_raw"

if not JSON_ROOT.exists():
    raise FileNotFoundError(f"Cannot find eightD_json_raw folder at: {BASE_DIR} or {BASE_DIR.parent}")

# Persist KB data folder
KB_DATA_ROOT = BASE_DIR.parent / "KB_motor_drives_8d"
SENTENCE_KB_DIR = KB_DATA_ROOT / "sentence_kb"
FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"
# CAUSE_KB_DIR = KB_DATA_ROOT / "cause_kb"


for p in [SENTENCE_KB_DIR, FAILURE_KB_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) Init KBs
# =========================================================
sentence_kb = SentenceKB(persist_dir=SENTENCE_KB_DIR)
failure_kb = FailureKB(persist_dir=FAILURE_KB_DIR)
# cause_kb = CauseKB(persist_dir=CAUSE_KB_DIR)
meta_kb = FileMetaStore(persist_dir=KB_DATA_ROOT)

# =========================================================
# 3) Ingest all 8D JSON files
# =========================================================
json_files = sorted(JSON_ROOT.glob("*.json"))
print(f"[INFO] JSON_ROOT = {JSON_ROOT}")
print(f"[INFO] Found {len(json_files)} 8D JSON files")

for jp in json_files:
    print(f"[INGEST] {jp.name}")
    ingest_8d_json(
        json_path=jp,
        failure_kb=failure_kb,
        # cause_kb=cause_kb,
        sentence_kb=sentence_kb,
        meta_kb=meta_kb
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


