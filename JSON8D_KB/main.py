from pathlib import Path

from kb_structure import FailureKB, CauseKB, SentenceKB
from ingest_8d import ingest_8d_json
from query import query_failure_to_cause


# =========================================================
# 1) Resolve paths (robust)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# JSON_ROOT = BASE_DIR.parent / "eightD_json_raw"
JSON_ROOT = Path(r'C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_test\failure_identification').resolve()
# if not JSON_ROOT.exists():
#     JSON_ROOT = BASE_DIR.parent / "eightD_json_raw"

if not JSON_ROOT.exists():
    raise FileNotFoundError(f"Cannot find eightD_json_raw folder at: {BASE_DIR} or {BASE_DIR.parent}")

# Persist KB data folder
KB_DATA_ROOT = BASE_DIR / "kb_data"
SENTENCE_KB_DIR = KB_DATA_ROOT / "sentence_kb"
FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"
CAUSE_KB_DIR = KB_DATA_ROOT / "cause_kb"

for p in [SENTENCE_KB_DIR, FAILURE_KB_DIR, CAUSE_KB_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) Init KBs
# =========================================================
sentence_kb = SentenceKB(persist_dir=SENTENCE_KB_DIR)
failure_kb = FailureKB(persist_dir=FAILURE_KB_DIR)
cause_kb = CauseKB(persist_dir=CAUSE_KB_DIR)


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
        cause_kb=cause_kb,
        sentence_kb=sentence_kb,
    )

print("[INFO] Ingest finished")
print(f"Sentence KB count : {sentence_kb.collection.count()}")
print(f"Failure KB count  : {failure_kb.collection.count()}")
print(f"Cause KB count    : {cause_kb.collection.count()}")


# # =========================================================
# # 4) Run an end-to-end query
# # =========================================================
# query = "current spike destroyed power supply"
# print("\n================ QUERY ================")
# print(query)

# results = query_failure_to_cause(
#     query_text=query,
#     failure_kb=failure_kb,
#     cause_kb=cause_kb,
#     sentence_kb=sentence_kb,
#     k_failure=3,
#     k_cause=3,
# )

# # =========================================================
# # 5) Display results
# # =========================================================
# print("\n================ RESULTS ================")

# for i, r in enumerate(results, start=1):
#     f = r["failure"]
#     c = r["cause"]
#     evidence = r["evidence"]

#     print("\n" + "=" * 80)
#     print(f"[{i}] FAILURE")
#     print(f"ID      : {f['failure_id']}")
#     print(f"Mode    : {f['failure_mode']}")
#     print(f"Element : {f['failure_element']}")
#     print(f"Status  : {f['status']}")

#     print("\n→ ROOT CAUSE")
#     print(f"ID         : {c['cause_id']}")
#     print(f"Cause      : {c['root_cause']}")
#     print(f"Level      : {c['cause_level']}")
#     print(f"Discipline : {c['discipline']}")
#     print(f"Confidence : {c['confidence']}")

#     print("\n→ SUPPORTING EVIDENCE")
#     for s in evidence:
#         print(f"- {s}")
