from pathlib import Path

from kb_structure import FailureKB, CauseKB, SentenceKB, FileMetaStore
from ingest_8d import ingest_8d_json


def resolve_paths():
    base = Path(__file__).resolve().parent

    kb_data = base / "kb_data_motor_drives"
    sentence_dir = kb_data / "sentence_kb"
    failure_dir = kb_data / "failure_kb"
    cause_dir = kb_data / "cause_kb"
    meta_dir = kb_data / "meta_kb"
    for p in [sentence_dir, failure_dir, cause_dir]:
        p.mkdir(parents=True, exist_ok=True)

    
    # json_root = base.parent / "eightD_json_V2"
    json_root = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_MD\failure_identification")
    return sentence_dir, failure_dir, cause_dir, meta_dir, json_root


def main():
    sentence_dir, failure_dir, cause_dir, meta_dir, json_root = resolve_paths()

    json_path = json_root / "8D6001175615R01.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    # Init KBs
    sentence_kb = SentenceKB(persist_dir=sentence_dir)
    failure_kb = FailureKB(persist_dir=failure_dir)
    cause_kb = CauseKB(persist_dir=cause_dir)
    meta_kb = FileMetaStore(persist_dir=meta_dir)

    print(f"[INFO] Ingest single JSON: {json_path.name}")
    ingest_8d_json(
        json_path=json_path,
        failure_kb=failure_kb,
        cause_kb=cause_kb,
        sentence_kb=sentence_kb,
        meta_kb= meta_kb,
    )

    print("[INFO] Done.")
    print(f"Sentence KB count : {sentence_kb.collection.count()}")
    print(f"Failure KB count  : {failure_kb.collection.count()}")
    print(f"Cause KB count    : {cause_kb.collection.count()}")


if __name__ == "__main__":
    main()
