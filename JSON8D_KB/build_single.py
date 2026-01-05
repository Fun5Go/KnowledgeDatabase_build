from pathlib import Path

from kb_structure import FailureKB, CauseKB, SentenceKB
from ingest_8d import ingest_8d_json


def resolve_paths():
    base = Path(__file__).resolve().parent

    kb_data = base / "kb_data"
    sentence_dir = kb_data / "sentence_kb"
    failure_dir = kb_data / "failure_kb"
    cause_dir = kb_data / "cause_kb"
    for p in [sentence_dir, failure_dir, cause_dir]:
        p.mkdir(parents=True, exist_ok=True)

    
    json_root = base.parent / "eightD_json_raw"
    return sentence_dir, failure_dir, cause_dir, json_root


def main():
    sentence_dir, failure_dir, cause_dir, json_root = resolve_paths()

    json_path = json_root / "8D6016160115R01.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    # Init KBs
    sentence_kb = SentenceKB(persist_dir=sentence_dir)
    failure_kb = FailureKB(persist_dir=failure_dir)
    cause_kb = CauseKB(persist_dir=cause_dir)

    print(f"[INFO] Ingest single JSON: {json_path.name}")
    ingest_8d_json(
        json_path=json_path,
        failure_kb=failure_kb,
        cause_kb=cause_kb,
        sentence_kb=sentence_kb,
    )

    print("[INFO] Done.")
    print(f"Sentence KB count : {sentence_kb.collection.count()}")
    print(f"Failure KB count  : {failure_kb.collection.count()}")
    print(f"Cause KB count    : {cause_kb.collection.count()}")


if __name__ == "__main__":
    main()
