from pathlib import Path
from kb_structure import FMEAFailureKB, FMEACauseKB
from ingest_fmea import ingest_fmea_json

def resolve_paths():
    base = Path(__file__).resolve().parent

    kb_data = base / "kb_data"
    failure_dir = kb_data / "failure_kb"
    cause_dir = kb_data / "cause_kb"

    for p in [failure_dir, cause_dir]:
        p.mkdir(parents=True, exist_ok=True)

    json_root = base.parent / "fmea_json_raw"
    return failure_dir, cause_dir, json_root


def main():
    failure_dir, cause_dir, json_root = resolve_paths()

    json_path = json_root / "FMEA6022160004R03.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    # Init KBs
    failure_kb = FMEAFailureKB(persist_dir=failure_dir)
    cause_kb = FMEACauseKB(persist_dir=cause_dir)

    print(f"[INFO] Ingest single FMEA JSON: {json_path.name}")

    ingest_fmea_json(
        json_path=json_path,
        failure_kb=failure_kb,
        cause_kb=cause_kb,
    )

    print("[INFO] Done.")
    print(f"Failure KB count : {failure_kb.collection.count()}")
    print(f"Cause KB count   : {cause_kb.collection.count()}")


if __name__ == "__main__":
    main()
