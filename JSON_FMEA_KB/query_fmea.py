from pathlib import Path
from typing import Optional
from pprint import pprint

from JSON_FMEA_KB.kb_structure import FMEAFailureKB, FMEACauseKB, FailureRetriever
# No module named 'kb_structure'

import json



# =========================================================
# Path resolver
# =========================================================
def resolve_paths():
    base = Path(__file__).resolve().parent
    kb_data = base / "kb_data"
    return kb_data / "failure_kb", kb_data / "cause_kb"

def resolve_paths_motor_drivesKB():
    base = Path(__file__).resolve().parent
    kb_data = base / "kb_motor_drives"
    return kb_data / "failure_kb", kb_data / "cause_kb"


# =========================================================
# Failure retrieval (independent)
# =========================================================
def retrieve_failures(
    failure_mode: Optional[str] = None,
    failure_element: Optional[str] = None,
    failure_effect: Optional[str] = None,
    top_k: int = 3,
):
    failure_dir, _ = resolve_paths()
    failure_kb = FMEAFailureKB(persist_dir=failure_dir)

    return failure_kb.search(
        failure_mode=failure_mode,
        failure_element=failure_element,
        failure_effect=failure_effect,
        k=top_k,
    ), failure_kb


# =========================================================
# Cause retrieval (independent)
# =========================================================
def retrieve_causes(
    cause_query: str,
    failure_id: Optional[str] = None,
    top_k: int = 5,
):
    _, cause_dir = resolve_paths_motor_drivesKB()
    cause_kb = FMEACauseKB(persist_dir=cause_dir)

    if failure_id:
        return cause_kb.search_under_failure(
            query=cause_query,
            failure_id=failure_id,
            k=top_k,
        ), cause_kb

    return cause_kb.search(
        query=cause_query,
        k=top_k,
    ), cause_kb


def eightD_fmea_search(
    signals: list[dict],
    productPnID: Optional[int] = None,   # 
):
    BASE_DIR = Path(__file__).resolve().parent
    # FAILURE_KB_DIR = BASE_DIR / "KB_motor_drives" / "failure_kb"
    failure_dir, cause_dir = resolve_paths_motor_drivesKB()
    failure_kb = FMEAFailureKB(persist_dir=failure_dir)
    cause_kb = FMEACauseKB(persist_dir=cause_dir)
    

    retriever = FailureRetriever(
        persist_dir=failure_dir,
        collection_name="fmea_failure_kb",
        store=failure_kb.store,
    )

    # aggregate query
    aggregate_text = "\n".join(
        s["text"] for s in signals
        if s.get("source_section") in ("D2", "D4")
    )

    failure_ids = retriever.search_from_8d(
        text=aggregate_text,
        d_stage="D2+D4",
        productPnID=productPnID,
        k=3,
    )


    print(f"Matched ID: {failure_ids}")
    results = []
    for rank, fid in enumerate(failure_ids, start=1):
        failure = failure_kb.store.get(fid, {})

        cause_ids, cause_kb = retrieve_causes(
            cause_query="",
            failure_id=fid,
            top_k=5,
        )
        causes = []
        for cid in cause_ids:
            cause = cause_kb.store.get(cid, {})
            causes.append({
                "failure_cause": cause.get("failure_cause"),
            })
        results.append({
            "Failure element": failure.get('failure_element'),
            "Failure mode": failure.get('failure_mode'),
            "Failure effect": failure.get('failure_effect'),
            "causes": causes,
        })

    return results


# =========================================================
# High-level demo pipeline
# =========================================================
def query_fmea_demo():
    # -------------------------
    # 1) Failure query (STRUCTURE)
    # -------------------------
    failure_ids, failure_kb = retrieve_failures(
        failure_element="",
        failure_mode="",
        failure_effect="", 
        top_k=5,
    )

    if not failure_ids:
        print("No similar failures found.")
        return

    print("=" * 80)
    print("FAILURE RESULTS")
    print("=" * 80)

    for rank, fid in enumerate(failure_ids, start=1):
        failure = failure_kb.get(fid)

        print("-" * 80)
        print(f"[Failure #{rank}] {fid}")
        print(f"Mode     : {failure.get('failure_mode')}")
        print(f"Element  : {failure.get('failure_element')}")
        print(f"Effect   : {failure.get('failure_effect')}")
        print(f"Severity : {failure.get('severity')} | RPN: {failure.get('rpn')}")

        # -------------------------
        # 2) Cause query (MECHANISM)
        # -------------------------
        cause_ids, cause_kb = retrieve_causes(
            cause_query=  "",
            failure_id=fid,
            top_k=5,
        )

        print(f"\n→ Linked causes ({len(cause_ids)}):")
        if not cause_ids:
            print("  (no causes retrieved)")
            continue

        for cid in cause_ids:
            cause = cause_kb.store.get(cid, {})
            print(f"  - Cause ID   : {cid}")
            print(f"    Cause text : {cause.get('failure_cause')}")
            print(f"    Discipline : {cause.get('discipline')}")
            print()

    print("=" * 80)


# =========================================================
if __name__ == "__main__":

    # query_fmea_demo()
    BASE_DIR = Path(__file__).resolve().parent
    # Persist KB data folders
    KB_DATA_ROOT = BASE_DIR / "KB_motor_drives"
    FAILURE_KB_DIR = KB_DATA_ROOT / "failure_kb"
    retriever = FailureRetriever(
    persist_dir=FAILURE_KB_DIR,
    collection_name="fmea_failure_kb",
    store=[],
)
    ids = retriever.search_from_8d(
    text= "Problem description\nAccording to Fallbrook Technologies Inc. (FTI) the connection between the Output Speed Sensor and the Main PCB is not adequate for HHI product use. Tolerance studies at FTI show that in nominal conditions, there is very little interference between the mating components (Figure 2-1):\nAs the parts deviate from nominal conditions, the components will not interfere (0.15mm gap), resulting in little to no connection and problematic performance.\nThe Output Speed Sensors mating header requires a 0.46mm x 0.46 mm pin:\nThe current Output Speed Sensor at FTI has a 0.43mm x 0.41mm pin, which is smaller than required by the header.\nDetails of components\nCurrent assembled header (AME PN 5050-0007-0015):\nProduct description: Archer M52 - 1.27mm (0.05”) Pitch - SIL Vertical Socket;\nManufacturer: Harwin;\nManufacturer PN: M52-5010345;\nPinning: 1x3p;\nMating pin size: 0.46mm square;\nConnector height: 8.5mm;\nFinish: Gold over nickel.\nCurrent assembled hall-sensor (AME PN 1520-0001-0019):\nProduct description: Hall-effect Latch A3213EUA-T;\nManufacturer: Allegro;\nManufacturer PN: A3213EUA-T;\nPinning: 1x3p;\nPin size: 0.36 ~ 0.48 x 0.35 ~ 0.44mm rectangular;\nPin height: 15.24 ~ 16.26mm;\nFinish: 100% matte tine plated.",
    d_stage=["D2", "D4"],
)
    print(ids)
