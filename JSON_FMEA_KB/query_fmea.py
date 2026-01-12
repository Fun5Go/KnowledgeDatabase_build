from pathlib import Path
from typing import Optional
from pprint import pprint

from kb_structure import FMEAFailureKB, FMEACauseKB



# =========================================================
# Path resolver
# =========================================================
def resolve_paths():
    base = Path(__file__).resolve().parent
    kb_data = base / "kb_data"
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
    _, cause_dir = resolve_paths()
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


# =========================================================
# High-level demo pipeline
# =========================================================
def query_fmea_demo():
    # -------------------------
    # 1) Failure query (STRUCTURE)
    # -------------------------
    failure_ids, failure_kb = retrieve_failures(
        failure_element="",
        failure_mode="Current measurement damaged",
        failure_effect="Motor drive damaged", 
        top_k=3,
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
            cause_query="Power surge",
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
    query_fmea_demo()
