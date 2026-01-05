from pathlib import Path
from pprint import pprint

from kb_structure import FMEAFailureKB, FMEACauseKB


def resolve_paths():
    base = Path(__file__).resolve().parent

    kb_data = base / "kb_data"
    failure_dir = kb_data / "failure_kb"
    cause_dir = kb_data / "cause_kb"

    return failure_dir, cause_dir


def query_fmea(
    failure_query: str,
    top_k_failure: int = 3,
    top_k_cause: int = 5,
):
    failure_dir, cause_dir = resolve_paths()

    failure_kb = FMEAFailureKB(persist_dir=failure_dir)
    cause_kb = FMEACauseKB(persist_dir=cause_dir)

    print("=" * 80)
    print("FAILURE QUERY:")
    print(failure_query)
    print("=" * 80)

    # Search failures
    failure_ids = failure_kb.search(
        query=failure_query,
        k=top_k_failure,
    )

    if not failure_ids:
        print("No similar failures found.")
        return

    print(f"\nTop {len(failure_ids)} similar failures:\n")

    for rank, fid in enumerate(failure_ids, start=1):
        failure = failure_kb.store.get(fid)

        print("-" * 80)
        print(f"[Failure #{rank}]")
        print(f"Failure ID    : {fid}")
        print(f"Failure mode  : {failure.get('failure_mode')}")
        print(f"Element       : {failure.get('failure_element')}")
        print(f"Function      : {failure.get('function')}")
        print(f"Effect        : {failure.get('failure_effect')}")
        print(f"Severity / RPN: {failure.get('severity')} / {failure.get('rpn')}")

        cause_ids = failure.get("cause_ids", [])
        print(f"\n→ Linked causes ({len(cause_ids)}):")

        # Search the causes under the failure
        cause_results = cause_kb.search_under_failure(
            query=failure_query,
            failure_id=fid,
            k=top_k_cause,
        )

        if not cause_results:
            print("  (no causes retrieved)")
            continue

        for cid in cause_results:
            cause = cause_kb.store.get(cid, {})
            print(f"  - Cause ID   : {cid}")
            print(f"    Cause text : {cause.get('failure_cause')}")
            print(f"    Discipline : {cause.get('discipline')}")
            print()

    print("=" * 80)


if __name__ == "__main__":
    query_fmea(
        failure_query="no connection between control module and control card",
        top_k_failure=3,
        top_k_cause=5,
    )
