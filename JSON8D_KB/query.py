# query_pipeline.py
from kb_structure import FailureKB, CauseKB, SentenceKB
from pathlib import Path



def retrieve_failures(
    *,
    failure_kb: FailureKB,
    failure_mode: str | None,
    failure_element: str | None,
    failure_effect: str | None,
    k: int = 3,
):
    return failure_kb.search(
        failure_mode=failure_mode,
        failure_element=failure_element,
        failure_effect=failure_effect,
        k=k,
    )


def retrieve_causes(
    *,
    cause_kb: CauseKB,
    cause_query: str,
    failure_id: str,
    k: int = 5,
):
    return cause_kb.search_under_failure(
        query=cause_query,
        failure_id=failure_id,
        k=k,
    )

def failure_to_cause_pipeline(
    *,
    failure_mode: str | None,
    failure_element: str | None,
    failure_effect: str | None,
    cause_query: str,
    failure_kb: FailureKB,
    cause_kb: CauseKB,
    sentence_kb: SentenceKB,
    k_failure=3,
    k_cause=5,
):
    results = []

    # -----------------------------
    # 1) Failure (WHAT broke)
    # -----------------------------
    failure_ids = retrieve_failures(
        failure_kb=failure_kb,
        failure_mode=failure_mode,
        failure_element=failure_element,
        failure_effect=failure_effect,
        k=k_failure,
    )

    for fid in failure_ids:
        failure = failure_kb.store.get(fid)
        if not failure:
            continue

        # -----------------------------
        # 2) Cause (WHY it broke)
        # -----------------------------
        cause_ids = retrieve_causes(
            cause_kb=cause_kb,
            cause_query=cause_query,
            failure_id=fid,
            k=k_cause,
        )

        causes = []
        for cid in cause_ids:
            cause = cause_kb.store.get(cid)
            if not cause:
                continue

            evidence = sentence_kb.get_by_ids(
                cause.get("supporting_sentence_ids", [])
            )

            causes.append({
                "cause": cause,
                "evidence": [s.text for s in evidence],
            })

        if causes:
            results.append({
                "failure": failure,
                "causes": causes,
            })

    return results

def detail_print_results(results: list[dict]):

    print("\n================ RESULTS ================")

    if not results:
        print("[WARN] No results. (Maybe you haven't ingested any JSON yet?)")
        return

    for i, r in enumerate(results, start=1):
        f = r.get("failure", {})
        causes = r.get("causes", [])

        print("\n" + "=" * 80)
        print(f"[{i}] FAILURE")
        print(f"ID      : {f.get('failure_id', '')}")
        print(f"Mode    : {f.get('failure_mode', '')}")
        print(f"Element : {f.get('failure_element', '')}")
        print(f"Effect  : {f.get('failure_effect', '')}")
        print(f"Status  : {f.get('status', '')}")

        for j, cblock in enumerate(causes, start=1):
            c = cblock.get("cause", {})
            evidence = cblock.get("evidence", [])

            print("\n→ ROOT CAUSE", f"(#{j})")
            print(f"ID         : {c.get('cause_id', '')}")
            print(f"Cause      : {c.get('root_cause', '')}")
            print(f"Level      : {c.get('cause_level', '')}")
            print(f"Discipline : {c.get('discipline', '')}")
            print(f"Confidence : {c.get('confidence', '')}")

            print("\n→ SUPPORTING EVIDENCE")
            for s in evidence:
                print(f"- {s}")


def resolve_paths():
    base = Path(__file__).resolve().parent
    kb_data = base / "kb_data"
    sentence_dir = kb_data / "sentence_kb"
    failure_dir = kb_data / "failure_kb"
    cause_dir = kb_data / "cause_kb"

    # Check and create directories if not exist
    for p in [sentence_dir, failure_dir, cause_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return sentence_dir, failure_dir, cause_dir


def main():
    sentence_dir, failure_dir, cause_dir = resolve_paths()

    sentence_kb = SentenceKB(sentence_dir)
    failure_kb = FailureKB(failure_dir)
    cause_kb = CauseKB(cause_dir)

    results = failure_to_cause_pipeline(
        failure_mode="motor fails to restart",
        failure_element="",
        failure_effect="",
        cause_query="incorrect start-up state machine",
        failure_kb=failure_kb,
        cause_kb=cause_kb,
        sentence_kb=sentence_kb,
        k_failure=3,
        k_cause=3,
    )
    detail_print_results(results)

    # for r in results:
    #     f = r["failure"]
    #     print("\nFAILURE:", f["failure_id"], f["failure_mode"])

    #     for c in r["causes"]:
    #         print("  CAUSE:", c["cause"]["root_cause"])

if __name__ == "__main__":
    main()
