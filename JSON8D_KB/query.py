# query_pipeline.py
from kb_structure import FailureKB, CauseKB, SentenceKB
from pathlib import Path



def query_failure_to_cause(
    query_text: str,
    failure_kb: FailureKB,
    cause_kb: CauseKB,
    sentence_kb: SentenceKB,
    k_failure=3,
    k_cause=5,
):
    # =====================================================
    # 1. Search failures
    # =====================================================
    failure_ids = failure_kb.search(query_text, k=k_failure)

    grouped_results = []

    for fid in failure_ids:
        failure_obj = failure_kb.store.get(fid)
        if failure_obj is None:
            print(f"[WARN] failure_id {fid} not found in store")
            continue

        # =================================================
        # 2. Search causes under this failure
        # =================================================
        cause_ids = cause_kb.search_under_failure(
            query=query_text,
            failure_id=fid,
            k=k_cause,
        )

        causes = []

        for cid in cause_ids:
            cause_obj = cause_kb.store.get(cid)
            if cause_obj is None:
                print(f"[WARN] cause_id {cid} not found in store")
                continue

            evidence_sents = sentence_kb.get_by_ids(
                cause_obj["supporting_sentence_ids"]
            )

            causes.append({
                "cause": cause_obj,
                "evidence": [s.text for s in evidence_sents],
            })

        # If the failure does not have any valid causes, skip it
        if not causes:
            continue

        grouped_results.append({
            "failure": failure_obj,
            "causes": causes,
        })

    return grouped_results





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

    # Load existing persisted KBs
    sentence_kb = SentenceKB(persist_dir=sentence_dir)
    failure_kb = FailureKB(persist_dir=failure_dir)
    cause_kb = CauseKB(persist_dir=cause_dir)

    # Query text
    query_text = "The failure mode is short problem and the failure cause is bootstrap capacitor"
    # query_text = "current spike destroyed power supply"

    print("\n================ QUERY ================")
    print(query_text)

    results = query_failure_to_cause(
        query_text=query_text,
        failure_kb=failure_kb,
        cause_kb=cause_kb,
        sentence_kb=sentence_kb,
        k_failure=3,
        k_cause=3,
    )

    print("\n================ RESULTS ================")
    if not results:
        print("[WARN] No results. (Maybe you haven't ingested any JSON yet?)")
        return

    for i, r in enumerate(results, start=1):
        f = r["failure"]
        causes = r["causes"]

        print("\n" + "=" * 80)
        print(f"[{i}] FAILURE")
        print(f"ID      : {f['failure_id']}")
        print(f"Mode    : {f['failure_mode']}")
        print(f"Element : {f['failure_element']}")
        print(f"Status  : {f['status']}")

        for j, cblock in enumerate(causes, start=1):
            c = cblock["cause"]
            evidence = cblock["evidence"]

            print("\n→ ROOT CAUSE", f"(#{j})")
            print(f"ID         : {c['cause_id']}")
            print(f"Cause      : {c['root_cause']}")
            print(f"Level      : {c['cause_level']}")
            print(f"Discipline : {c['discipline']}")
            print(f"Confidence : {c['confidence']}")

            print("\n→ SUPPORTING EVIDENCE")
            for s in evidence:
                print(f"- {s}")


if __name__ == "__main__":
    main()
