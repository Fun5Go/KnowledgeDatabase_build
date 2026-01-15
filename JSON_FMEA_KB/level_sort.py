PROCESS_WORDS = {
    # manufacturing / assembly
    "assembly", "assemble", "mount", "install", "weld", "solder",
    "coating", "curing",

    # human / operation
    "operator", "manual", "handling",

    # logistics
    "pack", "packing", "shipment",

    # quality / process control
    "torque", "fixture", "alignment"
}

def score_process_cause(cause_text: str) -> int:
    if not cause_text:
        return 0
    text = cause_text.lower()
    return sum(1 for w in PROCESS_WORDS if w in text)


def detect_fmea_level(record, threshold: int = 1) -> str:
    """
    Return: 'process' | 'non-process'
    """
    cause = record.get("cause_text", "")
    process_score = score_process_cause(cause)

    if process_score >= threshold:
        return "process"
    return "non-process"

def detect_fmea_level_with_debug(record, threshold=1):
    cause = record.get("cause_text", "")
    text = cause.lower()

    hits = [w for w in PROCESS_WORDS if w in text]
    level = "process" if len(hits) >= threshold else "non-process"

    return {
        "level": level,
        "process_score": len(hits),
        "process_hits": hits
    }