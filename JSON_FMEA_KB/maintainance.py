from kb_structure import FMEAFailureKB

from pathlib import Path
from copy import deepcopy

KB_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_expand\failure_kb")

failure_kb = FMEAFailureKB(persist_dir=KB_PATH)

# failure_kb.update_merge(FMEAFailure(
#     failure_id="FMEA6799210115R03__F12",
#     failure_mode="Leak at connector",
#     failure_element="Connector seal",
#     failure_effect="Loss of pressure",
# ))

# failure_kb.delete("FMEA6799210115R03__F8", delete_causes=True)
# raw = failure_kb.store.get("FMEA6799210115R03__F16")


# updated = deepcopy(raw)
# updated["failure_effect"] = "function not active"


# failure = FMEAFailure(**updated)

# failure_kb.update_failure(FMEAFailure(
#     failure_id="FMEA6799210115R03__F16",
#     failure_mode="Leak at connector",
#     failure_element="Connector seal",
#     failure_effect="Loss of pressure",
# ))

# failure_kb.update_cause(
#     cause_id="FMEA6799210115R03__F6_C1",
#     failure_cause="Leak at connector",
#     # failure_element="Connector seal",
#     # failure_effect="Loss of pressure",
# )


# failure_kb.delete_failure(failure_id="FMEA6799210115R03__F20",delete_linked_causes=True)
failure_kb.delete_by_semantic_node("element:0c4432535cbf")
