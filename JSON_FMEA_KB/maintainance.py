from kb_structure import FMEAFailureKB, FMEAFailure

from pathlib import Path
from copy import deepcopy

KB_PATH =  Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives\failure_kb")

failure_kb = FMEAFailureKB(persist_dir=KB_PATH)

# failure_kb.update_merge(FMEAFailure(
#     failure_id="FMEA6799210115R03__F12",
#     failure_mode="Leak at connector",
#     failure_element="Connector seal",
#     failure_effect="Loss of pressure",
# ))

# failure_kb.delete("FMEA6799210115R03__F8", delete_causes=True)
raw = failure_kb.failure_store.get("FMEA6799210115R03__F16")


updated = deepcopy(raw)
updated["failure_effect"] = "STO function not active"


failure = FMEAFailure(**updated)

failure_kb.update_failure(
    failure,
    delete_stale_roles=True
)