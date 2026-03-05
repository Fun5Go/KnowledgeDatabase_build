from .entity import structure_input_powertrain, structure_input_motorcontrol
from typing import Dict, List, Union, Optional, Any
from collections import defaultdict
from pathlib import Path
from .retrieval import generate_failure_chains_from_structure, build_ground_truth_input

# Vector KB folder
KB_PATH = Path(
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_discipline\failure_kb"
)

results = generate_failure_chains_from_structure(
        #INPUT
        persist_dir=KB_PATH,
        structure_input=structure_input_powertrain,
        #The number of top candidates to retrieve per field type
        top_k_per_field=30,
        # Similarity threshold
        min_similarity=0.45,
        top_n=100, # Return list of failure entity number
        replace=True,
        # Weight score
        weight_element = 0.2,
        weight_mode= 1.0,
        weight_cause = 1.0,
        weight_effect= 1.0,
    )
structrued_res = build_ground_truth_input(results, target_n=20) # To better read
print(structrued_res)