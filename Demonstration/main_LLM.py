from .RAG import RAG_pipeline, save_failure_candidates_to_json
from .entity import structure_input_powertrain, structure_input_motorcontrol
from pathlib import Path

KB_PATH = Path(
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KB_motor_drives_MOTORCONTROL\failure_kb"
)

result,OUTPUT_PATH = RAG_pipeline(structure_input=structure_input_motorcontrol, KB_PATH=KB_PATH, top_k_per_field=30, top_n=50,
                               target_n=30, weight_element = 0.6, min_similarity=0.45, RAG = False, FILL = True)
print("\n================ FAILURE CANDIDATES ================\n")
# print(json.dumps(result, indent=4))
save_failure_candidates_to_json(result, OUTPUT_PATH)