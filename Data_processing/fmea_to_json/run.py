import os
from Data_processing.fmea_to_json.xlsm_parser import process_dfmea_xlsm
from Data_processing.fmea_to_json.xlsx_parser import process_old_fmea_xlsx
from Data_processing.fmea_to_json.common_utils import load_fmea_index
INPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\RAW\FMEA_ALL"
OUTPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\FMEA_JSON_ALL"
FMEA_INDEX_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\Orion_list\FMEA\FMEA_with_filename.json"
FMEA_INDEX = load_fmea_index(FMEA_INDEX_PATH)


os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    path = os.path.join(INPUT_DIR, file)
    name, ext = os.path.splitext(file)

    if ext.lower() not in [".xlsm", ".xlsx"]:
        continue

    output_json = os.path.join(OUTPUT_DIR, name + ".json")
    print(f"Processing {file}")

    try:
        if ext.lower() == ".xlsm":
            process_dfmea_xlsm(path, output_json,sheet_index=1, fmea_index=FMEA_INDEX)
        else:
            process_old_fmea_xlsx(path, output_json,fmea_index=FMEA_INDEX)

        print("  ✔ Done")

    except Exception as e:
        print("  ✖ Error:", e)

        
