import os
from fmea_to_json.xlsm_parser import process_dfmea_xlsm
from fmea_to_json.xlsx_parser import process_old_fmea_xlsx

INPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\FMEA\Motor"
OUTPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\fmea_json_raw"

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
            process_dfmea_xlsm(path, output_json)
        else:
            process_old_fmea_xlsx(path, output_json)

        print("  ✔ Done")

    except Exception as e:
        print("  ✖ Error:", e)
