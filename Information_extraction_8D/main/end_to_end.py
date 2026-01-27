import os
import json
import copy
from Information_extraction_8D.main.eight_D_agent import build_8d_case_from_json
from typing import List

# ===== directories =====
SENTENCE_OUTPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_MD\sentence_selected"
FAILURE_OUTPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_MD\failure_identification"

# =============================
# Core run function (single file)
# =============================
def run(json_path: str) -> None:
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # ---- ensure output dirs ----
    os.makedirs(SENTENCE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FAILURE_OUTPUT_DIR, exist_ok=True)

    result_path = os.path.join(FAILURE_OUTPUT_DIR, f"{base_name}.json")
    if os.path.exists(result_path):
        print(f"Skip (already processed): {base_name}")
        return

    print(f"Start to process the file: {base_name}")

    result, output_iter1 = build_8d_case_from_json(json_path)

    # ---- save Iteration 1 (sentence selection) ----
    iter1_path = os.path.join(
        SENTENCE_OUTPUT_DIR, f"{base_name}_sentences.json"
    )
    with open(iter1_path, "w", encoding="utf-8") as f:
        json.dump(
            output_iter1.model_dump(),
            f,
            indent=2,
            ensure_ascii=False
        )

    # ---- save final EightDCase (failure identification) ----
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            result.model_dump(),
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Processed: {base_name}")


# =============================
# Batch processing
# =============================
def batch_run(folder_path: str) -> None:
    json_files: List[str] = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".json")
            and not f.startswith(".")
        ],
        key=lambda x: os.path.basename(x)
    )

    if not json_files:
        print("No .json files found.")
        return

    print(f"Found {len(json_files)} json files.")

    for i, json_path in enumerate(json_files[:10]):
        try:
            print(f"Processing {i+1}/{len(json_files)}: {os.path.basename(json_path)}")
            run(json_path)
            print(f"✔ Success: {os.path.basename(json_path)}")
        except Exception as e:
            print(f"✖ Failed: {os.path.basename(json_path)}")
            print(f"  Reason: {e}")

# =============================
# Entry point
# =============================
if __name__ == "__main__":
    MOTOR_EXAMPLE_DIR = (
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_raw_meta\motor_drives"
    )
    batch_run(MOTOR_EXAMPLE_DIR)