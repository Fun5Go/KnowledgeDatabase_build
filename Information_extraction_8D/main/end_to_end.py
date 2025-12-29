import os
import json
import copy
from Information_extraction_8D.main.eight_D_agent import build_8d_case_from_docx
from typing import List

# ===== directories =====
SENTENCE_OUTPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_test\sentence_selected"
FAILURE_OUTPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_test\failure_identification"

# =============================
# Core run function (single file)
# =============================
def run(doc_path: str) -> None:
    result, output_iter1 = build_8d_case_from_docx(doc_path)

    base_name = os.path.splitext(os.path.basename(doc_path))[0]

    # ---- ensure output dirs ----
    os.makedirs(SENTENCE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FAILURE_OUTPUT_DIR, exist_ok=True)

    # ---- save Iteration 1 (sentence selection) ----
    iter1_path = os.path.join(
        SENTENCE_OUTPUT_DIR, f"{base_name}_iter1.json"
    )
    with open(iter1_path, "w", encoding="utf-8") as f:
        json.dump(output_iter1.model_dump(), f, indent=2, ensure_ascii=False)

    # ---- save final EightDCase (failure identification) ----
    result_path = os.path.join(
        FAILURE_OUTPUT_DIR, f"{base_name}.json"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"✔ Processed: {base_name}")


# =============================
# Batch processing
# =============================
def batch_run(folder_path: str) -> None:
    docx_files: List[str] = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".docx")
    ]

    if not docx_files:
        print("⚠ No .docx files found.")
        return

    print(f"Found {len(docx_files)} docx files.")

    for doc_path in docx_files:
        try:
            run(doc_path)
        except Exception as e:
            print(f"✖ Failed: {os.path.basename(doc_path)}")
            print(f"  Reason: {e}")


# =============================
# Entry point
# =============================
if __name__ == "__main__":
    MOTOR_EXAMPLE_DIR = (
        r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D\Motor example"
    )
    batch_run(MOTOR_EXAMPLE_DIR)