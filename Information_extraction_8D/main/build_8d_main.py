import json
from Information_extraction_8D.main.eight_D_agent import build_8d_case_from_docx
import os

SENTENCE_OUTPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_test\sentence_selected"
FAILURE_OUTPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_test\failure_identification"
OUTPUT_DIR = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\eightD_json_raw"
def run(doc_path: str) -> str:
    result, output_iter1 = build_8d_case_from_docx(doc_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(doc_path))[0]

    # ---- 保存最终 EightDCase ----
    output_case_path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
    with open(output_case_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    # ---- 保存 Iteration 1 ----
    output_iter1_path = os.path.join(
        OUTPUT_DIR, f"{base_name}_iter1.json"
    )
    with open(output_iter1_path, "w", encoding="utf-8") as f:
        json.dump(output_iter1.model_dump(), f, indent=2, ensure_ascii=False)

    return output_case_path


if __name__ == "__main__":
    path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D\Motor example\8D6298190081R02.docx"
    run(path)