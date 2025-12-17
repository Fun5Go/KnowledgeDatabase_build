import json
from Information_extraction_8D.main.eight_D_agent import build_8d_case_from_docx

def run(doc_path):
    result = build_8d_case_from_docx(doc_path)

    # out = doc_path.replace(".docx", ".json")
    output_name = "8D_test.json"
    with open(output_name, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    # print("Generated:", out)


if __name__ == "__main__":
    path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D\Motor example\8D6782170310R02 - Motor noise.docx"
    run(path)