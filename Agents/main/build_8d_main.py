import json
from Agents.main.eight_D_agent import build_8d_case_from_docx

def run(doc_path):
    result = build_8d_case_from_docx(doc_path)

    out = doc_path.replace(".docx", ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    print("Generated:", out)


if __name__ == "__main__":
    path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D\8D6264240043R03.docx"
    run(path)