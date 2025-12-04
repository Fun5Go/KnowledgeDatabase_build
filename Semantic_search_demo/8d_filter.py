
import requests
import json
from docx import Document


def split_by_headings(doc_path, keep_levels=("Heading 1", "Heading 2")):
    doc = Document(doc_path)
    sections = []
    current_title = None
    current_text = []

    for para in doc.paragraphs:
        # If paragraph is exactly one of the allowed heading levels
        if para.style.name in keep_levels:
            # save previous section
            if current_title is not None:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_text)
                })
                current_text = []

            current_title = para.text
        else:
            # only collect body text
            if current_title is not None:
                current_text.append(para.text)

    # last section
    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_text)
        })

    return sections

def filter_sections(sections, want_prefixes):
    filtered_section= [
        sec for sec in sections
        if any(sec["title"].startswith(p) for p in want_prefixes)
    ]
    return filtered_section

def call_ollama(prompt, model="llama3.1:8b"):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]

def main():
    file_path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D\8D6015220004R01.docx" # Replace with your DOCX file path
    sections = split_by_headings(file_path, keep_levels=("Heading 1", "Heading 2"))
    # print(sections)
    # for sec in sections:
    #     print(sec["title"])

    want_prefixes = ["Introduction", "D2", "D3", "D4", "D5", "D6"]
    filtered_sections = filter_sections(sections, want_prefixes)
    for sec in filtered_sections:
        print(" -", sec["title"])

        # 4) Use LLM (Gemma 3 via Ollama) to extract info from each section
    for sec in filtered_sections:
        print("\n==============================")
        print("Processing section:", sec["title"])
        print("==============================")

        prompt = f"""
You are an 8D quality expert.

Extract structured information from the following 8D section.

Section title: {sec["title"]}

Section content:
{sec["content"]}

Return ONLY valid JSON with the following keys:
- "section_title": repeat the section title
- "summary": 1–3 sentence summary of this section
- "problem": for D2 if present, else null
- "quick_fix": for D3 if present, else null
- "root_cause": for D4 if present, else null
- "solution": for D5 if present, else null
- "implementation": for D6 if present, else null
- "evidence": list of important measurements / tests / facts mentioned
- "risks": list of any risks or limitations mentioned

If a field does not apply, set it to null.
Return ONLY JSON, no explanation.
"""
        try:
            result = call_ollama(prompt)
            print(result)  # this is the JSON string returned by Gemma
        except requests.exceptions.RequestException as e:
            print("Error calling Ollama:", e)


if __name__ == "__main__":
    main()
