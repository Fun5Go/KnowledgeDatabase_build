
import requests
import json
from docx import Document
import os
import getpass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langsmith import traceable

# Read key from .env
load_dotenv()
# Manual input
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "8D-LLM-Monitoring"

@traceable(run_type="llm", name="8D-Section-Extraction")
def call_litellm(prompt, model="azure/gpt-4.1", api_base="http://litellm.ame.local"):
    url = f"{api_base}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "stream": False,
    }

    resp = requests.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]




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


# --------------------------
# section type
# --------------------------
def get_section_type(title):
    mapping = {
        "Introduction": "Intro",
        "1 Introduction": "Intro",
        "D1": "D1", "D2": "D2", "D3": "D3",
        "D4": "D4", "D5": "D5", "D6": "D6"
    }
    for key in mapping:
        if title.startswith(key):
            return mapping[key]
    return "Unknown"


# --------------------------
# active fields mapping
# --------------------------
def get_active_fields(section_type):
    mapping = {
        "Intro": ["summary"],
        "D1": ["summary"],
        "D2": ["problem", "symptoms"],
        "D3": ["containment_actions"],
        "D4": ["root_causes"],
        "D5": ["solutions"],
        "D6": ["implementation"]
    }
    return mapping.get(section_type, [])


# --------------------------
# PRODUCT extraction from Introduction
# --------------------------
def extract_product_name(intro_text):
    prompt = f"""
Extract product/system name from Introduction.

INTRO TEXT:
{intro_text}

Return ONLY JSON:
{{
  "product": "",
  "alternate_names": []
}}
"""
    data = json.loads(call_litellm(prompt))
    return data.get("product", "")


# --------------------------
# MAIN PIPELINE
# --------------------------
def convert_8d_to_json(doc_path):

    sections = split_by_headings(doc_path)

    # extract intro text
    intro_text = ""
    for sec in sections:
        if get_section_type(sec["title"]) == "Intro":
            intro_text = sec["content"]
            break

    product = extract_product_name(intro_text)

    result = {
        "case_id": os.path.basename(doc_path).replace(".docx", ""),
        "product": product,
        "sections": []
    }

    # PROCESS EACH SECTION
    for sec in sections:
        section_type = get_section_type(sec["title"])
        active_fields = get_active_fields(section_type)

        prompt = f"""
You are an expert in 8D root cause analysis.

Extract structured data ONLY for the given section.
STRICT RULES:
- Only fill fields relevant for this section_type.
- DO NOT invent information from other sections.
- DO NOT repeat information from other sections.
- All fields NOT relevant for this section_type MUST be null or empty list.
- "raw_text" must contain the exact section text without modification.

SECTION TYPE: {section_type}
SECTION TITLE: {sec["title"]}

RAW SECTION TEXT:
{sec["content"]}

Allowed fields per section_type:

Intro / D1:
  fill: summary
  others: null/empty

D2 (Define Problem & Symptoms):
  fill: problem, symptoms, summary
  others: null/empty

D3 (Containment Actions):
  fill: containment_actions, summary
  others: null/empty

D4 (Root Cause Analysis):
  fill: root_causes, evidence, summary
  others: null/empty

D5 (Solution Proposal):
  fill: solutions, summary
  others: null/empty

D6 (Implementation / Verification):
  fill: implementation, summary
  others: null/empty

Return JSON ONLY in this schema:

{{
  "section_title": "",
  "section_type": "",
  "summary": "",
  "raw_text": "",
  "problem": null,
  "symptoms": [],
  "root_causes": [],
  "containment_actions": [],
  "solutions": [],
  "implementation": [],
  "evidence": [],
  "risks": [],
  "active_fields": []
}}

IMPORTANT:
- Never duplicate content from another 8D section.
- Never fill fields not allowed for this section type.
- Only use information inside RAW SECTION TEXT.
"""
        parsed = json.loads(call_litellm(prompt))

        # Always include raw text
        parsed["raw_text"] = sec["content"]

        # Inject active fields
        parsed["active_fields"] = active_fields

        result["sections"].append(parsed)

    return result


# --------------------------
# SAVE TO JSON FILE
# --------------------------
def save_json(data, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --------------------------
# Example
# --------------------------
if __name__ == "__main__":
    fp = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D\8D6015220004R01.docx"
    final_json = convert_8d_to_json(fp)

    save_path = fp.replace(".docx", ".json")
    save_json(final_json, save_path)

    print("Saved to:", save_path)

    # def filter_sections(sections, want_prefixes):
#     filtered_section= [
#         sec for sec in sections
#         if any(sec["title"].startswith(p) for p in want_prefixes)
#     ]
#     return filtered_section


# def call_ollama(prompt, model="llama3.1:8b"):
#     url = "http://localhost:11434/api/chat"
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "user", "content": prompt}
#         ],
#         "stream": False
#     }
#     resp = requests.post(url, json=payload)
#     resp.raise_for_status()
#     data = resp.json()
#     return data["message"]["content"]

# 


#     want_prefixes = ["Introduction", "D2", "D3", "D4", "D5", "D6"]
#     filtered_sections = filter_sections(sections, want_prefixes)
#     for sec in filtered_sections:
#         print(" -", sec["title"])

#         # 4) Use LLM (Gemma 3 via Ollama) to extract info from each section
#     for sec in filtered_sections:
#         print("\n==============================")
#         print("Processing section:", sec["title"])
#         print("==============================")

#         prompt = f"""
# You are an 8D quality expert.

# Extract structured information from the following 8D section.

# Section title: {sec["title"]}

# Section content:
# {sec["content"]}

# Return ONLY valid JSON with the following keys:
# - "section_title": repeat the section title
# - "summary": 1–3 sentence summary of this section
# - "problem": for D2 if present, else null
# - "quick_fix": for D3 if present, else null
# - "root_cause": for D4 if present, else null
# - "solution": for D5 if present, else null
# - "implementation": for D6 if present, else null
# - "evidence": list of important measurements / tests / facts mentioned
# - "risks": list of any risks or limitations mentioned

# If a field does not apply, set it to null.
# Return ONLY JSON, no explanation.
# """
#         try:
#             result = call_litellm(prompt)
#             print(result)  # this is the JSON string returned by Gemma
#         except requests.exceptions.RequestException as e:
#             print("Error calling:", e)