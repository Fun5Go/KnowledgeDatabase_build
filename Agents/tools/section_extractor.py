from langchain.tools import tool
from Agents.Prompts.eightD_extract_prompt import D2_prompt, D4_prompt
from Agents.tools.doc_parser import safe_json
from Agents.Prompts.eightD_prompt_integrate import Prompt
from Agents.main.llm import llm
from docx import Document
from typing import Optional, List, Literal

# @tool
# def extract_d2(section_text: str):
#     """Extract structured D2 information."""
#     prompt = D2_prompt.format(content=section_text)
#     resp = llm.invoke(prompt)
#     return safe_json(resp.content)
@tool
def extract_d2(section_text: str):
    """Extract structured D2 information."""

    messages = [
        {
            "role": "system",
            "content": "You are an expert reliability and systems engineer. Always output STRICT JSON."
        },
        {
            "role": "user",
            "content": D2_prompt.format(content=section_text)
        }
    ]

    resp = llm.invoke(messages)
    return safe_json(resp.content)


@tool
def extract_d4(data: dict):
    """Extract D4 root cause JSON, using D2 info as structured context."""

    # Tool input is always ONE dict — extract fields manually
    section_text = data.get("section_text", "")
    d2_info = data.get("d2_info", {}) or {}

    d2_context = build_d2_context(d2_info)
    # print("flatten D2 context:", d2_context,"\n")


    # Insert BOTH content and d2_context in the prompt
    messages = [
        {
            "role": "system",
            "content": "You are an expert root cause analysis engineer. Extract structured D4 information from the following 8D report text and the former section D2: Define problems and Symptoms.."
        },
        {
            "role": "user",
            "content":D4_prompt.format(
            content=section_text,
            d2_context=d2_context
            )
        }
    ]
    # Call LLM
    resp = llm.invoke(messages)

    # resp.content may not exist depending on client
    text = resp.content if hasattr(resp, "content") else resp

    return safe_json(text)

@tool
def extract_failure_d234(data: dict) -> dict:
    """Analyze D2, D3, D4 sections to extract failures """
    d2_text = data.get("d2_raw")
    d3_text = data.get("d3_raw")
    d4_text = data.get("d4_raw")
    messages = [
        {
            "role": "system",
            "content": "You are an expert root cause analysis engineer. Extract structured D2, D3, D4 information from the following 8D report text."
        },
        {
            "role": "user",
            "content":Prompt.format(
            d2_text =d2_text,
            d3_text=d3_text,
            d4_text = d4_text
            )
        }
    ]
    resp = llm.invoke(messages)
    return safe_json(resp.content)

@tool
def parse_8d_doc(doc_path: str) -> dict:
    """
    Parse a DOCX 8D document into raw sections (title + content).
    Returns {"sections": [{"title": ..., "content": ...}, ...]}.
    """
    doc = Document(doc_path)
    sections = []
    current_title = None
    current_text: List[str] = []

    for para in doc.paragraphs:
        style = para.style.name

        if style == "Heading 2":   # e.g., "2.2 D2", "2.4 D4"
            if current_title:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_text).strip()
                })
            current_title = para.text.strip()
            current_text = []
        else:
            if current_title and para.text.strip():
                current_text.append(para.text.strip())

    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_text).strip()
        })

    return {"sections": sections}

def build_d2_context(d2_info: dict) -> str:
    failures = d2_info.get("failures", [])

    lines = []
    for f in failures:
        lines.append(f"""
  - Element: {f.get('system_element')}
  - Mode: {f.get('failure_mode')}
  - Effect: {f.get('failure_effect')}
""")

    failures_block = "\n".join(lines)

    context_block = f"""
System Name: {d2_info.get('system_name') or d2_info.get('System_name')}
Problem Symptoms: {d2_info.get('problem_symptoms')}

Failures:
{failures_block}
"""
    return context_block