from langchain.tools import tool
from Information_extraction_8D.Prompts.eightD_extract_prompt import D2_prompt, D4_prompt
from Information_extraction_8D.tools.doc_parser import safe_json

from Information_extraction_8D.Prompts.eightD_prompt_integrate import Prompt
from Information_extraction_8D.Prompts.eightD_prompt_iteration import iter_prompt_1
from Information_extraction_8D.Prompts.eightD_prompt_iteration2 import iter_prompt_2
from Information_extraction_8D.main.llm import get_llm_backend
from docx import Document
from typing import Optional, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from Information_extraction_8D.Schemas.eightD_sentence_schema_V2 import Iteration1Output
from langsmith import get_current_run_tree, traceable
import os


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
    d2_text = data.get("d2_raw", "")
    d3_text = data.get("d3_raw", "")
    d4_text = data.get("d4_raw", "")
    llm = get_llm_backend(
        backend="openai",
        model="azure/gpt-4.1",
        json_mode=True,
        temperature=0,
    )
    extractor_system = "You are an expert reliability and systems engineer. Always output STRICT JSON."

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", extractor_system),
        ("user", Prompt),  
    ])
    prompt = extractor_prompt.invoke({
            "d2": d2_text,
            "d3": d3_text,
            "d4": d4_text
    })
    resp = llm.invoke(prompt.to_messages())
    parser = JsonOutputParser()
    return parser.parse(resp.content)

@tool
def extract_iteration_1(data: dict) -> dict:
    """Analyze D2, D3, D4 sections to extract failures by LLM iteration """
    d2_text = data.get("d2_raw", "")
    d3_text = data.get("d3_raw", "")
    d4_text = data.get("d4_raw", "")
    llm = get_llm_backend(
        backend="openai",
        model="azure/gpt-4.1",
        json_mode=True,
        temperature=0,
    )
    iteration1_system_prompt= """
    You are an expert quality and reliability engineer specializing in 8D problem solving, FMEA, and failure analysis.
    Your task is STRICTLY LIMITED to selecting HIGH VALUE sentences from the input text D2,D3,D4
    """ 

    iteration_prompt_1 = ChatPromptTemplate.from_messages([
        ("system", iteration1_system_prompt),
        ("user", iter_prompt_1),  
    ])
    prompt = iteration_prompt_1.invoke({
            "d2": d2_text,
            "d3": d3_text,
            "d4": d4_text
    })
    resp = llm.invoke(prompt.to_messages())
    parsed = JsonOutputParser().parse(resp.content)

    # Optional: strict validation
    validated = Iteration1Output(**parsed)
    return validated

@tool
def extract_iteration_2(data: dict) -> dict:
    """Analyze D2, D3, D4 sections to extract failures by LLM iteration """
    llm = get_llm_backend(
        # backend="local",
        # model="llama3.1:8b",
        # temperature=0,
        backend="openai",
        model="azure/gpt-4.1",
        json_mode=True,
        temperature=0,
    )

    iteration_system_2 = """
    You are an expert reliability and FMEA engineer.

    You are given extracted text signals from an 8D report:
    - D2: Define problem and symptoms
    - D3: Interim containment / quick fix
    - D4: Root cause analysis

    Core constraints (MANDATORY):
    - The entire input represents ONE failure case.
    - Do NOT split into multiple failures.
    - Use ONLY the provided signals.
    - Do NOT invent or infer facts beyond the signals.
    - Output STRICT JSON only.
"""
    signals_bullets = "\n".join(
        f"- [id:{s.get('id','')}]"
        f"[{s.get('source_section','?')}]"
        f"[{s.get('entity_type','?')}]"
        f"[{s.get('assertion_level','?')}] "
        f"{s.get('text','')} "
        f"(faithful_score={s.get('faithful_score','?')}, "
        f"type={s.get('faithful_type','?')})"
        for s in data.get("signals", [])
    )
    iteration_prompt_2 = ChatPromptTemplate.from_messages([
        ("system", iteration_system_2),
        ("user", iter_prompt_2),  
    ])
    prompt = iteration_prompt_2.invoke({
        "signals": signals_bullets
    })
    resp = llm.invoke(prompt.to_messages())
    parser = JsonOutputParser()
    return parser.parse(resp.content)


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