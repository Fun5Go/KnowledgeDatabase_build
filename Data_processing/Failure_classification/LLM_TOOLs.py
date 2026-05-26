from langchain.tools import tool
from docx import Document
from typing import Optional, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from llm import get_llm_backend
from prompt_identify_classify import domain_type_prompt
from product_schema import DomainTypeIdentificationOutput


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

@tool
def domain_type_identification_LLMcall(data: dict) -> dict:
    """Identify the 8D case type: Motor Drive related and Process Failure"""
    d2_text = data.get("d2_raw", "")
    d3_text = data.get("d3_raw", "")
    d4_text = data.get("d4_raw", "")
    # LLM initialization
    llm = get_llm_backend(
        backend="openai",
        model="azure/gpt-4.1",
        json_mode=True, # enable json response parsing
        temperature=0, # deterministic output
    )
    # System prompt
    identification_system = """You are an expert in motor drive systems, reliability engineering,
        and FMEA classification. Only perform domain and FMEA type identification."""
    # Build a chat prompt template with: system + user messages
    identification_prompt= ChatPromptTemplate.from_messages({
        ("system", identification_system),
        ("user", domain_type_prompt),  
    })
    # Fill the prompt variables with extracted text fields
    prompt = identification_prompt.invoke({
            "d2": d2_text,
            "d3": d3_text,
            "d4": d4_text
    })
    # Call LLM and parse output
    resp = llm.invoke(prompt.to_messages())
    parser = JsonOutputParser(pydantic_object=DomainTypeIdentificationOutput)
    return parser.parse(resp.content)