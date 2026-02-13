from langchain.tools import tool
from docx import Document
from typing import Optional, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .llm_init import get_llm_backend
from .failure_schema import FailureCandidates
from .prompt_failure_generation import failure_inference_prompt


@tool
def failure_inference_generation(data: dict) -> dict:
    """Generate failure candidates according to GT example and structure analysis"""
    structure_analysis =  data.get("structure_analysis", "")
    gt_example = data.get("gt_example", "")
    llm = get_llm_backend(
        backend="openai",
        model="azure/gpt-4.1",
        json_mode=True, # enable json response parsing
        temperature=0, # deterministic output
    )
    # System prompt
    system_prompt = """You are an expert in motor drive systems, reliability engineering,
        and FMEA classification. Only perform infer most relevant FMEA failure chain from the given structure analysis according to the ground truth example."""
    #     # Build a chat prompt template with: system + user messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", failure_inference_prompt),
    ])
    formatted_prompt = prompt.invoke({
        "gt_example": gt_example,
        "structure_analysis": structure_analysis,
    })
        # Call LLM and parse output
    resp = llm.invoke(formatted_prompt.to_messages())
    parser = JsonOutputParser(pydantic_object=FailureCandidates)
    return parser.parse(resp.content)
