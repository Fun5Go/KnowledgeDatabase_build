from langchain.tools import tool
from docx import Document
from typing import Optional, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .llm_init import get_llm_backend
from .failure_schema import FailureCandidates_RAG, FailureCandidates_PURE, FailureCandidates_RAG_FILL
from .prompt_failure_generation import failure_inference_prompt_RAG,failure_inference_prompt_PURE,failure_inference_prompt_RAG_FILL,failure_evaluate_prompt


@tool
def failure_inference_generation_RAG(data: dict) -> dict:
    """Generate failure candidates according to GT example and structure analysis"""
    structure_analysis =  data.get("structure_analysis", "")
    gt_example = data.get("gt_example", "")
    sentences =  data.get("sentences", "")
    llm = get_llm_backend(
        backend="openai",
        model="azure/gpt-4.1",
        json_mode=True, # enable json response parsing
        temperature=0, # deterministic output
    )
    # System prompt
    system_prompt = """You are a senior motor-drive system reliability engineer and FMEA architect.
    This is NOT a semantic similarity task.
    This is a physics-driven failure mechanism reconstruction task.
    You must think like a system failure investigator:
    Infer the most technically coherent and causally valid failure graph.
    Output strictly valid JSON according to the required schema.
    No explanations outside JSON.
    """
    #     # Build a chat prompt template with: system + user messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", failure_inference_prompt_RAG),
    ])
    formatted_prompt = prompt.invoke({
        "gt_example": gt_example,
        "8D_sentences": sentences,
        "structure_analysis": structure_analysis,
    })
        # Call LLM and parse output
    resp = llm.invoke(formatted_prompt.to_messages())
    parser = JsonOutputParser(pydantic_object=FailureCandidates_RAG)
    return parser.parse(resp.content)


@tool
def failure_inference_generation_RAG_FILL(data: dict) -> dict:
    """Generate failure candidates according to GT example and structure analysis"""
    structure_analysis =  data.get("structure_analysis", "")
    fill_failure = data.get("fill_failure", "")
    llm = get_llm_backend(
        backend="openai",
        model="azure/gpt-4.1",
        json_mode=True, # enable json response parsing
        temperature=0, # deterministic output
    )
    # System prompt
    system_prompt = """You are an expert in motor drive systems, reliability engineering,
        and FMEA classification. Only perform infer and fill most relevant FMEA failure text from the given structure analysis to the blank block."""
    #     # Build a chat prompt template with: system + user messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", failure_inference_prompt_RAG_FILL),
    ])
    formatted_prompt = prompt.invoke({
        "to_be_fill_failure": fill_failure,
        "structure_analysis": structure_analysis,
    })
        # Call LLM and parse output
    resp = llm.invoke(formatted_prompt.to_messages())
    parser = JsonOutputParser(pydantic_object=FailureCandidates_RAG_FILL)
    return parser.parse(resp.content)


@tool
def failure_inference_generation_PURE(data: dict) -> dict:
    """Generate failure candidates according to GT example and structure analysis"""
    structure_analysis =  data.get("structure_analysis", "")
    llm = get_llm_backend(
        backend="openai",
        model="azure/gpt-4.1",
        json_mode=True, # enable json response parsing
        temperature=0, # deterministic output
    )
    # System prompt
    system_prompt = """You are an expert in motor drive systems, reliability engineering,
        and FMEA classification. Only perform infer most relevant FMEA failure chain from the given structure analysis."""
    #     # Build a chat prompt template with: system + user messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", failure_inference_prompt_PURE),
    ])
    formatted_prompt = prompt.invoke({
        "structure_analysis": structure_analysis,
    })
        # Call LLM and parse output
    resp = llm.invoke(formatted_prompt.to_messages())
    parser = JsonOutputParser(pydantic_object=FailureCandidates_PURE)
    return parser.parse(resp.content)

@tool
def output_evaluation(data: dict) -> dict:
    """Evalutate the output results"""
    unmatched_predictions =  data.get("unmatched_predictions", "")
    llm = get_llm_backend(
        backend="openai",
        model="azure/gpt-4.1",
        json_mode=True, # enable json response parsing
        temperature=0, # deterministic output
    )
    # System prompt
    system_prompt = """You are an expert in motor drive systems, reliability engineering,
        and FMEA classification. You need to evaluate the reasonableness of the predictions generated by LLM."""
    #     # Build a chat prompt template with: system + user messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", failure_evaluate_prompt),
    ])
    formatted_prompt = prompt.invoke({
        "unmatched_predictions": unmatched_predictions,
    })
        # Call LLM and parse output
    resp = llm.invoke(formatted_prompt.to_messages())
    parser = JsonOutputParser(pydantic_object=FailureCandidates_PURE)
    return parser.parse(resp.content)





