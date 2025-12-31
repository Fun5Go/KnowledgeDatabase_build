
from langchain.agents import create_agent
from Information_extraction_8D.tools.section_extractor import extract_d2, extract_d4,parse_8d_doc, extract_failure_d234,extract_iteration_1,extract_iteration_2
from Information_extraction_8D.tools.doc_parser import extract_product
# from tools.doc_parser import parse_8d_doc
# from Agents.main.llm import llm
from Information_extraction_8D.Schemas.eigthD_schema_json_v3 import DocumentInfo,MaintenaceTag, EightDCase, EightDSections, D2Section, D4Section,D3Section, D5Section,D6Section,FailureChain
import os, re
from Information_extraction_8D.Schemas.eightD_sentence_schema_V2 import Iteration1Output
import copy
from typing import List, Dict, Any
from langsmith import traceable, get_current_run_tree
from Information_extraction_8D.Evaluation.evaluation_tool import check_faithfulness

def build_iteration2_input(iter1_output) -> dict:
    return {
        "signals": [
            {
                "id": s.id,
                "text": s.text,
                "entity_type": s.annotations.entity_type,
                "assertion_level": s.annotations.assertion_level,
                "source_section": s.source_section,
                "faithful_score": s.annotations.faithful_score,
                "faithful_type": s.annotations.faithful_type
            }
            for s in iter1_output.selected_sentences
        ],
    }


def assign_sentence_ids(items: List[Dict[str, Any]], doc_prefix: str) -> List[Dict[str, Any]]:
    """
    Assign deterministic sequential IDs grouped by section.
    Example: <doc_prefix>_D2_S001, <doc_prefix>_D4_S012
    """
    counters = {"D2": 0, "D3": 0, "D4": 0}

    for item in items:
        sec = item.source_section
        if sec not in counters:
            raise ValueError(f"Unexpected source_section: {sec}")

        counters[sec] += 1
        item.id = f"{doc_prefix}_{sec}_S{counters[sec]:03d}"

    return items


def annotate_faithfulness_for_sentences(
    selected_sentences: List[Any],
    *,
    d2_raw: str,
    d3_raw: str,
    d4_raw: str,
) -> List[Any]:
    """
    Adds faithful_score & faithful_type into each selected_sentence.annotations
    """

    source_text = "\n".join([
        d2_raw or "",
        d3_raw or "",
        d4_raw or "",
    ])

    for sent in selected_sentences:
        # sent.text is the extracted atomic sentence
        result = check_faithfulness(
            sentence=sent.text,
            source_text=source_text,
        )

        # --- ensure annotations exists ---
        if getattr(sent, "annotations", None) is None:
            sent.annotations = {}

        sent.annotations.faithful_type = result["type"]
        sent.annotations.faithful_score = result["score"]

    return selected_sentences


@traceable(name="8d-extraction-demo")
def build_8d_case_from_docx(doc_path: str) -> EightDCase:

    # 1) Parse document sections using the parse_8d_doc tool
    product_name = extract_product(doc_path)
    print("Product name:", product_name)

    # 2) Extract 8D ID from file name
    base_name = os.path.splitext(os.path.basename(doc_path))[0]
    print("fileID:",base_name)
    document_info = DocumentInfo(
        file_name=base_name,
        product_name=product_name,
    )

    run = get_current_run_tree()
    if run:
        run.metadata.update({
            "filename": base_name,
            # "doc_path": doc_path,
            "product_name": product_name,
        })
    parsed = parse_8d_doc.invoke({"doc_path": doc_path})
    sections = parsed["sections"]


    # Initialization
    d2_raw = None
    d3_raw = None
    d4_raw = None


    # 3) Loop through parsed sections
    for sec in sections:
        title = sec["title"]
        content = sec["content"]

        # --------------------
        # Extract D2 section
        # --------------------
        if title.startswith("D2"):
            d2_raw = content
            d2_section = D2Section(raw_context=d2_raw)

        # --------------------
        # Extract D3 section
        # --------------------
        if title.startswith("D3"):
            print("Copying D3 section...")
            d3_raw = content
            d3_section = D3Section(raw_context=d3_raw)
        # --------------------
        # Extract D4 section
        # --------------------
        if title.startswith("D4"):
            d4_raw = content
            d4_section = D4Section(raw_context=d4_raw)


        # --------------------
        # Extract D5 section
        # --------------------
        if title.startswith("D5"):
            print("Copying D5 section...")
            d5_raw = content
            d5_section = D5Section(raw_context=d5_raw)
        # --------------------
        # Extract D6 section
        # --------------------
        if title.startswith("D6"):
            print("Copying D6 section...")
            d6_raw = content
            d6_section = D6Section(raw_context=d6_raw)

    print("LLM iteration 1:")
    output_iter1 =  extract_iteration_1.invoke({
            "data": {
                "d2_raw": d2_raw or "",
                "d3_raw": d3_raw or "",
                "d4_raw": d4_raw or "",
        }
    })


    #output_iter1 = Iteration1Output(**output_iter1)
    #Add ids to sentences
    output_iter1.selected_sentences = assign_sentence_ids(
    output_iter1.selected_sentences,
    doc_prefix=base_name
)
    output_iter1.selected_sentences = annotate_faithfulness_for_sentences(
    output_iter1.selected_sentences,
    d2_raw=d2_raw,
    d3_raw=d3_raw,
    d4_raw=d4_raw,
) 
    # print(output_iter1)


    input_iter2 = build_iteration2_input(output_iter1)
    print("Done!LLM iteration 2:")
    output_iter2 = extract_iteration_2.invoke({"data":input_iter2})
    system_name = output_iter2.get("system_name") or ""
    print("system name:",system_name)


    failure_dict = copy.deepcopy(output_iter2)

    failure_dict["failure_ID"] = f"{base_name}_F1"
    failure_dict.setdefault("failure_level", "sub_system")
    failure_dict.setdefault("root_causes", [])

    allowed_disciplines = {"HW", "ESW", "MCH", "Other"}

    for c_idx, cause in enumerate(failure_dict["root_causes"]):

        if hasattr(cause, "model_dump"):
            cause = cause.model_dump()

        cause.setdefault("cause_level", "unknown")
        cause.setdefault("discipline_type", "Other")
        cause.setdefault("confidence", 0.5)

        if cause["discipline_type"] not in allowed_disciplines:
            cause["discipline_type"] = "Other"

        cause.setdefault("failure_mechanism", None)
        cause.setdefault("supporting_entities", [])
        cause.setdefault("inferred_insight", None)

        if not cause.get("failure_cause"):
            cause["failure_cause"] = "Unknown cause (LLM incomplete)"

        cause["cause_ID"] = f"{base_name}_F1_C{c_idx + 1}"

        failure_dict["root_causes"][c_idx] = cause

    failure = FailureChain(**failure_dict)

    # 4) Build top-level EightDCase object
    case = EightDCase(
        documents=[document_info],
        maintenance_tag=MaintenaceTag(
            review_status= "pending",
            Version = "V1"
        ),   # add extract_product() later if needed
        system_name=system_name,
        failure= failure,
        sections=EightDSections(
                                D2=d2_section,
                                D3=d3_section,
                                D4=d4_section,
                                D5=d5_section,
                                D6=d6_section,
                            ),
        selected_sentences=output_iter1.selected_sentences,
    )
        

    return case,output_iter1