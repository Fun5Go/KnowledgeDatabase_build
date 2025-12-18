
from langchain.agents import create_agent
from Information_extraction_8D.tools.section_extractor import extract_d2, extract_d4,parse_8d_doc, extract_failure_d234,extract_iteration_1,extract_iteration_2
from Information_extraction_8D.tools.doc_parser import extract_product
# from tools.doc_parser import parse_8d_doc
# from Agents.main.llm import llm
from Information_extraction_8D.Schemas.eightD_schema_json_v2 import EightDCase, EightDSections, D2Section, D4Section,D3Section, D5Section,D6Section,FailureItem
import os, re

def build_eight_d_agent():

    tools = [extract_d2,extract_d4,parse_8d_doc]

    agent = create_agent(
        model= llm,
        tools= tools
    )
    return agent

def build_iteration2_input(iter1_output: dict, min_conf: float = 0.6) -> dict:
    return {
        "signals": [
            {
                "text": s["sentence"],
                "hint": s["signal_type"],
                "confidence": s["confidence"],
                "source": s.get("source"),
            }
            for s in iter1_output.get("selected_sentences", [])
            if s.get("confidence", 0) >= min_conf
        ],
    }




def build_8d_case_from_docx(doc_path: str) -> EightDCase:

    # 1) Parse document sections using the parse_8d_doc tool
    product_name = extract_product(doc_path)
    print("Product name:", product_name)
    parsed = parse_8d_doc.invoke({"doc_path": doc_path})
    sections = parsed["sections"]

    # 2) Extract 8D ID from file name
    base = os.path.basename(doc_path)
    m = re.findall(r"8D\\d+", base)
    d8_id = m[0] if m else base


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
    system_name = output_iter1.get("system_name") or ""
    input_iter2 = build_iteration2_input(output_iter1)
    print("Done!LLM iteration 2:")
    output_iter2 = extract_iteration_2.invoke({"data":input_iter2})

    raw_failures = output_iter2.get("failures", [])
    failures = []
    for f in raw_failures:
        try:
            failures.append(FailureItem(**f))
        except Exception as e:
            print("Skipping invalid failure:", e)
    # 4) Build top-level EightDCase object
    case = EightDCase(
        d8_id=d8_id,
        product_name=product_name,   # add extract_product() later if needed
        system_name=system_name,
        failures= failures,
        sections=EightDSections(
                                D2=d2_section,
                                D3=d3_section,
                                D4=d4_section,
                                D5=d5_section,
                                D6=d6_section,
                            )
    )

    return case