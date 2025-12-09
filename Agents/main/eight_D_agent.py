
from langchain.agents import create_agent
from Agents.tools.section_extractor import extract_d2, extract_d4,parse_8d_doc
from Agents.tools.doc_parser import extract_product
# from tools.doc_parser import parse_8d_doc
from Agents.main.llm import llm
from Agents.Schemas.eightD_schema_json import EightDCase, EightDSections, D2Section, D4Section, FailureItem
import os, re

def build_eight_d_agent():

    tools = [extract_d2,extract_d4,parse_8d_doc]

    agent = create_agent(
        model= llm,
        tools= tools
    )
    return agent


    return context_block


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

    d2_section = None
    d4_section = None

    # 3) Loop through parsed sections
    for sec in sections:
        title = sec["title"]
        content = sec["content"]

        # --------------------
        # Extract D2 section
        # --------------------
        if title.startswith("D2"):
            print("Extracting D2 section...")
            d2_raw = extract_d2.invoke({"section_text": content})
            d2_section = D2Section(**d2_raw)


        # --------------------
        # Extract D4 section
        # --------------------
        if title.startswith("D4"):
            print("Extracting D4 section...")
            d4_raw = extract_d4.invoke({
            "data": {
                "section_text": content,
                "d2_info": d2_raw
            }
        })
            d4_section = D4Section(**d4_raw)

    # 4) Build top-level EightDCase object
    case = EightDCase(
        d8_id=d8_id,
        product_name=product_name,   # add extract_product() later if needed
        sections=EightDSections(D2=d2_section, D4=d4_section)
    )

    return case