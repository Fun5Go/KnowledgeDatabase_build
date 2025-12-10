
from langchain.agents import create_agent
from Agents.tools.section_extractor import extract_d2, extract_d4,parse_8d_doc, extract_failure_d234
from Agents.tools.doc_parser import extract_product
# from tools.doc_parser import parse_8d_doc
from Agents.main.llm import llm
from Agents.Schemas.eightD_schema_json_v2 import EightDCase, EightDSections, D2Section, D4Section,D3Section, D5Section,D6Section,FailureItem
import os, re

def build_eight_d_agent():

    tools = [extract_d2,extract_d4,parse_8d_doc]

    agent = create_agent(
        model= llm,
        tools= tools
    )
    return agent




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
            d2_raw = content
            d2_section = D2Section(raw_context=d2_raw)
        #     print("Extracting D2 section...")
        #     d2_extraction = extract_d2.invoke({"section_text": content})
        #     d2_section = D2Section(**d2_extraction)

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
        #     print("Extracting D4 section...")
        #     d4_raw = extract_d4.invoke({
        #     "data": {
        #         "section_text": content,
        #         "d2_info": d2_extraction
        #     }
        # })
        #     d4_section = D4Section(**d4_raw)

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

    print("Identifying failures from D2 to D4 section")
    # failure_list = extract_failure_d234.invoke({
    #         "data": {
    #             "d2_raw": d2_raw,
    #             "d3_raw": d3_raw,
    #             "d4_raw": d4_raw

    #     }
    # })
    failure_list = {
  "system_name": "Electronic Subassembly",
  "failures": [
    {
      "system_element": "Component X",
      "failure_mode": "Short Circuit",
      "failure_effect": "System Failure",
      "discipline_type": "HW",
      "root_cause": "D2: this is d2 sample text",
      "infer_context": "D3: this is d3 sample text"
    }
  ]
    }
    raw_failures = failure_list.get("failures", [])
    failures = [FailureItem(**f) for f in raw_failures]
    # 4) Build top-level EightDCase object
    case = EightDCase(
        d8_id=d8_id,
        product_name=product_name,   # add extract_product() later if needed
        failures= failures,
        sections=EightDSections(D2=d2_section, D4=d4_section, D5=d5_section, D6=d6_section, D3=d3_section, )
    )

    return case