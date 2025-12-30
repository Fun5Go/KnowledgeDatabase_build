
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
    # output_iter1 =  extract_iteration_1.invoke({
    #         "data": {
    #             "d2_raw": d2_raw or "",
    #             "d3_raw": d3_raw or "",
    #             "d4_raw": d4_raw or "",
    #     }
    # })
    output_iter1 = {
  "selected_sentences": [
    {
      "text": "The DUT blew up during the voltage dips/interrupt test.",
      "source_section": "D2",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The test that destroyed the DUT was during a custom test (not a test as specified in EN 61000-4-11).",
      "source_section": "D2",
      "annotations": {
        "entity_type": "condition",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The following components were destroyed: Fuse (F100), PFC MOSFET (T101), DC-link filter capacitor (C108).",
      "source_section": "D2",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The following voltage dip test destroyed another DUT.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The destroyed DUT had similar components that were destroyed: Fuse (F100), PFC MOSFET (T101).",
      "source_section": "D4",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed"
      }
    },
    {
      "text": "Only C108 didn’t get destroyed this time.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The gate control circuit has been changed in order to improve EMC.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "investigation",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The gate circuit was reverted back to the previous ATPM 300W design.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "investigation",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The same custom test as mentioned above was performed on the DUT with the reverted components.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "investigation",
        "assertion_level": "observed"
      }
    },
    {
      "text": "This time the DUT didn’t blow up.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The gate signal and DC-link voltage are measured at the moment the DUT will be destroyed during the voltage dips test.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "investigation",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The input current is measured to detect when the overcurrent trip of the PSU is activated.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "investigation",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The current limit of the PSU is set to 3A (fuse will break at 4A).",
      "source_section": "D4",
      "annotations": {
        "entity_type": "condition",
        "assertion_level": "observed"
      }
    },
    {
      "text": "In both scope images it can be seen that the current rises to about 4A for a short time due to the inrush current when a voltage dip occurs.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed"
      }
    },
    {
      "text": "The current clips at about 4A due to the soft overcurrent limit from the PFC chip, which reduces the duty cycle of the PFC MOSFET gate signal.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "investigation",
        "assertion_level": "observed"
      }
    },
    {
      "text": "With only R104 = 220Ω, the gate signal doesn’t return to 0V.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "investigation",
        "assertion_level": "observed"
      }
    },
    {
      "text": "With all the other components assembled, the gate signal does return to approximately 0V.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "investigation",
        "assertion_level": "observed"
      }
    }
  ]
}

    output_iter1 = Iteration1Output(**output_iter1)
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


    # input_iter2 = build_iteration2_input(output_iter1)
    # print("Done!LLM iteration 2:")
    # output_iter2 = extract_iteration_2.invoke({"data":input_iter2})
    output_iter2 = {
  "system_name": "",
  "failure_element": "power supply",
  "failure_mode": "component destruction",
  "failure_effect": "device blown up",
  "failure_level": "sub_system",
  "supporting_entities": [
    {
      "id": "8D6298190081R02_D2_S001",
      "text": "The DUT blew up during the voltage dips/interrupt test.",
      "source_section": "D2",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed",
        "faithful_score": 100,
        "faithful_type": "exact"
      }
    },
    {
      "id": "8D6298190081R02_D2_S003",
      "text": "The following components were destroyed: Fuse (F100), PFC MOSFET (T101), DC-link filter capacitor (C108).",
      "source_section": "D2",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed",
        "faithful_score": 100,
        "faithful_type": "fuzzy"
      }
    },
    {
      "id": "8D6298190081R02_D4_S001",
      "text": "The following voltage dip test destroyed another DUT.",
      "source_section": "D4",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed",
        "faithful_score": 100,
        "faithful_type": "fuzzy"
      }
    },
    {
      "id": "8D6298190081R02_D4_S002",
      "text": "The destroyed DUT had similar components that were destroyed: Fuse (F100), PFC MOSFET (T101).",
      "source_section": "D4",
      "annotations": {
        "entity_type": "symptom",
        "assertion_level": "observed",
        "faithful_score": 100,
        "faithful_type": "fuzzy"
      }
    }
  ],
  "root_causes": [
    {
      "cause_level": "design",
      "failure_cause": "gate control circuit modification",
      "discipline_type": "HW",
      "supporting_entities": [
        {
          "id": "8D6298190081R02_D4_S004",
          "text": "The gate control circuit has been changed in order to improve EMC.",
          "source_section": "D4",
          "annotations": {
            "entity_type": "investigation",
            "assertion_level": "observed",
            "faithful_score": 100,
            "faithful_type": "fuzzy"
          }
        },
        {
          "id": "8D6298190081R02_D4_S005",
          "text": "The gate circuit was reverted back to the previous ATPM 300W design.",
          "source_section": "D4",
          "annotations": {
            "entity_type": "investigation",
            "assertion_level": "observed",
            "faithful_score": 100,
            "faithful_type": "fuzzy"
          }
        },
        {
          "id": "8D6298190081R02_D4_S006",
          "text": "The same custom test as mentioned above was performed on the DUT with the reverted components.",
          "source_section": "D4",
          "annotations": {
            "entity_type": "investigation",
            "assertion_level": "observed",
            "faithful_score": 100,
            "faithful_type": "exact"
          }
        },
        {
          "id": "8D6298190081R02_D4_S007",
          "text": "This time the DUT didn’t blow up.",
          "source_section": "D4",
          "annotations": {
            "entity_type": "symptom",
            "assertion_level": "observed",
            "faithful_score": 100,
            "faithful_type": "exact"
          }
        }
      ],
      "inferred_insight": "Destruction of power supply components occurred after gate control circuit modification; reverting to previous design prevented failure.",
      "confidence": "high"
    }
  ]
}
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