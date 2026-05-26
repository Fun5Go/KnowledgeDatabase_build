
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
from datetime import datetime
import unicodedata

from JSON_FMEA_KB.query_fmea import eightD_fmea_search
from pathlib import Path
import json

# Helper functions


def normalize_text(text: str) -> str:
    if not text:
        return ""

    # Normalize unicode (quotes, accents, etc.)
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Convert bullet variants to "-"
    text = re.sub(r"[•●▪▫–—]", "-", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive newlines (keep max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def build_iteration2_input(iter1_output) -> dict:
    signals = []

    for s in iter1_output.selected_sentences:
        signal = {
            "sentence_id": s.sentence_id,
            "text": s.text,
            "status": s.annotations.status,
            "source_section": s.source_section,
            "faithful_score": s.annotations.faithful_score,
            "faithful_type": s.annotations.faithful_type,
        }

        # Only add subject if it exists (typically for D4)
        if getattr(s.annotations, "subject", None):
            signal["subject"] = s.annotations.subject

        signals.append(signal)

    return {"signals": signals}


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
        item.sentence_id = f"{doc_prefix}_{sec}_S{counters[sec]:03d}"

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

#======= Build FMEA example text
def default_maintenance_tag(
    review_status="pending",
    version="V1",
):
    return MaintenaceTag(
        review_status=review_status,
        Version=version,
        last_updated=datetime.utcnow().isoformat(),
        supersedes=None,
    )


def resolve_supporting_entities(id_refs, sentence_index):
    """
    id_refs: [{"sentence_id": "..."}]
    sentence_index: dict[id -> Sentence]
    """
    resolved = []
    for ref in id_refs:
        sid = ref["sentence_id"]
        if sid in sentence_index:
            s = sentence_index[sid]
            resolved.append({
                "sentence_id": s.sentence_id,
                "text": s.text,
                "source_section": s.source_section,
                "annotations": s.annotations
            })
    return resolved


def failures_to_fmea_style_text(failures: list[dict]) -> str:
    blocks = []

    for f in failures:
        lines = []

        element = f.get("Failure element")
        mode = f.get("Failure mode")
        effect = f.get("Failure effect")

        if element:
            lines.append(f"Failure element: {element}")
        if mode:
            lines.append(f"Failure mode: {mode}")
        if effect:
            lines.append(f"Failure effect: {effect}")

        causes = f.get("causes", [])
        if causes:
            if len(causes) == 1:
                cause_text = causes[0].get("failure_cause")
                if cause_text:
                    lines.append(f"Failure cause: {cause_text}")
            else:
                lines.append("Failure cause:")
                for c in causes:
                    ct = c.get("failure_cause")
                    if ct:
                        lines.append(f"- {ct}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)

@traceable(name="8d-extraction-MD")
def build_8d_case_from_json(json_path: str) -> EightDCase:

    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    raw_context = data.get("raw_context", {})
    metadata = data.get("metadata", {})

    # ---------- basic sanity check ----------
    if not raw_context:
        raise ValueError(f"Missing raw_context in {json_path.name}")
    file_name = metadata.get("file_name")
    document_info = DocumentInfo(
        file_name= file_name,
        product_name=metadata.get("product_name"),
        released_date=metadata.get("released_date"),
        product_domain=metadata.get("product_domain"),
        productPnId=metadata.get("productPnId"),
        parent_PN=metadata.get("parent_PN"),
        project_name=metadata.get("project_name"),
        fmea_type=metadata.get("fmea_type"),
    )
    # parsed = parse_8d_doc.invoke({"doc_path": doc_path})
    # sections = parsed["sections"]


    # Initialization
    d2_raw = raw_context.get("D2")
    d3_raw = raw_context.get("D3")
    d4_raw = raw_context.get("D4")

    print("Parsed sections")
    # 3) Loop through parsed sections
    d2_section = D2Section(
        raw_context=d2_raw
    ) if d2_raw else None

    d3_section = D3Section(
        raw_context=d3_raw
    ) if d3_raw else None

    d4_section = D4Section(
        raw_context=d4_raw
    ) if d4_raw else None

    d5_section = D5Section(
        raw_context=raw_context.get("D5")
    ) if raw_context.get("D5") else None

    d6_section = D6Section(
        raw_context=raw_context.get("D6")
    ) if raw_context.get("D6") else None

    print("LLM iteration 1")
    output_iter1 =  extract_iteration_1.invoke({
            "data": {
                "d2_raw": d2_raw or "",
                "d3_raw": d3_raw or "",
                "d4_raw": d4_raw or "",
        }
    })

    # output_iter1 = Iteration1Output(**output_iter1)

    #Add ids to sentences
    output_iter1.selected_sentences = assign_sentence_ids(
    output_iter1.selected_sentences,
    doc_prefix=document_info.file_name
)
    
    # ------ Faithfulness annotation ------
    output_iter1.selected_sentences = annotate_faithfulness_for_sentences(
    output_iter1.selected_sentences,
    d2_raw=d2_raw,
    d3_raw=d3_raw,
    d4_raw=d4_raw,
) 
    # print(output_iter1)
    

    input_iter2 = build_iteration2_input(output_iter1)

    # FMEA similar case search with productPnID
    results,failure_ids = eightD_fmea_search(
    signals=input_iter2["signals"],
    productPnID=document_info.productPnId,
)
    similar_fmea = failure_ids
    examples = failures_to_fmea_style_text(results)
    print("LLM iteration 2")
    output_iter2 = extract_iteration_2.invoke({
        "data": {
            "signals": input_iter2["signals"], # Sentences with annotations
            "examples": examples, # Similar FMEA cases in text format
        }
    })
    # output_iter2 = extract_iteration_2.invoke({"data":input_iter2})

    sentence_index = {s.sentence_id: s for s in output_iter1.selected_sentences}

    #-----Failure entity building------
    system_name = output_iter2.get("system_name") or ""
    print("system name:",system_name)


    failure_dict = copy.deepcopy(output_iter2)

    failure_dict["supporting_entities"] = resolve_supporting_entities(
    failure_dict.get("supporting_entities", []),
    sentence_index)

    failure_dict["failure_ID"] = f"{document_info.file_name}_F1"
    failure_dict.setdefault("failure_level", "sub_system")
    failure_dict.setdefault("root_causes", [])

    failure_dict["maintenance_tag"] = default_maintenance_tag(
    review_status="pending",
    version="V1",)  # MaintenaceTag(version="V1", review_status="pending")

    allowed_disciplines = {"HW", "ESW", "MCH", "Other"}


    #-----Root cause building------
    for c_idx, cause in enumerate(failure_dict["root_causes"]):

        if hasattr(cause, "model_dump"):
            cause = cause.model_dump()

        cause.setdefault("cause_level", "unknown")
        cause.setdefault("discipline_type", "Other")
        cause.setdefault("confidence", "medium")

        if cause["discipline_type"] not in allowed_disciplines:
            cause["discipline_type"] = "Other"

        cause.setdefault("failure_mechanism", None)
        cause.setdefault("supporting_entities", [])
        cause.setdefault("inferred_insight", None)

        if not cause.get("failure_cause"):
            cause["failure_cause"] = "Unknown cause (LLM incomplete)"

        cause["cause_ID"] = f"{file_name}_F1_C{c_idx + 1}"  # Add ID

        # Add surpporting text back
        cause["supporting_entities"] = resolve_supporting_entities(
        cause.get("supporting_entities", []),
        sentence_index)

        cause["maintenance_tag"] = default_maintenance_tag(
        review_status="pending",
        version="V1",) # MaintenaceTag(version="V1", review_status="pending")

        failure_dict["root_causes"][c_idx] = cause

    failure = FailureChain(**failure_dict) # Convert to FailureChain model

    # 4) Build top-level EightDCase object
    case = EightDCase(
        documents=[document_info],
        maintenance_tag=MaintenaceTag(
            review_status= "pending",
            Version = "V1"
        ),   # add extract_product() later if needed
        system_name=system_name,
        failure= failure,
        fmea_connection = similar_fmea,
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


# def sentence_search_test():
#     output_iter1 = {
#   "selected_sentences": [
#     {
#       "sentence_id": "8D ECO bridge_D2_S001",
#       "text": "When the motor bridge (3900-0005-0023) used on the APTM 300W (6298-1900-0503) went obsolete, this was communicated to the customer later than desired.",
#       "source_section": "D2",
#       "annotations": {
#         "status": "support",
#         "subject": "",
#         "faithful_score": 100,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D2_S002",
#       "text": "This resulted in preventable worry and potential supply issues.",
#       "source_section": "D2",
#       "annotations": {
#         "status": "support",
#         "subject": "",
#         "faithful_score": 100,
#         "faithful_type": "exact"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D3_S001",
#       "text": "Communication has started with the customer.",
#       "source_section": "D3",
#       "annotations": {
#         "status": "support",
#         "subject": "",
#         "faithful_score": 100,
#         "faithful_type": "exact"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D3_S002",
#       "text": "A selection of solutions has been presented, both solutions with a shorter lead time for a temporary solution and structured solutions which should be more future proof.",
#       "source_section": "D3",
#       "annotations": {
#         "status": "support",
#         "subject": "",
#         "faithful_score": 92,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D3_S003",
#       "text": "Since this is a UL rated product, the final solution also needs to be UL certified.",
#       "source_section": "D3",
#       "annotations": {
#         "status": "support",
#         "subject": "",
#         "faithful_score": 100,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D3_S004",
#       "text": "Since UL certification can take several months to complete, parallel paths are suggested which are only for markets other than the US market, where UL certification is not necessary.",
#       "source_section": "D3",
#       "annotations": {
#         "status": "support",
#         "subject": "",
#         "faithful_score": 100,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S001",
#       "text": "02-23: Shortage due to allocation noted.",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 93,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S002",
#       "text": "10-03-23: No significant free stock quantities of alternatives, no testing initiated.",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 87,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S003",
#       "text": "Original alternative options were FSB50660SFS (also end-of-life), NFA50460R47 (1-on-1, but no UL E number yet), IM241-M6S1J (requires redesign), and IM241-M6S1B (requires redesign).",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 88,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S004",
#       "text": "13-03-23: PCN received LTB for ordering 30-03-23.",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 91,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S005",
#       "text": "28-03-23: LTB placed by AME to cover known demand till 10-23.",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 95,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S006",
#       "text": "Requested UL E number for NFA50460R47 as that was least impactful option.",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 95,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S007",
#       "text": "20-06-23: UL E number for NFA50460R47 received after multiple reminders.",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 93,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S008",
#       "text": "05-07-23: Samples NFA50460R47 requested.",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 88,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S009",
#       "text": "20-07-23: Samples NFA50460R47 confirmed by OnSemi (1-2 week delivery).",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 93,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S010",
#       "text": "Testing NFA50460R47 planned after finalization testing of alternative for obsolete FFD08S60S-F085 (Diode on ATPM 300W).",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 97,
#         "faithful_type": "fuzzy"
#       }
#     },
#     {
#       "sentence_id": "8D ECO bridge_D4_S011",
#       "text": "23-10-23: First results testing NFA50460R47 for review (EMC and thermal testing).",
#       "source_section": "D4",
#       "annotations": {
#         "status": "support",
#         "subject": "motor bridge and alternatives",
#         "faithful_score": 95,
#         "faithful_type": "fuzzy"
#       }
#     }
#   ]
# }
#     output_iter1 = Iteration1Output(**output_iter1)
#     input_iter2 = build_iteration2_input(output_iter1)
#     results = eightD_fmea_search(
#     signals=input_iter2["signals"],
#     productPnID=57154
#     )
#     # print(results)
#     examples = failures_to_fmea_style_text(results)
#     print(examples)

# if __name__ == "__main__":
#     sentence_search_test()