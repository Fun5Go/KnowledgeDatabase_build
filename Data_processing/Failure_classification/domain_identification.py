import os, re
from LLM_TOOLs import parse_8d_doc, domain_type_identification_LLMcall
from langchain.agents import create_agent
import copy
from typing import List, Dict, Any, Optional, Union
from langsmith import traceable, get_current_run_tree
from datetime import datetime
import unicodedata
import json
from pathlib import Path


def _safe_dirname(name: str) -> str:
    """Make a filesystem-safe directory name."""
    if not name:
        return "unknown"
    name = name.strip().lower()
    name = re.sub(r"[^\w\-]+", "_", name)   # replace weird chars with _
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unknown"

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

@traceable(name="8d-domain-identification-test-store")
def domain_identification_from_docx(doc_path: str) ->dict:


    # 2) Extract 8D ID from file name
    file_name = os.path.basename(doc_path) 
    # print("fileID:",base_name)

    run = get_current_run_tree()
    if run:
        run.metadata.update({
            "filename": file_name,
            "stage": "domain_identification"
        })
    parsed = parse_8d_doc.invoke({"doc_path": doc_path})
    sections = parsed["sections"]


    # Initialization
    d2_raw = None
    d3_raw = None
    d4_raw = None
    d5_raw = None
    d6_raw = None

    print("Parsed sections")
    # 3) Loop through parsed sections
    for sec in sections:
        title = normalize_text(sec["title"])
        content = normalize_text(sec["content"])

        if title.startswith("D2"):
            d2_raw = content
        elif title.startswith("D3"):
            d3_raw = content
        elif title.startswith("D4"):
            d4_raw = content
        elif title.startswith("D5"):
            d5_raw = content
        elif title.startswith("D6"):
            d6_raw = content
    print("Start to classify the domain")
    output_label =  domain_type_identification_LLMcall.invoke({
            "data": {
                "d2_raw": d2_raw or "",
                "d3_raw": d3_raw or "",
                "d4_raw": d4_raw or "",
        }
    })

    label_item = {
        "market" :output_label.get("market", "unknown"),
        "FileName": file_name,
        "product_domain": output_label.get("product_domain", "unknown"),
        "fmea_type": output_label.get("fmea_type", "unknown"),
        "inferred_content": output_label.get("inferred_content", {})
    }
    if run:
        run.outputs = {"domain": label_item["product_domain"]}

    raw_context = {
        "D2": d2_raw,
        "D3": d3_raw,
        "D4": d4_raw,
        "D5": d5_raw,
        "D6": d6_raw,
    }

    return label_item,raw_context

def write_domain_labels_store(
    output_path: Union[str, Path],
    store_path: Union[str, Path],
    label_item: Dict,
    raw_context: Dict
):
    """
    Append one domain identification result to a JSON file at output_path.
    Also save raw_context + derived metadata into store_path/<filename>.json
    (filename derived from label_item["FileName"]).
    Match by copiedFileName if already exists.
    """

    output_path = Path(output_path)
    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)

    # ---- 1) 
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            data: List[Dict] = json.load(f)
    else:
        data = []

    # ---- 2) 
    updated = False
    metadata: Dict = {}  # 

    # Get filename
    file_value = label_item.get("FileName", "unknown")
    file_stem = Path(file_value).stem  # 兼容 str / Path

    for item in data:
        if item.get("copiedFileName") == file_value:
            item.update(label_item)
            updated = True

            # Save the motor drives product metadata
            metadata = {
                "file_name": file_stem,
                "released_date": item.get("released"),

                "product_name": item.get("productName"),
                "product_domain": label_item.get("product_domain"),
                "productPnId": item.get("productPnId"),
                "parent_PN": item.get("parentPn"),

                "project_name": item.get("name"),
                "fmea_type": label_item.get("fmea_type"),
            }
            break

    if not updated:
        data.append(label_item)

    # ---- 3) Rewrite output_path（----
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # ---- 4) Write in store_path/<filename>.json：store raw_context + metadata ----
    product_domain = _safe_dirname(label_item.get("product_domain", "unknown"))
    domain_dir = store_path / product_domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    store_payload = {
        "raw_context": raw_context,
        "metadata": metadata,
    }

    store_file = domain_dir / f"{file_stem}.json"
    with store_file.open("w", encoding="utf-8") as f:
        json.dump(store_payload, f, ensure_ascii=False, indent=2)

def batch_process(doc_dir_path, json_output_path, store_path, batch_size: Optional[int] = None):
    docx_files: List[str] = sorted(
        [
            os.path.join(doc_dir_path, f)
            for f in os.listdir(doc_dir_path)
            if f.lower().endswith(".docx")        
            and not f.startswith("~$")        # Exclude locked file
            and not f.startswith(".")         # 
        ],
        key=lambda x: os.path.basename(x)
    )
    if not docx_files:
        print(" No .docx files found.")
        return

    print(f"Found {len(docx_files)} docx files.")
    for i, doc_path in enumerate(docx_files[400:]):
        try:
            print(f"Processing {i+1}/{len(docx_files)}: {os.path.basename(doc_path)}")
            label_item,raw_context = domain_identification_from_docx(doc_path)
            write_domain_labels_store(json_output_path, store_path,label_item,raw_context)
            print(f"✔ Success: {os.path.basename(doc_path)}")
        except Exception as e:
            print(f"✖ Failed: {os.path.basename(doc_path)}")
            print(f"  Reason: {e}")

if __name__ == "__main__":
    DOC_DIR_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\RAW\8D_ALL")
    JSON_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\Orion_list\8D\8D_with_filename.json")
    STORE_PATH = Path(r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_raw_meta")
    batch_process(DOC_DIR_PATH,JSON_PATH,STORE_PATH)
