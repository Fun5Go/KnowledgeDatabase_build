# tools/doc_parser.py

from docx import Document
import re
from Agents.Schemas.eightD_schema_json import EightDCase
import json
from pydantic import ValidationError

def split_sections(doc_path):
    doc = Document(doc_path)
    sections = []
    current_title = None
    current_text = []

    for para in doc.paragraphs:
        style = para.style.name

        if style == "Heading 2":
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

    return sections


def extract_product(doc_path):
    doc = Document(doc_path)
    for table in doc.tables:
        for row in table.rows:
            if len(row.cells) < 2: continue
            key = row.cells[0].text.lower()
            if "product" in key:
                return row.cells[1].text.strip()
    return None


def find_8d_id(filename):
    m = re.findall(r"8D\d+", filename)
    return m[0] if m else filename




def safe_json(resp: str):
    """Extract JSON from LLM response."""
    text = resp.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    start = text.find("{")
    end = text.rfind("}")
    return json.loads(text[start:end+1])

def validate_eight_d(json_obj):
    try:
        parsed = EightDCase.model_validate(json_obj)
        return parsed
    except ValidationError as e:
        return {"validation_error": str(e)}
