# KG_qualification.py

import os
import re
import json
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

import pdfplumber
from neo4j import GraphDatabase
from chromadb.utils import embedding_functions


# =========================================================
# CONFIG
# =========================================================
QD_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\QD6303220037R03.pdf"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"

BATCH_SIZE = 100

HEADER_CROP = 40
Y_TOLERANCE = 3.0
LEFT_QD_BOUNDARY = 220.0

DEBUG_PRINT_PAGE_LINES = False
DEBUG_SAVE_PARSED_JSON = True
PARSED_JSON_PATH = r"C:\Users\FW\Desktop\qualification_parsed.json"

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_embedding_cache: Dict[str, List[float]] = {}


# =========================================================
# EMBEDDING
# =========================================================
def embed(text: str) -> Optional[List[float]]:
    text = safe_text(text)
    if not text:
        return None
    if text in _embedding_cache:
        return _embedding_cache[text]

    vec = embedder([text])[0]
    vec = [float(x) for x in vec]
    _embedding_cache[text] = vec
    return vec


# =========================================================
# BASIC UTILS
# =========================================================
def normalize_spaces(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_line(text: str) -> str:
    return normalize_spaces(text)


def clean_multiline_text(text: str) -> str:
    if not text:
        return ""
    lines = [clean_line(x) for x in text.splitlines()]
    lines = [x for x in lines if x]
    return "\n".join(lines).strip()


def flatten_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = clean_multiline_text(text)
    if not text:
        return None
    return re.sub(r"\s*\n\s*", " ", text).strip()


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def save_json(data, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_noise_line(text: str) -> bool:
    t = clean_line(text).lower()
    if not t:
        return True

    noise_exact = {
        "company confidential",
        "page",
        "date",
        "document title",
        "document id / release",
        "document id",
        "release",
    }
    if t in noise_exact:
        return True

    if re.fullmatch(r"\d+", t):
        return True

    if t.startswith("document title"):
        return True
    if t.startswith("document id"):
        return True
    if t.startswith("date "):
        return True

    return False


# =========================================================
# PATTERNS
# =========================================================
QD_ID_PATTERN = re.compile(r"\bQD_\d+\b")
TST_ID_PATTERN = re.compile(r"\bTST_\d+\b")
CHO_ID_PATTERN = re.compile(r"\bCHO_\d+\b")
REQ_ID_PATTERN = re.compile(r"\bREQ_\d+\b")
DRQ_ID_PATTERN = re.compile(r"\bDRQ_\d+\*?\b")
REF_PATTERN = re.compile(r"\[\d+\]")

QD_TITLE_PATTERN = re.compile(r"^QD\s*:\s*(.+)$", re.I)
TOC_ENTRY_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\.{2,}\s*(\d+)\s*$")
SECTION_HEADING_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$")

SECTION_LABELS = {"Objectives", "Preconditions"}


# =========================================================
# PDF LOW-LEVEL HELPERS
# =========================================================
def group_words_to_lines(words: List[Dict], y_tolerance: float = Y_TOLERANCE) -> List[List[Dict]]:
    if not words:
        return []

    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines = []
    current_line = [words[0]]
    current_top = words[0]["top"]

    for w in words[1:]:
        if abs(w["top"] - current_top) <= y_tolerance:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
            current_top = w["top"]

    lines.append(current_line)
    return lines


def line_to_info(line_words: List[Dict]) -> Dict:
    line_words = sorted(line_words, key=lambda x: x["x0"])
    text = " ".join(w["text"] for w in line_words).strip()

    return {
        "text": clean_line(text),
        "x0": min(w["x0"] for w in line_words),
        "x1": max(w["x1"] for w in line_words),
        "top": min(w["top"] for w in line_words),
        "bottom": max(w["bottom"] for w in line_words),
        "words": line_words,
    }


def words_to_text(words: List[Dict]) -> str:
    if not words:
        return ""
    words = sorted(words, key=lambda x: x["x0"])
    return clean_line(" ".join(w["text"] for w in words))


def words_left_of(words: List[Dict], x: float) -> str:
    selected = [w for w in words if w["x1"] <= x]
    return words_to_text(selected)


def words_in_column(words: List[Dict], x_left: float, x_right: float) -> str:
    selected = []
    for w in words:
        center = (w["x0"] + w["x1"]) / 2.0
        if x_left <= center < x_right:
            selected.append(w)
    return words_to_text(selected)


def debug_print_lines(line_infos: List[Dict]):
    print("\n" + "=" * 120)
    print("[DEBUG] PAGE LINES")
    print("=" * 120)
    for i, line in enumerate(line_infos):
        print(f"{i:03d} | top={line['top']:.1f} bottom={line['bottom']:.1f} | {line['text']}")


# =========================================================
# LABEL PARSERS
# =========================================================
def extract_verified_ids(text: str) -> List[str]:
    ids = (
        CHO_ID_PATTERN.findall(text)
        + REQ_ID_PATTERN.findall(text)
        + DRQ_ID_PATTERN.findall(text)
    )
    return unique_keep_order(ids)


def extract_refs(text: str) -> List[str]:
    return unique_keep_order(REF_PATTERN.findall(text))


def parse_qd_id(text: str) -> Optional[str]:
    m = QD_ID_PATTERN.search(text or "")
    return m.group(0) if m else None


def parse_tst_left_label(text: str) -> Dict:
    tst_id = next(iter(TST_ID_PATTERN.findall(text)), None)

    return {
        "tst_id": tst_id,
        "verified_ids": extract_verified_ids(text),
        "refs": extract_refs(text),
        "raw_label": clean_line(text),
    }


def parse_toc_line(text: str) -> Optional[Dict[str, Any]]:
    match = TOC_ENTRY_PATTERN.match(clean_line(text))
    if not match:
        return None

    number = match.group(1)
    title = clean_line(match.group(2))
    page = int(match.group(3))

    return {
        "number": number,
        "title": title,
        "page": page,
        "level": min(number.count(".") + 1, 4),
        "full_text": clean_line(f"{number} {title}"),
    }


def extract_toc_entries_from_pdf(pdf) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    in_toc = False
    started_collecting = False
    non_match_streak = 0

    for page in pdf.pages[:10]:
        cropped = page.crop((0, HEADER_CROP, page.width, page.height))
        page_text = cropped.extract_text() or ""
        if not page_text:
            continue

        lines = [clean_line(line) for line in page_text.splitlines() if clean_line(line)]
        for line in lines:
            low = line.lower()
            if "table of contents" in low:
                in_toc = True
                non_match_streak = 0
                continue

            if not in_toc:
                continue

            entry = parse_toc_line(line)
            if entry:
                entries[entry["full_text"].lower()] = entry
                started_collecting = True
                non_match_streak = 0
                continue

            if started_collecting:
                non_match_streak += 1
                if re.match(r"^\s*1\s+", line) or non_match_streak >= 8:
                    return entries

    return entries


def get_toc_section_list(toc_entries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(toc_entries.values(), key=lambda item: (item["page"], item["number"]))


def find_toc_entry_for_heading(text: str, toc_entries: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    if not toc_entries:
        return None
    return toc_entries.get(clean_line(text).lower())


def parse_section_heading_info(
    text: str,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    t = clean_line(text)
    if not t:
        return None

    toc_entry = find_toc_entry_for_heading(t, toc_entries)
    if toc_entry is not None:
        return {
            "number": toc_entry["number"],
            "title": toc_entry["title"],
            "level": toc_entry["level"],
            "full_text": toc_entry["full_text"],
        }

    match = SECTION_HEADING_PATTERN.match(t)
    if not match:
        return None

    number = clean_line(match.group(1))
    title = clean_line(match.group(2))
    if not number:
        return None

    return {
        "number": number,
        "title": title,
        "level": min(number.count(".") + 1, 4),
        "full_text": clean_line(f"{number} {title}") if title else number,
    }


def get_section_heading_level(
    text: str,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[int]:
    info = parse_section_heading_info(text, toc_entries=toc_entries)
    return info["level"] if info else None


# =========================================================
# TABLE HEADER DETECTION
# =========================================================
def is_tst_table_header(line_text: str) -> bool:
    t = re.sub(r"\s+", " ", line_text.strip().lower())
    return (
        "action" in t
        and "expected result" in t
        and "observed result" in t
        and "state" in t
    )


def find_table_header_line(line_infos: List[Dict]) -> Optional[int]:
    for i, line in enumerate(line_infos):
        if is_tst_table_header(line["text"]):
            return i
    return None


def find_word_x(words: List[Dict], token: str) -> Optional[float]:
    token_low = token.lower()
    for w in sorted(words, key=lambda x: x["x0"]):
        if w["text"].strip().lower() == token_low:
            return w["x0"]
    return None


def infer_table_columns(header_line: Dict) -> Dict[str, float]:
    words = header_line["words"]

    x_num = find_word_x(words, "#")
    x_action = find_word_x(words, "Action")
    x_expected = find_word_x(words, "Expected")
    x_observed = find_word_x(words, "Observed")
    x_state = find_word_x(words, "State")

    if None in (x_num, x_action, x_expected, x_observed, x_state):
        raise ValueError(
            f"Failed to infer table columns. "
            f"x_num={x_num}, x_action={x_action}, x_expected={x_expected}, "
            f"x_observed={x_observed}, x_state={x_state}"
        )

    return {
        "x_left_label_end": x_num,
        "x_num": x_num,
        "x_action": x_action,
        "x_expected": x_expected,
        "x_observed": x_observed,
        "x_state": x_state,
    }


# =========================================================
# TEXT APPEND HELPERS
# =========================================================
def append_field(obj: Dict, key: str, text: str):
    text = clean_line(text)
    if not text:
        return
    if obj.get(key):
        obj[key] += "\n" + text
    else:
        obj[key] = text


def make_empty_qd_block(
    section_info: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    block = {
        "qd_id": None,
        "verified_ids": [],
        "refs": [],
        "qd_title": "",
        "objectives": "",
        "preconditions": "",
        "raw_lines": [],
        "raw_left_label_lines": [],
        "tests": [],
        "section_tag": safe_text(section_info.get("full_text")) if section_info else "",
        "section_number": safe_text(section_info.get("number")) if section_info else "",
        "section_title": safe_text(section_info.get("title")) if section_info else "",
        "section_level": section_info.get("level") if section_info else None,
        "section_node_id": "",
    }

    if section_info and section_info.get("number") and document_id:
        block["section_node_id"] = get_section_node_semantic_id(document_id, section_info["number"])

    return block


def apply_section_info_to_block(
    block: Dict[str, Any],
    section_info: Optional[Dict[str, Any]],
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not section_info:
        return block

    block["section_tag"] = safe_text(section_info.get("full_text"))
    block["section_number"] = safe_text(section_info.get("number"))
    block["section_title"] = safe_text(section_info.get("title"))
    block["section_level"] = section_info.get("level")
    block["section_node_id"] = ""
    if section_info.get("number") and document_id:
        block["section_node_id"] = get_section_node_semantic_id(document_id, section_info["number"])
    return block


def block_to_section_info(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    section_number = safe_text(block.get("section_number"))
    section_title = safe_text(block.get("section_title"))
    section_tag = safe_text(block.get("section_tag"))
    section_level = block.get("section_level")

    if not section_number and not section_tag:
        return None

    return {
        "number": section_number,
        "title": section_title,
        "level": section_level,
        "full_text": section_tag or clean_line(f"{section_number} {section_title}").strip(),
    }


# =========================================================
# QD BLOCK PARSER
# =========================================================
def parse_qd_blocks(
    qd_lines: List[Dict],
    left_boundary: float = LEFT_QD_BOUNDARY,
    section_info: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict]:
    blocks: List[Dict[str, Any]] = []
    result = make_empty_qd_block(section_info=section_info, document_id=document_id)
    current_section_info = section_info
    current_content_section = None
    pending_qd_title = ""

    def finalize_current_block(block: Dict[str, Any]):
        block["verified_ids"] = unique_keep_order(block["verified_ids"])
        block["refs"] = unique_keep_order(block["refs"])
        block["objectives"] = clean_multiline_text(block["objectives"])
        block["preconditions"] = clean_multiline_text(block["preconditions"])
        blocks.append(block)

    for line in qd_lines:
        full_text = clean_line(line["text"])
        if not full_text or is_noise_line(full_text):
            continue

        line_section = parse_section_heading_info(full_text, toc_entries=toc_entries)
        if line_section is not None:
            current_section_info = line_section
            has_block_content = bool(
                result["qd_id"]
                or result["raw_lines"]
                or result["verified_ids"]
                or result["refs"]
                or result["qd_title"]
                or result["objectives"]
                or result["preconditions"]
            )
            if has_block_content:
                finalize_current_block(result)
                result = make_empty_qd_block(section_info=line_section, document_id=document_id)
                current_content_section = None
                pending_qd_title = ""
            else:
                apply_section_info_to_block(result, line_section, document_id=document_id)
            current_content_section = None
            continue

        left_words = [w for w in line["words"] if w["x0"] < left_boundary]
        right_words = [w for w in line["words"] if w["x0"] >= left_boundary]

        left_text = words_to_text(left_words)
        right_text = words_to_text(right_words)

        qd_id = parse_qd_id(left_text)
        if not qd_id and not result["qd_id"]:
            qd_id = parse_qd_id(full_text)

        if qd_id and qd_id != result["qd_id"]:
            if result["qd_id"]:
                finalize_current_block(result)
                result = make_empty_qd_block(section_info=current_section_info, document_id=document_id)
                current_content_section = None
            result["qd_id"] = qd_id
            if pending_qd_title and not result["qd_title"]:
                result["qd_title"] = pending_qd_title
                pending_qd_title = ""

        if not result["qd_id"] and qd_id:
            result["qd_id"] = qd_id

        result["raw_lines"].append(full_text)

        if left_text:
            result["raw_left_label_lines"].append(left_text)
            result["verified_ids"].extend(extract_verified_ids(left_text))
            result["refs"].extend(extract_refs(left_text))

        qd_inline_text = ""
        active_qd_id = result["qd_id"] or qd_id
        if active_qd_id:
            qd_inline_text = ""
            pattern = rf"^{re.escape(active_qd_id)}\s*(.*)$"
            m_inline = re.match(pattern, left_text or full_text, re.I)
            if m_inline:
                qd_inline_text = clean_line(m_inline.group(1))

        m_title = QD_TITLE_PATTERN.match(right_text) or QD_TITLE_PATTERN.match(full_text)
        if m_title:
            new_title = clean_line(m_title.group(1))
            if result["qd_id"] and result["qd_title"] and result["qd_title"] != new_title:
                finalize_current_block(result)
                result = make_empty_qd_block(section_info=current_section_info, document_id=document_id)
                current_content_section = None
                result["qd_title"] = new_title
                pending_qd_title = new_title
            else:
                pending_qd_title = new_title
                if result["qd_id"] and not result["qd_title"]:
                    result["qd_title"] = pending_qd_title
                    pending_qd_title = ""
            continue

        if right_text in SECTION_LABELS:
            current_content_section = right_text.lower()
            continue

        if full_text in SECTION_LABELS:
            current_content_section = full_text.lower()
            continue

        content = full_text
        if qd_inline_text:
            content = qd_inline_text

        if current_content_section == "objectives":
            append_field(result, "objectives", content)
        elif current_content_section == "preconditions":
            append_field(result, "preconditions", content)
        elif qd_inline_text and not current_content_section:
            append_field(result, "objectives", content)

        if pending_qd_title and result["qd_id"] and not result["qd_title"]:
            result["qd_title"] = pending_qd_title

    if result["qd_id"]:
        result["verified_ids"] = unique_keep_order(result["verified_ids"])
        result["refs"] = unique_keep_order(result["refs"])
        result["objectives"] = clean_multiline_text(result["objectives"])
        result["preconditions"] = clean_multiline_text(result["preconditions"])
        if pending_qd_title and not result["qd_title"]:
            result["qd_title"] = pending_qd_title
        blocks.append(result)

    return blocks


def parse_qd_block(
    qd_lines: List[Dict],
    left_boundary: float = LEFT_QD_BOUNDARY,
    section_info: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict:
    blocks = parse_qd_blocks(
        qd_lines,
        left_boundary=left_boundary,
        section_info=section_info,
        document_id=document_id,
        toc_entries=toc_entries,
    )
    return blocks[0] if blocks else make_empty_qd_block(section_info=section_info, document_id=document_id)


# =========================================================
# TST ROW GROUPING
# =========================================================
def split_table_lines_into_tst_groups(table_lines: List[Dict], cols: Dict[str, float]) -> List[List[Dict]]:
    groups: List[List[Dict]] = []
    current_group: List[Dict] = []

    for line in table_lines:
        words = line["words"]
        left_label = words_left_of(words, cols["x_left_label_end"])
        left_info = parse_tst_left_label(left_label)

        if left_info["tst_id"]:
            if current_group:
                groups.append(current_group)
            current_group = [line]
        else:
            if current_group:
                current_group.append(line)
            else:
                continue

    if current_group:
        groups.append(current_group)

    return groups


# =========================================================
# TST GROUP PARSER
# =========================================================
def parse_one_tst_group(group_lines: List[Dict], cols: Dict[str, float], page_width: float) -> Dict:
    first_line = group_lines[0]
    left_label_first = words_left_of(first_line["words"], cols["x_left_label_end"])
    left_info = parse_tst_left_label(left_label_first)

    row = {
        "tst_id": left_info["tst_id"],
        "verified_ids": left_info["verified_ids"],
        "refs": left_info["refs"],
        "raw_left_label": left_info["raw_label"],
        "step_no": "",
        "action": "",
        "expected_result": "",
        "observed_result": None,
        "state": "",
        "raw_group_lines": [],
    }

    observed_buffer = ""

    for line in group_lines:
        words = line["words"]
        full_text = clean_line(line["text"])
        if not full_text:
            continue

        row["raw_group_lines"].append(full_text)

        num_text = words_in_column(words, cols["x_num"], cols["x_action"])
        action_text = words_in_column(words, cols["x_action"], cols["x_expected"])
        expected_text = words_in_column(words, cols["x_expected"], cols["x_observed"])
        observed_text = words_in_column(words, cols["x_observed"], cols["x_state"])
        state_text = words_in_column(words, cols["x_state"], page_width + 5)

        if num_text and not row["step_no"]:
            row["step_no"] = clean_line(num_text)

        append_field(row, "action", action_text)
        append_field(row, "expected_result", expected_text)
        append_field(row, "state", state_text)

        observed_text = clean_line(observed_text)
        if observed_text:
            if observed_buffer:
                observed_buffer += "\n" + observed_text
            else:
                observed_buffer = observed_text

    row["action"] = clean_multiline_text(row["action"])
    row["expected_result"] = clean_multiline_text(row["expected_result"])
    row["state"] = clean_multiline_text(row["state"])

    observed_buffer = clean_multiline_text(observed_buffer)
    row["observed_result"] = observed_buffer if observed_buffer else None

    return row


def parse_tst_rows(table_lines: List[Dict], cols: Dict[str, float], page_width: float) -> List[Dict]:
    groups = split_table_lines_into_tst_groups(table_lines, cols)
    rows = []

    for group in groups:
        parsed = parse_one_tst_group(group, cols, page_width)
        if parsed.get("tst_id"):
            rows.append(parsed)

    return rows


def split_table_lines_for_tst_and_qd(
    table_lines: List[Dict],
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    tst_lines: List[Dict] = []
    qd_tail_lines: List[Dict] = []
    tail_started = False

    for line in table_lines:
        full_text = clean_line(line["text"])
        if not tail_started and parse_section_heading_info(full_text, toc_entries=toc_entries) is not None:
            tail_started = True

        if tail_started:
            qd_tail_lines.append(line)
        else:
            tst_lines.append(line)

    return tst_lines, qd_tail_lines


def attach_tests_to_qd_blocks(qd_blocks: List[Dict[str, Any]], tst_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not qd_blocks:
        return []

    for block in qd_blocks:
        block["tests"] = block.get("tests", [])

    qd_blocks[-1]["tests"].extend(tst_rows)
    return qd_blocks


# =========================================================
# PAGE PARSER
# =========================================================
def detect_page_section_info(line_infos: List[Dict], toc_entries: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    hits = collect_page_section_hits(line_infos, toc_entries=toc_entries)
    return hits[-1] if hits else None


def collect_page_section_hits(
    line_infos: List[Dict],
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    seen = set()

    for line in line_infos:
        info = parse_section_heading_info(line.get("text", ""), toc_entries=toc_entries)
        if info is None:
            continue
        key = clean_line(info["full_text"]).lower()
        if key in seen:
            continue
        seen.add(key)
        hits.append(info)

    return hits


def parse_one_page(
    page,
    page_no: int,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
    debug: bool = False,
) -> Optional[Dict]:
    cropped = page.crop((0, HEADER_CROP, page.width, page.height))

    words = cropped.extract_words(
        use_text_flow=False,
        keep_blank_chars=False,
        extra_attrs=["fontname", "size"],
    )

    if not words:
        return None

    lines = group_words_to_lines(words, y_tolerance=Y_TOLERANCE)
    line_infos = [line_to_info(line_words) for line_words in lines]

    if debug:
        debug_print_lines(line_infos)

    header_idx = find_table_header_line(line_infos)
    if header_idx is None:
        return None

    header_line = line_infos[header_idx]
    cols = infer_table_columns(header_line)
    document_id = make_qualification_document_id(QD_PATH)

    qd_lines = line_infos[:header_idx]
    table_lines = line_infos[header_idx + 1:]
    section_hits = collect_page_section_hits(line_infos, toc_entries=toc_entries)
    section_info = section_hits[-1] if section_hits else (detect_page_section_info(qd_lines, toc_entries=toc_entries) or detect_page_section_info(line_infos, toc_entries=toc_entries))
    section_source = "ocr" if section_hits else ("inherited" if current_section_info else "none")

    tst_lines, qd_tail_lines = split_table_lines_for_tst_and_qd(table_lines, toc_entries=toc_entries)

    qd_blocks = parse_qd_blocks(
        qd_lines,
        left_boundary=LEFT_QD_BOUNDARY,
        section_info=section_info,
        document_id=document_id,
        toc_entries=toc_entries,
    )
    tst_rows = parse_tst_rows(tst_lines, cols, page.width)
    qd_blocks = attach_tests_to_qd_blocks(qd_blocks, tst_rows)

    qd_tail_blocks: List[Dict[str, Any]] = []
    qd_tail_tests: List[Dict[str, Any]] = []
    if qd_tail_lines:
        tail_header_idx = find_table_header_line(qd_tail_lines)
        if tail_header_idx is not None:
            tail_qd_lines = qd_tail_lines[:tail_header_idx]
            tail_table_lines = qd_tail_lines[tail_header_idx + 1:]
            qd_tail_blocks = parse_qd_blocks(
                tail_qd_lines,
                left_boundary=LEFT_QD_BOUNDARY,
                section_info=section_info,
                document_id=document_id,
                toc_entries=toc_entries,
            )
            qd_tail_tests = parse_tst_rows(tail_table_lines, cols, page.width)
            qd_tail_blocks = attach_tests_to_qd_blocks(qd_tail_blocks, qd_tail_tests)
        else:
            qd_tail_blocks = parse_qd_blocks(
                qd_tail_lines,
                left_boundary=LEFT_QD_BOUNDARY,
                section_info=section_info,
                document_id=document_id,
                toc_entries=toc_entries,
            )

    if qd_tail_blocks:
        qd_blocks.extend(qd_tail_blocks)

    qd_data = qd_blocks[-1] if qd_blocks else make_empty_qd_block(section_info=section_info, document_id=document_id)
    tst_rows = tst_rows + qd_tail_tests
    effective_section_info = section_info
    if qd_tail_blocks:
        effective_section_info = block_to_section_info(qd_tail_blocks[-1]) or effective_section_info
    elif qd_blocks:
        effective_section_info = block_to_section_info(qd_blocks[-1]) or effective_section_info

    if not qd_data["qd_id"] and not tst_rows:
        return None

    return {
        "page": page_no,
        "qd": qd_blocks if len(qd_blocks) > 1 else qd_data,
        "table_columns": cols,
        "tests": [],
        "section_info": effective_section_info,
        "section_source": section_source,
        "section_hits": section_hits,
    }


# =========================================================
# DOCUMENT PARSER
# =========================================================
def print_identified_sections(
    toc_section_list: List[Dict[str, Any]],
    pages: List[Dict[str, Any]],
    page_tracks: Optional[List[Dict[str, Any]]] = None,
):
    print("\n" + "=" * 120)
    print("[IDENTIFIED SECTIONS]")
    print("=" * 120)

    if not toc_section_list:
        print("[INFO] No TOC sections found")
    else:
        for sec in toc_section_list:
            print(f"- {sec['number']} {sec['title']} (page {sec['page']}, level {sec['level']})")

    matched_sections = []
    seen_matched = set()
    if page_tracks:
        for track in page_tracks:
            for section_info in track.get("section_hits", []):
                if section_info.get("full_text"):
                    key = clean_line(section_info["full_text"]).lower()
                    if key not in seen_matched:
                        seen_matched.add(key)
                        matched_sections.append(section_info)

    print("\n" + "=" * 120)
    print("[OCR MATCHED SECTIONS]")
    print("=" * 120)
    if matched_sections:
        for sec in matched_sections:
            print(f"- {sec.get('full_text')} (page {sec.get('page', '?')}, level {sec.get('level', '?')})")
    else:
        print("[INFO] No OCR section matches found")

    toc_seen = {clean_line(sec.get("full_text", "")).lower() for sec in matched_sections if sec.get("full_text")}
    print("\n" + "=" * 120)
    print("[TOC SECTIONS NOT SEEN BY OCR]")
    print("=" * 120)
    unseen = [
        sec for sec in toc_section_list
        if clean_line(sec.get("full_text", "")).lower() not in toc_seen and safe_text(sec.get("number")) != "1"
    ]
    if unseen:
        for sec in unseen:
            print(f"- {sec['number']} {sec['title']} (page {sec['page']}, level {sec['level']})")
    else:
        print("[INFO] Every TOC section was seen by OCR at least once")

    print("\n" + "=" * 120)
    print("[PAGE SECTION TRACKING]")
    print("=" * 120)
    if page_tracks:
        for track in page_tracks:
            section_info = track.get("section_info") or {}
            section_text = section_info.get("full_text") or "[NO SECTION]"
            source = track.get("section_source") or "unknown"
            hits = track.get("section_hits", [])
            hit_text = ", ".join(sec.get("full_text", "") for sec in hits) if hits else ""
            if hit_text:
                print(f"Page {track['page']}: {section_text} [{source}] | hits: {hit_text}")
            else:
                print(f"Page {track['page']}: {section_text} [{source}]")
    else:
        for page in pages:
            section_info = page.get("section_info") or {}
            section_text = section_info.get("full_text") or "[NO SECTION]"
            print(f"Page {page['page']}: {section_text}")


def extract_qd_tst_from_pdf(file_path: str, debug: bool = False) -> Dict[str, Any]:
    results = []
    page_tracks = []

    with pdfplumber.open(file_path) as pdf:
        toc_entries = extract_toc_entries_from_pdf(pdf)
        toc_section_list = get_toc_section_list(toc_entries)
        current_section_info: Optional[Dict[str, Any]] = None
        for page_idx, page in enumerate(pdf.pages):
            parsed = parse_one_page(page, page_idx + 1, toc_entries=toc_entries, debug=debug)
            if parsed:
                if parsed.get("section_info"):
                    current_section_info = parsed["section_info"]
                elif current_section_info is not None:
                    parsed["section_info"] = current_section_info
                    parsed["section_source"] = "inherited"
                page_tracks.append({
                    "page": parsed["page"],
                    "section_info": parsed.get("section_info"),
                    "section_source": parsed.get("section_source"),
                    "section_hits": parsed.get("section_hits", []),
                })
                results.append(parsed)

    return {
        "toc_sections": toc_section_list,
        "pages": results,
        "page_tracks": page_tracks,
    }


# =========================================================
# AGGREGATION
# =========================================================
def make_qualification_document_id(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def get_section_node_label(level: int) -> str:
    if level <= 1:
        return "Chapter"
    if level == 2:
        return "Section"
    if level == 3:
        return "Subsection"
    return "Subsubsection"


def get_section_relationship_name(level: int) -> str:
    if level <= 1:
        return "HAS_CHAPTER"
    if level == 2:
        return "HAS_SECTION"
    if level == 3:
        return "HAS_SUBSECTION"
    return "HAS_SUBSUBSECTION"


def get_section_node_semantic_id(document_id: str, section_number: str) -> str:
    return f"{document_id}::section::{section_number}"


def make_qdchunk_name(file_name: str, qd_id: str) -> str:
    base_name = os.path.splitext(file_name)[0]
    return f"{base_name}_{qd_id}"


def make_tstchunk_name(file_name: str, tst_id: str) -> str:
    base_name = os.path.splitext(file_name)[0]
    return f"{base_name}_{tst_id}"


def build_qd_text(qd: Dict) -> str:
    parts = []
    if qd.get("qd_title"):
        parts.append(f"QD Title: {qd['qd_title']}")
    if qd.get("objectives"):
        parts.append(f"Objectives: {flatten_text(qd['objectives'])}")
    if qd.get("preconditions"):
        parts.append(f"Preconditions: {flatten_text(qd['preconditions'])}")
    return "\n".join(parts).strip()


def build_tst_text(tst: Dict) -> str:
    parts = []
    if tst.get("action"):
        parts.append(f"Action: {flatten_text(tst['action'])}")
    if tst.get("expected_result"):
        parts.append(f"Expected Result: {flatten_text(tst['expected_result'])}")
    if tst.get("observed_result"):
        parts.append(f"Observed Result: {flatten_text(tst['observed_result'])}")
    return "\n".join(parts).strip()


def aggregate_qd_results(parsed_data: Any, file_path: str) -> Dict[str, Any]:
    file_name = os.path.basename(file_path)
    document_id = make_qualification_document_id(file_path)

    if isinstance(parsed_data, dict):
        parsed_pages = parsed_data.get("pages", [])
        toc_sections = parsed_data.get("toc_sections", [])
    else:
        parsed_pages = parsed_data
        toc_sections = []

    qd_map: Dict[str, Dict[str, Any]] = {}
    tst_rows: List[Dict[str, Any]] = []

    global_tst_seq = 1

    def iter_qd_entries_from_page(page: Dict[str, Any]) -> List[Dict[str, Any]]:
        qd_value = page.get("qd")
        if isinstance(qd_value, list):
            return qd_value
        if isinstance(qd_value, dict):
            return [qd_value]
        return []

    def iter_tests_from_qd_entries(qd_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for qd in qd_entries:
            tests = qd.get("tests", [])
            if isinstance(tests, list):
                rows.extend([{"qd": qd, "test": tst} for tst in tests if isinstance(tst, dict)])
        return rows

    for page in parsed_pages:
        page_no = page["page"]
        qd_entries = iter_qd_entries_from_page(page)
        if not qd_entries:
            continue

        for qd in qd_entries:
            qd_id = safe_text(qd.get("qd_id"))
            if not qd_id:
                continue

            if qd_id not in qd_map:
                qd_map[qd_id] = {
                    "name": make_qdchunk_name(file_name, qd_id),
                    "qd_id": qd_id,
                    "verified_ids": [],
                    "refs": [],
                    "qd_title": "",
                    "objectives": "",
                    "preconditions": "",
                    "pages": [],
                    "tests": [],
                    "section_tag": "",
                    "section_level": None,
                    "section_node_id": None,
                    "type": "QD",
                }

            target = qd_map[qd_id]
            target["verified_ids"].extend(qd.get("verified_ids", []))
            target["refs"].extend(qd.get("refs", []))

            if qd.get("qd_title") and not target["qd_title"]:
                target["qd_title"] = qd["qd_title"]

            if qd.get("objectives"):
                target["objectives"] = "\n".join(
                    x for x in [target["objectives"], qd["objectives"]] if x
                )

            if qd.get("preconditions"):
                target["preconditions"] = "\n".join(
                    x for x in [target["preconditions"], qd["preconditions"]] if x
                )

            if page_no not in target["pages"]:
                target["pages"].append(page_no)

            section_tag = safe_text(qd.get("section_tag"))
            section_level = qd.get("section_level")
            section_node_id = safe_text(qd.get("section_node_id"))
            section_number = safe_text(qd.get("section_number"))

            if not section_tag:
                section_info = page.get("section_info") or {}
                section_tag = safe_text(section_info.get("full_text"))
                section_level = section_info.get("level")
                section_number = safe_text(section_info.get("number"))
                if section_tag and not section_node_id and section_number:
                    section_node_id = get_section_node_semantic_id(document_id, section_number)

            if section_tag and not target.get("section_tag"):
                target["section_tag"] = section_tag
                target["section_level"] = section_level
                target["section_node_id"] = section_node_id

            for item in iter_tests_from_qd_entries([qd]):
                tst = item["test"]
                tst_id = safe_text(tst.get("tst_id"))
                if not tst_id:
                    continue

                row = {
                    "name": make_tstchunk_name(file_name, tst_id),
                    "tst_id": tst_id,
                    "qd_chunk_name": qd_map[qd_id]["name"] if qd_id in qd_map else "",
                    "verified_ids": unique_keep_order(tst.get("verified_ids", [])),
                    "refs": unique_keep_order(tst.get("refs", [])),
                    "step_no": safe_text(tst.get("step_no")),
                    "action": flatten_text(tst.get("action")),
                    "expected_result": flatten_text(tst.get("expected_result")),
                    "observed_result": flatten_text(tst.get("observed_result")),
                    "state": flatten_text(tst.get("state")),
                    "pages": [page_no],
                    "section_tag": section_tag,
                    "section_level": section_level,
                    "section_node_id": section_node_id,
                    "type": "TST",
                }

                row["text"] = build_tst_text(row)
                if row["text"]:
                    row["embedding"] = embed(row["text"])
                else:
                    row["embedding"] = None

                tst_rows.append(row)
                target["tests"].append(row)
                global_tst_seq += 1

    qd_rows = []
    for qd_id, row in qd_map.items():
        row["verified_ids"] = unique_keep_order(row["verified_ids"])
        row["refs"] = unique_keep_order(row["refs"])
        row["objectives"] = clean_multiline_text(row["objectives"])
        row["preconditions"] = clean_multiline_text(row["preconditions"])
        row["text"] = build_qd_text(row)
        row["embedding"] = embed(row["text"]) if row["text"] else None
        row["tests"] = row.get("tests", [])
        qd_rows.append(row)

    return {
        "qd_rows": qd_rows,
        "tst_rows": tst_rows,
        "toc_sections": toc_sections,
    }


# =========================================================
# KG BUILDER
# =========================================================
class QualificationKGBuilder:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def setup_schema(self):
        queries = [
            """
            CREATE CONSTRAINT qualification_document_semantic_id_unique IF NOT EXISTS
            FOR (n:QualificationDocumentation)
            REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT qdchunk_name_unique IF NOT EXISTS
            FOR (n:QDChunk)
            REQUIRE n.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT tstchunk_name_unique IF NOT EXISTS
            FOR (n:TSTChunk)
            REQUIRE n.name IS UNIQUE
            """,
        ]

        with self.driver.session(database=self.database) as session:
            for q in queries:
                session.run(q)

        print("[INFO] Neo4j qualification schema ready")

    def upsert_document_node(
        self,
        document_id: str,
        file_path: str,
        section_tags: Optional[List[str]] = None,
    ):
        file_name = os.path.basename(file_path)
        section_tags = unique_keep_order([safe_text(x) for x in (section_tags or []) if safe_text(x)])

        props = {
            "semantic_id": document_id,
            "file_name": file_name,
            "source": file_path,
            "doc_type": "qualification",
            "section_tags": section_tags,
            "section_count": len(section_tags),
        }

        query = """
        MERGE (d:QualificationDocumentation {semantic_id: $document_id})
        SET d = $props
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, document_id=document_id, props=props)

    def purge_document_subgraph(self, document_id: str):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (d:QualificationDocumentation {semantic_id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHAPTER|HAS_SECTION|HAS_SUBSECTION|HAS_SUBSUBSECTION*1..]->(sec)<-[:PART_OF]-(q:QDChunk)<-[:BELONGS_TO]-(t:TSTChunk)
                WITH collect(DISTINCT t) AS tst_nodes
                UNWIND tst_nodes AS t
                WITH t WHERE t IS NOT NULL
                DETACH DELETE t
                """,
                document_id=document_id,
            )

            session.run(
                """
                MATCH (d:QualificationDocumentation {semantic_id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHAPTER|HAS_SECTION|HAS_SUBSECTION|HAS_SUBSUBSECTION*1..]->(sec)<-[:PART_OF]-(q:QDChunk)
                WITH collect(DISTINCT q) AS qd_nodes
                UNWIND qd_nodes AS q
                WITH q WHERE q IS NOT NULL
                DETACH DELETE q
                """,
                document_id=document_id,
            )

            session.run(
                """
                MATCH (d:QualificationDocumentation {semantic_id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHAPTER|HAS_SECTION|HAS_SUBSECTION|HAS_SUBSUBSECTION*1..]->(n)
                WITH collect(DISTINCT n) AS nodes
                UNWIND nodes AS node
                WITH node WHERE node IS NOT NULL
                DETACH DELETE node
                """,
                document_id=document_id,
            )

        print(f"[INFO] Purged old qualification subgraph for document: {document_id}")

    def _collect_section_node_rows(self, document_id: str, toc_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        exact_infos: Dict[str, Dict[str, Any]] = {}

        for sec in toc_sections:
            number = safe_text(sec.get("number"))
            title = safe_text(sec.get("title"))
            if not number:
                continue
            if number == "1":
                continue
            exact_infos[number] = {
                "number": number,
                "title": title,
                "level": min(int(sec.get("level") or (number.count(".") + 1)), 4),
                "full_text": safe_text(sec.get("full_text")) or clean_line(f"{number} {title}"),
            }

        rows: List[Dict[str, Any]] = []
        for number, info in exact_infos.items():
            level = min(int(info["level"]), 4)
            parent_semantic_id = None

            if level > 1:
                parent_number = ".".join(number.split(".")[:-1])
                while parent_number:
                    if parent_number in exact_infos:
                        parent_semantic_id = get_section_node_semantic_id(document_id, parent_number)
                        break
                    if "." not in parent_number:
                        parent_number = ""
                    else:
                        parent_number = ".".join(parent_number.split(".")[:-1])

            rows.append(
                {
                    "semantic_id": get_section_node_semantic_id(document_id, number),
                    "document_id": document_id,
                    "number": number,
                    "title": safe_text(info["title"]) or number,
                    "section_tag": safe_text(info["full_text"]) or number,
                    "hierarchy_level": str(level),
                    "level": level,
                    "parent_semantic_id": parent_semantic_id,
                }
            )

        return rows

    def _collect_section_tags(self, toc_sections: List[Dict[str, Any]]) -> List[str]:
        tags = []
        for sec in toc_sections:
            number = safe_text(sec.get("number"))
            if not number or number == "1":
                continue
            tags.append(safe_text(sec.get("full_text")) or clean_line(f"{number} {safe_text(sec.get('title'))}"))
        return unique_keep_order(tags)

    def import_section_nodes(self, document_id: str, toc_sections: List[Dict[str, Any]], batch_size: int = 100):
        rows = self._collect_section_node_rows(document_id, toc_sections)
        if not rows:
            print("[INFO] No section hierarchy rows to import")
            return

        rows_by_level: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}
        for row in rows:
            rows_by_level[int(row["level"])].append(row)

        with self.driver.session(database=self.database) as session:
            for level in range(1, 5):
                level_rows = rows_by_level[level]
                if not level_rows:
                    continue

                label = get_section_node_label(level)
                create_query = f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{semantic_id: row.semantic_id}})
                SET n = {{
                    semantic_id: row.semantic_id,
                    document_id: row.document_id,
                    number: row.number,
                    title: row.title,
                    section_tag: row.section_tag,
                    hierarchy_level: row.hierarchy_level
                }}
                """

                for i in range(0, len(level_rows), batch_size):
                    batch = level_rows[i:i + batch_size]
                    session.run(create_query, rows=batch)

                if level == 1:
                    rel_query = """
                    UNWIND $rows AS row
                    MATCH (d:QualificationDocumentation {semantic_id: $document_id})
                    MATCH (n:Chapter {semantic_id: row.semantic_id})
                    MERGE (d)-[:HAS_CHAPTER]->(n)
                    """
                else:
                    parent_label = get_section_node_label(level - 1)
                    rel_name = get_section_relationship_name(level)
                    rel_query = f"""
                    UNWIND $rows AS row
                    MATCH (p:{parent_label} {{semantic_id: row.parent_semantic_id}})
                    MATCH (n:{label} {{semantic_id: row.semantic_id}})
                    MERGE (p)-[:{rel_name}]->(n)
                    """

                for i in range(0, len(level_rows), batch_size):
                    batch = level_rows[i:i + batch_size]
                    session.run(rel_query, rows=batch, document_id=document_id)

    def import_qdchunks(self, document_id: str, qd_rows: List[Dict], batch_size: int = 100):
        if not qd_rows:
            print("[INFO] No QDChunk rows to import")
            return

        query = """
        UNWIND $rows AS row
        MERGE (q:QDChunk {name: row.name})
        SET q = {
            name: row.name,
            qd_id: row.qd_id,
            verified_ids: row.verified_ids,
            refs: row.refs,
            qd_title: row.qd_title,
            objectives: row.objectives,
            preconditions: row.preconditions,
            text: row.text,
            embedding: row.embedding,
            pages: row.pages,
            section_tag: row.section_tag,
            section_level: row.section_level,
            section_node_id: row.section_node_id,
            type: row.type
        }
        WITH q, row
        OPTIONAL MATCH (sec {semantic_id: row.section_node_id})
        FOREACH (_ IN CASE WHEN row.section_node_id IS NULL OR sec IS NULL THEN [] ELSE [1] END |
            MERGE (q)-[:PART_OF]->(sec)
        )
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(qd_rows), batch_size):
                batch = qd_rows[i:i + batch_size]
                session.run(query, rows=batch, document_id=document_id)
                print(f"[INFO] Imported QDChunk batch {i + 1} - {i + len(batch)} / {len(qd_rows)}")

    def import_tstchunks(self, tst_rows: List[Dict], batch_size: int = 100):
        if not tst_rows:
            print("[INFO] No TSTChunk rows to import")
            return

        query = """
        UNWIND $rows AS row
        MERGE (t:TSTChunk {name: row.name})
        SET t = {
            name: row.name,
            tst_id: row.tst_id,
            verified_ids: row.verified_ids,
            refs: row.refs,
            step_no: row.step_no,
            action: row.action,
            expected_result: row.expected_result,
            observed_result: row.observed_result,
            state: row.state,
            text: row.text,
            embedding: row.embedding,
            pages: row.pages,
            section_tag: row.section_tag,
            section_level: row.section_level,
            section_node_id: row.section_node_id,
            type: row.type
        }
        WITH t, row
        MATCH (q:QDChunk {name: row.qd_chunk_name})
        MERGE (t)-[:BELONGS_TO]->(q)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(tst_rows), batch_size):
                batch = tst_rows[i:i + batch_size]
                session.run(query, rows=batch)
                print(f"[INFO] Imported TSTChunk batch {i + 1} - {i + len(batch)} / {len(tst_rows)}")

    def import_tst_verified_links(self, tst_rows: List[Dict], batch_size: int = 100):
        rows = []

        for row in tst_rows:
            tst_name = row["name"]
            for verified_id in row.get("verified_ids", []):
                if verified_id:
                    rows.append({
                        "tst_name": tst_name,
                        "verified_id": verified_id,
                    })

        if not rows:
            print("[INFO] No VERIFIED links to import")
            return

        query = """
        UNWIND $rows AS row
        MATCH (t:TSTChunk {name: row.tst_name})
        MATCH (ts:TSChunk:ESWTSChunk {primary_requirement_id: row.verified_id})
        MERGE (t)-[:VERIFIED]->(ts)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch)
                print(f"[INFO] Imported VERIFIED batch {i + 1} - {i + len(batch)} / {len(rows)}")

    def import_qd_verified_links(self, qd_rows: List[Dict], batch_size: int = 100):
        rows = []

        for row in qd_rows:
            qd_name = row["name"]
            for verified_id in row.get("verified_ids", []):
                if verified_id:
                    rows.append({
                        "qd_name": qd_name,
                        "verified_id": verified_id,
                    })

        if not rows:
            print("[INFO] No QD VERIFIED links to import")
            return

        query = """
        UNWIND $rows AS row
        MATCH (q:QDChunk {name: row.qd_name})
        MATCH (ts:TSChunk:ESWTSChunk {primary_requirement_id: row.verified_id})
        MERGE (q)-[:VERIFIED]->(ts)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch)
                print(f"[INFO] Imported QD VERIFIED batch {i + 1} - {i + len(batch)} / {len(rows)}")

    def refresh_qualification_document(self, file_path: str, qd_rows: List[Dict], tst_rows: List[Dict], toc_sections: List[Dict[str, Any]], batch_size: int = 100):
        document_id = make_qualification_document_id(file_path)
        section_tags = self._collect_section_tags(toc_sections)

        self.upsert_document_node(document_id, file_path, section_tags=section_tags)
        self.purge_document_subgraph(document_id)

        self.import_section_nodes(document_id, toc_sections, batch_size=batch_size)
        self.upsert_document_node(document_id, file_path, section_tags=section_tags)
        self.import_qdchunks(document_id, qd_rows, batch_size=batch_size)
        self.import_tstchunks(tst_rows, batch_size=batch_size)
        self.import_qd_verified_links(qd_rows, batch_size=batch_size)
        self.import_tst_verified_links(tst_rows, batch_size=batch_size)
        

        print(f"[INFO] Refreshed qualification document: {document_id}")


# =========================================================
# PREVIEW
# =========================================================
def print_summary(qd_rows: List[Dict], tst_rows: List[Dict]):
    print("\n" + "=" * 120)
    print("[QD SUMMARY]")
    print("=" * 120)

    for q in qd_rows:
        print(f"\nQDChunk: {q['name']}")
        print(f"qd_id: {q['qd_id']}")
        print(f"verified_ids: {q['verified_ids']}")
        print(f"refs: {q['refs']}")
        print(f"qd_title: {q['qd_title']}")
        print(f"objectives: {q['objectives']}")
        print(f"preconditions: {q['preconditions']}")
        print(f"text: {q['text']}")
        print(f"pages: {q['pages']}")
        tests = q.get("tests", [])
        if tests:
            print("tests:")
            for t in tests:
                print(f"  - {t['tst_id']} -> {t['name']}")


def print_section_qd_summary(toc_sections: List[Dict[str, Any]], qd_rows: List[Dict]):
    print("\n" + "=" * 120)
    print("[SECTION -> QD IDS]")
    print("=" * 120)

    if not toc_sections:
        print("[INFO] No section tags found")
        return

    section_order: List[str] = []
    section_meta: Dict[str, Dict[str, Any]] = {}
    for sec in toc_sections:
        section_key = safe_text(sec.get("full_text"))
        if not section_key:
            continue
        section_meta[section_key] = sec
        if section_key not in section_order:
            section_order.append(section_key)

    section_to_qds: Dict[str, List[str]] = defaultdict(list)
    fallback_key = "[NO SECTION]"

    for q in qd_rows:
        section_key = safe_text(q.get("section_tag")) or fallback_key
        qd_id = safe_text(q.get("qd_id"))
        if not qd_id:
            continue
        section_to_qds[section_key].append(qd_id)

    for section_key in section_order:
        qd_ids = unique_keep_order(section_to_qds.get(section_key, []))
        if not qd_ids:
            continue
        sec = section_meta.get(section_key, {})
        page = sec.get("page")
        level = sec.get("level")
        print(f"{section_key} -> page {page} -> level {level}")
        for qd_id in qd_ids:
            print(f"  - {qd_id}")

    extra_keys = [k for k in section_to_qds.keys() if k not in section_meta and k != fallback_key]
    for section_key in extra_keys:
        qd_ids = unique_keep_order(section_to_qds.get(section_key, []))
        if not qd_ids:
            continue
        print(section_key)
        for qd_id in qd_ids:
            print(f"  - {qd_id}")

    if section_to_qds.get(fallback_key):
        print(fallback_key)
        for qd_id in unique_keep_order(section_to_qds.get(fallback_key, [])):
            print(f"  - {qd_id}")


def print_section_summary(toc_sections: List[Dict[str, Any]], page_tracks: List[Dict[str, Any]]):
    print("\n" + "=" * 120)
    print("[SECTION SUMMARY]")
    print("=" * 120)

    if not toc_sections:
        print("[INFO] No section tags found")
        return

    for sec in toc_sections:
        section_tag = safe_text(sec.get("full_text"))
        if not section_tag or safe_text(sec.get("number")) == "1":
            continue
        print(f"{section_tag} -> page {sec.get('page')} -> level {sec.get('level')}")

    if page_tracks:
        print("\n" + "=" * 120)
        print("[PAGE SECTION TRACKING]")
        print("=" * 120)
        for track in page_tracks:
            section_info = track.get("section_info") or {}
            section_text = section_info.get("full_text") or "[NO SECTION]"
            source = track.get("section_source") or "unknown"
            hits = track.get("section_hits", [])
            hit_text = ", ".join(sec.get("full_text", "") for sec in hits) if hits else ""
            if hit_text:
                print(f"Page {track['page']}: {section_text} [{source}] | hits: {hit_text}")
            else:
                print(f"Page {track['page']}: {section_text} [{source}]")


# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.isfile(QD_PATH):
        raise FileNotFoundError(f"Qualification PDF not found: {QD_PATH}")

    parsed_data = extract_qd_tst_from_pdf(QD_PATH, debug=DEBUG_PRINT_PAGE_LINES)

    if DEBUG_SAVE_PARSED_JSON:
        save_json(parsed_data, PARSED_JSON_PATH)
        print(f"[INFO] Parsed JSON saved to: {PARSED_JSON_PATH}")

    aggregated = aggregate_qd_results(parsed_data, QD_PATH)
    qd_rows = aggregated["qd_rows"]
    tst_rows = aggregated["tst_rows"]
    toc_sections = aggregated.get("toc_sections", [])
    print_section_qd_summary(toc_sections, qd_rows)

    kg_builder = QualificationKGBuilder(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    try:
        kg_builder.setup_schema()
        kg_builder.refresh_qualification_document(
            file_path=QD_PATH,
            qd_rows=qd_rows,
            tst_rows=tst_rows,
            toc_sections=toc_sections,
            batch_size=BATCH_SIZE,
        )
        print("[DONE]")
    finally:
        kg_builder.close()


if __name__ == "__main__":
    main()
