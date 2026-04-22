import os
import re
import json
from typing import List, Dict, Optional, Tuple, Any

import pdfplumber


# =========================================================
# CONFIG
# =========================================================
FILE_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\QD6303220037R03.pdf"
OUTPUT_JSON = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\qd_tst_parsed.json"

HEADER_CROP = 40
Y_TOLERANCE = 3.0
LEFT_QD_BOUNDARY = 220.0


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
# BASIC UTILS
# =========================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

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


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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


def debug_print_lines(line_infos: List[Dict], max_lines: Optional[int] = None):
    total = len(line_infos) if max_lines is None else min(max_lines, len(line_infos))
    print("\n" + "=" * 120)
    print("[DEBUG] PAGE LINES")
    print("=" * 120)
    for i in range(total):
        line = line_infos[i]
        print(f"{i:03d} | top={line['top']:.1f} bottom={line['bottom']:.1f} | {line['text']}")


# =========================================================
# COMMON LABEL PARSERS
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


def strip_qd_id_prefix(text: str, qd_id: str) -> str:
    t = clean_line(text)
    q = clean_line(qd_id)
    if not t or not q:
        return ""
    pattern = rf"^{re.escape(q)}\s*(.*)$"
    m = re.match(pattern, t, re.I)
    if m:
        return clean_line(m.group(1))
    return ""


def parse_tst_left_label(text: str) -> Dict:
    tst_id = next(iter(TST_ID_PATTERN.findall(text)), None)
    verified_ids = extract_verified_ids(text)
    refs = extract_refs(text)

    return {
        "tst_id": tst_id,
        "verified_ids": verified_ids,
        "refs": refs,
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


def match_toc_section_info(
    text: str,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    toc_entry = find_toc_entry_for_heading(text, toc_entries)
    if toc_entry is None:
        return None

    return {
        "number": toc_entry["number"],
        "title": toc_entry["title"],
        "level": toc_entry["level"],
        "full_text": toc_entry["full_text"],
    }


def parse_section_heading_info(
    text: str,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    t = clean_line(text)
    if not t:
        return None

    toc_info = match_toc_section_info(t, toc_entries=toc_entries)
    if toc_info is not None:
        return toc_info

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


def detect_page_section_info(
    line_infos: List[Dict],
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    for line in line_infos:
        info = match_toc_section_info(line.get("text", ""), toc_entries=toc_entries)
        if info is not None:
            return info
    return None


def is_qd_boundary_line(text: str, toc_entries: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
    t = clean_line(text)
    if not t:
        return False
    if QD_TITLE_PATTERN.match(t):
        return True
    return match_toc_section_info(t, toc_entries=toc_entries) is not None


def collect_page_section_hits(
    line_infos: List[Dict],
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    seen = set()
    for line in line_infos:
        info = match_toc_section_info(line.get("text", ""), toc_entries=toc_entries)
        if info is None:
            continue
        key = clean_line(info["full_text"]).lower()
        if key in seen:
            continue
        seen.add(key)
        hits.append(info)
    return hits


def split_table_lines_for_tst_and_qd(
    table_lines: List[Dict],
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    tst_lines: List[Dict] = []
    qd_tail_lines: List[Dict] = []
    tail_started = False

    for line in table_lines:
        full_text = clean_line(line["text"])
        if not tail_started and is_qd_boundary_line(full_text, toc_entries=toc_entries):
            tail_started = True

        if tail_started:
            qd_tail_lines.append(line)
        else:
            tst_lines.append(line)

    return tst_lines, qd_tail_lines


def make_qualification_document_id(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def get_section_node_semantic_id(document_id: str, section_number: str) -> str:
    return f"{document_id}::section::{section_number}"


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
    """
    解析表格上方的 QD 描述区域:
      - 左侧: QD_49 + verified ids + refs
      - 右侧: QD : title / Objectives / Preconditions
    """
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

        line_section = match_toc_section_info(full_text, toc_entries=toc_entries)
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
            qd_inline_text = strip_qd_id_prefix(full_text, active_qd_id)
            if not qd_inline_text:
                qd_inline_text = strip_qd_id_prefix(left_text, active_qd_id)

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
def split_table_lines_into_tst_groups(
    table_lines: List[Dict],
    cols: Dict[str, float],
) -> List[List[Dict]]:
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
def parse_one_tst_group(
    group_lines: List[Dict],
    cols: Dict[str, float],
    page_width: float,
    section_info: Optional[Dict[str, Any]] = None,
) -> Dict:
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
        "observed_result": "",
        "state": "",
        "raw_group_lines": [],
    }

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
        append_field(row, "observed_result", observed_text)
        append_field(row, "state", state_text)

    row["action"] = clean_multiline_text(row["action"])
    row["expected_result"] = clean_multiline_text(row["expected_result"])
    row["observed_result"] = clean_multiline_text(row["observed_result"])
    row["state"] = clean_multiline_text(row["state"])

    return row


def parse_tst_rows(
    table_lines: List[Dict],
    cols: Dict[str, float],
    page_width: float,
    section_info: Optional[Dict[str, Any]] = None,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict]:
    tst_lines, _ = split_table_lines_for_tst_and_qd(table_lines, toc_entries=toc_entries)
    groups = split_table_lines_into_tst_groups(tst_lines, cols)
    rows = []

    for group in groups:
        parsed = parse_one_tst_group(group, cols, page_width, section_info=section_info)
        if parsed.get("tst_id"):
            rows.append(parsed)

    return rows


def attach_tests_to_qd_blocks(qd_blocks: List[Dict[str, Any]], tst_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not qd_blocks:
        return []

    for block in qd_blocks:
        block["tests"] = block.get("tests", [])

    anchor_block = qd_blocks[-1]
    anchor_block["tests"].extend(tst_rows)
    return qd_blocks


# =========================================================
# PAGE PARSER
# =========================================================
def parse_one_page(
    page,
    page_no: int,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
    current_section_info: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
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

    qd_lines = line_infos[:header_idx]
    table_lines = line_infos[header_idx + 1:]
    section_hits = collect_page_section_hits(line_infos, toc_entries=toc_entries)
    ocr_section_info = section_hits[-1] if section_hits else detect_page_section_info(qd_lines, toc_entries=toc_entries) or detect_page_section_info(line_infos, toc_entries=toc_entries)
    page_section_info = ocr_section_info or current_section_info
    section_source = "ocr" if ocr_section_info else ("inherited" if current_section_info else "none")
    tst_lines, qd_tail_lines = split_table_lines_for_tst_and_qd(table_lines, toc_entries=toc_entries)

    qd_blocks = parse_qd_blocks(
        qd_lines,
        left_boundary=LEFT_QD_BOUNDARY,
        section_info=page_section_info,
        document_id=document_id,
        toc_entries=toc_entries,
    )
    tst_rows = parse_tst_rows(tst_lines, cols, page.width, section_info=page_section_info, toc_entries=toc_entries)
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
                section_info=page_section_info,
                document_id=document_id,
                toc_entries=toc_entries,
            )
            qd_tail_tests = parse_tst_rows(
                tail_table_lines,
                cols,
                page.width,
                section_info=page_section_info,
                toc_entries=toc_entries,
            )
            qd_tail_blocks = attach_tests_to_qd_blocks(qd_tail_blocks, qd_tail_tests)
        else:
            qd_tail_blocks = parse_qd_blocks(
                qd_tail_lines,
                left_boundary=LEFT_QD_BOUNDARY,
                section_info=page_section_info,
                document_id=document_id,
                toc_entries=toc_entries,
            )

    if qd_tail_blocks:
        qd_blocks.extend(qd_tail_blocks)

    qd_data = qd_blocks[-1] if qd_blocks else make_empty_qd_block(section_info=page_section_info, document_id=document_id)
    tst_rows = tst_rows + qd_tail_tests

    if not qd_data["qd_id"] and not tst_rows:
        return None

    return {
        "page": page_no,
        "qd": qd_blocks if len(qd_blocks) > 1 else qd_data,
        "table_columns": cols,
        "section_info": page_section_info,
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
            section_info = track.get("section_info") or {}
            if track.get("section_source") == "ocr" and section_info.get("full_text"):
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
            print(f"Page {track['page']}: {section_text} [{source}]")
    else:
        for page in pages:
            section_info = page.get("section_info") or {}
            section_text = section_info.get("full_text") or "[NO SECTION]"
            print(f"Page {page['page']}: {section_text}")


def extract_qd_tst_from_pdf(file_path: str, debug: bool = False) -> Dict[str, Any]:
    results = []
    page_tracks = []
    document_id = make_qualification_document_id(file_path)

    with pdfplumber.open(file_path) as pdf:
        toc_entries = extract_toc_entries_from_pdf(pdf)
        toc_section_list = get_toc_section_list(toc_entries)
        current_section_info: Optional[Dict[str, Any]] = None

        for page_idx, page in enumerate(pdf.pages):
            parsed = parse_one_page(
                page,
                page_idx + 1,
                toc_entries=toc_entries,
                current_section_info=current_section_info,
                document_id=document_id,
                debug=debug,
            )
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
                output_page = dict(parsed)
                output_page.pop("section_info", None)
                output_page.pop("section_source", None)
                output_page.pop("section_hits", None)
                results.append(output_page)

    results_meta = {
        "toc_sections": toc_section_list,
        "pages": results,
        "page_tracks": page_tracks,
    }

    return results_meta


def iter_qd_entries_from_page(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    qd_value = page.get("qd")
    if isinstance(qd_value, list):
        return qd_value
    if isinstance(qd_value, dict):
        return [qd_value]
    return []


# =========================================================
# JSON SAVE
# =========================================================
def save_json(data, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# PRETTY PRINT
# =========================================================
def print_summary(results_meta: Dict[str, Any]):
    pages = results_meta.get("pages", [])
    toc_sections = results_meta.get("toc_sections", [])
    page_tracks = results_meta.get("page_tracks", [])
    print("\n" + "=" * 120)
    print("[SUMMARY]")
    print("=" * 120)

    print("\n" + "=" * 120)
    print("[TOC SECTIONS]")
    print("=" * 120)
    if toc_sections:
        for sec in toc_sections:
            print(f"{sec['number']} {sec['title']} -> page {sec['page']} -> level {sec['level']}")
    else:
        print("[INFO] No TOC sections found")

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
        sec for sec in toc_sections
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


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    results = extract_qd_tst_from_pdf(FILE_PATH, debug=False)
    save_json(results, OUTPUT_JSON)
    print_summary(results)

    print("\nDone.")
    print(f"JSON saved to: {OUTPUT_JSON}")
