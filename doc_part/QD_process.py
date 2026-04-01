import os
import re
import json
from typing import List, Dict, Optional, Tuple

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

SECTION_LABELS = {"Objectives", "Preconditions"}


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


# =========================================================
# QD BLOCK PARSER
# =========================================================
def parse_qd_block(
    qd_lines: List[Dict],
    left_boundary: float = LEFT_QD_BOUNDARY,
) -> Dict:
    """
    解析表格上方的 QD 描述区域:
      - 左侧: QD_49 + verified ids + refs
      - 右侧: QD : title / Objectives / Preconditions
    """
    result = {
        "qd_id": None,
        "verified_ids": [],
        "refs": [],
        "qd_title": "",
        "objectives": "",
        "preconditions": "",
        "raw_lines": [],
        "raw_left_label_lines": [],
    }

    current_section = None

    for line in qd_lines:
        full_text = clean_line(line["text"])
        if not full_text or is_noise_line(full_text):
            continue

        left_words = [w for w in line["words"] if w["x0"] < left_boundary]
        right_words = [w for w in line["words"] if w["x0"] >= left_boundary]

        left_text = words_to_text(left_words)
        right_text = words_to_text(right_words)

        result["raw_lines"].append(full_text)

        if left_text:
            result["raw_left_label_lines"].append(left_text)

            if not result["qd_id"]:
                qd_id = parse_qd_id(left_text)
                if qd_id:
                    result["qd_id"] = qd_id

            result["verified_ids"].extend(extract_verified_ids(left_text))
            result["refs"].extend(extract_refs(left_text))

        if not result["qd_id"]:
            qd_id = parse_qd_id(full_text)
            if qd_id:
                result["qd_id"] = qd_id

        m_title = QD_TITLE_PATTERN.match(right_text) or QD_TITLE_PATTERN.match(full_text)
        if m_title:
            result["qd_title"] = clean_line(m_title.group(1))
            current_section = None
            continue

        if right_text in SECTION_LABELS:
            current_section = right_text.lower()
            continue

        if full_text in SECTION_LABELS:
            current_section = full_text.lower()
            continue

        content = right_text if right_text else full_text

        if current_section == "objectives":
            append_field(result, "objectives", content)
        elif current_section == "preconditions":
            append_field(result, "preconditions", content)

    result["verified_ids"] = unique_keep_order(result["verified_ids"])
    result["refs"] = unique_keep_order(result["refs"])
    result["objectives"] = clean_multiline_text(result["objectives"])
    result["preconditions"] = clean_multiline_text(result["preconditions"])

    return result


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
) -> List[Dict]:
    groups = split_table_lines_into_tst_groups(table_lines, cols)
    rows = []

    for group in groups:
        parsed = parse_one_tst_group(group, cols, page_width)
        if parsed.get("tst_id"):
            rows.append(parsed)

    return rows


# =========================================================
# PAGE PARSER
# =========================================================
def parse_one_page(page, page_no: int, debug: bool = False) -> Optional[Dict]:
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

    qd_data = parse_qd_block(qd_lines, left_boundary=LEFT_QD_BOUNDARY)
    tst_rows = parse_tst_rows(table_lines, cols, page.width)

    if not qd_data["qd_id"] and not tst_rows:
        return None

    return {
        "page": page_no,
        "qd": qd_data,
        "table_columns": cols,
        "tests": tst_rows,
    }


# =========================================================
# DOCUMENT PARSER
# =========================================================
def extract_qd_tst_from_pdf(file_path: str, debug: bool = False) -> List[Dict]:
    results = []

    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            parsed = parse_one_page(page, page_idx + 1, debug=debug)
            if parsed:
                results.append(parsed)

    return results


# =========================================================
# JSON SAVE
# =========================================================
def save_json(data, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# PRETTY PRINT
# =========================================================
def print_summary(results: List[Dict]):
    print("\n" + "=" * 120)
    print("[SUMMARY]")
    print("=" * 120)

    for page in results:
        qd = page["qd"]
        print(f"\n[PAGE {page['page']}]")
        print(f"QD ID: {qd.get('qd_id')}")
        print(f"verified_ids: {qd.get('verified_ids')}")
        print(f"refs: {qd.get('refs')}")
        print(f"QD Title: {qd.get('qd_title')}")
        print(f"Objectives: {qd.get('objectives')}")
        print(f"Preconditions: {qd.get('preconditions')}")
        print(f"Tests: {len(page['tests'])}")

        for i, t in enumerate(page["tests"], start=1):
            print("-" * 100)
            print(f"Test #{i}")
            print(f"tst_id: {t.get('tst_id')}")
            print(f"verified_ids: {t.get('verified_ids')}")
            print(f"refs: {t.get('refs')}")
            print(f"step_no: {t.get('step_no')}")
            print(f"action: {t.get('action')}")
            print(f"expected_result: {t.get('expected_result')}")
            print(f"observed_result: {t.get('observed_result')}")
            print(f"state: {t.get('state')}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    results = extract_qd_tst_from_pdf(FILE_PATH, debug=False)
    save_json(results, OUTPUT_JSON)
    print_summary(results)

    print("\nDone.")
    print(f"JSON saved to: {OUTPUT_JSON}")