# KG_qualification.py

import os
import re
import json
from typing import List, Dict, Optional, Any
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
def parse_qd_block(qd_lines: List[Dict], left_boundary: float = LEFT_QD_BOUNDARY) -> Dict:
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
# AGGREGATION
# =========================================================
def make_qualification_document_id(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


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


def aggregate_qd_results(parsed_pages: List[Dict], file_path: str) -> Dict[str, Any]:
    file_name = os.path.basename(file_path)

    qd_map: Dict[str, Dict[str, Any]] = {}
    tst_rows: List[Dict[str, Any]] = []

    global_tst_seq = 1

    for page in parsed_pages:
        page_no = page["page"]
        qd = page["qd"]

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

        for tst in page.get("tests", []):
            tst_id = safe_text(tst.get("tst_id"))
            if not tst_id:
                continue

            row = {
                "name": make_tstchunk_name(file_name, tst_id),
                "tst_id": tst_id,
                "qd_chunk_name": target["name"],
                "verified_ids": unique_keep_order(tst.get("verified_ids", [])),
                "refs": unique_keep_order(tst.get("refs", [])),
                "step_no": safe_text(tst.get("step_no")),
                "action": flatten_text(tst.get("action")),
                "expected_result": flatten_text(tst.get("expected_result")),
                "observed_result": flatten_text(tst.get("observed_result")),
                "state": flatten_text(tst.get("state")),
                "pages": [page_no],
                "type": "TST",
            }

            row["text"] = build_tst_text(row)
            if row["text"]:
                row["embedding"] = embed(row["text"])
            else:
                row["embedding"] = None

            tst_rows.append(row)
            global_tst_seq += 1

    qd_rows = []
    for qd_id, row in qd_map.items():
        row["verified_ids"] = unique_keep_order(row["verified_ids"])
        row["refs"] = unique_keep_order(row["refs"])
        row["objectives"] = clean_multiline_text(row["objectives"])
        row["preconditions"] = clean_multiline_text(row["preconditions"])
        row["text"] = build_qd_text(row)
        row["embedding"] = embed(row["text"]) if row["text"] else None
        qd_rows.append(row)

    return {
        "qd_rows": qd_rows,
        "tst_rows": tst_rows,
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

    def upsert_document_node(self, document_id: str, file_path: str):
        file_name = os.path.basename(file_path)

        props = {
            "semantic_id": document_id,
            "file_name": file_name,
            "source": file_path,
            "doc_type": "qualification",
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
                MATCH (t:TSTChunk)-[:BELONGS_TO]->(q:QDChunk)-[:PART_OF]->(d:QualificationDocumentation {semantic_id: $document_id})
                DETACH DELETE t
                """,
                document_id=document_id,
            )

            session.run(
                """
                MATCH (q:QDChunk)-[:PART_OF]->(d:QualificationDocumentation {semantic_id: $document_id})
                DETACH DELETE q
                """,
                document_id=document_id,
            )

        print(f"[INFO] Purged old qualification subgraph for document: {document_id}")

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
            type: row.type
        }
        WITH q
        MATCH (d:QualificationDocumentation {semantic_id: $document_id})
        MERGE (q)-[:PART_OF]->(d)
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
        MATCH (ts:TSChunk {primary_requirement_id: row.verified_id})
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
        MATCH (ts:TSChunk {primary_requirement_id: row.verified_id})
        MERGE (q)-[:VERIFIED]->(ts)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch)
                print(f"[INFO] Imported QD VERIFIED batch {i + 1} - {i + len(batch)} / {len(rows)}")

    def refresh_qualification_document(self, file_path: str, qd_rows: List[Dict], tst_rows: List[Dict], batch_size: int = 100):
        document_id = make_qualification_document_id(file_path)

        self.upsert_document_node(document_id, file_path)
        self.purge_document_subgraph(document_id)

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

    print("\n" + "=" * 120)
    print("[TST SUMMARY]")
    print("=" * 120)

    for t in tst_rows[:20]:
        print(f"\nTSTChunk: {t['name']}")
        print(f"tst_id: {t['tst_id']}")
        print(f"belongs_to: {t['qd_chunk_name']}")
        print(f"verified_ids: {t['verified_ids']}")
        print(f"refs: {t['refs']}")
        print(f"step_no: {t['step_no']}")
        print(f"action: {t['action']}")
        print(f"expected_result: {t['expected_result']}")
        print(f"observed_result: {t['observed_result']}")
        print(f"state: {t['state']}")
        print(f"text: {t['text']}")
        print(f"pages: {t['pages']}")


# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.isfile(QD_PATH):
        raise FileNotFoundError(f"Qualification PDF not found: {QD_PATH}")

    parsed_pages = extract_qd_tst_from_pdf(QD_PATH, debug=DEBUG_PRINT_PAGE_LINES)

    if DEBUG_SAVE_PARSED_JSON:
        save_json(parsed_pages, PARSED_JSON_PATH)
        print(f"[INFO] Parsed JSON saved to: {PARSED_JSON_PATH}")

    aggregated = aggregate_qd_results(parsed_pages, QD_PATH)
    qd_rows = aggregated["qd_rows"]
    tst_rows = aggregated["tst_rows"]

    print_summary(qd_rows, tst_rows)

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
            batch_size=BATCH_SIZE,
        )
        print("[DONE]")
    finally:
        kg_builder.close()


if __name__ == "__main__":
    main()