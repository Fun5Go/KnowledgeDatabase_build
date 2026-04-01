import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

import pdfplumber
from docx import Document as DocxDocument
from dotenv import load_dotenv
from neo4j import GraphDatabase
from chromadb.utils import embedding_functions


# =========================================================
# CONFIG
# =========================================================
FS_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\FS6303220015R11.pdf"
TS_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\TS6303220021R05.pdf"

# ---- PDF layout settings ----
HEADER_CROP = 90
LEFT_ID_THRESHOLD = 160
PREFIX_MERGE_MAX_GAP = 6.0

# ---- start parsing from this heading ----
START_HEADING_PATTERN = re.compile(r"^\s*1\s+Introduction\s*$", re.IGNORECASE)

# ---- requirement ids ----
REQ_ID_CORE = r"(?:REQ|DRQ|LIM|CHO|ASM)_\d+(?:\*|\[[A-Za-z0-9_-]+\])?"
REQ_AT_LINE_START_PATTERN = re.compile(
    rf"^({REQ_ID_CORE})(?:\s+({REQ_ID_CORE}))?(?:\s+(.*))?$"
)
REQ_TOKEN_PATTERN = re.compile(REQ_ID_CORE)
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)+\s+[A-Z][^\n]*$"
)


# ---- normal text chunking ----
NORMAL_CHUNK_SIZE = 1200
NORMAL_CHUNK_OVERLAP = 150

# ---- debug ----
DEBUG_HEADER_FOOTER = True
DEBUG_PREVIEW_CHUNKS = True
PREVIEW_CHUNK_COUNT = 20

# ---- batch ----
BATCH_SIZE = 100


# =========================================================
# ENV / NEO4J / EMBEDDING
# =========================================================
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_embedding_cache: Dict[str, List[float]] = {}


# =========================================================
# SIMPLE CHUNK CONTAINER
# =========================================================
@dataclass
class ChunkDoc:
    page_content: str
    metadata: Dict[str, Any]


# =========================================================
# UTILS
# =========================================================
def safe_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def normalize_spaces(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_line(text: str) -> str:
    return normalize_spaces(text)


def is_noise_line(text: str) -> bool:
    low = text.lower().strip()

    if low in {
        "company confidential",
        "page",
        "date",
        "document title",
        "document id / release",
        "document id",
        "release",
    }:
        return True

    if re.fullmatch(r"\d+", low):
        return True

    if low.startswith("document title"):
        return True
    if low.startswith("document id"):
        return True
    if low.startswith("date "):
        return True

    return False


def merge_hyphenated_lines(lines: List[str]) -> List[str]:
    merged = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if i < len(lines) - 1:
            nxt = lines[i + 1].strip()
            if cur.endswith("-") and nxt:
                merged.append(cur[:-1] + nxt)
                i += 2
                continue
        merged.append(cur)
        i += 1
    return merged


def clean_chunk_text(text: str) -> str:
    lines = [clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line and not is_noise_line(line)]
    lines = merge_hyphenated_lines(lines)
    return "\n".join(lines).strip()


def split_text_with_overlap(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            para_break = text.rfind("\n\n", start, end)
            line_break = text.rfind("\n", start, end)
            sentence_break = text.rfind(". ", start, end)
            cut = max(para_break, line_break, sentence_break)
            if cut > start + chunk_size // 2:
                end = cut + (2 if cut == sentence_break else 0)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(end - overlap, start + 1)

    return chunks


def parse_requirement_line(text: str):
    m = REQ_AT_LINE_START_PATTERN.match(text.strip())
    if not m:
        return None

    id1 = m.group(1)
    id2 = m.group(2)
    remainder = (m.group(3) or "").strip()

    ids = [id1]
    if id2:
        ids.append(id2)

    return {
        "ids": ids,
        "remainder": remainder
    }


def parse_requirement_ids(text: str) -> List[str]:
    return REQ_TOKEN_PATTERN.findall(text or "")


def is_requirement_label_text(text: str) -> bool:
    if not text:
        return False

    # 至少得有一个 requirement id
    tokens = parse_requirement_ids(text)
    if not tokens:
        return False

    # 允许的内容：
    # 1. requirement id，如 REQ_13 / DRQ_15 / CHO_35
    # 2. 引用标记，如 [11]
    # 3. 空白
    remainder = text

    # 去掉 requirement ids
    remainder = REQ_TOKEN_PATTERN.sub("", remainder)

    # 去掉引用标记 [11]
    remainder = re.sub(r"\[\d+\]", "", remainder)

    # 去掉所有空白
    remainder = re.sub(r"\s+", "", remainder)

    # 如果什么都不剩，说明整块左栏本质上就是 label 区
    return remainder == ""


def looks_like_rationale_label(text: str) -> bool:
    return text.strip().lower() == "rationale"


def looks_like_continuation(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if not t:
        return False

    if re.match(r"^[0-9:;,\.\)\]-]", t):
        return True
    if re.match(r"^[a-z]", t):
        return True

    return False

def looks_like_section_heading(text: str) -> bool:
    if not text:
        return False

    t = clean_line(text)

    # avoid matching requirement ids accidentally
    if parse_requirement_ids(t):
        return False

    return bool(SECTION_HEADING_PATTERN.match(t))


def ends_with_hyphen(text_lines: List[str]) -> bool:
    for line in reversed(text_lines):
        s = line.strip()
        if s:
            return s.endswith("-")
    return False


def make_document_id(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


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


def extract_requirement_only_text(chunk_text: str) -> str:
    lines = chunk_text.splitlines()
    req_parts = []

    for line in lines:
        line = clean_line(line)
        if not line:
            continue
        if line.startswith("Requirement:"):
            req_parts.append(line[len("Requirement:"):].strip())

    return clean_line(" ".join(req_parts))


def extract_rationale_only_text(chunk_text: str) -> str:
    lines = chunk_text.splitlines()
    rationale_parts = []

    for line in lines:
        line = clean_line(line)
        if not line:
            continue
        if line.startswith("Rationale:"):
            rationale_parts.append(line[len("Rationale:"):].strip())

    return clean_line(" ".join(rationale_parts))



def make_chunk_name(file_name: str, requirement_id: str) -> str:
    return f"{file_name}_{requirement_id}"

def infer_spec_type(file_path: str) -> str:
    file_name = os.path.basename(file_path).upper()

    if file_name.startswith("FS"):
        return "FS"
    if file_name.startswith("TS"):
        return "TS"

    raise ValueError(f"Cannot infer spec type from file name: {file_name}. Expected prefix FS or TS.")


# =========================================================
# PDF LOW-LEVEL HELPERS
# =========================================================
def group_words_to_lines(words: List[Dict], y_tolerance: float = 3.0) -> List[List[Dict]]:
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


def words_to_text(words: List[Dict]) -> str:
    if not words:
        return ""
    words = sorted(words, key=lambda x: x["x0"])
    return clean_line(" ".join(w["text"] for w in words))


def split_line_left_right(line_words: List[Dict], threshold: float = LEFT_ID_THRESHOLD) -> Tuple[str, str]:
    left_words = [w for w in line_words if w["x0"] < threshold]
    right_words = [w for w in line_words if w["x0"] >= threshold]

    left_text = words_to_text(left_words)
    right_text = words_to_text(right_words)
    return left_text, right_text


def line_to_info(line_words: List[Dict]) -> Dict:
    line_words = sorted(line_words, key=lambda x: x["x0"])
    text = " ".join(w["text"] for w in line_words).strip()
    x0 = min(w["x0"] for w in line_words)
    x1 = max(w["x1"] for w in line_words)
    top = min(w["top"] for w in line_words)
    bottom = max(w["bottom"] for w in line_words)

    return {
        "text": clean_line(text),
        "x0": x0,
        "x1": x1,
        "top": top,
        "bottom": bottom,
        "words": line_words,
    }


def lines_are_close(prev_line: Dict, curr_line: Dict, max_gap: float = PREFIX_MERGE_MAX_GAP) -> bool:
    gap = curr_line["top"] - prev_line["bottom"]
    return gap <= max_gap


def is_same_requirement_prefix_block(
    prev_line_info: Optional[Dict],
    curr_line_info: Dict,
    left_text: str,
    right_text: str,
    current_ids: Optional[List[str]],
    current_req_lines: List[str],
    max_gap: float = PREFIX_MERGE_MAX_GAP,
) -> bool:
    if not current_ids:
        return False

    left_ids = parse_requirement_ids(left_text)
    if not left_ids or not is_requirement_label_text(left_text):
        return False

    close_to_prev = False
    if prev_line_info is not None:
        close_to_prev = lines_are_close(prev_line_info, curr_line_info, max_gap=max_gap)

    continuation_hint = (
        not right_text
        or looks_like_continuation(right_text)
        or ends_with_hyphen(current_req_lines)
    )

    return close_to_prev and continuation_hint


def extract_region_text(page, bbox, label: str = "") -> str:
    try:
        region = page.crop(bbox)
        txt = region.extract_text() or ""
        txt = normalize_spaces(txt)
        return txt
    except Exception as e:
        return f"[ERROR extracting {label}: {e}]"


def print_pdf_header_footer_debug(file_path: str, max_pages: Optional[int] = None):
    print("\n" + "=" * 120)
    print("[DEBUG] HEADER / FOOTER CHECK")
    print("=" * 120)

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        pages_to_check = total_pages if max_pages is None else min(max_pages, total_pages)

        for page_idx in range(pages_to_check):
            page = pdf.pages[page_idx]

            header_bbox = (0, 0, page.width, HEADER_CROP)
            footer_bbox = (0, page.height - 120, page.width, page.height)

            header_text = extract_region_text(page, header_bbox, "header")
            footer_text = extract_region_text(page, footer_bbox, "footer")

            print(f"\n--- PAGE {page_idx + 1} ---")
            print("[HEADER]")
            print(header_text if header_text else "[EMPTY]")
            print("[FOOTER]")
            print(footer_text if footer_text else "[EMPTY]")


# =========================================================
# FLUSH HELPERS
# =========================================================
def flush_normal_buffer(
    chunks: List[ChunkDoc],
    normal_lines: List[str],
    file_path: str,
    pages: List[int],
):
    text = clean_chunk_text("\n".join(normal_lines))
    if not text:
        return

    split_parts = split_text_with_overlap(
        text,
        chunk_size=NORMAL_CHUNK_SIZE,
        overlap=NORMAL_CHUNK_OVERLAP,
    )

    for idx, part in enumerate(split_parts, start=1):
        chunks.append(
            ChunkDoc(
                page_content=part,
                metadata={
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "pages": pages[:],
                    "type": "FS_section",
                    "chunk_mode": "normal",
                    "chunk_index": idx,
                },
            )
        )


def flush_req_buffer(
    chunks: List[ChunkDoc],
    current_ids: Optional[List[str]],
    current_text_lines: List[str],
    file_path: str,
    pages: List[int],
    spec_type: str,
):
    if not current_ids or not current_text_lines:
        return

    text = clean_chunk_text("\n".join(current_text_lines))
    if not text:
        return

    primary_requirement_id = current_ids[0]
    related_requirement_ids = current_ids[1:] if len(current_ids) > 1 else []

    chunks.append(
        ChunkDoc(
            page_content=text,
            metadata={
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "pages": pages[:],
                "type": spec_type,
                "spec_type": spec_type,
                "chunk_mode": "requirement",
                "requirement_ids": current_ids[:],
                "primary_requirement_id": primary_requirement_id,
                "related_requirement_ids": related_requirement_ids,
            },
        )
    )


def update_requirement_id_header(current_req_lines: List[str], current_ids: List[str]) -> List[str]:
    header = f"Requirement IDs: {', '.join(current_ids)}"
    if current_req_lines:
        current_req_lines[0] = header
    else:
        current_req_lines = [header]
    return current_req_lines


def append_requirement_text(current_req_lines: List[str], text: str):
    if not text:
        return

    if len(current_req_lines) >= 2 and current_req_lines[-1].startswith("Requirement: "):
        current_req_lines[-1] += " " + text
    else:
        current_req_lines.append(f"Requirement: {text}")


def append_rationale_text(current_req_lines: List[str], text: str):
    if not text:
        return

    if len(current_req_lines) >= 2 and current_req_lines[-1].startswith("Rationale: "):
        current_req_lines[-1] += " " + text
    else:
        current_req_lines.append(f"Rationale: {text}")


# =========================================================
# PDF PARSER
# =========================================================
def extract_spec_chunks_from_pdf(file_path: str) -> List[ChunkDoc]:
    chunks: List[ChunkDoc] = []
    spec_type = infer_spec_type(file_path)

    started = False
    in_requirement_mode = False

    normal_lines: List[str] = []
    normal_pages: List[int] = []

    current_ids: Optional[List[str]] = None
    current_req_lines: List[str] = []
    current_req_pages: List[int] = []

    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_no = page_idx + 1

            cropped = page.crop((0, HEADER_CROP, page.width, page.height))

            words = cropped.extract_words(
                use_text_flow=False,
                keep_blank_chars=False,
                extra_attrs=["fontname", "size"],
            )

            if not words:
                continue

            lines = group_words_to_lines(words, y_tolerance=3.0)
            line_infos = [line_to_info(line_words) for line_words in lines]

            prev_line_info = None

            for line_info in line_infos:
                full_text = line_info["text"]
                left_text, right_text = split_line_left_right(line_info["words"], LEFT_ID_THRESHOLD)

                if not full_text or is_noise_line(full_text):
                    prev_line_info = line_info
                    continue

                if not started:
                    if START_HEADING_PATTERN.match(full_text):
                        started = True
                        normal_lines.append(full_text)
                        if page_no not in normal_pages:
                            normal_pages.append(page_no)
                    prev_line_info = line_info
                    continue

                left_ids = parse_requirement_ids(left_text)

                # -------------------------------------------------
                # Case 1: 左侧是 requirement prefix
                # -------------------------------------------------
                if left_ids and is_requirement_label_text(left_text):
                    if is_same_requirement_prefix_block(
                        prev_line_info=prev_line_info,
                        curr_line_info=line_info,
                        left_text=left_text,
                        right_text=right_text,
                        current_ids=current_ids,
                        current_req_lines=current_req_lines,
                        max_gap=PREFIX_MERGE_MAX_GAP,
                    ):
                        for rid in left_ids:
                            if rid not in current_ids:
                                current_ids.append(rid)

                        current_req_lines = update_requirement_id_header(current_req_lines, current_ids)

                        if right_text:
                            append_requirement_text(current_req_lines, right_text)

                        if page_no not in current_req_pages:
                            current_req_pages.append(page_no)

                        prev_line_info = line_info
                        continue

                    # 新 requirement
                    if not in_requirement_mode:
                        flush_normal_buffer(chunks, normal_lines, file_path, normal_pages)
                        normal_lines = []
                        normal_pages = []
                        in_requirement_mode = True

                    flush_req_buffer(
                        chunks, current_ids, current_req_lines, file_path, current_req_pages, spec_type
                    )

                    current_ids = left_ids[:]
                    current_req_pages = [page_no]
                    current_req_lines = [f"Requirement IDs: {', '.join(current_ids)}"]

                    if right_text:
                        append_requirement_text(current_req_lines, right_text)

                    prev_line_info = line_info
                    continue

                # -------------------------------------------------
                # Case 2: 左侧是 Rationale
                # -------------------------------------------------
                if looks_like_rationale_label(left_text):
                    if in_requirement_mode and current_ids:
                        if right_text:
                            append_rationale_text(current_req_lines, right_text)
                        else:
                            current_req_lines.append("Rationale:")

                        if page_no not in current_req_pages:
                            current_req_pages.append(page_no)
                    else:
                        normal_lines.append(full_text)
                        if page_no not in normal_pages:
                            normal_pages.append(page_no)

                    prev_line_info = line_info
                    continue

                # -------------------------------------------------
                # Case 3: section heading -> end current requirement block
                # -------------------------------------------------
                if looks_like_section_heading(full_text):
                    if in_requirement_mode and current_ids:
                        flush_req_buffer(
                            chunks, current_ids, current_req_lines, file_path, current_req_pages, spec_type
                        )

                        current_ids = None
                        current_req_lines = []
                        current_req_pages = []
                        in_requirement_mode = False

                    normal_lines.append(full_text)
                    if page_no not in normal_pages:
                        normal_pages.append(page_no)

                    prev_line_info = line_info
                    continue

                # -------------------------------------------------
                # Case 4: requirement 正文续行
                # -------------------------------------------------
                if in_requirement_mode and current_ids:
                    continuation = right_text if right_text else full_text

                    if continuation and not is_noise_line(continuation):
                        if current_req_lines and current_req_lines[-1].startswith("Rationale:"):
                            append_rationale_text(current_req_lines, continuation)
                        else:
                            append_requirement_text(current_req_lines, continuation)

                        if page_no not in current_req_pages:
                            current_req_pages.append(page_no)

                    prev_line_info = line_info
                    continue

                prev_line_info = line_info

        if in_requirement_mode:
            flush_req_buffer(
                chunks, current_ids, current_req_lines, file_path, current_req_pages, spec_type
            )
        else:
            flush_normal_buffer(chunks, normal_lines, file_path, normal_pages)

    print(f"[INFO] Extracted {len(chunks)} chunks from PDF ({spec_type})")
    return chunks

# =========================================================
# DOCX PARSER
# =========================================================
def extract_spec_chunks_from_docx(file_path: str) -> List[ChunkDoc]:
    chunks: List[ChunkDoc] = []
    spec_type = infer_spec_type(file_path)

    started = False
    in_requirement_mode = False

    normal_lines: List[str] = []

    current_ids: Optional[List[str]] = None
    current_req_lines: List[str] = []

    doc = DocxDocument(file_path)

    for para in doc.paragraphs:
        text = clean_line(para.text)
        if not text or is_noise_line(text):
            continue

        if not started:
            if START_HEADING_PATTERN.match(text):
                started = True
                normal_lines.append(text)
            continue

        parsed = parse_requirement_line(text)

        if parsed:
            if not in_requirement_mode:
                txt = clean_chunk_text("\n".join(normal_lines))
                if txt:
                    split_parts = split_text_with_overlap(
                        txt,
                        chunk_size=NORMAL_CHUNK_SIZE,
                        overlap=NORMAL_CHUNK_OVERLAP,
                    )
                    for idx, part in enumerate(split_parts, start=1):
                        chunks.append(
                            ChunkDoc(
                                page_content=part,
                                metadata={
                                    "source": file_path,
                                    "file_name": os.path.basename(file_path),
                                    "type": f"{spec_type}_section",
                                    "spec_type": spec_type,
                                    "chunk_mode": "normal",
                                    "chunk_index": idx,
                                },
                            )
                        )
                normal_lines = []
                in_requirement_mode = True

            flush_req_buffer(chunks, current_ids, current_req_lines, file_path, [], spec_type)

            current_ids = parsed["ids"][:]
            current_req_lines = [f"Requirement IDs: {', '.join(current_ids)}"]

            if parsed["remainder"]:
                append_requirement_text(current_req_lines, parsed["remainder"])

        else:
            if in_requirement_mode and current_ids:
                if looks_like_rationale_label(text):
                    current_req_lines.append("Rationale:")
                else:
                    if current_req_lines and current_req_lines[-1].startswith("Rationale:"):
                        append_rationale_text(current_req_lines, text)
                    else:
                        append_requirement_text(current_req_lines, text)
            else:
                normal_lines.append(text)

    if in_requirement_mode:
        flush_req_buffer(chunks, current_ids, current_req_lines, file_path, [], spec_type)
    else:
        txt = clean_chunk_text("\n".join(normal_lines))
        if txt:
            split_parts = split_text_with_overlap(
                txt,
                chunk_size=NORMAL_CHUNK_SIZE,
                overlap=NORMAL_CHUNK_OVERLAP,
            )
            for idx, part in enumerate(split_parts, start=1):
                chunks.append(
                    ChunkDoc(
                        page_content=part,
                        metadata={
                            "source": file_path,
                            "file_name": os.path.basename(file_path),
                            "type": f"{spec_type}_section",
                            "spec_type": spec_type,
                            "chunk_mode": "normal",
                            "chunk_index": idx,
                        },
                    )
                )

    print(f"[INFO] Extracted {len(chunks)} chunks from DOCX ({spec_type})")
    return chunks



# =========================================================
# PREPARE CHUNKS
# =========================================================
def prepare_spec_chunks(file_path: str) -> List[ChunkDoc]:
    lower = file_path.lower()

    if lower.endswith(".pdf"):
        chunks = extract_spec_chunks_from_pdf(file_path)
    elif lower.endswith(".docx"):
        chunks = extract_spec_chunks_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")

    requirement_chunks = [
        ch for ch in chunks
        if ch.metadata.get("chunk_mode") == "requirement"
    ]

    spec_type = infer_spec_type(file_path)
    print(f"[INFO] Total requirement chunks ready for KG import ({spec_type}): {len(requirement_chunks)}")
    return requirement_chunks


# =========================================================
# KG BUILDER
# =========================================================

def make_document_id(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def make_fschunk_name(file_name: str, requirement_id: str) -> str:
    base_name = os.path.splitext(file_name)[0]
    return f"{base_name}_{requirement_id}"

def make_tschunk_name(file_name: str, requirement_id: str, seq: int) -> str:
    base_name = os.path.splitext(file_name)[0]
    return f"{base_name}_{requirement_id}_TS_{seq}"

def make_rationale_chunk_name(file_name: str, requirement_id: str) -> str:
    return f"{file_name}_{requirement_id}_rationale"


class VectorKGBuilder:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def setup_schema(self):
        queries = [
            """
            CREATE CONSTRAINT specification_document_semantic_id_unique IF NOT EXISTS
            FOR (n:SpecificationDocument)
            REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT fschunk_name_unique IF NOT EXISTS
            FOR (n:FSChunk)
            REQUIRE n.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT tschunk_name_unique IF NOT EXISTS
            FOR (n:TSChunk)
            REQUIRE n.name IS UNIQUE
            """,
            """
            CREATE CONSTRAINT rationale_chunk_name_unique IF NOT EXISTS
            FOR (n:RationaleChunk)
            REQUIRE n.name IS UNIQUE
            """
        ]
        with self.driver.session(database=self.database) as session:
            for q in queries:
                session.run(q)

        print("[INFO] Neo4j schema ready")

    def upsert_document_node(self, document_id: str, file_path: str):
        file_name = os.path.basename(file_path)
        spec_type = infer_spec_type(file_path)

        props = {
            "semantic_id": document_id,
            "file_name": file_name,
            "source": file_path,
            "spec_type": spec_type,
        }

        query = """
        MERGE (d:SpecificationDocument {semantic_id: $document_id})
        SET d = $props
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, document_id=document_id, props=props)

    def purge_document_subgraph(self, document_id: str):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (ra:RationaleChunk)-[:RATIONALE_FOR]->(fs:FSChunk)-[:PART_OF]->(d:SpecificationDocument {semantic_id: $document_id})
                DETACH DELETE ra
                """,
                document_id=document_id,
            )

            session.run(
                """
                MATCH (n)-[:PART_OF]->(d:SpecificationDocument {semantic_id: $document_id})
                DETACH DELETE n
                """,
                document_id=document_id,
            )

        print(f"[INFO] Purged old subgraph for document: {document_id}")

    def cleanup_orphan_requirement_nodes(self):
        """
        清理旧版本遗留下来的 Requirement 节点。
        只删除孤立节点，安全。
        """
        query = """
        MATCH (r:Requirement)
        WHERE NOT (r)--()
        DELETE r
        """
        with self.driver.session(database=self.database) as session:
            session.run(query)

        print("[INFO] Cleaned orphan Requirement nodes")

    def _build_fschunk_row(self, chunk) -> Optional[Dict[str, Any]]:
        file_name = safe_text(chunk.metadata.get("file_name"))
        primary_requirement_id = safe_text(chunk.metadata.get("primary_requirement_id"))
        related_requirement_ids = chunk.metadata.get("related_requirement_ids", [])
        pages = chunk.metadata.get("pages", [])
        chunk_type = safe_text(chunk.metadata.get("type", "FS"))

        if not isinstance(pages, list):
            pages = []

        if not isinstance(related_requirement_ids, list):
            related_requirement_ids = []

        if not file_name or not primary_requirement_id:
            return None

        cleaned_file_name = os.path.splitext(file_name)[0]

        requirement_text = extract_requirement_only_text(chunk.page_content)
        if not requirement_text:
            return None

        related_fschunk_names = [
            make_fschunk_name(cleaned_file_name, rid)
            for rid in related_requirement_ids
            if rid and rid != primary_requirement_id
        ]

        return {
            "name": make_fschunk_name(cleaned_file_name, primary_requirement_id),
            "primary_requirement_id": primary_requirement_id,
            "related_requirement_ids": related_requirement_ids,
            "related_fschunk_names": related_fschunk_names,
            "embedding": embed(requirement_text),
            "text": requirement_text,
            "pages": pages,
            "type": chunk_type,
        }
    
    def _build_tschunk_row(self, chunk, seq: int) -> Optional[Dict[str, Any]]:
        file_name = safe_text(chunk.metadata.get("file_name"))
        primary_requirement_id = safe_text(chunk.metadata.get("primary_requirement_id"))
        raw_ids = chunk.metadata.get("requirement_ids", [])
        pages = chunk.metadata.get("pages", [])
        chunk_type = safe_text(chunk.metadata.get("type", "TS"))

        if not isinstance(pages, list):
            pages = []

        if not isinstance(raw_ids, list):
            raw_ids = []

        if not file_name or not primary_requirement_id:
            return None

        cleaned_file_name = os.path.splitext(file_name)[0]

        requirement_text = extract_requirement_only_text(chunk.page_content)
        if not requirement_text:
            return None

        # 去掉自己的主 ID，并顺便去重
        referred_fs_ids = []
        seen = set()

        for rid in raw_ids:
            rid = safe_text(rid)
            if not rid:
                continue
            if rid == primary_requirement_id:
                continue
            if rid in seen:
                continue
            seen.add(rid)
            referred_fs_ids.append(rid)

        return {
            "name": make_tschunk_name(cleaned_file_name, primary_requirement_id, seq),
            "primary_requirement_id": primary_requirement_id,
            "referred_fs_ids": referred_fs_ids,
            "embedding": embed(requirement_text),
            "text": requirement_text,
            "pages": pages,
            "type": chunk_type,
        }

    def _build_rationale_row(self, chunk, spec_kind: str, seq: Optional[int] = None) -> Optional[Dict[str, Any]]:
        file_name = safe_text(chunk.metadata.get("file_name"))
        primary_requirement_id = safe_text(chunk.metadata.get("primary_requirement_id"))
        pages = chunk.metadata.get("pages", [])
        chunk_type = safe_text(chunk.metadata.get("type", spec_kind))

        if not isinstance(pages, list):
            pages = []

        if not file_name or not primary_requirement_id:
            return None

        cleaned_file_name = os.path.splitext(file_name)[0]

        rationale_text = extract_rationale_only_text(chunk.page_content)
        if not rationale_text:
            return None

        if spec_kind == "TS":
            if seq is None:
                return None

            parent_chunk_name = make_tschunk_name(cleaned_file_name, primary_requirement_id, seq)

            # rationale 自己的名字不带 _TS_{seq}
            rationale_name = f"{cleaned_file_name}_{primary_requirement_id}_rationale"
        else:
            parent_chunk_name = make_fschunk_name(cleaned_file_name, primary_requirement_id)
            rationale_name = f"{cleaned_file_name}_{primary_requirement_id}_rationale"

        return {
            "name": rationale_name,
            "parent_chunk_name": parent_chunk_name,
            "embedding": embed(rationale_text),
            "text": rationale_text,
            "pages": pages,
            "type": chunk_type,
        }

    def import_fschunks(self, document_id: str, chunks: List[Any], batch_size: int = 100):
        rows = []
        for ch in chunks:
            row = self._build_fschunk_row(ch)
            if row is not None:
                rows.append(row)

        if not rows:
            print("[INFO] No FSChunk rows to import")
            return

        query = """
        UNWIND $rows AS row
        MERGE (c:FSChunk {name: row.name})
        SET c = {
            name: row.name,
            primary_requirement_id: row.primary_requirement_id,
            related_requirement_ids: row.related_requirement_ids,
            embedding: row.embedding,
            text: row.text,
            pages: row.pages,
            type: row.type
        }
        WITH c
        MATCH (d:SpecificationDocument {semantic_id: $document_id})
        MERGE (c)-[:PART_OF]->(d)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch, document_id=document_id)
                print(f"[INFO] Imported FSChunk batch {i + 1} - {i + len(batch)} / {len(rows)}")

    def import_tschunks(self, document_id: str, chunks: List[Any], batch_size: int = 100):
        rows = []
        seq = 1

        for ch in chunks:
            row = self._build_tschunk_row(ch, seq=seq)
            if row is not None:
                rows.append(row)
                seq += 1

        if not rows:
            print("[INFO] No TSChunk rows to import")
            return

        query = """
        UNWIND $rows AS row
        MERGE (c:TSChunk {name: row.name})
        SET c = {
            name: row.name,
            primary_requirement_id: row.primary_requirement_id,
            referred_fs_ids: row.referred_fs_ids,
            embedding: row.embedding,
            text: row.text,
            pages: row.pages,
            type: row.type
        }
        WITH c
        MATCH (d:SpecificationDocument {semantic_id: $document_id})
        MERGE (c)-[:PART_OF]->(d)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch, document_id=document_id)
                print(f"[INFO] Imported TSChunk batch {i + 1} - {i + len(batch)} / {len(rows)}")

    def import_rationale_chunks(
        self,
        document_id: str,
        chunks: List[Any],
        parent_label: str,
        spec_kind: str,
        batch_size: int = 100
    ):
        rows = []
        seq = 1

        for ch in chunks:
            if spec_kind == "TS":
                row = self._build_rationale_row(ch, spec_kind="TS", seq=seq)
                if row is not None:
                    rows.append(row)
                    seq += 1
            else:
                row = self._build_rationale_row(ch, spec_kind="FS")
                if row is not None:
                    rows.append(row)

        if not rows:
            print(f"[INFO] No {spec_kind} RationaleChunk rows to import")
            return

        query = f"""
        UNWIND $rows AS row
        MATCH (p:{parent_label} {{name: row.parent_chunk_name}})
        MATCH (d:SpecificationDocument {{semantic_id: $document_id}})
        MERGE (ra:RationaleChunk {{name: row.name}})
        SET ra = {{
            name: row.name,
            embedding: row.embedding,
            text: row.text,
            pages: row.pages,
            type: row.type
        }}
        MERGE (ra)-[:RATIONALE_FOR]->(p)
        MERGE (ra)-[:PART_OF]->(d)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch, document_id=document_id)
                print(f"[INFO] Imported {spec_kind} RationaleChunk batch {i + 1} - {i + len(batch)} / {len(rows)}")

    def import_related_links(self, chunks: List[Any], batch_size: int = 100):
        rows = []

        for ch in chunks:
            row = self._build_fschunk_row(ch)
            if row is None:
                continue

            source_name = row["name"]
            for target_name in row.get("related_fschunk_names", []):
                if target_name and target_name != source_name:
                    rows.append({
                        "source_name": source_name,
                        "target_name": target_name,
                    })

        if not rows:
            print("[INFO] No RELATED links to import")
            return

        query = """
        UNWIND $rows AS row
        MATCH (src:FSChunk {name: row.source_name})
        MATCH (dst:FSChunk {name: row.target_name})
        MERGE (src)-[:RELATED]->(dst)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch)
                print(f"[INFO] Imported RELATED batch {i + 1} - {i + len(batch)} / {len(rows)}")

    def import_ts_implement_links(self, fs_file_path: str, ts_chunks: List[Any], batch_size: int = 100):
        fs_file_name = os.path.basename(fs_file_path)
        fs_base_name = os.path.splitext(fs_file_name)[0]

        rows = []
        seq = 1

        for ch in ts_chunks:
            row = self._build_tschunk_row(ch, seq=seq)
            if row is None:
                continue

            ts_name = row["name"]
            for fs_req_id in row.get("referred_fs_ids", []):
                if fs_req_id:
                    rows.append({
                        "ts_name": ts_name,
                        "fs_name": make_fschunk_name(fs_base_name, fs_req_id),
                    })

            seq += 1

        if not rows:
            print("[INFO] No IMPLEMENT links to import")
            return

        query = """
        UNWIND $rows AS row
        MATCH (ts:TSChunk {name: row.ts_name})
        MATCH (fs:FSChunk {name: row.fs_name})
        MERGE (ts)-[:IMPLEMENT]->(fs)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch)
                print(f"[INFO] Imported IMPLEMENT batch {i + 1} - {i + len(batch)} / {len(rows)}")

    def refresh_fs_document(self, file_path: str, chunks: List[Any], batch_size: int = 100):
        document_id = make_document_id(file_path)

        self.upsert_document_node(document_id, file_path)
        self.purge_document_subgraph(document_id)

        self.import_fschunks(document_id, chunks, batch_size=batch_size)
        self.import_related_links(chunks, batch_size=batch_size)
        self.import_rationale_chunks(
            document_id=document_id,
            chunks=chunks,
            parent_label="FSChunk",
            spec_kind="FS",
            batch_size=batch_size,
        )
        self.cleanup_orphan_requirement_nodes()

        print(f"[INFO] Refreshed FS document: {document_id}")


    def refresh_ts_document(self, ts_file_path: str, fs_file_path: str, ts_chunks: List[Any], batch_size: int = 100):
        document_id = make_document_id(ts_file_path)

        self.upsert_document_node(document_id, ts_file_path)
        self.purge_document_subgraph(document_id)

        self.import_tschunks(document_id, ts_chunks, batch_size=batch_size)
        self.import_rationale_chunks(
            document_id=document_id,
            chunks=ts_chunks,
            parent_label="TSChunk",
            spec_kind="TS",
            batch_size=batch_size,
        )
        self.import_ts_implement_links(fs_file_path, ts_chunks, batch_size=batch_size)
        self.cleanup_orphan_requirement_nodes()

        print(f"[INFO] Refreshed TS document: {document_id}")

# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.isfile(FS_PATH):
        raise FileNotFoundError(f"FS file not found: {FS_PATH}")

    if not os.path.isfile(TS_PATH):
        raise FileNotFoundError(f"TS file not found: {TS_PATH}")

    if DEBUG_HEADER_FOOTER and FS_PATH.lower().endswith(".pdf"):
        print_pdf_header_footer_debug(FS_PATH, max_pages=5)

    if DEBUG_HEADER_FOOTER and TS_PATH.lower().endswith(".pdf"):
        print_pdf_header_footer_debug(TS_PATH, max_pages=5)

    fs_chunks = prepare_spec_chunks(FS_PATH)
    ts_chunks = prepare_spec_chunks(TS_PATH)

    if DEBUG_PREVIEW_CHUNKS:
        print("\n" + "=" * 120)
        print("[DEBUG] FS REQUIREMENT CHUNK PREVIEW")
        print("=" * 120)
        for i, ch in enumerate(fs_chunks[:PREVIEW_CHUNK_COUNT], start=1):
            print(f"\n--- FS CHUNK {i} ---")
            print("METADATA:")
            print(ch.metadata)
            print("RAW CONTENT:")
            print(ch.page_content[:2000])
            print("REQUIREMENT TEXT ONLY:")
            print(extract_requirement_only_text(ch.page_content))

        print("\n" + "=" * 120)
        print("[DEBUG] TS REQUIREMENT CHUNK PREVIEW")
        print("=" * 120)
        for i, ch in enumerate(ts_chunks[:PREVIEW_CHUNK_COUNT], start=1):
            print(f"\n--- TS CHUNK {i} ---")
            print("METADATA:")
            print(ch.metadata)
            print("RAW CONTENT:")
            print(ch.page_content[:2000])
            print("REQUIREMENT TEXT ONLY:")
            print(extract_requirement_only_text(ch.page_content))

    kg_builder = VectorKGBuilder(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    try:
        kg_builder.setup_schema()

        # 先导 FS，保证 TS -> FS 能匹配到目标节点
        kg_builder.refresh_fs_document(
            file_path=FS_PATH,
            chunks=fs_chunks,
            batch_size=BATCH_SIZE,
        )

        # 再导 TS，并建立 IMPLEMENT -> FSChunk
        kg_builder.refresh_ts_document(
            ts_file_path=TS_PATH,
            fs_file_path=FS_PATH,
            ts_chunks=ts_chunks,
            batch_size=BATCH_SIZE,
        )

        print("[DONE]")
    finally:
        kg_builder.close()

if __name__ == "__main__":
    main()