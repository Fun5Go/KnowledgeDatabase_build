import os
import re
from collections import Counter
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
ESW_TS_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\TS6303220021R05.pdf"
HW_TS_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\TS6303220029R09.pdf"

# ---- PDF layout settings ----
HEADER_CROP = 90
FOOTER_CROP = 70
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
TOP_LEVEL_SECTION_HEADING_PATTERN = re.compile(
    r"^\s*\d+\s+[A-Z][^\n]*$"
)
SUBSECTION_HEADING_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)+\s+[A-Z][^\n]*$"
)
TOC_ENTRY_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\.{2,}\s*(\d+)\s*$"
)


# ---- normal text chunking ----
NORMAL_CHUNK_SIZE = 1200
NORMAL_CHUNK_OVERLAP = 150

# ---- debug ----
DEBUG_HEADER_FOOTER = True
DEBUG_PREVIEW_CHUNKS = False
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


def normalize_section_key(text: str) -> str:
    text = clean_line(text)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


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
    if "company confidential" in low:
        return True
    if re.fullmatch(r"\d+\s+company confidential", low):
        return True
    if re.fullmatch(r"\d+\s+page", low):
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

def looks_like_section_heading(text: str, toc_entries: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
    return get_section_heading_level(text, toc_entries=toc_entries) is not None


def get_section_heading_level(
    text: str,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[int]:
    if not text:
        return None

    t = clean_line(text)
    toc_entry = find_toc_entry_for_heading(t, toc_entries)
    if toc_entry is not None:
        return toc_entry["level"]
    if toc_entries:
        return None

    # avoid matching requirement ids accidentally
    if parse_requirement_ids(t):
        return None

    if SUBSECTION_HEADING_PATTERN.match(t):
        return 2
    if TOP_LEVEL_SECTION_HEADING_PATTERN.match(t):
        return 1
    return None


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


def make_section_chunk_name(file_name: str, section_name: str) -> str:
    base_name = os.path.splitext(file_name)[0]
    return f"{base_name}_{clean_line(section_name)}"

def infer_spec_type(file_path: str) -> str:
    file_name = os.path.basename(file_path).upper()

    if file_name.startswith("FS"):
        return "FS"
    if file_name.startswith("TS"):
        return "TS"

    raise ValueError(f"Cannot infer spec type from file name: {file_name}. Expected prefix FS or TS.")


def normalize_discipline(discipline: Optional[str]) -> Optional[str]:
    discipline = safe_text(discipline).upper()
    if not discipline:
        return None
    if discipline not in {"ESW", "HW"}:
        raise ValueError(f"Unsupported discipline: {discipline}. Expected ESW or HW.")
    return discipline


def get_document_primary_label(spec_type: str) -> str:
    if spec_type == "FS":
        return "FunctionalSpecification"
    if spec_type == "TS":
        return "TechnicalSpecification"
    raise ValueError(f"Unsupported spec type for document label: {spec_type}")


def get_document_secondary_label(spec_type: str, discipline: Optional[str]) -> Optional[str]:
    discipline = normalize_discipline(discipline)
    if spec_type == "TS" and discipline:
        return f"{discipline}TechnicalSpecification"
    return None


def get_chunk_secondary_label(base_label: str, discipline: Optional[str]) -> Optional[str]:
    discipline = normalize_discipline(discipline)
    if not discipline:
        return None
    if base_label == "TSChunk":
        return f"{discipline}TSChunk"
    return None


def get_rationale_label(spec_kind: str) -> str:
    if spec_kind == "FS":
        return "FSRationaleChunk"
    if spec_kind == "TS":
        return "TSRationaleChunk"
    raise ValueError(f"Unsupported spec kind for rationale label: {spec_kind}")


def get_rationale_discipline_label(spec_kind: str, discipline: Optional[str]) -> Optional[str]:
    discipline = normalize_discipline(discipline)
    if spec_kind != "TS" or not discipline:
        return None
    return f"{discipline}RationaleChunk"


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
        "level": number.count(".") + 1,
        "full_text": clean_line(f"{number} {title}"),
    }


def extract_toc_entries_from_pdf(pdf) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    in_toc = False
    started_collecting = False
    non_match_streak = 0

    for page_idx, page in enumerate(pdf.pages[:10]):
        cropped = page.crop((0, HEADER_CROP, page.width, page.height - FOOTER_CROP))
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
                key = normalize_section_key(entry["full_text"])
                entries[key] = entry
                started_collecting = True
                non_match_streak = 0
                continue

            if started_collecting:
                non_match_streak += 1
                if START_HEADING_PATTERN.match(line) or non_match_streak >= 8:
                    return entries

    return entries


def find_toc_entry_for_heading(
    text: str,
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if not toc_entries:
        return None

    normalized = normalize_section_key(text)
    return toc_entries.get(normalized)


def is_introduction_section(section_tag: Optional[str]) -> bool:
    section_tag = clean_line(section_tag or "")
    if not section_tag:
        return False
    return bool(re.match(r"^1(?:\.\d+)*\s+", section_tag))


def should_start_from_section(
    section_tag: Optional[str],
    toc_entries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    section_tag = clean_line(section_tag or "")
    if not section_tag:
        return False

    if parse_toc_line(section_tag):
        return False

    toc_entry = find_toc_entry_for_heading(section_tag, toc_entries)
    if toc_entries:
        if toc_entry is None:
            return False
        return not safe_text(toc_entry.get("number")).startswith("1")

    match = re.match(r"^(\d+(?:\.\d+)*)\s+", section_tag)
    if not match:
        return False

    return not match.group(1).startswith("1")


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
            footer_bbox = (0, page.height - FOOTER_CROP, page.width, page.height)

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
    discipline: Optional[str] = None,
    spec_type: Optional[str] = None,
    section_tag: Optional[str] = None,
):
    text = clean_chunk_text("\n".join(normal_lines))
    if not text:
        return

    section_tag = safe_text(section_tag)
    if not section_tag or is_introduction_section(section_tag):
        return

    chunk_spec_type = spec_type or infer_spec_type(file_path)
    file_name = os.path.basename(file_path)
    chunks.append(
        ChunkDoc(
            page_content=text,
            metadata={
                "source": file_path,
                "file_name": file_name,
                "pages": pages[:],
                "type": f"{chunk_spec_type}_section",
                "spec_type": chunk_spec_type,
                "discipline": normalize_discipline(discipline),
                "chunk_mode": "section",
                "chunk_index": 1,
                "section_tag": section_tag,
                "name": make_section_chunk_name(file_name, section_tag),
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
    discipline: Optional[str] = None,
    section_tag: Optional[str] = None,
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
                "discipline": normalize_discipline(discipline),
                "chunk_mode": "requirement",
                "requirement_ids": current_ids[:],
                "primary_requirement_id": primary_requirement_id,
                "related_requirement_ids": related_requirement_ids,
                "section_tag": safe_text(section_tag),
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
def extract_spec_chunks_from_pdf(file_path: str, discipline: Optional[str] = None) -> List[ChunkDoc]:
    chunks: List[ChunkDoc] = []
    spec_type = infer_spec_type(file_path)
    discipline = normalize_discipline(discipline)

    started = False
    in_requirement_mode = False
    toc_entries: Dict[str, Dict[str, Any]] = {}

    normal_lines: List[str] = []
    normal_pages: List[int] = []

    current_ids: Optional[List[str]] = None
    current_req_lines: List[str] = []
    current_req_pages: List[int] = []
    current_section_tag: Optional[str] = None
    current_major_section_tag: Optional[str] = None

    with pdfplumber.open(file_path) as pdf:
        toc_entries = extract_toc_entries_from_pdf(pdf)

        for page_idx, page in enumerate(pdf.pages):
            page_no = page_idx + 1

            cropped = page.crop((0, HEADER_CROP, page.width, page.height - FOOTER_CROP))

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

                if parse_toc_line(full_text):
                    prev_line_info = line_info
                    continue

                if not started:
                    toc_entry = find_toc_entry_for_heading(full_text, toc_entries=toc_entries)
                    if should_start_from_section(full_text, toc_entries=toc_entries) and toc_entry is not None:
                        started = True
                        canonical_section_tag = safe_text(toc_entry.get("full_text")) or full_text
                        current_major_section_tag = canonical_section_tag
                        current_section_tag = canonical_section_tag
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
                        flush_normal_buffer(
                            chunks,
                            normal_lines,
                            file_path,
                            normal_pages,
                            discipline=discipline,
                            spec_type=spec_type,
                            section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
                        )
                        normal_lines = []
                        normal_pages = []
                        in_requirement_mode = True

                    flush_req_buffer(
                        chunks,
                        current_ids,
                        current_req_lines,
                        file_path,
                        current_req_pages,
                        spec_type,
                        discipline=discipline,
                        section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
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
                toc_entry = find_toc_entry_for_heading(full_text, toc_entries=toc_entries)
                heading_level = get_section_heading_level(full_text, toc_entries=toc_entries)
                if heading_level is not None:
                    canonical_section_tag = safe_text(toc_entry.get("full_text")) if toc_entry else full_text

                    if not in_requirement_mode:
                        flush_normal_buffer(
                            chunks,
                            normal_lines,
                            file_path,
                            normal_pages,
                            discipline=discipline,
                            spec_type=spec_type,
                            section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
                        )
                        normal_lines = []
                        normal_pages = []

                    if in_requirement_mode and current_ids:
                        flush_req_buffer(
                            chunks,
                            current_ids,
                            current_req_lines,
                            file_path,
                            current_req_pages,
                            spec_type,
                            discipline=discipline,
                            section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
                        )

                        current_ids = None
                        current_req_lines = []
                        current_req_pages = []
                        in_requirement_mode = False

                    current_section_tag = canonical_section_tag
                    if heading_level == 1:
                        current_major_section_tag = canonical_section_tag

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

                normal_lines.append(full_text)
                if page_no not in normal_pages:
                    normal_pages.append(page_no)

                prev_line_info = line_info

        if in_requirement_mode:
            flush_req_buffer(
                chunks,
                current_ids,
                current_req_lines,
                file_path,
                current_req_pages,
                spec_type,
                discipline=discipline,
                section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
            )
        else:
            flush_normal_buffer(
                chunks,
                normal_lines,
                file_path,
                normal_pages,
                discipline=discipline,
                spec_type=spec_type,
                section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
            )

    print(f"[INFO] Extracted {len(chunks)} chunks from PDF ({spec_type})")
    return chunks

# =========================================================
# DOCX PARSER
# =========================================================
def extract_spec_chunks_from_docx(file_path: str, discipline: Optional[str] = None) -> List[ChunkDoc]:
    chunks: List[ChunkDoc] = []
    spec_type = infer_spec_type(file_path)
    discipline = normalize_discipline(discipline)

    started = False
    in_requirement_mode = False

    normal_lines: List[str] = []

    current_ids: Optional[List[str]] = None
    current_req_lines: List[str] = []
    current_section_tag: Optional[str] = None
    current_major_section_tag: Optional[str] = None

    doc = DocxDocument(file_path)

    for para in doc.paragraphs:
        text = clean_line(para.text)
        if not text or is_noise_line(text):
            continue

        if not started:
            if should_start_from_section(text):
                started = True
                current_major_section_tag = text
                current_section_tag = text
            continue

        parsed = parse_requirement_line(text)

        if parsed:
            if not in_requirement_mode:
                txt = clean_chunk_text("\n".join(normal_lines))
                if txt:
                    flush_normal_buffer(
                        chunks,
                        normal_lines,
                        file_path,
                        [],
                        discipline=discipline,
                        spec_type=spec_type,
                        section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
                    )
                normal_lines = []
                in_requirement_mode = True

            flush_req_buffer(
                chunks,
                current_ids,
                current_req_lines,
                file_path,
                [],
                spec_type,
                discipline=discipline,
                section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
            )

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
                heading_level = get_section_heading_level(text)
                if heading_level is not None:
                    flush_normal_buffer(
                        chunks,
                        normal_lines,
                        file_path,
                        [],
                        discipline=discipline,
                        spec_type=spec_type,
                        section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
                    )
                    normal_lines = []
                    current_section_tag = text
                    if heading_level == 1:
                        current_major_section_tag = text
                else:
                    normal_lines.append(text)

    if in_requirement_mode:
        flush_req_buffer(
            chunks,
            current_ids,
            current_req_lines,
            file_path,
            [],
            spec_type,
            discipline=discipline,
            section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
        )
    else:
        flush_normal_buffer(
            chunks,
            normal_lines,
            file_path,
            [],
            discipline=discipline,
            spec_type=spec_type,
            section_tag=get_effective_section_tag(current_section_tag, current_major_section_tag),
        )

    print(f"[INFO] Extracted {len(chunks)} chunks from DOCX ({spec_type})")
    return chunks



# =========================================================
# PREPARE CHUNKS
# =========================================================
def prepare_spec_chunks(file_path: str, discipline: Optional[str] = None) -> List[ChunkDoc]:
    lower = file_path.lower()
    discipline = normalize_discipline(discipline)

    if lower.endswith(".pdf"):
        chunks = extract_spec_chunks_from_pdf(file_path, discipline=discipline)
    elif lower.endswith(".docx"):
        chunks = extract_spec_chunks_from_docx(file_path, discipline=discipline)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")

    requirement_chunks = [
        ch for ch in chunks
        if ch.metadata.get("chunk_mode") == "requirement"
    ]
    section_chunks = [
        ch for ch in chunks
        if ch.metadata.get("chunk_mode") == "section"
    ]

    spec_type = infer_spec_type(file_path)
    print(f"[INFO] Total requirement chunks ready for KG import ({spec_type}): {len(requirement_chunks)}")
    print(f"[INFO] Total section chunks ready for KG import ({spec_type}): {len(section_chunks)}")
    return chunks


def print_section_tag_debug(chunks: List[ChunkDoc], label: str):
    counts = Counter()

    for ch in chunks:
        section_tag = safe_text(ch.metadata.get("section_tag")) or "[EMPTY]"
        counts[section_tag] += 1

    print("\n" + "=" * 120)
    print(f"[DEBUG] SECTION TAG COUNTS - {label}")
    print("=" * 120)

    if not counts:
        print("[DEBUG] No section tags found")
        return

    for section_tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{count:>4}  {section_tag}")


def get_effective_section_tag(
    current_section_tag: Optional[str],
    current_major_section_tag: Optional[str],
) -> str:
    return safe_text(current_section_tag) or safe_text(current_major_section_tag)


SECTION_HIERARCHY_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$")


def parse_section_hierarchy(section_tag: Optional[str]) -> Optional[Dict[str, str]]:
    section_tag = safe_text(section_tag)
    if not section_tag:
        return None

    match = SECTION_HIERARCHY_PATTERN.match(section_tag)
    if not match:
        return None

    prefix = match.group(1)
    title = clean_line(match.group(2))
    level = len(prefix.split("."))

    node_type_map = {
        1: "Chapter",
        2: "Section",
        3: "Subsection",
        4: "Subsubsection",
    }
    node_type = node_type_map.get(level)
    if not node_type:
        return None

    return {
        "section_prefix": prefix,
        "section_title": title,
        "hierarchy_level": str(level),
        "node_type": node_type,
    }


def make_section_node_semantic_id(document_id: str, section_prefix: str) -> str:
    return f"{document_id}::{section_prefix}"


def section_prefix_sort_key(prefix: str) -> Tuple[int, ...]:
    parts = []
    for part in safe_text(prefix).split("."):
        if not part:
            continue
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


# =========================================================
# KG BUILDER
# =========================================================

def make_document_id(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def make_fschunk_name(file_name: str, requirement_id: str) -> str:
    base_name = os.path.splitext(file_name)[0]
    return f"{base_name}_{requirement_id}"

def make_tschunk_name(file_name: str, requirement_id: str, seq: Optional[int] = None) -> str:
    base_name = os.path.splitext(file_name)[0]
    return f"{base_name}_{requirement_id}"

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
            CREATE CONSTRAINT functional_specification_semantic_id_unique IF NOT EXISTS
            FOR (n:FunctionalSpecification)
            REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT technical_specification_semantic_id_unique IF NOT EXISTS
            FOR (n:TechnicalSpecification)
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
            """,
            """
            CREATE CONSTRAINT chapter_semantic_id_unique IF NOT EXISTS
            FOR (n:Chapter)
            REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT section_semantic_id_unique IF NOT EXISTS
            FOR (n:Section)
            REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT subsection_semantic_id_unique IF NOT EXISTS
            FOR (n:Subsection)
            REQUIRE n.semantic_id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT subsubsection_semantic_id_unique IF NOT EXISTS
            FOR (n:Subsubsection)
            REQUIRE n.semantic_id IS UNIQUE
            """
        ]
        with self.driver.session(database=self.database) as session:
            for q in queries:
                session.run(q)

        print("[INFO] Neo4j schema ready")

    def upsert_document_node(self, document_id: str, file_path: str, discipline: Optional[str] = None):
        file_name = os.path.basename(file_path)
        spec_type = infer_spec_type(file_path)
        discipline = normalize_discipline(discipline)
        document_label = get_document_primary_label(spec_type)
        secondary_label = get_document_secondary_label(spec_type, discipline)
        secondary_label_set = f"SET d:{secondary_label}" if secondary_label else ""

        props = {
            "semantic_id": document_id,
            "file_name": file_name,
            "source": file_path,
            "spec_type": spec_type,
            "discipline": discipline,
        }

        query = f"""
        MERGE (d:{document_label} {{semantic_id: $document_id}})
        {secondary_label_set}
        SET d = $props
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, document_id=document_id, props=props)

    def purge_document_subgraph(self, document_id: str):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (ra:RationaleChunk)-[:RATIONALE_FOR]->(fs:FSChunk)-[:PART_OF]->(d {semantic_id: $document_id})
                DETACH DELETE ra
                """,
                document_id=document_id,
            )

            session.run(
                """
                MATCH (n)-[:PART_OF]->(d {semantic_id: $document_id})
                DETACH DELETE n
                """,
                document_id=document_id,
            )

            session.run(
                """
                MATCH (d {semantic_id: $document_id})-[:HAS_CHAPTER|HAS_SECTION|HAS_SUBSECTION|HAS_SUBSUBSECTION*1..]->(h)
                OPTIONAL MATCH (related)-[:PART_OF]->(h)
                WITH collect(DISTINCT h) AS hierarchy_nodes, collect(DISTINCT related) AS related_nodes
                UNWIND hierarchy_nodes + related_nodes AS n
                WITH DISTINCT n
                WHERE n IS NOT NULL
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

    def _build_fschunk_row(self, document_id: str, chunk) -> Optional[Dict[str, Any]]:
        file_name = safe_text(chunk.metadata.get("file_name"))
        chunk_mode = safe_text(chunk.metadata.get("chunk_mode"))
        primary_requirement_id = safe_text(chunk.metadata.get("primary_requirement_id"))
        related_requirement_ids = chunk.metadata.get("related_requirement_ids", [])
        pages = chunk.metadata.get("pages", [])
        chunk_type = safe_text(chunk.metadata.get("type", "FS"))
        discipline = normalize_discipline(chunk.metadata.get("discipline"))
        section_tag = safe_text(chunk.metadata.get("section_tag"))
        chunk_name = safe_text(chunk.metadata.get("name"))
        section_info = parse_section_hierarchy(section_tag)

        if not isinstance(pages, list):
            pages = []

        if not isinstance(related_requirement_ids, list):
            related_requirement_ids = []

        if chunk_mode == "section":
            section_text = clean_chunk_text(chunk.page_content)
            if not file_name or not section_tag or not section_text:
                return None

            return {
                "name": chunk_name or make_section_chunk_name(file_name, section_tag),
                "primary_requirement_id": None,
                "related_requirement_ids": [],
                "related_fschunk_names": [],
                "embedding": embed(section_text),
                "text": section_text,
                "pages": pages,
                "type": chunk_type,
                "discipline": discipline,
                "section_tag": section_tag,
                "section_node_semantic_id": make_section_node_semantic_id(document_id, section_info["section_prefix"]) if section_info else None,
                "chunk_mode": "section",
            }

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
            "discipline": discipline,
            "section_tag": section_tag,
            "section_node_semantic_id": make_section_node_semantic_id(document_id, section_info["section_prefix"]) if section_info else None,
            "chunk_mode": "requirement",
        }
    
    def _build_tschunk_row(self, document_id: str, chunk, seq: int) -> Optional[Dict[str, Any]]:
        file_name = safe_text(chunk.metadata.get("file_name"))
        chunk_mode = safe_text(chunk.metadata.get("chunk_mode"))
        primary_requirement_id = safe_text(chunk.metadata.get("primary_requirement_id"))
        raw_ids = chunk.metadata.get("requirement_ids", [])
        pages = chunk.metadata.get("pages", [])
        chunk_type = safe_text(chunk.metadata.get("type", "TS"))
        discipline = normalize_discipline(chunk.metadata.get("discipline"))
        section_tag = safe_text(chunk.metadata.get("section_tag"))
        chunk_name = safe_text(chunk.metadata.get("name"))
        section_info = parse_section_hierarchy(section_tag)

        if not isinstance(pages, list):
            pages = []

        if not isinstance(raw_ids, list):
            raw_ids = []

        if chunk_mode == "section":
            section_text = clean_chunk_text(chunk.page_content)
            if not file_name or not section_tag or not section_text:
                return None

            return {
                "name": chunk_name or make_section_chunk_name(file_name, section_tag),
                "primary_requirement_id": None,
                "referred_fs_ids": [],
                "embedding": embed(section_text),
                "text": section_text,
                "pages": pages,
                "type": chunk_type,
                "discipline": discipline,
                "section_tag": section_tag,
                "section_node_semantic_id": make_section_node_semantic_id(document_id, section_info["section_prefix"]) if section_info else None,
                "chunk_mode": "section",
            }

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
            "discipline": discipline,
            "section_tag": section_tag,
            "section_node_semantic_id": make_section_node_semantic_id(document_id, section_info["section_prefix"]) if section_info else None,
            "chunk_mode": "requirement",
        }

    def _build_rationale_row(self, document_id: str, chunk, spec_kind: str, seq: Optional[int] = None) -> Optional[Dict[str, Any]]:
        file_name = safe_text(chunk.metadata.get("file_name"))
        primary_requirement_id = safe_text(chunk.metadata.get("primary_requirement_id"))
        pages = chunk.metadata.get("pages", [])
        chunk_type = safe_text(chunk.metadata.get("type", spec_kind))
        discipline = normalize_discipline(chunk.metadata.get("discipline"))
        section_tag = safe_text(chunk.metadata.get("section_tag"))
        section_info = parse_section_hierarchy(section_tag)

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
            "discipline": discipline,
            "section_tag": section_tag,
            "section_node_semantic_id": make_section_node_semantic_id(document_id, section_info["section_prefix"]) if section_info else None,
        }

    def _build_section_hierarchy_rows(self, document_id: str, file_path: str, chunks: List[Any]) -> List[Dict[str, Any]]:
        rows_by_id: Dict[str, Dict[str, Any]] = {}
        toc_entries: Dict[str, Dict[str, Any]] = {}

        if file_path.lower().endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                toc_entries = extract_toc_entries_from_pdf(pdf)

        if toc_entries:
            ordered_entries = sorted(
                toc_entries.values(),
                key=lambda entry: (section_prefix_sort_key(entry["number"]), entry["level"], entry["page"]),
            )

            for entry in ordered_entries:
                section_info = parse_section_hierarchy(entry["full_text"])
                if not section_info:
                    continue

                semantic_id = make_section_node_semantic_id(document_id, section_info["section_prefix"])
                if semantic_id in rows_by_id:
                    continue

                rows_by_id[semantic_id] = {
                    "semantic_id": semantic_id,
                    "name": safe_text(section_info["section_title"]) or safe_text(entry["title"]) or entry["full_text"],
                    "prefix": section_info["section_prefix"],
                    "hierarchy_level": section_info["hierarchy_level"],
                    "parent_semantic_id": make_section_node_semantic_id(
                        document_id,
                        ".".join(section_info["section_prefix"].split(".")[:-1]),
                    ) if section_info["hierarchy_level"] != "1" else None,
                }

            return list(rows_by_id.values())

        for ch in chunks:
            section_tag = safe_text(ch.metadata.get("section_tag"))
            section_info = parse_section_hierarchy(section_tag)
            if not section_info:
                continue

            semantic_id = make_section_node_semantic_id(document_id, section_info["section_prefix"])
            if semantic_id in rows_by_id:
                continue

            rows_by_id[semantic_id] = {
                "semantic_id": semantic_id,
                "name": safe_text(section_info["section_title"]) or section_tag,
                "prefix": section_info["section_prefix"],
                "hierarchy_level": section_info["hierarchy_level"],
                "parent_semantic_id": make_section_node_semantic_id(
                    document_id,
                    ".".join(section_info["section_prefix"].split(".")[:-1]),
                ) if section_info["hierarchy_level"] != "1" else None,
            }

        return list(rows_by_id.values())

    def import_section_hierarchy(self, document_id: str, file_path: str, chunks: List[Any], batch_size: int = 100):
        rows = self._build_section_hierarchy_rows(document_id, file_path, chunks)
        if not rows:
            print("[INFO] No section hierarchy rows to import")
            return

        level_configs = [
            ("1", "Chapter", "HAS_CHAPTER"),
            ("2", "Section", "HAS_SECTION"),
            ("3", "Subsection", "HAS_SUBSECTION"),
            ("4", "Subsubsection", "HAS_SUBSUBSECTION"),
        ]

        with self.driver.session(database=self.database) as session:
            for level, label, rel_type in level_configs:
                level_rows = [row for row in rows if row["hierarchy_level"] == level]
                if not level_rows:
                    continue

                if level == "1":
                    query = f"""
                    UNWIND $rows AS row
                    MERGE (n:{label} {{semantic_id: row.semantic_id}})
                    SET n = {{
                        semantic_id: row.semantic_id,
                        name: row.name,
                        prefix: row.prefix,
                        hierarchy_level: row.hierarchy_level,
                    }}
                    WITH n
                    MATCH (d {{semantic_id: $document_id}})
                    MERGE (d)-[:{rel_type}]->(n)
                    """
                else:
                    query = f"""
                    UNWIND $rows AS row
                    MATCH (parent {{semantic_id: row.parent_semantic_id}})
                    MERGE (n:{label} {{semantic_id: row.semantic_id}})
                    SET n = {{
                        semantic_id: row.semantic_id,
                        name: row.name,
                        prefix: row.prefix,
                        hierarchy_level: row.hierarchy_level,
                    }}
                    MERGE (parent)-[:{rel_type}]->(n)
                    """

                for i in range(0, len(level_rows), batch_size):
                    batch = level_rows[i:i + batch_size]
                    session.run(query, rows=batch, document_id=document_id)
                    print(
                        f"[INFO] Imported {label} batch {i + 1} - {i + len(batch)} / {len(level_rows)}"
                    )

    def import_fschunks(self, document_id: str, chunks: List[Any], batch_size: int = 100):
        rows = []
        for ch in chunks:
            row = self._build_fschunk_row(document_id, ch)
            if row is not None:
                rows.append(row)

        if not rows:
            print("[INFO] No FSChunk rows to import")
            return

        query = """
        UNWIND $rows AS row
        MERGE (c:FSChunk {name: row.name})
        MATCH (section {semantic_id: row.section_node_semantic_id})
        SET c = {
            name: row.name,
            primary_requirement_id: row.primary_requirement_id,
            related_requirement_ids: row.related_requirement_ids,
            embedding: row.embedding,
            text: row.text,
            pages: row.pages,
            type: row.type,
            discipline: row.discipline,
            section_tag: row.section_tag,
            chunk_mode: row.chunk_mode
        }
        WITH c, section
        MERGE (c)-[:PART_OF]->(section)
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
            row = self._build_tschunk_row(document_id, ch, seq=seq)
            if row is not None:
                rows.append(row)
                seq += 1

        if not rows:
            print("[INFO] No TSChunk rows to import")
            return

        with self.driver.session(database=self.database) as session:
            for discipline in [None, "ESW", "HW"]:
                discipline_rows = [row for row in rows if row.get("discipline") == discipline]
                if not discipline_rows:
                    continue

                discipline_label = get_chunk_secondary_label("TSChunk", discipline)
                discipline_label_set = f"SET c:{discipline_label}" if discipline_label else ""
                query = f"""
                UNWIND $rows AS row
                MERGE (c:TSChunk {{name: row.name}})
                MATCH (section {{semantic_id: row.section_node_semantic_id}})
                {discipline_label_set}
                SET c = {{
                    name: row.name,
                    primary_requirement_id: row.primary_requirement_id,
                    referred_fs_ids: row.referred_fs_ids,
                    embedding: row.embedding,
                    text: row.text,
                    pages: row.pages,
                    type: row.type,
                    discipline: row.discipline,
                    section_tag: row.section_tag,
                    chunk_mode: row.chunk_mode
                }}
                WITH c, section
                MERGE (c)-[:PART_OF]->(section)
                """

                for i in range(0, len(discipline_rows), batch_size):
                    batch = discipline_rows[i:i + batch_size]
                    session.run(query, rows=batch, document_id=document_id)
                    print(f"[INFO] Imported TSChunk batch {i + 1} - {i + len(batch)} / {len(discipline_rows)} ({discipline or 'GENERIC'})")

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
                row = self._build_rationale_row(document_id, ch, spec_kind="TS", seq=seq)
                if row is not None:
                    rows.append(row)
                seq += 1
            else:
                row = self._build_rationale_row(document_id, ch, spec_kind="FS")
                if row is not None:
                    rows.append(row)

        if not rows:
            print(f"[INFO] No {spec_kind} RationaleChunk rows to import")
            return

        query = f"""
        UNWIND $rows AS row
        MATCH (p:{parent_label} {{name: row.parent_chunk_name}})
        MATCH (section {{semantic_id: row.section_node_semantic_id}})
        MERGE (ra:RationaleChunk:{get_rationale_label(spec_kind)} {{name: row.name}})
        SET ra = {{
            name: row.name,
            embedding: row.embedding,
            text: row.text,
            pages: row.pages,
            type: row.type,
            section_tag: row.section_tag
        }}
        MERGE (ra)-[:RATIONALE_FOR]->(p)
        MERGE (ra)-[:PART_OF]->(section)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch, document_id=document_id)
                print(f"[INFO] Imported {spec_kind} RationaleChunk batch {i + 1} - {i + len(batch)} / {len(rows)}")

            if spec_kind == "TS":
                for discipline in ["ESW", "HW"]:
                    discipline_rows = []
                    for row in rows:
                        discipline_label = get_rationale_discipline_label(spec_kind, row.get("discipline"))
                        if discipline_label == f"{discipline}RationaleChunk":
                            discipline_rows.append(row)

                    if not discipline_rows:
                        continue

                    label_query = f"""
                    UNWIND $rows AS row
                    MATCH (ra:RationaleChunk:TSRationaleChunk {{name: row.name}})
                    SET ra:{discipline}RationaleChunk
                    """
                    session.run(label_query, rows=discipline_rows)
                    print(f"[INFO] Labeled {len(discipline_rows)} TS rationale nodes as {discipline}RationaleChunk")

    def print_rationale_label_counts(self):
        query = """
        MATCH (ra:RationaleChunk)
        WITH
            count(ra) AS total_count,
            count(CASE WHEN ra:FSRationaleChunk THEN 1 END) AS fs_count,
            count(CASE WHEN ra:TSRationaleChunk THEN 1 END) AS ts_count,
            count(CASE WHEN ra:ESWRationaleChunk THEN 1 END) AS esw_count,
            count(CASE WHEN ra:HWRationaleChunk THEN 1 END) AS hw_count
        RETURN total_count, fs_count, ts_count, esw_count, hw_count
        """

        with self.driver.session(database=self.database) as session:
            record = session.run(query).single()

        if not record:
            print("[INFO] No RationaleChunk nodes found")
            return

        print("[RATIONALE LABEL COUNTS]")
        print(f"RationaleChunk: {record['total_count']}")
        print(f"FSRationaleChunk: {record['fs_count']}")
        print(f"TSRationaleChunk: {record['ts_count']}")
        print(f"ESWRationaleChunk: {record['esw_count']}")
        print(f"HWRationaleChunk: {record['hw_count']}")

    def import_related_links(self, chunks: List[Any], batch_size: int = 100):
        rows = []

        for ch in chunks:
            if safe_text(ch.metadata.get("chunk_mode")) != "requirement":
                continue
            file_name = safe_text(ch.metadata.get("file_name"))
            row = self._build_fschunk_row(make_document_id(file_name) if file_name else "", ch)
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
            if safe_text(ch.metadata.get("chunk_mode")) != "requirement":
                continue
            file_name = safe_text(ch.metadata.get("file_name"))
            row = self._build_tschunk_row(make_document_id(file_name) if file_name else "", ch, seq=seq)
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

        self.import_section_hierarchy(document_id, file_path, chunks, batch_size=batch_size)
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


    def refresh_ts_document(
        self,
        ts_file_path: str,
        fs_file_path: str,
        ts_chunks: List[Any],
        discipline: Optional[str] = None,
        batch_size: int = 100,
    ):
        document_id = make_document_id(ts_file_path)
        discipline = normalize_discipline(discipline)

        self.upsert_document_node(document_id, ts_file_path, discipline=discipline)
        self.purge_document_subgraph(document_id)

        self.import_section_hierarchy(document_id, ts_file_path, ts_chunks, batch_size=batch_size)
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

        print(f"[INFO] Refreshed TS document: {document_id} ({discipline or 'GENERIC'})")

# =========================================================
# MAIN
# =========================================================
def main():
    if not os.path.isfile(FS_PATH):
        raise FileNotFoundError(f"FS file not found: {FS_PATH}")

    if not os.path.isfile(ESW_TS_PATH):
        raise FileNotFoundError(f"TS file not found: {ESW_TS_PATH}")

    if not os.path.isfile(HW_TS_PATH):
        raise FileNotFoundError(f"TS file not found: {HW_TS_PATH}")

    if DEBUG_HEADER_FOOTER and FS_PATH.lower().endswith(".pdf"):
        print_pdf_header_footer_debug(FS_PATH, max_pages=5)

    if DEBUG_HEADER_FOOTER and ESW_TS_PATH.lower().endswith(".pdf"):
        print_pdf_header_footer_debug(ESW_TS_PATH, max_pages=5)

    if DEBUG_HEADER_FOOTER and HW_TS_PATH.lower().endswith(".pdf"):
        print_pdf_header_footer_debug(HW_TS_PATH, max_pages=5)

    fs_chunks = prepare_spec_chunks(FS_PATH)
    esw_ts_chunks = prepare_spec_chunks(ESW_TS_PATH, discipline="ESW")
    hw_ts_chunks = prepare_spec_chunks(HW_TS_PATH, discipline="HW")

    print_section_tag_debug(fs_chunks, "FS")
    print_section_tag_debug(esw_ts_chunks, "ESW TS")
    print_section_tag_debug(hw_ts_chunks, "HW TS")

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
        print("[DEBUG] ESW TS REQUIREMENT CHUNK PREVIEW")
        print("=" * 120)
        for i, ch in enumerate(esw_ts_chunks[:PREVIEW_CHUNK_COUNT], start=1):
            print(f"\n--- ESW TS CHUNK {i} ---")
            print("METADATA:")
            print(ch.metadata)
            print("RAW CONTENT:")
            print(ch.page_content[:2000])
            print("REQUIREMENT TEXT ONLY:")
            print(extract_requirement_only_text(ch.page_content))

        print("\n" + "=" * 120)
        print("[DEBUG] HW TS REQUIREMENT CHUNK PREVIEW")
        print("=" * 120)
        for i, ch in enumerate(hw_ts_chunks[:PREVIEW_CHUNK_COUNT], start=1):
            print(f"\n--- HW TS CHUNK {i} ---")
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
            ts_file_path=ESW_TS_PATH,
            fs_file_path=FS_PATH,
            ts_chunks=esw_ts_chunks,
            discipline="ESW",
            batch_size=BATCH_SIZE,
        )

        kg_builder.refresh_ts_document(
            ts_file_path=HW_TS_PATH,
            fs_file_path=FS_PATH,
            ts_chunks=hw_ts_chunks,
            discipline="HW",
            batch_size=BATCH_SIZE,
        )

        kg_builder.print_rationale_label_counts()

        print("[DONE]")
    finally:
        kg_builder.close()

if __name__ == "__main__":
    main()
