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
FILE_PATH = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\FS6303220015R11.pdf"

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
    tokens = REQ_TOKEN_PATTERN.findall(text)
    if not tokens:
        return False

    compact = re.sub(r"\s+", "", text)
    compact_tokens = "".join(tokens)
    return compact == compact_tokens


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
):
    if not current_ids or not current_text_lines:
        return

    text = clean_chunk_text("\n".join(current_text_lines))
    if not text:
        return

    chunks.append(
        ChunkDoc(
            page_content=text,
            metadata={
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "pages": pages[:],
                "type": "FS",
                "chunk_mode": "requirement",
                "requirement_ids": current_ids[:],
                "primary_requirement_id": current_ids[0],
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
def extract_fs_chunks_from_pdf(file_path: str) -> List[ChunkDoc]:
    chunks: List[ChunkDoc] = []

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
                    # 如果像同一个 block 的附加 prefix，则并入当前 requirement
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

                    # 否则是真正的新 requirement
                    if not in_requirement_mode:
                        flush_normal_buffer(chunks, normal_lines, file_path, normal_pages)
                        normal_lines = []
                        normal_pages = []
                        in_requirement_mode = True

                    flush_req_buffer(chunks, current_ids, current_req_lines, file_path, current_req_pages)

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
                        flush_req_buffer(chunks, current_ids, current_req_lines, file_path, current_req_pages)

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
                        # 如果上一行是 Rationale，就继续拼到 Rationale
                        if current_req_lines and current_req_lines[-1].startswith("Rationale:"):
                            append_rationale_text(current_req_lines, continuation)
                        else:
                            append_requirement_text(current_req_lines, continuation)

                        if page_no not in current_req_pages:
                            current_req_pages.append(page_no)

                    prev_line_info = line_info
                    continue

        if in_requirement_mode:
            flush_req_buffer(chunks, current_ids, current_req_lines, file_path, current_req_pages)
        else:
            flush_normal_buffer(chunks, normal_lines, file_path, normal_pages)

    print(f"[INFO] Extracted {len(chunks)} chunks from PDF")
    return chunks


# =========================================================
# DOCX PARSER
# =========================================================
def extract_fs_chunks_from_docx(file_path: str) -> List[ChunkDoc]:
    chunks: List[ChunkDoc] = []

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
                                    "type": "FS_section",
                                    "chunk_mode": "normal",
                                    "chunk_index": idx,
                                },
                            )
                        )
                normal_lines = []
                in_requirement_mode = True

            flush_req_buffer(chunks, current_ids, current_req_lines, file_path, [])

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
        flush_req_buffer(chunks, current_ids, current_req_lines, file_path, [])
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
                            "type": "FS_section",
                            "chunk_mode": "normal",
                            "chunk_index": idx,
                        },
                    )
                )

    print(f"[INFO] Extracted {len(chunks)} chunks from DOCX")
    return chunks



# =========================================================
# PREPARE CHUNKS
# =========================================================
def prepare_fs_chunks(file_path: str) -> List[ChunkDoc]:
    lower = file_path.lower()

    if lower.endswith(".pdf"):
        chunks = extract_fs_chunks_from_pdf(file_path)
    elif lower.endswith(".docx"):
        chunks = extract_fs_chunks_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")

    requirement_chunks = [
        ch for ch in chunks
        if ch.metadata.get("chunk_mode") == "requirement"
    ]

    print(f"[INFO] Total requirement chunks ready for KG import: {len(requirement_chunks)}")
    return requirement_chunks


# =========================================================
# KG BUILDER
# =========================================================

def make_document_id(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def make_fschunk_name(file_name: str, requirement_id: str) -> str:
    new_filename = file_name.replace(".pdf", "")
    return f"{new_filename}_{requirement_id}"


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

        props = {
            "semantic_id": document_id,
            "file_name": file_name,
            "source": file_path,
        }

        query = """
        MERGE (d:SpecificationDocument {semantic_id: $document_id})
        SET d = $props
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, document_id=document_id, props=props)

    def purge_document_subgraph(self, document_id: str):
        with self.driver.session(database=self.database) as session:
            # 先删新结构下的 rationale -> FSChunk
            session.run(
                """
                MATCH (ra:RationaleChunk)-[:RATIONALE_FOR]->(fs:FSChunk)-[:PART_OF]->(d:SpecificationDocument {semantic_id: $document_id})
                DETACH DELETE ra
                """,
                document_id=document_id,
            )

            # 再删所有 PART_OF 到该 document 的 chunk
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
        pages = chunk.metadata.get("pages", [])
        chunk_type = safe_text(chunk.metadata.get("type", "FS"))

        if not isinstance(pages, list):
            pages = []

        if not file_name or not primary_requirement_id:
            return None

        if not file_name or not primary_requirement_id:
            return None

        # Clean the file_name by removing ".pdf" if it exists
        cleaned_file_name = file_name.replace(".pdf", "")

        requirement_text = extract_requirement_only_text(chunk.page_content)
        if not requirement_text:
            return None

        return {
            "name": make_fschunk_name(cleaned_file_name, primary_requirement_id),  # Use cleaned file_name here
            "embedding": embed(requirement_text),
            "text": requirement_text,
            "pages": pages,
            "type": chunk_type,
        }

    def _build_rationale_row(self, chunk) -> Optional[Dict[str, Any]]:
        file_name = safe_text(chunk.metadata.get("file_name"))
        primary_requirement_id = safe_text(chunk.metadata.get("primary_requirement_id"))
        pages = chunk.metadata.get("pages", [])
        chunk_type = safe_text(chunk.metadata.get("type", "FS"))

        if not isinstance(pages, list):
            pages = []

        if not file_name or not primary_requirement_id:
            return None

        # Clean the file_name by removing ".pdf" if it exists
        cleaned_file_name = file_name.replace(".pdf", "")

        rationale_text = extract_rationale_only_text(chunk.page_content)
        if not rationale_text:
            return None

        return {
            "name": make_rationale_chunk_name(cleaned_file_name, primary_requirement_id),
            "fschunk_name": make_fschunk_name(cleaned_file_name, primary_requirement_id),
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

    def import_rationale_chunks(self, chunks: List[Any], batch_size: int = 100):
        rows = []
        for ch in chunks:
            row = self._build_rationale_row(ch)
            if row is not None:
                rows.append(row)

        if not rows:
            print("[INFO] No RationaleChunk rows to import")
            return

        query = """
        UNWIND $rows AS row
        MATCH (fs:FSChunk {name: row.fschunk_name})
        MERGE (ra:RationaleChunk {name: row.name})
        SET ra = {
            name: row.name,
            embedding: row.embedding,
            text: row.text,
            pages: row.pages,
            type: row.type
        }
        MERGE (ra)-[:RATIONALE_FOR]->(fs)
        """

        with self.driver.session(database=self.database) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(query, rows=batch)
                print(f"[INFO] Imported RationaleChunk batch {i + 1} - {i + len(batch)} / {len(rows)}")

    def refresh_document(self, file_path: str, chunks: List[Any], batch_size: int = 100):
        document_id = make_document_id(file_path)

        self.upsert_document_node(document_id, file_path)
        self.purge_document_subgraph(document_id)

        # 先导入 FSChunk，再导入 RationaleChunk
        self.import_fschunks(document_id, chunks, batch_size=batch_size)
        self.import_rationale_chunks(chunks, batch_size=batch_size)

        # 清理旧模型残留
        self.cleanup_orphan_requirement_nodes()

        print(f"[INFO] Refreshed document: {document_id}")

# =========================================================
# MAIN
# =========================================================
def main():
    file_path = FILE_PATH

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if DEBUG_HEADER_FOOTER and file_path.lower().endswith(".pdf"):
        print_pdf_header_footer_debug(file_path, max_pages=5)

    chunks = prepare_fs_chunks(file_path)

    if DEBUG_PREVIEW_CHUNKS:
        print("\n" + "=" * 120)
        print("[DEBUG] REQUIREMENT CHUNK PREVIEW")
        print("=" * 120)

        for i, ch in enumerate(chunks[:PREVIEW_CHUNK_COUNT], start=1):
            print(f"\n--- CHUNK {i} ---")
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
        kg_builder.refresh_document(
            file_path=file_path,
            chunks=chunks,
            batch_size=BATCH_SIZE,
        )
        print("[DONE]")
    finally:
        kg_builder.close()

if __name__ == "__main__":
    main()