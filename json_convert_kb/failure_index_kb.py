from dataclasses import dataclass, asdict, is_dataclass
from typing import Dict, Any, List
from pathlib import Path
import json
import chromadb
from chromadb.utils import embedding_functions
import math


# ============================================================
# Keyword helpers
# ============================================================

MECHANISM_KEYWORDS = [
    "current",
    "gate",
    "voltage",
    "duty",
    "overcurrent",
    "inrush",
    "mosfet",
    "return to",
    "doesn't return",
    "clip",
]


def looks_like_mechanism(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in MECHANISM_KEYWORDS)

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ============================================================
# Data models
# ============================================================

@dataclass
class Sentence:
    id: str
    text: str
    source_section: str
    annotations: Dict[str, Any]
    case_id: str


@dataclass
class FailureIndex:
    failure_id: str
    failure_element: str
    failure_mode: str
    status: str
    supporting_sentence_ids: List[str]
    case_id: str
    root_causes: List[Dict[str, Any]]
    extra_sentence_ids: List[str]  


# ============================================================
# Failure evaluation
# ============================================================

def evaluate_failure(
    sentences: List[Sentence],
    min_faithful: int = 90,
    allow_levels=("observed", "confirmed"),
) -> str:
    for s in sentences:
        if s.source_section != "D2":
            continue
        ann = s.annotations or {}
        if ann.get("assertion_level") in allow_levels and int(
            ann.get("faithful_score", 0)
        ) >= min_faithful:
            return "supported"
    return "hypothesis"


# ============================================================
# Sentence KB
# ============================================================

class SentenceKB:
    def __init__(self, persist_dir):
        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="sentences",
            embedding_function=self.embedder,
        )

    def add(
        self,
        sentence: Sentence,
        failure_id: str,
        sentence_role: str,
        cause_id: str | None = None,
    ):
        self.collection.add(
            ids=[sentence.id],
            documents=[sentence.text],
            metadatas=[
                {
                    "case_id": sentence.case_id,
                    "failure_id": failure_id,
                    "cause_id": cause_id or "",
                    "sentence_role": sentence_role,
                    "source_section": sentence.source_section,
                    "entity_type": sentence.annotations.get("entity_type"),
                    "assertion_level": sentence.annotations.get("assertion_level"),
                    "faithful_score": int(
                        sentence.annotations.get("faithful_score", 0)
                    ),
                }
            ],
        )

    # --------------------------------------------------------
    # 🔑 D4 candidate search
    # --------------------------------------------------------
    def search_d4_candidates(
        self,
        query_text: str,
        case_id: str,
        exclude_ids: set[str],
        top_k: int = 5,
    ):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where={
                "$and": [
                    {"case_id": case_id},
                    {"source_section": "D4"},
                ]
            },
        )

        filtered = []
        for sid, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            if sid in exclude_ids:
                continue
            filtered.append((sid, doc, meta))
        return filtered


# ============================================================
# Failure Index KB
# ============================================================

class FailureIndexKB:
    def __init__(self, store_path):
        self.store_path = Path(store_path).resolve()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        self.failures = {}

        if self.store_path.exists():
            with self.store_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self.failures[obj["failure_id"]] = obj

    def add(self, failure):
        if is_dataclass(failure):
            obj = asdict(failure)
        elif isinstance(failure, dict):
            obj = failure
        else:
            obj = failure.__dict__

        fid = obj["failure_id"]
        self.failures[fid] = obj

        with self.store_path.open("w", encoding="utf-8") as f:
            for item in self.failures.values():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# Ingest logic (核心)
# ============================================================

def ingest_failure_record(
    record: dict,
    sentence_kb: SentenceKB,
    failure_kb: FailureIndexKB,
) -> FailureIndex:

    # -------- case id --------
    docs = record.get("documents", [])
    if not docs:
        raise ValueError("Missing documents in 8D JSON")
    case_id = docs[0].get("file_name")

    failure = record["failure"]

    extra_sentence_ids: set[str] = set()

    # ========================================================
    # 1) Failure sentences
    # ========================================================
    failure_sentences: List[Sentence] = []
    failure_sentence_ids: List[str] = []

    for ent in failure.get("supporting_entities", []):
        s = Sentence(
            id=ent["id"],
            text=ent["text"],
            source_section=ent.get("source_section", ""),
            case_id=case_id,
            annotations=ent.get("annotations", {}) or {},
        )
        failure_sentences.append(s)
        failure_sentence_ids.append(s.id)

        sentence_kb.add(
            sentence=s,
            failure_id=failure.get("failure_ID"),
            sentence_role="failure_sentence",
        )

    status = evaluate_failure(failure_sentences)

    # ========================================================
    # 2) Collect iteration2 cause sentence IDs
    # ========================================================
    cause_primary_ids = set()
    for cause in failure.get("root_causes", []):
        for ent in cause.get("supporting_entities", []):
            cause_primary_ids.add(ent["id"])

    # ========================================================
    # 3) Root causes + cause similarity expansion
    # ========================================================
    root_cause_indexes: List[Dict[str, Any]] = []

    for cause in failure.get("root_causes", []):
        cause_id = cause.get("cause_ID", "")
        cause_supporting_ids: List[str] = []

        # -------- primary cause sentences (anchors) --------
        cause_anchor_sentences: List[Sentence] = []

        for ent in cause.get("supporting_entities", []):
            s = Sentence(
                id=ent["id"],
                text=ent["text"],
                source_section=ent.get("source_section", ""),
                case_id=case_id,
                annotations=ent.get("annotations", {}) or {},
            )
            cause_anchor_sentences.append(s)
            cause_supporting_ids.append(s.id)

            sentence_kb.add(
                sentence=s,
                failure_id=failure.get("failure_ID"),
                sentence_role="cause_sentence",
                cause_id=cause_id,
            )

        # -------- 🔑 expand cause support from D4 by cause similarity --------
        cause_extra_ids = set(cause_supporting_ids)

                # build lookup
        selected_sentences = {
            s["id"]: s for s in record.get("selected_sentences", [])
        }

        for cs in cause_anchor_sentences:
            cs_vec = sentence_kb.embedder([cs.text])[0]

            for sid, sdict in selected_sentences.items():
                if sdict["source_section"] != "D4":
                    continue
                if sid in cause_primary_ids:
                    continue
                if sid in cause_extra_ids:
                    continue

                text = sdict["text"]
                if not looks_like_mechanism(text):
                    continue

                cand_vec = sentence_kb.embedder([text])[0]
                sim = cosine_similarity(cs_vec, cand_vec)

                if sim < 0.3:
                    continue

                # 🔑 这里才是真正新增的 D4 mechanism sentence
                s = Sentence(
                    id=sid,
                    text=text,
                    source_section="D4",
                    case_id=case_id,
                    annotations=sdict.get("annotations", {}),
                )

                sentence_kb.add(
                    sentence=s,
                    failure_id=failure.get("failure_ID"),
                    sentence_role="cause_sentence",
                    cause_id=cause_id,
                )

                cause_supporting_ids.append(sid)
                cause_extra_ids.add(sid)
                extra_sentence_ids.add(sid)

        root_cause_indexes.append(
            {
                "cause_id": cause_id,
                "cause_level": cause.get("cause_level", ""),
                "failure_cause": cause.get("failure_cause", ""),
                "failure_mechanism": cause.get("failure_mechanism", ""),
                "discipline_type": cause.get("discipline_type", ""),
                "confidence": cause.get("confidence", ""),
                "inferred_insight": cause.get("inferred_insight", ""),
                "supporting_sentence_ids": cause_supporting_ids,
            }
        )

    # ========================================================
    # 4) Failure index
    # ========================================================
    f_index = FailureIndex(
        failure_id=failure["failure_ID"],
        failure_element=failure.get("failure_element", ""),
        failure_mode=failure.get("failure_mode", ""),
        status=status,
        supporting_sentence_ids=failure_sentence_ids,
        case_id=case_id,
        root_causes=root_cause_indexes,
        extra_sentence_ids=sorted(extra_sentence_ids), 
    )

    failure_kb.add(f_index)
    return f_index


# ============================================================
# Example runner
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
KB_ROOT = (BASE_DIR.parent / "kb").resolve()
SENTENCE_KB_DIR = (KB_ROOT / "sentence_kb").resolve()
FAILURE_INDEX_PATH = (KB_ROOT / "failure_index_kb.jsonl").resolve()

JSON_ROOT = (BASE_DIR.parent / "eightD_json_raw").resolve()
json_path = (JSON_ROOT / "8D6318110147R01.json").resolve()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sentence_kb = SentenceKB(persist_dir=SENTENCE_KB_DIR)
failure_kb = FailureIndexKB(store_path=FAILURE_INDEX_PATH)

f_index = ingest_failure_record(data, sentence_kb, failure_kb)
print(f_index)
print("COUNT AFTER INGEST:", sentence_kb.collection.count())
