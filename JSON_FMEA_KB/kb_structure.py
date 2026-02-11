from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Literal,Iterable

from pathlib import Path
import json

import chromadb
from chromadb.utils import embedding_functions

from dataclasses import asdict
from collections import defaultdict
from dataclasses import asdict, is_dataclass, field

FilterValue = Union[str, List[str]]
DStage = Literal["D2", "D4"]
FailureFieldType = Literal["element", "mode", "effect", "cause"]

#======= Helper =========
def is_valid_embed_text(text: Optional[str]) -> bool:
    if text is None:
        return False
    t = text.strip()
    if not t:
        return False
    if t.lower() in {
        "-", "n/a", "na", "null", "none", "tbd", "to be defined"
    }:
        return False
    return True

@dataclass
class FileMeta: #General metadata for a FMEA worksheet
    source_type: str #new_fmea/ old_fmea/ 8D
    released: Optional[str] #released date
    # productId: Optional[int] 
    # productPnId: Optional[int]
    productPnID: Optional[int]
    productName: Optional[str]
    # project_description: Optional[str]
    file_name: str
    product_domain: Optional[str]
    # failure_id: str #FMEA61843..._F1

@dataclass
class Sentence:
    id: str
    text: str
    sentence_role: str          # failure | cause | effect
    source_type: str            # old_fmea | new_fmea | 8d
    file_name: str
    case_id: str
    metadata: Dict[str, Any]


@dataclass
class FailureSemanticNode:
    """
    Unique semantic failure concept.
    THIS is embeddable.
    """

    semantic_id: str                     # unique ID for this semantic node
    field_type: FailureFieldType         # element | mode | effect | cause

    text: str                            # normalized semantic text

    # All failures that map to this semantic concept
    failure_ids: List[str] = field(default_factory=list)
    source_type: str = ""   
    # Optional bookkeeping
    source_count: int = 0



@dataclass
class FailureEntity:
    """
    One FMEA row / record.
    NOT embeddable.
    """

    # ===== identifiers =====
    failure_id: str

    # ===== links to semantic nodes =====
    mode_id: Optional[str]
    element_id: Optional[str]
    effect_id: Optional[str]
    cause_id: Optional[str]

    # ===== original raw text (traceability) =====
    failure_mode_text: Optional[str]
    failure_element_text: Optional[str]
    failure_effect_text: Optional[str]
    failure_cause_text: Optional[str]

    # ===== process context =====
    process_step: Optional[str] = None
    system: Optional[str] = None
    function: Optional[str] = None
    discipline: Optional[str] = None

    # ===== ratings =====
    severity: Optional[float] = None
    occurrence: Optional[float] = None
    detection: Optional[float] = None
    rpn: Optional[float] = None

    # ===== controls & actions =====
    prevention: Optional[str] = None
    detection_method: Optional[str] = None
    recommended_action: Optional[str] = None

    # ===== source info =====
    source_type: str = ""                # Old / New FMEA
    fmea_type: Optional[str] = None

    productPnID: Optional[int] = None
    product_domain: Optional[str] = None
    released_year: Optional[int] = None

class FileMetaStore:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.store_path = self.persist_dir / "file_meta_store.json"
        self.store: dict[str, dict] = {}

        if self.store_path.exists():
            self.store = json.loads(self.store_path.read_text(encoding="utf-8"))

    def add(self, meta: FileMeta):
        new_val = asdict(meta)
        old_val = self.store.get(meta.file_name)

        if old_val == new_val:
            return

        self.store[meta.file_name] = new_val
        self.store_path.write_text(
            json.dumps(self.store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    


class FMEAFailureKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ---------- structured store ----------
        self.field_store_path = self.persist_dir / "fmea_field_store.json"
        self.field_store: Dict[str, dict] = {}
        if self.field_store_path.exists():
            self.field_store = json.loads(self.field_store_path.read_text(encoding="utf-8"))

        self.entity_store_path = self.persist_dir / "fmea_entity_store.json"
        self.entity_store: dict[str, dict] = {}
        if self.entity_store_path.exists():
            self.entity_store = json.loads(self.entity_store_path.read_text(encoding="utf-8"))

        # ---------- vector store ----------
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="failure_semantic_kb",
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------------------------
    # existing: add failure
    # ---------------------------
def upsert_semantic_node(
    self,
    *,
    semantic_id: str,
    field_type: str,
    text: str,
    failure_ids: list[str],
    source_type: str,
) -> None:

    if not is_valid_embed_text(text):
        return

    # =========================================
    # 1️⃣ CHECK EXISTING
    # =========================================
    existing = self.field_store.get(semantic_id)

    if existing:
        # merge failure_ids
        old_ids = existing.get("failure_ids", [])
        merged_ids = list(dict.fromkeys(old_ids + failure_ids))

        existing["failure_ids"] = merged_ids
        existing["count"] = len(merged_ids)

        # optional: merge source_type
        old_source = existing.get("source_type")
        if old_source != source_type:
            existing["source_type"] = "mixed"

        self.field_store[semantic_id] = existing

    else:
        # create new
        merged_ids = list(dict.fromkeys(failure_ids))

        self.field_store[semantic_id] = {
            "semantic_id": semantic_id,
            "field_type": field_type,
            "text": text,
            "failure_ids": merged_ids,
            "count": len(merged_ids),
            "source_type": source_type,
        }

    # =========================================
    # 2️⃣ SAVE JSON
    # =========================================
    self.field_store_path.write_text(
        json.dumps(self.field_store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # =========================================
    # 3️⃣ VECTOR UPSERT
    # =========================================
    # ⚠ 只需要用当前 canonical text
    self.collection.upsert(
        ids=[semantic_id],
        documents=[text],
        metadatas=[{
            "field_type": field_type,
            "count": self.field_store[semantic_id]["count"],
        }],
    )

    # ---------------------------
    # existing: add cause
    # ---------------------------
    def upsert_failure_entity(self, entity):
        """
        entity: FailureEntity
        """
        self.entity_store[entity.failure_id] = asdict(entity)
        self.entity_store_path.write_text(
            json.dumps(self.entity_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
