from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Literal,Iterable

from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

from dataclasses import asdict
from collections import defaultdict
from dataclasses import asdict, is_dataclass, field

FilterValue = Union[str, List[str]]
DStage = Literal["D2", "D4"]
FailureFieldType = Literal["element", "mode", "effect", "cause"]


class BGEEmbeddingFunction:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", normalize=True):
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)

    # 
    def __call__(self, input):
        # input: List[str]
        return self.model.encode(
            input,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        ).tolist()

    def name(self) -> str:
        return f"st::{self.model_name}::norm={self.normalize}"

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
    """
    Unified embeddable semantic unit
    Used for both 8D and FMEA.
    """
    failure_id: str
    # ---- main semantic text (what gets embedded) ----
    text: str
    # ---- semantic classification ----
    source_type: str            # 8D | old_fmea | new_fmea
    # ---- structural traceability ----
    product_domain: Optional[str] = None




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
    file_name: str

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

    same_id: Optional[List[str]] = field(default_factory=list)

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
    

class SentenceKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.store_path = self.persist_dir / "sentence_store.json"
        self.store: dict[str, dict] = {}
        if self.store_path.exists():
            self.store = json.loads(
                self.store_path.read_text(encoding="utf-8")
            )

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        self.embedder = BGEEmbeddingFunction("BAAI/bge-base-en-v1.5", normalize=True)

        self.collection = self.client.get_or_create_collection(
            name="sentences",
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )
    def add_sentence(self, sentence_id: str, sentence: Sentence, *, overwrite: bool = False) -> None:
        """
        Add a sentence into:
        1) JSON store (structured truth)
        2) Chroma collection (vector index)

        overwrite:
            - False: skip if exists
            - True: replace existing (store + chroma)
        """
        exists = sentence_id in self.store

        if exists and not overwrite:
            return

        # If overwriting, remove old vector first to avoid duplicates
        if exists and overwrite:
            try:
                self.collection.delete(ids=[sentence_id])
            except Exception:
                # tolerate if not present in chroma for some reason
                pass

        # ---- update structured store ----
        self.store[sentence_id] = asdict(sentence)

        # ---- persist structured store immediately ----
        self.store_path.write_text(
            json.dumps(self.store, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # ---- add to vector DB ----
        # (Chroma will embed sentence.text via embedding_function)
        self.collection.add(
            ids=[sentence_id],
            documents=[sentence.text],
            metadatas=[{
                "failure_id": sentence.failure_id,
                "source_type": sentence.source_type,
                "product_domain": sentence.product_domain,
            }]
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

        self.entity_store_path = self.persist_dir / "entity_store.json"
        self.entity_store: dict[str, dict] = {}
        if self.entity_store_path.exists():
            self.entity_store = json.loads(self.entity_store_path.read_text(encoding="utf-8"))

        self.edge_store_path = self.persist_dir / "fmea_edge_store.json"
        self.edge_store: Dict[str, Dict[str, Dict[str, int]]] = {}
        if self.edge_store_path.exists():
            self.edge_store = json.loads(self.edge_store_path.read_text(encoding="utf-8"))
        else:
            self.edge_store = {
                "mode_to_cause": {},
                "mode_to_effect": {},
                "element_to_mode": {},
            }

        # ---------- vector store ----------
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        # self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        #     # model_name="all-MiniLM-L6-v2"
        # )
        self.embedder = BGEEmbeddingFunction("BAAI/bge-base-en-v1.5", normalize=True)
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

        # -------------------------------
        # Take the existing
        # -------------------------------
        existing = self.field_store.get(semantic_id)

        if existing:
            # merge failure_ids
            merged_ids = set(existing.get("failure_ids", []))
            merged_ids.update(failure_ids)
            failure_ids_unique = sorted(merged_ids)

            # merge source_type
            if existing.get("source_type") != source_type:
                existing_source = existing.get("source_type")
                if isinstance(existing_source, list):
                    source_types = set(existing_source)
                else:
                    source_types = {existing_source}
                source_types.add(source_type)
                merged_source_type = list(source_types)
            else:
                merged_source_type = existing.get("source_type")

        else:
            failure_ids_unique = sorted(set(failure_ids))
            merged_source_type = source_type

        count = len(failure_ids_unique)

        # -------------------------------
        # write structured store
        # -------------------------------
        self.field_store[semantic_id] = {
            "semantic_id": semantic_id,
            "field_type": field_type,
            "text": text,
            "failure_ids": failure_ids_unique,
            "count": count,
            "source_type": merged_source_type,
        }

        # -------------------------------
        #  JSON
        # -------------------------------
        self.field_store_path.write_text(
            json.dumps(self.field_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # -------------------------------
        # update vector store
        # -------------------------------

        if isinstance(merged_source_type, list):
            source_type_meta = ",".join(sorted(merged_source_type))
        else:
            source_type_meta = merged_source_type
        self.collection.upsert(
            ids=[semantic_id],
            documents=[text],
            metadatas=[{
                "field_type": field_type,
                "count": count,
                "source_type": source_type_meta
            }],
        )

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
        def _add_edge(src_type: str, src_id: str, tgt_id: str):
            if not src_id or not tgt_id:
                return
            self.edge_store.setdefault(src_type, {})
            self.edge_store[src_type].setdefault(src_id, {})
            self.edge_store[src_type][src_id].setdefault(tgt_id, 0)
            self.edge_store[src_type][src_id][tgt_id] += 1
                    # mode -> cause
        _add_edge("mode_to_cause", entity.mode_id, entity.cause_id)

        # mode -> effect
        _add_edge("mode_to_effect", entity.mode_id, entity.effect_id)

        # element -> mode
        _add_edge("element_to_mode", entity.element_id, entity.mode_id)

        # 3) persist edge store
        self.edge_store_path.write_text(
            json.dumps(self.edge_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
