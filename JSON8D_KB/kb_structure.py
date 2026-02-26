from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from typing import Optional
from dataclasses import asdict,field
from collections import defaultdict

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


# =========================================================
# Data models
# =========================================================
@dataclass
class MaintenanceTag:
    review_status: str        # reviewed | pending | rejected
    version: str              # e.g. V1
    last_updated: str
    supersedes: Optional[str] # previous ID


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
    version: Optional[int]


@dataclass
class Sentence:
    id: str
    text: str
    source_section: str
    case_id: str
    annotations: Dict[str, Any]

    failure_id: str = ""
    source_type: str = "8D"
    cause_id: Optional[str] = None
    sentence_role: str = ""
    #is_activate: bool = True # Keep the invalid sentences
    productPnID: Optional[int] = None
    product_domain: Optional[str] = None
    released_year: Optional[int] = None
 


@dataclass
class EightDSemanticNode:
    """
    Unique semantic concept extracted from 8D.
    THIS is embeddable.
    """

    semantic_id: str                  # unique ID
    field_type: str                   # element | mode | effect | cause

    text: str                         # normalized semantic text

    # All 8D failures that map to this concept
    failure_ids: List[str] = field(default_factory=list)
    source_type: List[str] = field(default_factory=list)

    # bookkeeping
    source_count: int = 0

@dataclass
class EightDFailureEntity:
    """
    One 8D failure record.
    NOT embeddable.
    """

    # ===== identifiers =====
    failure_id: str
    file_name:str

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

    # cause_ids: List[Dict[str, Any]] = field(default_factory=list)

    # ===== 8D-specific =====
    status: Optional[str] = None
    discipline: Optional[str] = None


    # ===== evidence =====
    supporting_sentence_ids: List[str] = field(default_factory=list)

    # ===== process / context =====
    fmea_type: Optional[str] = None
    source_type: str = "8D"

    # ===== product =====
    productPnID: Optional[int] = None
    product_domain: Optional[str] = None
    released_year: Optional[int] = None


# =========================================================
# Failure evaluation (used during ingest)
# =========================================================

def evaluate_failure(
    sentences: List[Sentence],
    min_faithful: int = 95,
    allow_levels=("support", "suspect"),
) -> str:
    """
    Decide whether a failure is supported or hypothesis
    based ONLY on failure-level sentences.
    """
    for s in sentences:
        ann = s.annotations or {}
        if s.source_section != "D2":
            continue
        if ann.get("assertion_level") in allow_levels and int(
            ann.get("faithful_score", 0)
        ) >= min_faithful:
            return "supported"
    return "hypothesis"


# =========================================================
# Sentence KB (facts only)
# =========================================================

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

        # self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        #     model_name="all-MiniLM-L6-v2"
        # )
        self.embedder = BGEEmbeddingFunction("BAAI/bge-base-en-v1.5", normalize=True)
        self.collection = self.client.get_or_create_collection(
            name="sentences",
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        sentence: Sentence,
        failure_id: str,
        sentence_role: str,
        cause_id: Optional[str] = None,
    ):
        meta = {
            "case_id": sentence.case_id,
            "failure_id": failure_id,
            "cause_id": cause_id or "",
            "sentence_role": sentence_role,
            "source_section": sentence.source_section,
            "status": sentence.annotations.get("status"),
            "subject": sentence.annotations.get("subject"),
            "faithful_score": int(sentence.annotations.get("faithful_score", 0)),
            "productPnID": sentence.productPnID,     
            "product_domain": sentence.product_domain,
            "released_year":sentence.released_year
        }

        # -------------------------------
        # 1) Chroma upsert
        # -------------------------------
        self.collection.upsert(
            ids=[sentence.id],
            documents=[sentence.text],
            metadatas=[meta],
        )

        # -------------------------------
        # 2) JSON store upsert
        # -------------------------------
        record = {
            "id": sentence.id,
            "text": sentence.text,
            "case_id": sentence.case_id,
            "source_section": sentence.source_section,
            "failure_id": failure_id,
            "cause_id": cause_id,
            "sentence_role": sentence_role,
            "annotations": sentence.annotations or {},
            "meta": meta,
        }

        old = self.store.get(sentence.id)
        if old != record:
            self.store[sentence.id] = record
            self.store_path.write_text(
                json.dumps(self.store, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def get_by_ids(self, ids: List[str]) -> List[Sentence]:
        if not ids:
            return []

        res = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        sentences = []
        for sid, text, meta in zip(
            res["ids"], res["documents"], res["metadatas"]
        ):
            sentences.append(
                Sentence(
                    id=sid,
                    text=text,
                    source_section=meta.get("source_section", ""),
                    case_id=meta.get("case_id", ""),
                    annotations={
                        "status": meta.get("status"),
                        "subject": meta.get("subject"),
                        "faithful_score": meta.get("faithful_score"),
                    },
                    failure_id=meta.get("failure_id", ""),
                    cause_id=meta.get("cause_id") or None,
                    sentence_role=meta.get("sentence_role", ""),
                )
            )
        return sentences
    
    def search(
        self,
        *,
        query: str,
        failure_id: str | None = None,
        cause_id: str | None = None,
        roles: list[str] | None = None,
        k: int = 5,
    ):
        filters = []

        if failure_id:
            filters.append({"failure_id": failure_id})

        if cause_id is not None:
            # Optional filter
            filters.append({"cause_id": cause_id})

        if roles:
            filters.append({"sentence_role": {"$in": roles}})

        if not filters:
            where = None
        elif len(filters) == 1:
            where = filters[0]
        else:
            where = {"$and": filters}

        return self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
    def get_record(self, sentence_id: str) -> Optional[dict]:
        return self.store.get(sentence_id) 
    def list_records(self) -> list[dict]: 
        return list(self.store.values())
    
    def list_by_failure(
        self,
        failure_id: str,
        roles: Optional[list[str]] = None,
        *,
        include_cause_id: bool | None = None,
    ) -> list[dict]:
        """
        List sentence records from JSON store by failure_id.

        Args:
            failure_id: target failure id
            roles: optional role filter, e.g. ["failure_sentence", "cause_sentence"]
            include_cause_id:
                - True: only sentences that have cause_id
                - False: only sentences that have no cause_id
                - None: no filter
        """
        out: list[dict] = []
        for rec in self.store.values():
            if rec.get("failure_id") != failure_id:
                continue

            if roles is not None and rec.get("sentence_role") not in roles:
                continue

            cid = rec.get("cause_id")
            has_cause = cid is not None and cid != ""
            if include_cause_id is True and not has_cause:
                continue
            if include_cause_id is False and has_cause:
                continue

            out.append(rec)
        return out

    def list_by_case(
        self,
        case_id: str,
        roles: Optional[list[str]] = None,
        *,
        failure_id: Optional[str] = None,
    ) -> list[dict]:
        """
        List sentence records from JSON store by case_id.

        Args:
            case_id: target case id (your file_name)
            roles: optional role filter
            failure_id: optional extra filter within the case
        """
        out: list[dict] = []
        for rec in self.store.values():
            if rec.get("case_id") != case_id:
                continue

            if failure_id is not None and rec.get("failure_id") != failure_id:
                continue

            if roles is not None and rec.get("sentence_role") not in roles:
                continue

            out.append(rec)
        return out


# =========================================================
# Failure KB (entry gate)
# =========================================================

class EightDFailureKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # =========================================================
        # SHARED SEMANTIC STORE  (same as FMEA)
        # =========================================================
        self.field_store_path = self.persist_dir / "fmea_field_store.json"
        self.field_store: Dict[str, dict] = {}

        if self.field_store_path.exists():
            self.field_store = json.loads(
                self.field_store_path.read_text(encoding="utf-8")
            )

        # =========================================================
        # 8D ENTITY STORE (separate)
        # =========================================================
        self.entity_store_path = self.persist_dir / "entity_store.json"
        self.entity_store: Dict[str, dict] = {}

        if self.entity_store_path.exists():
            self.entity_store = json.loads(
                self.entity_store_path.read_text(encoding="utf-8")
            )

        # =========================================================
        # SHARED VECTOR STORE (same collection as FMEA)
        # =========================================================
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        #     model_name="all-MiniLM-L6-v2"
        # )
        self.embedder = BGEEmbeddingFunction("BAAI/bge-base-en-v1.5", normalize=True)
        self.collection = self.client.get_or_create_collection(
            name="failure_semantic_kb",  # same as FMEA
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    # =========================================================
    # SEMANTIC NODE UPSERT (shared)
    # =========================================================
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

        existing = self.field_store.get(semantic_id)

        if existing:
            # merge failure_ids
            merged_ids = set(existing.get("failure_ids", []))
            merged_ids.update(failure_ids)
            failure_ids_unique = sorted(merged_ids)

            # merge source_type
            existing_sources = existing.get("source_type", [])
            if isinstance(existing_sources, str):
                existing_sources = [existing_sources]

            merged_sources = sorted(set(existing_sources + [source_type]))

        else:
            failure_ids_unique = sorted(set(failure_ids))
            merged_sources = [source_type]

        count = len(failure_ids_unique)

        self.field_store[semantic_id] = {
            "semantic_id": semantic_id,
            "field_type": field_type,
            "text": text,
            "failure_ids": failure_ids_unique,
            "count": count,
            "source_type": merged_sources,
        }

        self.field_store_path.write_text(
            json.dumps(self.field_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # unified vector store
        self.collection.upsert(
            ids=[semantic_id],
            documents=[text],
            metadatas=[{
                "field_type": field_type,
                "count": count,
                "source_type": ",".join(merged_sources)
            }],
        )
    # =========================================================
    # 8D FAILURE ENTITY UPSERT
    # =========================================================
    def upsert_failure_entity(self, entity: EightDFailureEntity):
        def make_failure_uid(failure_id: str, cause_id: str | None) -> str:
            return f"{failure_id}__{cause_id}" if cause_id else failure_id
        uid = make_failure_uid(entity.failure_id, getattr(entity, "cause_id", None))
        self.entity_store[uid] = asdict(entity)

        self.entity_store_path.write_text(
            json.dumps(self.entity_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )




