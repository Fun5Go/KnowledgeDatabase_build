from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import json
from typing import Optional
from dataclasses import asdict,field
from collections import defaultdict


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
    cause_id: Optional[str] = None
    sentence_role: str = ""
    #is_activate: bool = True # Keep the invalid sentences
    productPnID: Optional[int] = None
    product_domain: Optional[str] = None


@dataclass
class Failure:
    # ===== required =====
    failure_id: str
    failure_mode: str
    failure_element: str

    # ===== optional text =====
    failure_effect: Optional[str] = None
    status: Optional[str] = None

    # ===== context =====
    fmea_type: Optional[str] = None
    source_type: str = "8D"

    # ===== evidence / links =====
    supporting_sentence_ids: List[str] = field(default_factory=list)
    cause_ids: List[Dict[str, Any]] = field(default_factory=list)

    # ===== maintenance =====
    maintenance: Optional[MaintenanceTag] = None

    # ===== product =====
    productPnID: Optional[int] = None
    product_domain: Optional[str] = None
    # Maintenance
    # revision: int
    # last_updated: str


@dataclass
class Cause:
    cause_id: str
    failure_id: str
    failure_mode: str
    failure_element: str
    failure_effect: Optional[str]
    root_cause: str

    fmea_type: str

    discipline: str
    confidence: str
    supporting_sentence_ids: List[str]

    # Maintenance
    maintenance: MaintenanceTag
    # revision: int
    # last_updated: str
    source_type: str = "8D"      
    productPnID: Optional[int] = None
    product_domain: Optional[str] = None


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

        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

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

class FailureKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # -------- persistent store --------
        self.store_path = self.persist_dir / "8d_failure_store.json"
        self.store: Dict[str, Dict[str, Any]] = {}
        if self.store_path.exists():
            with open(self.store_path, "r", encoding="utf-8") as f:
                self.store = json.load(f)
        self.cause_store_path = self.persist_dir / "8d_cause_store.json"
        self.cause_store: dict[str, dict] = {}
        if self.cause_store_path.exists():
            self.cause_store = json.loads(
                self.cause_store_path.read_text(encoding="utf-8")
            )
        # -------- vector store --------
        # Initialize the embedding model
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="all_failure_kb",
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )
    # =========================================================
    # Add failure (Failure key embedding)
    # =========================================================
    def add(self, failure):
        self.store[failure.failure_id] = asdict(failure)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2, ensure_ascii=False)
        ids = []
        documents = []
        metadatas = []
        def add_field(text: Optional[str], role: str):
            if not is_valid_embed_text(text):
                return
            ids.append(f"{failure.failure_id}::{role}")
            documents.append(text) # doc for embedding
            metadatas.append({
                "failure_id": failure.failure_id,
                "role": role,
                "fmea_type": failure.fmea_type,
                "source_type": failure.source_type,
                "productPnID": failure.productPnID,     
                "product_domain": failure.product_domain,
                "review_status": failure.maintenance.review_status,
                "version": failure.maintenance.version,
                "last_updated": failure.maintenance.last_updated,
            })
        # ---- split embedding ----
        add_field(failure.failure_mode, "failure_mode")
        add_field(failure.failure_element, "failure_element")
        add_field(failure.failure_effect, "failure_effect")
        if ids:
            self.collection.upsert( # update or insert
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        
    def add_cause(self, cause: Cause):

        self.cause_store[cause.cause_id] = asdict(cause)
        self.cause_store_path.write_text(
        json.dumps(self.cause_store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
        embed_text = cause.root_cause
        if not is_valid_embed_text(embed_text):
            return

        self.collection.upsert(
            ids=[cause.cause_id],
            documents=[embed_text],
            metadatas=[{
                "failure_id": cause.failure_id,
                "cause_id": cause.cause_id,
                "role": "failure_cause",
                "discipline": cause.discipline or "",

                "productPnID": cause.productPnID,
                "product_domain": cause.product_domain,
                "fmea_type": cause.fmea_type,
                "source_type": cause.source_type,

            }],
        )

    # =========================================================
    # Low-level role-based search
    # =========================================================
    def search_by_role(
        self,
        query: str,
        role: str,
        k: int = 5,
    ):
        return self.collection.query(
            query_texts=[query],
            n_results=k,
            where={"role": role},
        )

    # =========================================================
    # High-level merged search (FMEA style)
    # =========================================================
    def search(
        self,
        failure_mode: Optional[str] = None,
        failure_element: Optional[str] = None,
        failure_effect: Optional[str] = None,
        k: int = 3,
    ) -> List[str]:
        """
        Return ranked failure_ids
        """

        merged = defaultdict(lambda: {
            "score": 0.0,
            "roles": set(),
        })

        def merge_hits(res, role, weight):
            if not res["ids"]:
                return
            for meta, dist in zip(
                res["metadatas"][0],
                res["distances"][0],
            ):
                fid = meta["failure_id"]
                merged[fid]["score"] += weight * (1 - dist)
                merged[fid]["roles"].add(role)

        # ---- role-aware retrieval ----
        if failure_mode:
            res = self.search_by_role(
                failure_mode, "failure_mode", k
            )
            merge_hits(res, "failure_mode", 0.5)

        if failure_element:
            res = self.search_by_role(
                failure_element, "failure_element", k
            )
            merge_hits(res, "failure_element", 0.4)

        if failure_effect:
            res = self.search_by_role(
                failure_effect, "failure_effect", k
            )
            merge_hits(res, "failure_effect", 0.3)

        ranked = sorted(
            merged.items(),
            key=lambda x: (x[1]["score"], len(x[1]["roles"])),
            reverse=True,
        )

        return [fid for fid, _ in ranked[:k]]

    # =========================================================
    # Get full failure object
    # =========================================================
    def get(self, failure_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get(failure_id)


# =========================================================
# Cause KB (strictly under failure)
# =========================================================

class CauseKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # -------- persistent store --------
        self.store_path = self.persist_dir / "cause_store.json"
        self.store: Dict[str, Dict[str, Any]] = {}
        if self.store_path.exists():
            with open(self.store_path, "r", encoding="utf-8") as f:
                self.store = json.load(f)

        # -------- vector store --------
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="cause_kb",
            embedding_function=self.embedder,
        )

    # Embedding function
    def add(self, cause: Cause):
        self.store[cause.cause_id] = asdict(cause)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.store, f, indent=2, ensure_ascii=False)

        embed_text = "\n".join([
            f"Root cause: {cause.root_cause}"
        ])

        self.collection.upsert(
            ids=[cause.cause_id],
            documents=[embed_text],
            metadatas=[{
                "failure_id": cause.failure_id,
                "discipline": cause.discipline,
                "cause_level": cause.cause_level,
                "confidence": cause.confidence,
                "review_status": cause.maintenance.review_status,
                "version": cause.maintenance.version,
            }],
        )
    # Search function
    def search_under_failure(
        self,
        query: str,
        failure_id: str,
        k: int = 5,
    ) -> List[str]:
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            where={"failure_id": failure_id},
        )
        return res["ids"][0] if res["ids"] else []
    
    def get_all_vectors(self): 
        res = self.collection.get(
            include=["embeddings", "metadatas", "ids"]
        )
        return res



