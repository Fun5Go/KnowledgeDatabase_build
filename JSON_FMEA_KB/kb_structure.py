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
class FMEAFailure:
    # ===== required =====
    failure_id: str
    failure_mode: str

    # ===== optional text fields =====
    failure_element: Optional[str] = None
    failure_effect: Optional[str] = None

    process_step: Optional[str] = None
    system: Optional[str] = None
    function: Optional[str] = None

    # ===== ratings =====
    severity: Optional[float] = None
    rpn: Optional[float] = None

    # ===== links: store cause id + text =====
    cause_ids: List[Dict[str, Any]] = field(default_factory=list)

    # ===== context =====
    source_type: str = ""               # Old/New FMEA
    fmea_type: Optional[str] = None

    productPnID: Optional[int] = None
    product_domain: Optional[str] = None
    released_year: Optional[int] = None


@dataclass
class FMEACause:
    cause_id: str
    failure_id: str
    failure_mode: str
    failure_element: Optional[str]
    failure_effect: Optional[str]

    failure_cause: str
    discipline: Optional[str]

    # Controls entity
    prevention: Optional[str]          # controls_prevention
    detection: Optional[str]            # current_detection
    detection_value: Optional[float]    # detection (number)

    occurrence: Optional[float]         # occurrence (number)
    recommended_action: Optional[str]   # recommended_action

    source_type: str # Old/New FMEA
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

        # ---------- structured store (unchanged) ----------
        self.store_path = self.persist_dir / "fmea_failure_store.json"
        self.store: Dict[str, dict] = {}
        if self.store_path.exists():
            self.store = json.loads(self.store_path.read_text(encoding="utf-8"))

        self.cause_store_path = self.persist_dir / "fmea_cause_store.json"
        self.cause_store: dict[str, dict] = {}
        if self.cause_store_path.exists():
            self.cause_store = json.loads(
                self.cause_store_path.read_text(encoding="utf-8")
            )

        # ---------- vector store ----------
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
    # Add failure (ROLE-AWARE embedding)
    # =========================================================
    def add(self, failure:FMEAFailure):
        # ---------- store structured ----------
        self.store[failure.failure_id] = asdict(failure)
        self.store_path.write_text(
            json.dumps(self.store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ids = []
        documents = []
        metadatas = []
    

        def add_field(text: Optional[str], role: str):
            if not is_valid_embed_text(text):
                return

            ids.append(f"{failure.failure_id}::{role}")
            documents.append(text)

            metadatas.append({
                "failure_id": failure.failure_id,
                "role": role,

                "system": failure.system or "",
                "fmea_type": failure.fmea_type or "",   

                "productPnID": failure.productPnID,     
                "product_domain": failure.product_domain,
                "source_type": failure.source_type,
                "released_date": failure.released_year,
            })

        # ---------- split embedding by role ----------
        add_field(failure.failure_mode, "failure_mode")
        add_field(failure.failure_element, "failure_element")
        add_field(failure.failure_effect, "failure_effect")

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

    def add_cause(self, cause: FMEACause):

        self.cause_store[cause.cause_id] = asdict(cause)
        self.cause_store_path.write_text(
        json.dumps(self.cause_store, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
        embed_text = cause.failure_cause
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
                "released_date": cause.released_year,

            }],
        )

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Any, List

import chromadb
from chromadb.utils import embedding_functions


# assumes you already have this
def is_valid_embed_text(text: Optional[str]) -> bool:
    return bool(text and text.strip())


class FMEAFailureKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ---------- structured store ----------
        self.store_path = self.persist_dir / "fmea_failure_store.json"
        self.store: Dict[str, dict] = {}
        if self.store_path.exists():
            self.store = json.loads(self.store_path.read_text(encoding="utf-8"))

        self.cause_store_path = self.persist_dir / "fmea_cause_store.json"
        self.cause_store: dict[str, dict] = {}
        if self.cause_store_path.exists():
            self.cause_store = json.loads(self.cause_store_path.read_text(encoding="utf-8"))

        # ---------- vector store ----------
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="all_failure_kb",
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------------------------
    # existing: add failure
    # ---------------------------
    def add(self, failure):
        self.store[failure.failure_id] = asdict(failure)
        self.store_path.write_text(
            json.dumps(self.store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        def add_field(text: Optional[str], role: str):
            if not is_valid_embed_text(text):
                return
            ids.append(f"{failure.failure_id}::{role}")
            documents.append(text)
            metadatas.append({
                "failure_id": failure.failure_id,
                "role": role,
                "system": failure.system or "",
                "fmea_type": failure.fmea_type or "",
                "productPnID": failure.productPnID,
                "product_domain": failure.product_domain,
                "source_type": failure.source_type,
                "released_year": failure.released_year,
            })

        add_field(failure.failure_mode, "failure_mode")
        add_field(failure.failure_element, "failure_element")
        add_field(failure.failure_effect, "failure_effect")

        if ids:
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # ---------------------------
    # existing: add cause
    # ---------------------------
    def add_cause(self, cause):
        self.cause_store[cause.cause_id] = asdict(cause)
        self.cause_store_path.write_text(
            json.dumps(self.cause_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        embed_text = cause.failure_cause
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
                "released_year": cause.released_year,
            }],
        )

    # =========================================================
    # UPDATE / DELETE (Failure)
    # =========================================================
    def update(self, failure) -> None:
        """
        Update a failure in both:
        - structured store
        - vector store (delete old role-fragments, then re-add)
        """
        # delete any existing vectors for this failure_id
        self._delete_failure_vectors(failure.failure_id)

        # replace structured + re-add vectors
        self.add(failure)

    def delete(self, failure_id: str, *, delete_causes: bool = False) -> None:
        """
        Delete a failure from:
        - structured failure store
        - vector store role-fragments
        Optionally also deletes all causes linked to this failure_id.
        """
        # structured store
        if failure_id in self.store:
            del self.store[failure_id]
            self.store_path.write_text(
                json.dumps(self.store, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # vectors
        self._delete_failure_vectors(failure_id)

        if delete_causes:
            self.delete_causes_by_failure_id(failure_id)

    def _delete_failure_vectors(self, failure_id: str) -> None:
        """
        Removes all role-fragment vectors for a failure_id.
        Uses deterministic ids created in add(): {failure_id}::{role}.
        """
        ids = [
            f"{failure_id}::failure_mode",
            f"{failure_id}::failure_element",
            f"{failure_id}::failure_effect",
        ]
        # Chroma delete ignores missing IDs in most versions; if yours errors, catch exceptions here.
        self.collection.delete(ids=ids)

    # =========================================================
    # UPDATE / DELETE (Cause)
    # =========================================================
    def update_cause(self, cause) -> None:
        """
        Update a cause in both stores.
        Safe approach: delete existing cause vector by cause_id then re-add.
        """
        self.delete_cause(cause.cause_id)
        self.add_cause(cause)

    def delete_cause(self, cause_id: str) -> None:
        """
        Delete a cause from:
        - structured cause store
        - vector store (single doc id == cause_id)
        """
        if cause_id in self.cause_store:
            del self.cause_store[cause_id]
            self.cause_store_path.write_text(
                json.dumps(self.cause_store, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        self.collection.delete(ids=[cause_id])

    def delete_causes_by_failure_id(self, failure_id: str) -> None:
        """
        Delete all causes linked to a failure_id.
        - removes matching causes from structured cause_store
        - deletes corresponding vectors from Chroma
        """
        # structured cause ids to delete
        to_delete = [
            cid for cid, c in self.cause_store.items()
            if c.get("failure_id") == failure_id
        ]

        if to_delete:
            for cid in to_delete:
                del self.cause_store[cid]
            self.cause_store_path.write_text(
                json.dumps(self.cause_store, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.collection.delete(ids=to_delete)
    def update_merge(self, failure: FMEAFailure) -> None:
        """
        Patch-style update:
        - keeps existing fields unless the new object provides a value
        - then calls existing update() which reindexes vectors
        """
        fid = failure.failure_id
        old = self.store.get(fid)
        if not old:
            # no existing record -> treat as add
            return self.add(failure)

        new = asdict(failure)

        # merge rule: only overwrite if new value is meaningful
        merged = dict(old)
        for k, v in new.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            if isinstance(v, list) and len(v) == 0:
                continue
            merged[k] = v

        self.update(FMEAFailure(**merged))
        

class FailureRetriever:
    """
    Retrieval layer for FMEA / 8D semantic matching
    """

    # ===============================
    # Role base weights (global)
    # ===============================

    ROLE_WEIGHT = {
        "failure_effect": 1.0,
        "failure_mode": 1.0,
        "failure_element": 0.9,
        "failure_cause": 1.1,
    }

    # ===============================
    # 8D stage → role bias
    # ===============================
    STAGE_ROLE_BIAS = {
        "D2": {
            "failure_effect": 0.8,
            "failure_mode": 1.2,
            "failure_element": 0.9,
            "failure_cause": 0.6,
        },
        "D4": {
            "failure_cause": 1.3,
            "failure_mode": 0.9,
            "failure_effect": 0.5,
            "failure_element": 0.7,
        },
    }
    def __init__(
        self,
        persist_dir,
        collection_name: str,
        store: Dict[str, dict],
    ):
        # ---------- vector store ----------
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder,
        )

        # ---------- structured failure store ----------
        self.store = store

    # =========================================================
    # Build Chroma where filter
    # =========================================================
    def _build_where(
        self,
        productPnID: Optional[FilterValue] = None,
        fmea_type: Optional[FilterValue] = None,
    ) -> Optional[Dict]:
        filters = []

        if productPnID:
            filters.append(
                {"productPnID": {"$in": productPnID}}
                if isinstance(productPnID, list)
                else {"productPnID": productPnID}
            )

        if fmea_type:
            filters.append(
                {"fmea_type": {"$in": fmea_type}}
                if isinstance(fmea_type, list)
                else {"fmea_type": fmea_type}
            )

        if not filters:
            return None
        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}

    # =========================================================
    # Role-agnostic vector search
    # =========================================================
    def search_free(
        self,
        query: str,
        productPnID: Optional[FilterValue] = None,
        fmea_type: Optional[FilterValue] = None,
        k: int = 10,
    ):
        """
        Do NOT constrain role.
        Let embedding model see the full 8D sentence.
        """
        where = self._build_where(productPnID, fmea_type)
        return self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
        )
    # =========================================================
    # Merge with role + stage bias
    # =========================================================
    def merge_hits_with_bias(
        self,
        res,
        d_stage: DStage | Iterable[DStage],
    ) -> Dict[str, dict]:
        """
        Merge vector hits into failure_id space
        Supports single stage or multiple stages (e.g. D2 + D4)
        """
        merged = defaultdict(lambda: {
            "score": 0.0,
            "roles": set(),
        })

        if not res or not res.get("ids") or not res["ids"][0]:
            return merged

        # ---- normalize stages ----
        if isinstance(d_stage, (list, tuple, set)):
            stages = d_stage
        else:
            stages = [d_stage]

        for meta, dist in zip(
            res["metadatas"][0],
            res["distances"][0],
        ):
            failure_id = meta["failure_id"]
            role = meta.get("role", "unknown")

            base_weight = self.ROLE_WEIGHT.get(role, 0.7)

            for stage in stages:
                stage_bias = self.STAGE_ROLE_BIAS.get(stage, {})
                stage_weight = stage_bias.get(role, 0.7)

                score = base_weight * stage_weight * (1 - dist)

                merged[failure_id]["score"] += score
                merged[failure_id]["roles"].add(role)

        return merged
    # =========================================================
    # 8D entry point (recommended)
    # =========================================================
    def search_from_8d(
        self,
        text: str,
        d_stage: DStage,
        productPnID: Optional[FilterValue] = None,
        fmea_type: Optional[FilterValue] = None,
        k: int = 5,
        raw_k: int = 15,
    ) -> List[str]:

        # Role-free vector search: top-N semantically similar fragments
        res = self.search_free(
            query=text,
            productPnID=productPnID,
            fmea_type=fmea_type,
            k=raw_k,
        )
        # Merge with semantic bias with role and stage weighted
        merged = self.merge_hits_with_bias(res, d_stage)
        # Rank by score + role coverage
        ranked = sorted(
            merged.items(),
            key=lambda x: (x[1]["score"], len(x[1]["roles"])),
            reverse=True,
        )
        return [fid for fid, _ in ranked[:k]]

    # =========================================================
    # Get full structured failure
    # =========================================================
    def get(self, failure_id: str) -> Optional[dict]:
        return self.store.get(failure_id)