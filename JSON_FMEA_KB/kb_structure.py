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
                "released_year": failure.released_year,
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
                "released_year": cause.released_year,

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
            where: Optional[Dict[str, Any]] = None,
        ):
            where = where or {"role": role}
            if "role" not in where:
                where = {**where, "role": role}

            return self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
            )
    
    def _build_where(
        self,
        role: str,
        productPnID: Optional[Union[str, List[str]]] = None,
        fmea_type: Optional[Union[str, List[str]]] = None,
    ):
        filters = [{"role": role}]

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

        if len(filters) == 1:
            return filters[0]

        return {"$and": filters}

    # =========================================================
    # High-level merged search (RAG entry point)
    # =========================================================
    def search(
        self,
        failure_mode: Optional[str] = None,
        failure_element: Optional[str] = None,
        failure_effect: Optional[str] = None,
        productPnID: Optional[FilterValue] = None,
        fmea_type: Optional[FilterValue] = None,
        k: int = 5,
        per_role_k: Optional[int] = None,  # Optional
    ) -> List[str]:
        """
        Return ranked failure_ids
        """
        per_role_k = per_role_k or k

        merged = defaultdict(lambda: {
            "score": 0.0,
            "roles": set(),
        })

        def merge_hits(res, role, weight):
            if not res or not res.get("ids") or not res["ids"][0]:
                return
            for meta, dist in zip(
                res["metadatas"][0],
                res["distances"][0],
            ):
                fid = meta["failure_id"]
                merged[fid]["score"] += weight * (1 - dist)
                merged[fid]["roles"].add(role)

        # ---------- role-specific retrieval with filters ----------
        if failure_mode:
            where = self._build_where("failure_mode", productPnID, fmea_type)
            res = self.search_by_role(failure_mode, "failure_mode", per_role_k, where=where)
            merge_hits(res, "failure_mode", 0.5)

        if failure_element:
            where = self._build_where("failure_element", productPnID, fmea_type)
            res = self.search_by_role(failure_element, "failure_element", per_role_k, where=where)
            merge_hits(res, "failure_element", 0.4)

        if failure_effect:
            where = self._build_where("failure_effect", productPnID, fmea_type)
            res = self.search_by_role(failure_effect, "failure_effect", per_role_k, where=where)
            merge_hits(res, "failure_effect", 0.3)

        # ---------- rank ----------
        ranked = sorted(
            merged.items(),
            key=lambda x: (x[1]["score"], len(x[1]["roles"])),
            reverse=True,
        )

        return [fid for fid, _ in ranked[:k]]

    # =========================================================
    # Get full failure object
    # =========================================================
    def get(self, failure_id: str) -> Optional[dict]:
        return self.store.get(failure_id)
    

    # def search_under_failure(self, query: str, failure_id: str, k: int = 5):
    #     res = self.collection.query(
    #         query_texts=[query],
    #         n_results=k,
    #         where={"failure_id": failure_id},
    #     )
    #     return res["ids"][0] if res["ids"] else []
    
    # def get_all_vectors(self): 
    #     res = self.collection.get(
    #         include=["embeddings", "metadatas", "ids"]
    #     )
    #     return res
    

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