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

    def update_failure(
        self,
        failure: FMEAFailure,
        *,
        delete_stale_roles: bool = True,
    ) -> None:
        """
        Update structured failure record and re-embed vectors.
        If delete_stale_roles=True, removes all vectors belonging to failure_id first.
        """

        # --- structured store ---
        self.store[failure.failure_id] = asdict(failure)
        self.store_path.write_text(
            json.dumps(self.store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # --- vector store ---
        if delete_stale_roles:
            self.collection.delete(where={"failure_id": failure.failure_id})

        # re-add embeddings
        self.add(failure)


    def update_cause(self, cause_id: str, **fields) -> None:
        """
        Partial update: merge fields into an existing cause record,
        then re-embed the cause vector.
        """
        if cause_id not in self.cause_store:
            raise KeyError(f"cause_id not found in cause_store: {cause_id}")

        # merge
        rec = dict(self.cause_store[cause_id])
        for k, v in fields.items():
            if v is not None:
                rec[k] = v

        # persist
        self.cause_store[cause_id] = rec
        self.cause_store_path.write_text(
            json.dumps(self.cause_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # re-embed: delete by id + upsert (only if text valid)
        self.collection.delete(ids=[cause_id])
        embed_text = rec.get("failure_cause")
        if is_valid_embed_text(embed_text):
            self.collection.upsert(
                ids=[cause_id],
                documents=[embed_text],
                metadatas=[{
                    "failure_id": rec.get("failure_id", ""),
                    "cause_id": cause_id,
                    "role": "failure_cause",
                    "discipline": rec.get("discipline", "") or "",
                    "productPnID": rec.get("productPnID"),
                    "product_domain": rec.get("product_domain"),
                    "fmea_type": rec.get("fmea_type", "") or "",
                    "source_type": rec.get("source_type", "") or "",
                    "released_year": rec.get("released_year"),
                }],
            )


    # -----------------------------
    # CRUD: delete
    # -----------------------------
    def delete_failure(
        self,
        failure_id: str,
        *,
        delete_linked_causes: bool = False,
    ) -> None:
        """
        Deletes failure structured record and all vectors with failure_id.
        Optionally deletes linked causes.
        """

        # --- structured failure ---
        if failure_id in self.store:
            del self.store[failure_id]
            self.store_path.write_text(
                json.dumps(self.store, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # --- vector store ---
        self.collection.delete(where={"failure_id": failure_id})

        # --- optionally delete linked causes ---
        if delete_linked_causes:
            to_delete = [
                cid for cid, c in self.cause_store.items()
                if c.get("failure_id") == failure_id
            ]
            for cid in to_delete:
                self.delete_cause(cid)


    def delete_cause(self, cause_id: str) -> None:
        """
        Deletes cause structured record and its vector.
        """

        if cause_id in self.cause_store:
            del self.cause_store[cause_id]
            self.cause_store_path.write_text(
                json.dumps(self.cause_store, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        self.collection.delete(ids=[cause_id])

    def delete_by_where(self, where: Dict[str, Any]) -> None:
        """
        Vector-only deletion by filter. Use carefully.
        Structured stores are NOT altered.
        """
        self.collection.delete(where=where)
        

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