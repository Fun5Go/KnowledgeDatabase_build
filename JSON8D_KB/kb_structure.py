from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import json
from typing import Optional
from dataclasses import asdict
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
class Sentence:
    id: str
    text: str
    source_section: str
    case_id: str
    annotations: Dict[str, Any]

    #is_activate: bool = True # Keep the invalid sentences


@dataclass
class Failure:
    failure_id: str
    failure_mode: str
    failure_element: str
    failure_effect: Optional[str]
    product: Optional[str]
    status: str
    supporting_sentence_ids: List[str]
    cause_ids: List[str]
    maintenance: MaintenanceTag

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
    cause_level: str
    discipline: str
    confidence: str
    supporting_sentence_ids: List[str]

    # Maintenance
    maintenance: MaintenanceTag
    # revision: int
    # last_updated: str


# =========================================================
# Failure evaluation (used during ingest)
# =========================================================

def evaluate_failure(
    sentences: List[Sentence],
    min_faithful: int = 95,
    allow_levels=("observed", "confirmed"),
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

class SentenceKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
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
        cause_id: Optional[str] = None,
    ):
        self.collection.add(
            ids=[sentence.id],
            documents=[sentence.text],
            metadatas=[{
                "case_id": sentence.case_id,
                "failure_id": failure_id,
                "cause_id": cause_id or "",
                "sentence_role": sentence_role,
                "source_section": sentence.source_section,
                "entity_type": sentence.annotations.get("entity_type"),
                "assertion_level": sentence.annotations.get("assertion_level"),
                "faithful_score": int(sentence.annotations.get("faithful_score", 0)),
            }],
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
                        "entity_type": meta.get("entity_type"),
                        "assertion_level": meta.get("assertion_level"),
                        "faithful_score": meta.get("faithful_score"),
                    },
                )
            )
        return sentences


# =========================================================
# Failure KB (entry gate)
# =========================================================

class FailureKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # -------- persistent store --------
        self.store_path = self.persist_dir / "failure_store.json"
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
            name="failure_kb",
            embedding_function=self.embedder,
        )

    # =========================================================
    # Add failure (ROLE-AWARE embedding)
    # =========================================================
    def add(self, failure):
        # ---- structured store ----
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
            documents.append(text)
            metadatas.append({
                "failure_id": failure.failure_id,
                "role": role,

                # keep your existing metadata
                "failure_mode": failure.failure_mode,
                "failure_element": failure.failure_element,
                "product": failure.product or "",
                "review_status": failure.maintenance.review_status,
                "version": failure.maintenance.version,
                "last_updated": failure.maintenance.last_updated,
            })

        # ---- split embedding ----
        add_field(failure.failure_mode, "failure_mode")
        add_field(failure.failure_element, "failure_element")
        add_field(failure.failure_effect, "failure_effect")

        if ids:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
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



