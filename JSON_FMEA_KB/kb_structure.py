from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from pathlib import Path
import json

import chromadb
from chromadb.utils import embedding_functions

from dataclasses import asdict
from collections import defaultdict

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
    failure_id: str
    failure_mode: str
    failure_element: Optional[str]
    failure_effect: Optional[str]

    system: Optional[str]
    function: Optional[str]

    severity: Optional[int]
    rpn: Optional[float]

    cause_ids: List[str]

    source_type: str # Old/New FMEA


@dataclass
class FMEACause:
    cause_id: str
    failure_id: str

    failure_cause: str
    discipline: Optional[str]

    # Controls entity
    prevention: Optional[str]          # controls_prevention
    detection: Optional[str]            # current_detection
    detection_value: Optional[float]    # detection (number)

    occurrence: Optional[float]         # occurrence (number)
    recommended_action: Optional[str]   # recommended_action


class FMEAFailureKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ---------- structured store (unchanged) ----------
        self.store_path = self.persist_dir / "fmea_failure_store.json"
        self.store: Dict[str, dict] = {}
        if self.store_path.exists():
            self.store = json.loads(self.store_path.read_text(encoding="utf-8"))

        # ---------- vector store ----------
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="fmea_failure_kb",
            embedding_function=self.embedder,
        )

    # =========================================================
    # Add failure (ROLE-AWARE embedding)
    # =========================================================
    def add(self, failure):
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
            if not text:
                return
            ids.append(f"{failure.failure_id}::{role}")
            documents.append(text)
            metadatas.append({
                "failure_id": failure.failure_id,
                "role": role,
                "system": failure.system or "",
                "severity": failure.severity or 0,
                "rpn": failure.rpn or 0,
                "type": failure.source_type,
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
    # High-level merged search (RAG entry point)
    # =========================================================
    def search(
        self,
        failure_mode: Optional[str] = None,
        failure_element: Optional[str] = None,
        failure_effect: Optional[str] = None,
        k: int = 5,
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

        # ---------- role-specific retrieval ----------
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
    
class FMEACauseKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.store_path = self.persist_dir / "fmea_cause_store.json"
        self.store = {}
        if self.store_path.exists():
            self.store = json.loads(self.store_path.read_text(encoding="utf-8"))

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="fmea_cause_kb",
            embedding_function=self.embedder,
        )

    def add(self, cause: FMEACause):
        self.store[cause.cause_id] = asdict(cause)
        self.store_path.write_text(
            json.dumps(self.store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        embed_text = "\n".join([
            # f"Failure ID: {cause.failure_id}",
            # f"Failure element: {cause.failure_element}",
            f"Failure cause: {cause.failure_cause}",
        ])

        self.collection.upsert(
            ids=[cause.cause_id],
            documents=[embed_text],
            metadatas=[{
                "failure_id": cause.failure_id,
                "discipline": cause.discipline or "",
            }],
        )

    def search_under_failure(self, query: str, failure_id: str, k: int = 5):
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            where={"failure_id": failure_id},
        )
        return res["ids"][0] if res["ids"] else []