from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from pathlib import Path
import json

import chromadb
from chromadb.utils import embedding_functions
from dataclasses import asdict


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

        self.store_path = self.persist_dir / "fmea_failure_store.json"
        self.store = {}
        if self.store_path.exists():
            self.store = json.loads(self.store_path.read_text(encoding="utf-8"))

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="fmea_failure_kb",
            embedding_function=self.embedder,
        )

    def add(self, failure: FMEAFailure):
        self.store[failure.failure_id] = asdict(failure)
        self.store_path.write_text(
            json.dumps(self.store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Minimize the noise
        embed_text = " | ".join(
            t for t in [
                failure.failure_mode,
                failure.failure_element,
                failure.failure_effect,
            ]
            if t
        )
        self.collection.upsert(
            ids=[failure.failure_id],
            documents=[embed_text],
            metadatas=[{
                "system": failure.system or "",
                "severity": failure.severity or 0,
                "rpn": failure.rpn or 0,
                "type": failure.source_type
            }],
        )

    def search(self, query: str, k: int = 3):
        res = self.collection.query(query_texts=[query], n_results=k)
        return res["ids"][0] if res["ids"] else []
    
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