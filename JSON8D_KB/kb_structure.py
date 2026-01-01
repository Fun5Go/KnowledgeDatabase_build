from dataclasses import dataclass
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import json



# ------- Failure knowledge base structrue --------
@dataclass
class Failure:
    failure_id: str
    failure_mode: str
    failure_element: str
    failure_effect: str | None
    product: str | None
    status: str                      # supported / hypothesis / reviewed
    supporting_sentence_ids: List[str]
    cause_ids: List[str]             # parent ID



class FailureKB:
    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))

        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="failure_kb",
            embedding_function=self.embedder,
        )

        self.store: dict[str, dict] = {}

    def add(self, failure: Failure):
        self.store[failure.failure_id] = failure.__dict__

        embed_text = "\n".join([
            f"Failure mode: {failure.failure_mode}",
            f"Failure element: {failure.failure_element}",
            f"Failure effect: {failure.failure_effect or ''}",
        ])

        self.collection.add(
            ids=[failure.failure_id],
            documents=[embed_text],
            metadatas=[{
                "failure_mode": failure.failure_mode,
                "failure_element": failure.failure_element,
                "status": failure.status,
                "product": failure.product or "",
            }]
        )

    def search(self, query: str, k=3):
        return self.collection.query(
            query_texts=[query],
            n_results=k,
        )
    

#----------- Cause knowledge base structure ----------
@dataclass
class Cause:
    cause_id: str
    failure_id: str
    failure_mode: str
    failure_element: str
    failure_effect: str | None

    root_cause: str
    cause_level: str
    discipline: str
    confidence: str

    supporting_sentence_ids: List[str]

class CauseKB:
    def __init__(self, persist_dir: Path):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="cause_kb",
            embedding_function=self.embedder,
        )

        self.store: dict[str, dict] = {}

    def add(self, cause: Cause):
        self.store[cause.cause_id] = cause.__dict__

        similarity_text = "\n".join([
            f"Failure mode: {cause.failure_mode}",
            f"Failure element: {cause.failure_element}",
            f"Failure effect: {cause.failure_effect or ''}",
            f"Root cause: {cause.root_cause}",
        ])

        self.collection.add(
            ids=[cause.cause_id],
            documents=[similarity_text],
            metadatas=[{
                "failure_id": cause.failure_id,
                "discipline": cause.discipline,
                "cause_level": cause.cause_level,
                "confidence": cause.confidence,
            }]
        )

    def search_under_failure(self, query: str, failure_id: str, k=5):
        return self.collection.query(
            query_texts=[query],
            n_results=k,
            where={"failure_id": failure_id}
        )
