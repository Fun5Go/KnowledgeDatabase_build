from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Literal,Iterable

from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import hashlib

from dataclasses import asdict
from collections import defaultdict
from dataclasses import asdict, is_dataclass, field

FilterValue = Union[str, List[str]]
DStage = Literal["D2", "D4"]
FailureFieldType = Literal["element", "mode", "effect", "cause"]
EdgeKind = Literal["hard", "soft"]


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

def make_edge_id(src_id: str, relation: str, tgt_id: str, kind: str) -> str:
    key = f"{src_id}|{relation}|{tgt_id}|{kind}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()

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
    #Special for cause
    discipline: Optional[str] = None


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

@dataclass
class GraphEdge:
    edge_id: str                         # stable unique id (hash)
    src_id: str
    src_type: str                        # element|mode|effect|cause|...
    relation: str                        # MODE_LEADS_TO_EFFECT ...
    tgt_id: str
    tgt_type: str

    kind: EdgeKind = "hard"              # hard or soft
    weight: float = 1.0                  # normalized strength (0..1 for soft; >=1 ok for hard)
    count: int = 0                       # occurrences / evidence count

    # provenance / evidence
    failure_ids: List[str] = field(default_factory=list)  # which FailureEntity records support it
    source_type: Optional[str] = None                     # Old/New
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # soft-edge special
    score_type: Optional[str] = None      # "cosine"|"pmi"|"lift"|"rule"...
    threshold: Optional[float] = None     # how it was formed
    metadata: Dict = field(default_factory=dict)

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
                # model_name="BAAI/bge-base-en-v1.5"
        )

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
                "cause_to_effect":{},
            }

        # ---------- vector store ----------
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
                # model_name="BAAI/bge-base-en-v1.5"
        )
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
    discipline: str | None = None,
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
            "discipline": discipline if field_type == "cause" else None,
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
        metadata = {
            "field_type": field_type,
            "count": count,
            "source_type": source_type_meta,
        }

        if field_type == "cause" and discipline:
            metadata["discipline"] = discipline

        self.collection.upsert(
            ids=[semantic_id],
            documents=[text],
            metadatas=[metadata],
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


        _add_edge("cause_to_effect", entity.cause_id, entity.effect_id)

        # 3) persist edge store
        self.edge_store_path.write_text(
            json.dumps(self.edge_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    def delete_failure(self, failure_id: str) -> None:

        entity = self.entity_store.get(failure_id)
        if not entity:
            print(f"[WARN] failure_id not found: {failure_id}")
            return

        mode_id = entity.get("mode_id")
        element_id = entity.get("element_id")
        effect_id = entity.get("effect_id")
        cause_id = entity.get("cause_id")

        # -----------------------------
        # 1 remove entity
        # -----------------------------
        del self.entity_store[failure_id]

        self.entity_store_path.write_text(
            json.dumps(self.entity_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # -----------------------------
        # 2 update field_store
        # -----------------------------
        for sid in [mode_id, element_id, effect_id, cause_id]:

            if not sid:
                continue

            node = self.field_store.get(sid)
            if not node:
                continue

            ids = set(node.get("failure_ids", []))
            ids.discard(failure_id)

            if not ids:
                # remove semantic node entirely
                del self.field_store[sid]

                try:
                    self.collection.delete(ids=[sid])
                except Exception:
                    pass
            else:
                node["failure_ids"] = sorted(ids)
                node["count"] = len(ids)
                self.field_store[sid] = node

        self.field_store_path.write_text(
            json.dumps(self.field_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # -----------------------------
        # 3 update edge_store
        # -----------------------------
        def _decrement(src_type, src, tgt):

            if not src or not tgt:
                return

            src_map = self.edge_store.get(src_type, {}).get(src)
            if not src_map:
                return

            if tgt in src_map:
                src_map[tgt] -= 1

                if src_map[tgt] <= 0:
                    del src_map[tgt]

            if not src_map:
                self.edge_store[src_type].pop(src, None)

        _decrement("mode_to_cause", mode_id, cause_id)
        _decrement("mode_to_effect", mode_id, effect_id)
        _decrement("element_to_mode", element_id, mode_id)
        _decrement("cause_to_effect", cause_id, effect_id)

        self.edge_store_path.write_text(
            json.dumps(self.edge_store, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[DELETE] failure {failure_id}")

    def delete_field_text(self, field_type: str, field_id: str):


        node = self.field_store.get(field_id)

        if not node:
            print("[WARN] field not found")
            return

        failure_ids = list(node.get("failure_ids", []))

        for fid in failure_ids:
            self.delete_failure(fid)


    def delete_by_element(self, field_id: str):

        node = self.field_store.get(field_id)

        if not node:
            print("[WARN] element not found")
            return

        failure_ids = list(node.get("failure_ids", []))

        for fid in failure_ids:
            self.delete_failure(fid)

                
