from dataclasses import dataclass,asdict, is_dataclass
from typing import Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import json
from typing import List



@dataclass
class Sentence:
    id: str
    text: str
    source_section: str
    annotations: Dict[str, Any]
    case_id: str


@dataclass
class FailureIndex:
    failure_id: str
    failure_element: str
    failure_mode: str
    status: str
    supporting_sentence_ids: List[str]
    case_id: str


def evaluate_failure(sentences: List[Sentence],
                     min_faithful: int = 90,
                     allow_levels=("observed", "confirmed")) -> str:
    """
    Failure 是否通过：
    - 至少一条 D2 的 observed / confirmed sentence
    """
    for s in sentences:
        if s.source_section != "D2":
            continue
        ann = s.annotations or {}
        if ann.get("assertion_level") in allow_levels \
           and int(ann.get("faithful_score", 0)) >= min_faithful:
            return "supported"
    return "hypothesis"



def ingest_failure_record(
    record: dict,
    sentence_kb,
    failure_kb
) -> FailureIndex:

    # 1) case_id：来自 documents
    docs = record.get("documents", [])
    if not docs:
        raise ValueError("Missing documents in 8D JSON")
    case_id = docs[0].get("file_name")

    failure = record["failure"]

    sentences: List[Sentence] = []
    supporting_ids: List[str] = []

    # 2) supporting_entities → Sentence KB
    for ent in failure.get("supporting_entities", []):
        s = Sentence(
            id=ent["id"],
            text=ent["text"],
            source_section=ent.get("source_section", ""),
            case_id=case_id,
            annotations=ent.get("annotations", {}) or {}
        )
        sentences.append(s)
        supporting_ids.append(s.id)

        # 入 sentence KB
        sentence_kb.add(s, failure.get("failure_ID"))
    # 3) failure 状态评估（基于 D2）
    status = evaluate_failure(sentences)

    # 4) Failure Index（索引层）
    f_index = FailureIndex(
        failure_id=failure["failure_ID"],
        failure_element=failure.get("failure_element", ""),
        failure_mode=failure.get("failure_mode", ""),
        status=status,
        supporting_sentence_ids=supporting_ids,
        case_id=case_id
    )

    failure_kb.add(f_index)
    return f_index



class SentenceKB:
    def __init__(self, persist_dir):
        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ✅ 你的版本应该支持这个
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name="sentences",
            embedding_function=self.embedder
        )

    def add(self, sentence,failure_id: str):
        self.collection.add(
            ids=[sentence.id],
            documents=[sentence.text],
            metadatas=[{
                "case_id": sentence.case_id,
                "failure_id": failure_id, 
                "source_section": sentence.source_section,
                "entity_type": sentence.annotations.get("entity_type"),
                "assertion_level": sentence.annotations.get("assertion_level"),
                "faithful_score": int(sentence.annotations.get("faithful_score", 0)),
            }]
        )

    def close(self):
        # 某些版本通过释放 client 触发 flush
        self.client = None

    def search(self, query: str, top_k=5, min_faithful=90):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"faithful_score": {"$gte": min_faithful}}
        )
        return results
    
import json
from pathlib import Path
from dataclasses import asdict, is_dataclass


class FailureIndexKB:
    def __init__(self, store_path):
        self.store_path = Path(store_path).resolve()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # failure_id -> failure dict
        self.failures = {}

        # load existing index if exists
        if self.store_path.exists():
            with self.store_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self.failures[obj["failure_id"]] = obj

    def add(self, failure):
        # normalize to dict
        if is_dataclass(failure):
            obj = asdict(failure)
        elif isinstance(failure, dict):
            obj = failure
        else:
            obj = failure.__dict__

        fid = obj["failure_id"]

        # 🔴 幂等写入：覆盖同一 failure_id
        self.failures[fid] = obj

        # 🔴 重写整个文件（而不是 append）
        with self.store_path.open("w", encoding="utf-8") as f:
            for item in self.failures.values():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def get(self, failure_id):
        return self.failures.get(failure_id)

    def all(self):
        return list(self.failures.values())



# record = {
#   "system_name": "",
#   "failure": {
#     "failure_ID": "8D6298190081R02_F1",
#     "failure_level": "sub_system",
#     "failure_element": "power supply",
#     "failure_mode": "component destruction",
#     "failure_effect": "device blown up",
#     "supporting_entities": [
#       {
#         "id": "8D6298190081R02_D2_S001",
#         "text": "The DUT blew up during the voltage dips/interrupt test.",
#         "source_section": "D2",
#         "annotations": {
#           "entity_type": "symptom",
#           "assertion_level": "observed",
#           "faithful_score": 100,
#           "faithful_type": "exact"
#         }
#       }
#     ]
#   }
# }

BASE_DIR = Path(__file__).resolve().parent


KB_ROOT = (BASE_DIR.parent / "kb").resolve()
SENTENCE_KB_DIR = (KB_ROOT / "sentence_kb").resolve()
FAILURE_INDEX_PATH = (KB_ROOT / "failure_index_kb.jsonl").resolve()
# SENTENCE_KB_DIR.mkdir(parents=True, exist_ok=True)

JSON_ROOT = (BASE_DIR.parent / "eightD_json_raw").resolve()

json_path = (JSON_ROOT / "8D6298190081R02.json").resolve()

# Path(r"..\eightD_json_raw\8D6298190081R02.json")

# -------- Load 8D JSON --------
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)


sentence_kb = SentenceKB(
    persist_dir=SENTENCE_KB_DIR
)

failure_kb = FailureIndexKB(
    store_path=FAILURE_INDEX_PATH
)

f_index = ingest_failure_record(data, sentence_kb, failure_kb)
print(f_index)
print("COUNT AFTER INGEST:", sentence_kb.collection.count())