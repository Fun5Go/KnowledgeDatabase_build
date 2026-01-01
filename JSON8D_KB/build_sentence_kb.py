# build_sentence_kb.py
from kb_structure import SentenceKB
from ingest_8d import ingest_8d_json

def build_sentence_kb(json_records, sentence_kb: SentenceKB):
    for record in json_records:
        ingest_8d_json(
            json_path=record,
            failure_kb=None,     
            cause_kb=None,       
            sentence_kb=sentence_kb,
        )