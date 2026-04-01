# config.py or utils.py

from typing import List
from config import embedder, _embedding_cache

def get_query_embedding(text: str) -> List[float]:
    if text in _embedding_cache:
        return _embedding_cache[text]

    emb = embedder([text])[0]
    _embedding_cache[text] = emb
    return emb



