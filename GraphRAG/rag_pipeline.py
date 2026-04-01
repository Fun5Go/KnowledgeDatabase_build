# chunk_rag.py

from typing import Dict, Any, List
from neo4j_retriever import ChunkRetriever


def looks_like_function_statement(text: str) -> bool:
    """
    Heuristic filter for function-like FSChunk text.
    """
    text_lower = text.lower()
    patterns = [
        "shall support",
        "shall provide",
        "shall operate",
        "shall protect",
        "shall allow",
        "shall include",
        "shall enable",
        "shall maintain",
        "shall measure",
        "shall control",
        "shall throw",
        "shall require",
        "shall turn",
        "shall include",
        "shall have",
    ]
    return any(p in text_lower for p in patterns)


class ChunkGraphRAG:
    def __init__(self):
        self.retriever = ChunkRetriever()

    def close(self):
        self.retriever.close()

    def answer_product_functions(
        self,
        product_name: str,
        query_text: str,
        top_k: int = 8
    ) -> Dict[str, Any]:
        """
        Answer function query by:
        1) retrieving FSChunk seeds
        2) expanding graph evidence around each FSChunk
        """
        seeds = self.retriever.retrieve_function_seeds(
            product_name=product_name,
            query_text=query_text,
            top_k=top_k
        )

        evidence = []
        functions = []

        for seed in seeds:
            ctx = self.retriever.expand_fs_context(seed["node_id"])
            fs_text = (ctx.get("fs_text") or seed.get("text") or "").strip()

            if not fs_text:
                continue

            item = {
                "fs_text": fs_text,
                "score": seed["rrf_score"],
                "sources": seed["sources"],
                "ts_texts": [x for x in ctx.get("ts_texts", []) if x],
                "qd_texts": [x for x in ctx.get("qd_texts", []) if x],
                "tst_texts": [x for x in ctx.get("tst_texts", []) if x],
                "rationale_texts": [x for x in ctx.get("rationale_texts", []) if x],
            }
            evidence.append(item)

            if looks_like_function_statement(fs_text) and fs_text not in functions:
                functions.append(fs_text)

        # fallback: if heuristic misses too much, still return top FSChunk texts
        if not functions:
            for e in evidence:
                if e["fs_text"] not in functions:
                    functions.append(e["fs_text"])

        return {
            "product_name": product_name,
            "query_text": query_text,
            "functions": functions,
            "evidence": evidence,
        }