from typing import List, Dict, Any, Literal, Optional
from neo4j_retriever import ChunkRetriever


QueryType = Literal[
    "product_function",
    "element_sentences",
]


def looks_like_function_statement(text: str) -> bool:
    """
    Heuristic filter for function-like requirement text.

    Notes
    -----
    This filter is intentionally permissive. It captures both:
    1. direct functional statements, e.g. "shall support"
    2. implementation-oriented requirement statements, e.g.
       "shall be implemented"

    In later stages, these statements can be further separated into
    function / feature / implementation categories if needed.
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
        "shall have",
        "shall be",
        "shall be implemented",
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
        Answer a product-level function query by:
        1. retrieving FSChunk seeds
        2. expanding graph evidence around each FSChunk

        Parameters
        ----------
        product_name : str
            Product name, e.g. "iPS3".
        query_text : str
            Product-level query text.
        top_k : int
            Number of FS seeds to retrieve.

        Returns
        -------
        Dict[str, Any]
            Product-level function evidence.
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

        # Fallback: return top FS texts even if the heuristic is too strict.
        if not functions:
            for e in evidence:
                if e["fs_text"] not in functions:
                    functions.append(e["fs_text"])

        return {
            "query_type": "product_function",
            "product_name": product_name,
            "query_text": query_text,
            "functions": functions,
            "evidence": evidence,
        }

    def query_element_sentences_from_ts(
        self,
        element_name: str,
        query_text: str = "",
        top_k: int = 8
    ) -> Dict[str, Any]:
        """
        Retrieve element-relevant sentences with TSChunk as the primary
        retrieval entry point.

        Parameters
        ----------
        element_name : str
            Element name from structure analysis, e.g. "Motor control".
        query_text : str
            Optional refinement such as "function", "protection",
            "verification", or "qualification".
        top_k : int
            Number of TS seed nodes to retrieve.

        Returns
        -------
        Dict[str, Any]
            Structured multi-view evidence centered on the queried element.
        """
        seeds = self.retriever.retrieve_element_ts_seeds(
            element_name=element_name,
            function_text=query_text,
            top_k=top_k
        )

        evidence = []
        ts_sentences = []
        fs_sentences = []
        qd_sentences = []
        tst_sentences = []
        rationale_sentences = []

        def _extend_unique(target: List[str], values: List[str]) -> None:
            """
            Append values to a target list while preserving uniqueness and order.
            """
            for value in values:
                if value and value not in target:
                    target.append(value)

        for seed in seeds:
            ctx = self.retriever.expand_ts_context(seed["node_id"])

            ts_text = (ctx.get("ts_text") or seed.get("text") or "").strip()
            fs_texts = [x for x in ctx.get("fs_texts", []) if x]
            qd_texts = [x for x in ctx.get("qd_texts", []) if x]
            tst_texts = [x for x in ctx.get("tst_texts", []) if x]
            rationale_texts = [x for x in ctx.get("rationale_texts", []) if x]

            item = {
                "ts_text": ts_text,
                "score": seed["rrf_score"],
                "sources": seed["sources"],
                "fs_texts": fs_texts,
                "qd_texts": qd_texts,
                "tst_texts": tst_texts,
                "rationale_texts": rationale_texts,
            }
            evidence.append(item)

            if ts_text and ts_text not in ts_sentences:
                ts_sentences.append(ts_text)

            _extend_unique(fs_sentences, fs_texts)
            _extend_unique(qd_sentences, qd_texts)
            _extend_unique(tst_sentences, tst_texts)
            _extend_unique(rationale_sentences, rationale_texts)

        return {
            "query_type": "element_sentences",
            "element_name": element_name,
            "query_text": query_text,
            "ts_sentences": ts_sentences,
            "fs_sentences": fs_sentences,
            "qd_sentences": qd_sentences,
            "tst_sentences": tst_sentences,
            "rationale_sentences": rationale_sentences,
            "evidence": evidence,
        }

    def query(
        self,
        query_type: QueryType,
        query_text: str,
        top_k: int = 8,
        product_name: Optional[str] = None,
        element_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Unified query entry point for multiple query functions.

        Supported query types
        ---------------------
        - product_function:
            retrieve product-level function evidence from FS-centered search
        - element_sentences:
            retrieve element-related sentences from TS-centered search

        Parameters
        ----------
        query_type : QueryType
            Type of query to execute.
        query_text : str
            User query text or retrieval refinement.
        top_k : int
            Retrieval depth.
        product_name : Optional[str]
            Required for product_function queries.
        element_name : Optional[str]
            Required for element_sentences queries.

        Returns
        -------
        Dict[str, Any]
            Structured query result.
        """
        if query_type == "product_function":
            if not product_name:
                raise ValueError("product_name is required for query_type='product_function'.")
            return self.answer_product_functions(
                product_name=product_name,
                query_text=query_text,
                top_k=top_k,
            )

        if query_type == "element_sentences":
            if not element_name:
                raise ValueError("element_name is required for query_type='element_sentences'.")
            return self.query_element_sentences_from_ts(
                element_name=element_name,
                query_text=query_text,
                top_k=top_k,
            )

        raise ValueError(f"Unsupported query_type: {query_type}")

    def batch_query(
        self,
        query_specs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple heterogeneous queries in one batch.

        Example
        -------
        query_specs = [
            {
                "query_type": "product_function",
                "product_name": "iPS3",
                "query_text": "What are the functions of iPS3?",
                "top_k": 50,
            },
            {
                "query_type": "element_sentences",
                "element_name": "Motor control",
                "query_text": "function protection verification",
                "top_k": 30,
            },
        ]

        Returns
        -------
        List[Dict[str, Any]]
            A list of structured query results.
        """
        results = []

        for spec in query_specs:
            result = self.query(
                query_type=spec["query_type"],
                query_text=spec.get("query_text", ""),
                top_k=spec.get("top_k", 8),
                product_name=spec.get("product_name"),
                element_name=spec.get("element_name"),
            )
            results.append(result)

        return results