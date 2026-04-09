from rag_pipeline import ChunkGraphRAG
from fmea_retriever import FMEASentenceRetriever


def print_score_breakdown(evidence: dict):
    breakdown = evidence.get("score_breakdown") or {}
    raw = breakdown.get("raw_components") or {}
    weighted = breakdown.get("weighted_components") or {}

    if not breakdown:
        return

    print(
        "    scores:"
        f" final={breakdown.get('final_score', evidence.get('score', 0.0)):.4f}"
        f" prior={breakdown.get('retrieval_prior', evidence.get('rrf_score', 0.0)):.4f}"
    )

    if raw:
        print(
            "    raw:"
            + "".join(
                f" {name}={value:.4f}"
                for name, value in raw.items()
            )
        )

    if weighted:
        print(
            "    weighted:"
            + "".join(
                f" {name}={value:.4f}"
                for name, value in weighted.items()
            )
        )


def print_product_function_result(result: dict):
    """
    Pretty-print the result of a product-level function query.
    Supports both old and new field names.
    """
    product_text = result.get("product_text", result.get("product_name", ""))
    function_text = result.get("function_text")

    print("=" * 100)
    print("Query Type :", result.get("query_type", ""))
    print("Product    :", product_text)
    print("Function   :", function_text)
    print("=" * 100)

    functions = result.get("functions", [])
    evidence = result.get("evidence", [])

    print("\nFunctions:")
    if not functions:
        print("  (none)")
    else:
        for i, func in enumerate(functions, start=1):
            print(f"{i}. {func}")

    print("\nEvidence:")
    if not evidence:
        print("  (none)")
        return

    for i, e in enumerate(evidence, start=1):
        print("-" * 100)
        print(
            f"[{i}] final_score={e.get('score', 0.0):.4f} "
            f"rrf_score={e.get('rrf_score', 0.0):.4f}"
        )
        print_score_breakdown(e)

        print(f"    fs_node_id={e.get('fs_node_id', '')}")
        print(f"    sources={e.get('sources', [])}")

        print("FS:")
        print(f"  {e.get('fs_text', '')}")

        print("TS:")
        ts_texts = e.get("ts_texts", [])
        if ts_texts:
            for j, text in enumerate(ts_texts, start=1):
                print(f"  {j}. {text}")
        else:
            print("  (none)")

        print("RELATED FS:")
        dfs_texts = e.get("dfs_texts", [])
        if dfs_texts:
            for j, text in enumerate(dfs_texts, start=1):
                print(f"  {j}. {text}")
        else:
            print("  (none)")

        # Backward compatibility with old QD output
        qd_texts = e.get("qd_texts", [])
        if qd_texts:
            print("QD:")
            for j, text in enumerate(qd_texts, start=1):
                print(f"  {j}. {text}")

        print("TST:")
        tst_texts = e.get("tst_texts", [])
        if tst_texts:
            for j, text in enumerate(tst_texts, start=1):
                print(f"  {j}. {text}")
        else:
            print("  (none)")

        print("RATIONALE:")
        rationale_texts = e.get("rationale_texts", [])
        if rationale_texts:
            for j, text in enumerate(rationale_texts, start=1):
                print(f"  {j}. {text}")
        else:
            print("  (none)")


def print_element_sentences_result(result):
    """
    Pretty-print the result of an element-level sentence query.
    """
    print("=" * 100)
    print("Query Type:", result["query_type"])
    print("Element:", result["element_name"])
    print("Query:", result["query_text"])
    print("=" * 100)

    print("\nTS Sentences:")
    for i, text in enumerate(result["ts_sentences"], start=1):
        print(f"{i}. {text}")

    print("\nFS Sentences:")
    for i, text in enumerate(result["fs_sentences"], start=1):
        print(f"{i}. {text}")

    print("\nQD Sentences:")
    for i, text in enumerate(result["qd_sentences"], start=1):
        print(f"{i}. {text}")

    print("\nTST Sentences:")
    for i, text in enumerate(result["tst_sentences"], start=1):
        print(f"{i}. {text}")

    print("\nRationale Sentences:")
    for i, text in enumerate(result["rationale_sentences"], start=1):
        print(f"{i}. {text}")

    print("\nEvidence:")
    for i, e in enumerate(result["evidence"], start=1):
        print("-" * 100)
        print(
            f"[{i}] final_score={e.get('score', 0.0):.4f} "
            f"rrf_score={e.get('rrf_score', 0.0):.4f} "
            f"sources={e['sources']}"
        )
        print_score_breakdown(e)
        print("TS:", e["ts_text"])
        print("FS:", e["fs_texts"])
        print("QD:", e["qd_texts"])
        print("TST:", e["tst_texts"])
        print("RATIONALE:", e["rationale_texts"])


def print_fmea_sentence_result(result):
    """
    Pretty-print the result of an FMEA sentence linking query.
    """
    print("=" * 100)
    print("Query Type: fmea_sentence_link")
    print("Fields:", result.get("query_fields", {}))
    print("=" * 100)

    print("\nDense Queries:")
    dense_queries = result.get("dense_queries", [])
    if dense_queries:
        for i, text in enumerate(dense_queries, start=1):
            print(f"{i}. {text}")
    else:
        print("  (none)")

    print("\nSparse Queries:")
    sparse_queries = result.get("sparse_queries", [])
    if sparse_queries:
        for i, text in enumerate(sparse_queries, start=1):
            print(f"{i}. {text}")
    else:
        print("  (none)")

    print("\nGroups:")
    groups = result.get("groups", [])
    if not groups:
        print("  (none)")
        return

    for i, group in enumerate(groups, start=1):
        print("-" * 100)
        print(
            f"[{i}] score={group.get('score', 0.0):.4f} "
            f"node_count={group.get('node_count', 0)}"
        )

        print("RELATIONSHIPS:")
        relationships = group.get("relationships", [])
        if relationships:
            for rel in relationships:
                print(
                    f"  {rel.get('source_id', '')} -> {rel.get('target_id', '')} "
                    f"{rel.get('relationships', [])}"
                )
        else:
            print("  (none)")

        print("NODES:")
        for node in group.get("nodes", []):
            label = node.get("label") or (node.get("labels") or [""])[0]
            print(
                f"  label={label} final_score={node.get('final_score', 0.0):.4f} "
                f"node_id={node.get('node_id', '')}"
            )
            weighted_scores = node.get("weighted_scores", {})
            if weighted_scores:
                print(
                    "    weighted="
                    + " ".join(
                        f"{name}={value:.4f}"
                        for name, value in weighted_scores.items()
                    )
                )
            print(f"    text={node.get('text', '')}")


def main():
    # ------------------------------------------------------------------
    # Choose the demo type here:
    #   1. "graphrag"
    #   2. "fmea_sentence_link"
    # ------------------------------------------------------------------
    demo_type = "graphrag"

    if demo_type == "graphrag":
        rag = ChunkGraphRAG()

        # ------------------------------------------------------------------
        # Choose the query type here:
        #   1. "product_function"
        #   2. "element_sentences"
        # ------------------------------------------------------------------
        query_spec = {
            "query_type": "product_function",
            "product_name": "iPS3",
            "function_text": "Soft starter",
            "top_k": 15,
        }

        ## Example for element query:
        # query_spec = {
        #     "query_type": "element_sentences",
        #     "element_name": "Relay switching",
        #     "function_text": "turn on/turn off",
        #     "top_k": 15,
        # }

        try:
            result = rag.query(
                query_type=query_spec["query_type"],
                query_text=query_spec.get("function_text", ""),
                top_k=query_spec.get("top_k", 8),
                product_text=query_spec.get("product_name"),
                element_name=query_spec.get("element_name"),
            )

            if result["query_type"] == "product_function":
                print_product_function_result(result)

            elif result["query_type"] == "element_sentences":
                print_element_sentences_result(result)

            else:
                raise ValueError(f"Unsupported result query type: {result['query_type']}")

        finally:
            rag.close()

    elif demo_type == "fmea_sentence_link":
        retriever = FMEASentenceRetriever()

        query_spec = {
            # "product_text": "iPS3",
            "element_text": "Motor control",
            "function_text": "Relay switching",
            "mode_text": " relay, close, turn on/turn off",
            # "effect_text": "current surge",
            # "cause_text": "overvoltage",
            "top_k": 10,
        }

        try:
            result = retriever.query_fmea_sentences(
                product_text=query_spec.get("product_text", ""),
                function_text=query_spec.get("function_text", ""),
                effect_text=query_spec.get("effect_text", ""),
                element_text=query_spec.get("element_text", ""),
                mode_text=query_spec.get("mode_text", ""),
                cause_text=query_spec.get("cause_text", ""),
                top_k=query_spec.get("top_k", 8),
            )
            print_fmea_sentence_result(result)
        finally:
            retriever.close()

    else:
        raise ValueError(f"Unsupported demo type: {demo_type}")


if __name__ == "__main__":
    main()
