from rag_pipeline import ChunkGraphRAG


def print_product_function_result(result):
    """
    Pretty-print the result of a product-level function query.
    """
    print("=" * 100)
    print("Query Type:", result["query_type"])
    print("Product:", result["product_name"])
    print("Query:", result["query_text"])
    print("=" * 100)

    print("\nFunctions:")
    for i, func in enumerate(result["functions"], start=1):
        print(f"{i}. {func}")

    print("\nEvidence:")
    for i, e in enumerate(result["evidence"], start=1):
        print("-" * 100)
        print(f"[{i}] score={e['score']:.4f} sources={e['sources']}")
        print("FS:", e["fs_text"])
        print("TS:", e["ts_texts"])
        print("QD:", e["qd_texts"])
        print("TST:", e["tst_texts"])
        print("RATIONALE:", e["rationale_texts"])


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
        print(f"[{i}] score={e['score']:.4f} sources={e['sources']}")
        print("TS:", e["ts_text"])
        print("FS:", e["fs_texts"])
        print("QD:", e["qd_texts"])
        print("TST:", e["tst_texts"])
        print("RATIONALE:", e["rationale_texts"])


def main():
    rag = ChunkGraphRAG()

    # ------------------------------------------------------------------
    # Choose the query type here:
    #   1. "product_function"
    #   2. "element_sentences"
    # ------------------------------------------------------------------
    # query_spec = {
    #     "query_type": "product_function",
    #     "product_name": "iPS3",
    #     "query_text": "What are the functions of this product: iPS3",
    #     "top_k": 100,
    # }

    # Example for element query:
    query_spec = {
        "query_type": "element_sentences",
        "element_name": "Motor control",
        "function_text": "Soft starter",
        "top_k": 15,
    }

    try:
        result = rag.query(
            query_type=query_spec["query_type"],
            query_text=query_spec.get("function_text", ""),
            top_k=query_spec.get("top_k", 8),
            product_name=query_spec.get("product_name"),
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


if __name__ == "__main__":
    main()