# main.py

from rag_pipeline import ChunkGraphRAG


def main():
    rag = ChunkGraphRAG()

    query = "What are the functions of this sub-element: iPS3"
    product_name = "iPS3"

    result = rag.answer_product_functions(
        product_name=product_name,
        query_text=query,
        top_k=50
    )

    print("=" * 100)
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

    rag.close()


if __name__ == "__main__":
    main()