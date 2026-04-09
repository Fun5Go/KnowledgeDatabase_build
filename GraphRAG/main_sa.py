from .fmea_retriever import FMEASentenceRetriever
from .fmea_retrieverV2 import FMEASentenceRetrieverV2


def print_structure_group_results(result: dict):
    print("=" * 100)
    print("Query Type: structure_to_fmea_text_groups")
    print(f"Flattened Queries: {len(result.get('queries', []))}")
    print("=" * 100)

    groups = result.get("groups", [])
    if not groups:
        print("\nGroups:")
        print("  (none)")
        return

    print("\nGroups:")
    for i, group in enumerate(groups, start=1):
        print("-" * 100)
        print(
            f"[{i}] score={group.get('score', 0.0):.4f} "
            f"hit_count={group.get('hit_count', 0)} "
            f"group_id={group.get('group_id', '')}"
        )

        print("Matched Queries:")
        matched_queries = group.get("matched_queries", [])
        if matched_queries:
            for match in matched_queries:
                print(
                    f"  [{match.get('query_type', '')}] {match.get('query_text', '')} "
                    f"| group_score={match.get('group_score', 0.0):.4f}"
                )
        else:
            print("  (none)")

        print("Relationships:")
        relationships = group.get("relationships", [])
        if relationships:
            for rel in relationships:
                print(
                    f"  {rel.get('source_id', '')} -> {rel.get('target_id', '')} "
                    f"{rel.get('relationships', [])}"
                )
        else:
            print("  (none)")

        print("Texts:")
        for node in group.get("nodes", []):
            label = node.get("label") or (node.get("labels") or [""])[0]
            print(
                f"  label={label} final_score={node.get('final_score', 0.0):.4f} "
                f"node_id={node.get('node_id', '')}"
            )
            print(f"    text={node.get('text', '')}")


def print_structure_sentence_support(result: dict):
    print("=" * 100)
    print("Query Type: structure_analysis_sentence_support")
    print("=" * 100)

    print("\nEffect Support:")
    effect_support = result.get("effect_support", [])
    if effect_support:
        for i, item in enumerate(effect_support, start=1):
            query_spec = item.get("query_spec", {})
            print("-" * 100)
            print(
                f"[{i}] function={query_spec.get('function_text', '')} "
                f"effect={query_spec.get('effect_text', '')}"
            )
            sentences = item.get("sentences", [])
            if not sentences:
                print("  (no supporting sentences)")
            for sentence in sentences:
                print(
                    f"  label={sentence.get('label', '')} "
                    f"score={sentence.get('final_score', 0.0):.4f} "
                    f"node_id={sentence.get('node_id', '')}"
                )
                print(f"    text={sentence.get('text', '')}")
    else:
        print("  (none)")

    print("\nMode Support:")
    mode_support = result.get("mode_support", [])
    if mode_support:
        for i, item in enumerate(mode_support, start=1):
            query_spec = item.get("query_spec", {})
            print("-" * 100)
            print(
                f"[{i}] element={query_spec.get('element_text', '')} "
                f"function={query_spec.get('function_text', '')} "
                f"mode={query_spec.get('mode_text', '')}"
            )
            sentences = item.get("sentences", [])
            if not sentences:
                print("  (no supporting sentences)")
            for sentence in sentences:
                print(
                    f"  label={sentence.get('label', '')} "
                    f"score={sentence.get('final_score', 0.0):.4f} "
                    f"node_id={sentence.get('node_id', '')}"
                )
                print(f"    text={sentence.get('text', '')}")
    else:
        print("  (none)")

    print("\nCause Support:")
    cause_support = result.get("cause_support", {})
    print(f"  {cause_support.get('status', 'TODO')}: {cause_support.get('message', '')}")


def print_effect_support(result: dict):
    print("=" * 100)
    print("Query Type: direct_effect_sentence_support")
    print(
        f"Product: {result.get('product_text', '')} | "
        f"Function: {result.get('function_text', '')}"
    )
    print("=" * 100)

    effect_support = result.get("effect_support", [])
    if not effect_support:
        print("\nEffect Support:")
        print("  (none)")
        return

    print("\nEffect Support:")
    for i, item in enumerate(effect_support, start=1):
        query_spec = item.get("query_spec", {})
        print("-" * 100)
        print(f"[{i}] effect={query_spec.get('effect_text', '')}")
        sentences = item.get("sentences", [])
        if not sentences:
            print("  (no supporting sentences)")
            continue
        for sentence in sentences:
            print(
                f"  label={sentence.get('label', '')} "
                f"score={sentence.get('final_score', 0.0):.4f} "
                f"node_id={sentence.get('node_id', '')}"
            )
            print(f"    text={sentence.get('text', '')}")


def print_v2_support(result: dict, support_key: str):
    print("=" * 100)
    print(f"Query Type: v2_{support_key}")
    print(
        f"Product: {result.get('product_text', '')} | "
        f"Element: {result.get('element_text', '')} | "
        f"Function: {result.get('function_text', '')}"
    )
    print("=" * 100)

    items = result.get(support_key, [])
    if not items:
        print(f"\n{support_key}:")
        print("  (none)")
        return

    print(f"\n{support_key}:")
    for i, item in enumerate(items, start=1):
        query_item = item.get("query_item", {})
        print("-" * 100)
        print(
            f"[{i}] target={query_item.get('target_text', '')} "
            f"| positive_variants={query_item.get('positive_target_variants', [])}"
        )

        evidence = item.get("evidence", [])
        if not evidence:
            print("  (no evidence)")
            continue

        for pack in evidence:
            primary = pack.get("primary_sentence", {})
            print(
                f"  primary label={primary.get('label', '')} "
                f"score={primary.get('score', 0.0):.4f} "
                f"node_id={primary.get('node_id', '')}"
            )
            print(f"    text={primary.get('text', '')}")

            supporting_context = pack.get("supporting_context", [])
            if supporting_context:
                for ctx in supporting_context:
                    print(
                        f"    support {ctx.get('label', '')} "
                        f"{ctx.get('relationship', '')}: {ctx.get('text', '')}"
                    )


def main():
    structure_input_motorcontrol = {
        "product_domain": "motor_drives",
        "nodes": [
            {
                "element_id": "E1",
                "failure_element": "Motor control",
                "modes": {
                    "Soft starter": [
                        "Component break-down",
                        "Unbalanced motor currents",
                    ],
                    "Zero-crossing detection": [
                        "Incorrect interpretation zero-crossing",
                        "Soft start too long",
                        "No detection",
                    ],
                    "Relay switching": [
                        "Welded relay",
                        "Relay cannot close",
                        "False turn-on / turn-off",
                    ],
                },
                "causes": {
                    "mechanics": [
                        "Cooling insufficient",
                        "Compressor vibrations",
                    ],
                    "hardware": [
                        "(Starting) Motor current too high for chosen components",
                        "Overvoltage due to motor disconnect",
                        "Under Voltage due to incorrect triggering",
                        "Live switching of relays",
                    ],
                    "software": [
                        "Priority zero-crossing interrupt too low",
                        "Open loop control",
                    ],
                    "other": [
                        "No (correctly designed) snubber design",
                        "Too high dT junction as a result of power cycling of component",
                    ],
                },
                "effects": [
                    "Motor cannot start",
                    "Overcurrent towards motor",
                    "Motor starts without soft start",
                ],
            }
        ],
    }

    # retriever = FMEASentenceRetriever()
    # try:
    #     direct_effect_support = retriever.query_effect_support(
    #         product_text="iPS3",
    #         function_text="Soft starter",
    #         effect_texts=[
    #             "Motor cannot start",
    #             "Motor starts without soft start",
    #             "Overcurrent towards motor",
    #         ],
    #         top_k_per_effect=6,
    #         sentence_top_k=5,
    #     )
    #     print_effect_support(direct_effect_support)

        # print("\n")
        # sentence_support = retriever.query_structure_analysis_support(
        #     structure_input=structure_input_motorcontrol,
        #     top_k_per_query=6,
        #     sentence_top_k=5,
        # )
        # print_structure_sentence_support(sentence_support)

        # print("\n")
        # result = retriever.query_structure_input(
        #     structure_input=structure_input_motorcontrol,
        #     top_k_per_query=6,
        #     aggregate_top_k=20,
        # )
        # print_structure_group_results(result)
    # finally:
    #     retriever.close()

    retriever_v2 = FMEASentenceRetrieverV2()
    try:
        # effect_support_v2 = retriever_v2.query_effect_support(
        #     product_text="iPS3",
        #     function_text="Soft starter",
        #     effect_texts=[
        #         "Motor cannot start",
        #         "Motor starts without soft start",
        #         "Overcurrent towards motor",
        #     ],
        #     top_k_per_effect=5,
        #     expand_positive_variants=True,
        # )
        # print("\n")
        # print_v2_support(effect_support_v2, "effect_support")

        mode_support_v2 = retriever_v2.query_mode_support(
            element_text="Motor control",
            function_text="Relay switching",
            mode_texts=[
                      "Priority zero-crossing interrupt too low",
                        "Relay cannot close",
                        # "False turn-on / turn-off",
            ],
            top_k_per_mode=5,
            expand_positive_variants=True,
        )
        print("\n")
        print_v2_support(mode_support_v2, "mode_support")
    finally:
        retriever_v2.close()


if __name__ == "__main__":
    main()
