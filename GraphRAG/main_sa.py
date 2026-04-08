from fmea_retriever import FMEASentenceRetriever


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

    retriever = FMEASentenceRetriever()
    try:
        result = retriever.query_structure_input(
            structure_input=structure_input_motorcontrol,
            top_k_per_query=6,
            aggregate_top_k=20,
        )
        print_structure_group_results(result)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
