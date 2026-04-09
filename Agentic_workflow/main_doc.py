import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langsmith import traceable

from Agentic_workflow.LLMs.entity_infer_agent import (
    FMEAEntityInferenceAgent,
    pretty_print_entity_inference,
)
from GraphRAG.fmea_retrieverV2 import FMEASentenceRetrieverV2


load_dotenv()


def build_structure_context(
    structure_input: Dict[str, Any],
    function_text: str,
) -> Dict[str, Any]:
    product_text = (structure_input.get("product_domain") or "").strip()

    for node in structure_input.get("nodes", []):
        modes = node.get("modes", {})
        if function_text not in modes:
            continue

        return {
            "product_text": product_text,
            "element_text": (node.get("failure_element") or "").strip(),
            "function_text": function_text,
            "candidate_modes": [
                value.strip()
                for value in modes.get(function_text, [])
                if isinstance(value, str) and value.strip()
            ],
            "candidate_effects": [
                value.strip()
                for value in node.get("effects", [])
                if isinstance(value, str) and value.strip()
            ],
            "candidate_causes": [
                value.strip()
                for values in (node.get("causes") or {}).values()
                if isinstance(values, list)
                for value in values
                if isinstance(value, str) and value.strip()
            ],
        }

    return {
        "product_text": product_text,
        "element_text": "",
        "function_text": function_text,
        "candidate_modes": [],
        "candidate_effects": [],
        "candidate_causes": [],
    }


def print_mode_support_summary(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("DOCUMENT MODE SUPPORT")
    print("=" * 100)

    for item in result.get("mode_support", []):
        query_item = item.get("query_item", {})
        print(f"\nMode: {query_item.get('target_text', '')}")
        for evidence in item.get("evidence", []):
            primary = evidence.get("primary_sentence", {})
            print(
                f"  primary [{primary.get('label', '')}] "
                f"score={primary.get('score', 0.0):.4f}"
            )
            print(f"    {primary.get('text', '')}")

            for support in evidence.get("supporting_context", []):
                print(
                    f"    -> {support.get('relationship', '')} "
                    f"[{support.get('label', '')}] {support.get('text', '')}"
                )


@traceable(
    run_type="chain",
    name="fmea_doc_mode_retrieval",
    tags=["fmea", "doc", "retrieval", "mode_support"],
)
def run_doc_mode_retrieval(
    retriever: FMEASentenceRetrieverV2,
    structure_context: Dict[str, Any],
    mode_texts: List[str],
    top_k_per_mode: int = 3,
) -> Dict[str, Any]:
    return retriever.query_mode_support(
        element_text=structure_context["element_text"],
        function_text=structure_context["function_text"],
        mode_texts=mode_texts,
        top_k_per_mode=top_k_per_mode,
        expand_positive_variants=True,
    )


@traceable(
    run_type="chain",
    name="fmea_doc_entity_inference",
    tags=["fmea", "doc", "llm", "entity_infer"],
)
def run_doc_entity_inference(
    agent: FMEAEntityInferenceAgent,
    mode_support_result: Dict[str, Any],
    structure_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return agent.infer_entities_from_mode_support(
        mode_support_result=mode_support_result,
        structure_context=structure_context,
    )


@traceable(
    run_type="chain",
    name="fmea_doc_experiment",
    tags=["fmea", "doc", "pipeline", "mode_support"],
)
def run_doc_experiment(
    retriever: FMEASentenceRetrieverV2,
    agent: FMEAEntityInferenceAgent,
    structure_context: Dict[str, Any],
    mode_texts: List[str],
    top_k_per_mode: int = 3,
) -> Dict[str, Any]:
    mode_support_result = run_doc_mode_retrieval(
        retriever=retriever,
        structure_context=structure_context,
        mode_texts=mode_texts,
        top_k_per_mode=top_k_per_mode,
    )
    inferred_entities = run_doc_entity_inference(
        agent=agent,
        mode_support_result=mode_support_result,
        structure_context=structure_context,
    )
    return {
        "structure_context": structure_context,
        "mode_support_result": mode_support_result,
        "inferred_entities": inferred_entities,
    }


def main() -> None:
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

    function_text = "Soft starter"
    mode_texts = [
                        "Component break-down",
                        "Unbalanced motor currents",
    ]

    structure_context = build_structure_context(
        structure_input=structure_input_motorcontrol,
        function_text=function_text,
    )

    retriever = FMEASentenceRetrieverV2()
    agent = FMEAEntityInferenceAgent(
        backend=os.getenv("LLM_BACKEND", "openai"),
        model=os.getenv("LLM_MODEL", "azure/gpt-4.1"),
    )

    try:
        result = run_doc_experiment(
            retriever=retriever,
            agent=agent,
            structure_context=structure_context,
            mode_texts=mode_texts,
            top_k_per_mode=3,
        )
    finally:
        retriever.close()

    mode_support_result = result["mode_support_result"]
    inferred = result["inferred_entities"]

    print_mode_support_summary(mode_support_result)
    pretty_print_entity_inference(inferred)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
