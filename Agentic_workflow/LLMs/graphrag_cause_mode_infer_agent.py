import json
import re
from typing import Any, Dict, List, Optional

from langsmith import traceable

from .llm_init import get_llm_backend


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON object found in LLM response.")

    return json.loads(match.group(0))


class GraphRAGCauseModeInferenceAgent:
    def __init__(self, backend: str = "openai", model: Optional[str] = None):
        self.llm = get_llm_backend(
            backend=backend,
            model=model,
            temperature=0.4,
            json_mode=True,
        )

    def _build_prompt(self, payload: Dict[str, Any]) -> str:
        schema = {
            "cause_text": "string",
            "discipline": "string",
            "failure_element": "string",
            "selected_modes": [
                {
                    "mode_text": "string",
                    "function_text": "string",
                    "selected": True,
                    "confidence": "high",
                    "support_type": "complete_entity",
                    "reason": "short explanation grounded in the sentences",
                    "supporting_sentence_ids": [1, 2],
                }
            ],
            "global_reasoning": "short summary for cause -> mode inference",
            "effect_inference_status": "pending",
        }

        return f"""
You are an FMEA reasoning agent for GraphRAG sentence evidence.

Task:
1. Read one cause_text with its sentence_group evidence.
2. Infer which candidate mode or modes are best supported by the cause evidence.
3. Treat the cause_text as fixed input. You must only choose from the provided mode candidates.
4. The candidates are grouped by function. One function can contain multiple modes.
5. Use the sentence_group first, then decide which modes are plausible under each function.
6. Prefer direct technical links over vague wording overlap.
7. Normally select 1 mode. Select 2 only when both are clearly supported.
8. Do not infer effects yet. Keep effect_inference_status as "pending".

Support type choices:
- complete_entity
- composed_from_multiple
- partial_pattern
- no_direct_gt

Confidence choices:
- high
- medium
- low

Decision guidance:
- "Priority zero-crossing interrupt too low" should prefer modes consistent with zero-crossing detection timing and interpretation.
- "Live switching of relays" should prefer relay-related modes or soft-start behavior only when the sentence evidence supports that causal chain.
- If sentence evidence is weak, still return the most plausible candidate but use lower confidence and weaker support type.

Read the input like this:
- cause_text, discipline, failure_element: fixed context
- evidence.sentence_group: merged sentence evidence in plain text
- evidence.function_mode_candidates: function -> mode_candidates mapping

Input:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}
""".strip()

    def _fallback_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        grouped_candidates = payload.get("evidence", {}).get("function_mode_candidates", [])
        best_candidate = {}
        for group in grouped_candidates:
            function_text = group.get("function_text", "")
            for mode_text in group.get("mode_candidates", []):
                best_candidate = {
                    "mode_text": mode_text,
                    "function_text": function_text,
                }
                break
            if best_candidate:
                break

        selected_modes = []
        if best_candidate:
            selected_modes.append({
                "mode_text": best_candidate.get("mode_text", ""),
                "function_text": best_candidate.get("function_text", ""),
                "selected": True,
                "confidence": "low",
                "support_type": "partial_pattern",
                "reason": "Fallback selected the first aggregated mode candidate.",
                "supporting_sentence_ids": [],
            })

        return {
            "cause_text": payload.get("cause", ""),
            "discipline": payload.get("discipline", ""),
            "failure_element": payload.get("failure_element", ""),
            "selected_modes": selected_modes,
            "global_reasoning": "Fallback used aggregated candidate order only.",
            "effect_inference_status": "pending",
        }

    def _clean_result(
        self,
        parsed: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        valid_support_types = {
            "complete_entity",
            "composed_from_multiple",
            "partial_pattern",
            "no_direct_gt",
        }
        valid_confidence = {"high", "medium", "low"}
        valid_mode_pairs = set()
        for group in payload.get("evidence", {}).get("function_mode_candidates", []):
            function_text = group.get("function_text", "")
            for mode_text in group.get("mode_candidates", []):
                valid_mode_pairs.add((mode_text, function_text))

        cleaned_modes = []
        for mode in parsed.get("selected_modes", []):
            mode_text = mode.get("mode_text", "")
            function_text = mode.get("function_text", "")
            if (mode_text, function_text) not in valid_mode_pairs:
                continue

            support_type = mode.get("support_type", "no_direct_gt")
            if support_type not in valid_support_types:
                support_type = "no_direct_gt"

            confidence = mode.get("confidence", "low")
            if confidence not in valid_confidence:
                confidence = "low"

            cleaned_modes.append({
                "mode_text": mode_text,
                "function_text": function_text,
                "selected": bool(mode.get("selected", True)),
                "confidence": confidence,
                "support_type": support_type,
                "reason": mode.get("reason", ""),
                "supporting_sentence_ids": [],
            })

        if not cleaned_modes and payload.get("evidence", {}).get("function_mode_candidates"):
            return self._fallback_result(payload)

        return {
            "cause_text": payload.get("cause", ""),
            "discipline": payload.get("discipline", ""),
            "failure_element": payload.get("failure_element", ""),
            "selected_modes": cleaned_modes,
            "global_reasoning": parsed.get("global_reasoning", ""),
            "effect_inference_status": "pending",
        }

    @traceable(
        run_type="chain",
        name="graphrag_cause_mode_inference",
        tags=["fmea", "graphrag", "cause_mode", "entity_infer"],
    )
    def infer_cause_to_mode(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not payload.get("evidence", {}).get("function_mode_candidates"):
            return {
                "cause_text": payload.get("cause", ""),
                "discipline": payload.get("discipline", ""),
                "failure_element": payload.get("failure_element", ""),
                "selected_modes": [],
                "global_reasoning": "No mode candidates were available for this cause group.",
                "effect_inference_status": "pending",
            }

        prompt = self._build_prompt(payload)

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = extract_json(content)
            return self._clean_result(parsed, payload)
        except Exception as exc:
            print(f"[WARN] cause -> mode inference failed, fallback used: {exc}")
            return self._fallback_result(payload)


def pretty_print_cause_mode_results(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("GRAPHRAG CAUSE -> MODE INFERENCE")
    print("=" * 80)

    for item in results:
        print(f"\nCause: {item.get('cause_text', '')}")
        print(f"Discipline: {item.get('discipline', '')}")
        print(f"Failure Element: {item.get('failure_element', '')}")
        for mode in item.get("selected_modes", []):
            print(f"  - Mode       : {mode.get('mode_text', '')}")
            print(f"    Function   : {mode.get('function_text', '')}")
            print(f"    Selected   : {mode.get('selected', True)}")
            print(f"    Confidence : {mode.get('confidence', '')}")
            print(f"    SupportType: {mode.get('support_type', '')}")
            print(f"    Reason     : {mode.get('reason', '')}")
            print(f"    Sentences  : {mode.get('supporting_sentence_ids', [])}")
        print(f"  Summary: {item.get('global_reasoning', '')}")
        print(f"  Effect inference: {item.get('effect_inference_status', '')}")

    print("=" * 80)
