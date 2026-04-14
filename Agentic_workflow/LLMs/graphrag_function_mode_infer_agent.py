import json
import re
from typing import Any, Dict, List, Optional, Tuple

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


def _normalize_text(value: str) -> str:
    value = (value or "").replace("\u0000", "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


class GraphRAGFunctionModeInferenceAgent:
    def __init__(self, backend: str = "openai", model: Optional[str] = None):
        self.llm = get_llm_backend(
            backend=backend,
            model=model,
            temperature=0.0,
            json_mode=True,
        )

    def _build_prompt(self, payload: Dict[str, Any]) -> str:
        schema = {
            "failure_element": "string",
            "inferred_modes": [
                {
                    "function_text": "string",
                    "mode_text": "string",
                    "selected": True,
                    "confidence": "high",
                    "reason": "short engineering explanation grounded in the text",
                    "evidence_sentences": [
                        "verbatim sentence or clause copied from the source text"
                    ],
                }
            ],
            "global_reasoning": "short summary",
        }

        return f"""
You are an FMEA reasoning agent for technical specification text.

Task:
1. Read one technical specification entry for a failure element.
2. The entry contains:
   - "TechnicalSpecification Choice of Motor control": the specification topic
   - "Choice": the original technical requirement text
   - "Rationale": the supporting explanation text, if available
3. Use "Choice" as the primary evidence.
4. Use "Rationale" as supporting context only when it helps clarify the intended failure implication.
5. The input contains several matched functions, and each function has its own candidate failure modes.
6. Infer the potential failure mode or modes implied by the technical specification entry.
7. You must ONLY choose from the provided candidates under the provided matched functions.
8. Return every clearly supported mode. Several modes may be valid.
9. For every selected mode, you must attach the function_text it belongs to.
10. For every selected mode, attach evidence_sentences copied from the original Choice or Rationale text.
11. Evidence must be verbatim text spans from the source text, not paraphrases.
12. Use engineering logic, but keep every decision grounded in the source text.
13. Do not invent new functions or new failure modes.

Decision guidance:
- One source text can support several failure modes across different functions.
- Prefer candidates that are directly implied by timing checks, detection logic, error reporting, startup behavior, unstable transitions, current behavior, or relay behavior in the text.
- If a function has no well-supported candidate mode, do not select any mode from that function.
- Keep reasons concise and practical.

Confidence choices:
- high
- medium
- low

Input:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}
""".strip()

    def _fallback_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "failure_element": payload.get("failure_element", ""),
            "inferred_modes": [],
            "global_reasoning": "Fallback returned no inferred mode because the LLM result was unavailable.",
        }

    def _valid_function_mode_pairs(self, payload: Dict[str, Any]) -> set[Tuple[str, str]]:
        valid_pairs: set[Tuple[str, str]] = set()
        for item in payload.get("matched_functions", []):
            function_text = (item.get("function_text") or "").strip()
            for mode_text in item.get("Failure Mode", []):
                cleaned_mode = (mode_text or "").strip()
                if function_text and cleaned_mode:
                    valid_pairs.add((function_text, cleaned_mode))
        return valid_pairs

    def _clean_result(self, parsed: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        valid_pairs = self._valid_function_mode_pairs(payload)
        valid_confidence = {"high", "medium", "low"}
        choice_text = _normalize_text(payload.get("Choice", ""))
        rationale_text = _normalize_text(payload.get("Rationale", ""))
        source_text = f"{choice_text} {rationale_text}".strip()

        cleaned_modes: List[Dict[str, Any]] = []
        seen_pairs = set()

        for item in parsed.get("inferred_modes", []):
            function_text = (item.get("function_text") or "").strip()
            mode_text = (item.get("mode_text") or "").strip()
            pair = (function_text, mode_text)
            if pair not in valid_pairs or pair in seen_pairs:
                continue

            confidence = (item.get("confidence") or "low").strip().lower()
            if confidence not in valid_confidence:
                confidence = "low"

            evidence_sentences: List[str] = []
            for evidence in item.get("evidence_sentences", []):
                evidence_text = (evidence or "").strip()
                if not evidence_text:
                    continue
                if _normalize_text(evidence_text) in source_text:
                    evidence_sentences.append(evidence_text)

            cleaned_modes.append({
                "function_text": function_text,
                "mode_text": mode_text,
                "selected": bool(item.get("selected", True)),
                "confidence": confidence,
                "reason": (item.get("reason") or "").strip(),
                "evidence_sentences": evidence_sentences,
            })
            seen_pairs.add(pair)

        return {
            "failure_element": payload.get("failure_element", ""),
            "inferred_modes": cleaned_modes,
            "global_reasoning": (parsed.get("global_reasoning") or "").strip(),
        }

    @traceable(
        run_type="chain",
        name="graphrag_function_mode_inference",
        tags=["fmea", "graphrag", "function_mode", "llm"],
    )
    def infer_function_to_mode(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not payload.get("matched_functions"):
            return {
                "failure_element": payload.get("failure_element", ""),
                "inferred_modes": [],
                "global_reasoning": "No matched-function candidate groups were available for this text.",
            }

        prompt = self._build_prompt(payload)

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = extract_json(content)
            return self._clean_result(parsed, payload)
        except Exception as exc:
            print(f"[WARN] grouped function -> mode inference failed, fallback used: {exc}")
            return self._fallback_result(payload)


def pretty_print_function_mode_results(results: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("GRAPHRAG FUNCTION -> MODE INFERENCE")
    print("=" * 80)

    for item in results:
        print(f"\nChunk: {item.get('chunk_name', '')}")
        print(f"Failure Element: {item.get('failure_element', '')}")
        for mode in item.get("inferred_modes", []):
            print(f"  - Function  : {mode.get('function_text', '')}")
            print(f"    Mode      : {mode.get('mode_text', '')}")
            print(f"    Confidence: {mode.get('confidence', '')}")
            print(f"    Reason    : {mode.get('reason', '')}")
            print(f"    Evidence  : {mode.get('evidence_sentences', [])}")
        print(f"  Summary: {item.get('global_reasoning', '')}")

    print("=" * 80)
