import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import traceable

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None
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


class FMEASelectionAgent:
    def __init__(self, backend: str = "openai", model: Optional[str] = None):
        self.llm = get_llm_backend(
            backend=backend,
            model=model,
            temperature=0.0,
            json_mode=True,
        )

    def _build_mode_to_effect_lookup(self, simplified_result: dict) -> Dict[str, List[dict]]:
        return {
            item.get("query_mode_text", ""): item.get("predicted_effects", [])
            for item in simplified_result.get("mode_to_effect", [])
        }

    def _build_grouped_cause_to_mode_payload(
        self,
        simplified_result: dict,
        top_n_modes: int = 4,
    ) -> List[dict]:
        grouped = []
        for item in simplified_result.get("cause_to_mode", []):
            cause = item.get("query_cause_text", "")
            predicted_modes = item.get("predicted_modes", [])[:top_n_modes]

            grouped.append({
                "element_text": item.get("element_text", ""),
                "cause_discipline": item.get("cause_discipline", ""),
                "cause": cause,
                "mode_candidates": [
                    {
                        "mode_text": m.get("mode_query_text", ""),
                        "function_text": m.get("function_text", ""),
                        "mode_rank": m.get("rank")
                    }
                    for m in predicted_modes
                ]
            })
        return grouped

    def _build_grouped_mode_to_effect_payload(
        self,
        simplified_result: dict,
        selected_modes: List[str],
        max_effects_per_mode: int = 3,
    ) -> List[dict]:
        lookup = self._build_mode_to_effect_lookup(simplified_result)
        grouped = []

        # 建一个 mode -> full item lookup
        mode_meta_lookup = {
            item.get("query_mode_text", ""): item
            for item in simplified_result.get("mode_to_effect", [])
        }

        for mode_text in selected_modes:
            effect_candidates = lookup.get(mode_text, [])[:max_effects_per_mode]
            mode_meta = mode_meta_lookup.get(mode_text, {})

            grouped.append({
                "element_text": mode_meta.get("element_text", ""),
                "function_text": mode_meta.get("function_text", ""),
                "mode": mode_text,
                "effect_candidates": [
                    {
                        "effect_text": e.get("effect_query_text", ""),
                        "effect_rank": e.get("rank")
                    }
                    for e in effect_candidates
                ]
            })
        return grouped

    def _build_grouped_cause_to_mode_prompt(self, grouped_payload: List[dict]) -> str:
        schema = {
            "results": [
                {
                    "cause": "string",
                    "selected_modes": [
                        {
                            "mode_text": "string",
                            "selected": True,
                            "reason": "why this mode is plausible for the cause under the given element/function context",
                            "confidence": 0.0
                        }
                    ],
                    "global_reasoning": "short summary"
                }
            ]
        }

        return f"""
    You are an FMEA reasoning agent.

    Task:
    1. For EACH cause, choose the most suitable failure mode or modes from the provided candidates.
    2. Use engineering semantic logic first.
    3. Use the following context carefully:
    - element_text = the higher-level failure element / subsystem
    - cause_discipline = the engineering source category of the cause
    - function_text = the sub-function or functional block where the candidate mode belongs
    4. Judge plausibility under the full context:
    cause within element -> candidate mode within function.
    5. Use rank only as supporting evidence:
    - rank 1 is stronger than rank 2
    - smaller rank means stronger retrieval preference
    6. Do not invent any mode outside the provided candidates.
    7. Normally select 1-2 mode per cause unless another is also clearly plausible.
    8. Keep reasoning concise and practical.

    Important:
    - A candidate mode may be textually similar but belong to an implausible function block.
    - Prefer modes whose function context is consistent with the cause and the element behavior.

    Input:
    {json.dumps(grouped_payload, indent=2, ensure_ascii=False)}

    Return ONLY valid JSON matching this schema:
    {json.dumps(schema, indent=2, ensure_ascii=False)}
    """.strip()

    def _build_grouped_mode_to_effect_prompt(self, grouped_payload: List[dict]) -> str:
        schema = {
            "results": [
                {
                    "mode": "string",
                    "selected_effects": [
                        {
                            "effect_text": "string",
                            "selected": True,
                            "reason": "why this effect is plausible for the mode under the given element/function context",
                            "confidence": 0.0
                        }
                    ],
                    "global_reasoning": "short summary"
                }
            ]
        }

        return f"""
You are an FMEA reasoning agent.

Task:
1. For EACH failure mode, choose the most suitable failure effect or effects from the provided candidates.
2. Use engineering semantic logic first.
3. Use the following context carefully:
   - element_text = the higher-level failure element / subsystem
   - function_text = the sub-function or functional block where the mode occurs
4. Judge plausibility under the full context:
   mode within function within element -> candidate effect.
5. Use rank only as supporting evidence:
   - rank 1 is stronger than rank 2
   - smaller rank means stronger retrieval preference
6. Do not invent any effect outside the provided candidates.
7. Normally select only 1 effect per mode unless another is also clearly plausible.
8. Keep reasoning concise and practical.
Important:
- The same mode text may lead to different effects depending on the function block and element context.
- Prefer effects that are system-logically consistent with the mode's function.

Input:
{json.dumps(grouped_payload, indent=2, ensure_ascii=False)}

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}
""".strip()

    def _fallback_grouped_cause_to_mode(
        self,
        grouped_payload: List[dict]
    ) -> List[dict]:
        results = []
        for item in grouped_payload:
            cause = item.get("cause", "")
            candidates = item.get("mode_candidates", [])
            if not candidates:
                results.append({
                    "cause": cause,
                    "selected_modes": [],
                    "global_reasoning": "No mode candidates found."
                })
                continue

            best = sorted(candidates, key=lambda x: x.get("mode_rank", 999))[0]
            results.append({
                "cause": cause,
                "selected_modes": [
                    {
                        "mode_text": best["mode_text"],
                        "selected": True,
                        "reason": "Fallback selected best-ranked mode.",
                        "confidence": 1.0 / max(best.get("mode_rank", 999), 1),
                    }
                ],
                "global_reasoning": "Fallback used best-ranked cause->mode candidate."
            })
        return results

    def _fallback_grouped_mode_to_effect(
        self,
        grouped_payload: List[dict]
    ) -> List[dict]:
        results = []
        for item in grouped_payload:
            mode = item.get("mode", "")
            candidates = item.get("effect_candidates", [])
            if not candidates:
                results.append({
                    "mode": mode,
                    "selected_effects": [],
                    "global_reasoning": "No effect candidates found."
                })
                continue

            best = sorted(candidates, key=lambda x: x.get("effect_rank", 999))[0]
            results.append({
                "mode": mode,
                "selected_effects": [
                    {
                        "effect_text": best["effect_text"],
                        "selected": True,
                        "reason": "Fallback selected best-ranked effect.",
                        "confidence": 1.0 / max(best.get("effect_rank", 999), 1),
                    }
                ],
                "global_reasoning": "Fallback used best-ranked mode->effect candidate."
            })
        return results

    def _clean_grouped_cause_to_mode_result(
        self,
        parsed: dict,
        grouped_payload: List[dict]
    ) -> List[dict]:
        valid_lookup = {
            item["cause"]: {m["mode_text"] for m in item.get("mode_candidates", [])}
            for item in grouped_payload
        }

        cleaned_results = []
        for item in parsed.get("results", []):
            cause = item.get("cause", "")
            valid_modes = valid_lookup.get(cause, set())

            cleaned_modes = []
            for mode in item.get("selected_modes", []):
                if mode.get("mode_text") in valid_modes:
                    cleaned_modes.append(mode)

            cleaned_results.append({
                "cause": cause,
                "selected_modes": cleaned_modes,
                "global_reasoning": item.get("global_reasoning", "")
            })

        # 补齐没返回的 cause
        returned_causes = {r["cause"] for r in cleaned_results}
        for payload_item in grouped_payload:
            if payload_item["cause"] not in returned_causes:
                cleaned_results.append({
                    "cause": payload_item["cause"],
                    "selected_modes": [],
                    "global_reasoning": "Model did not return this cause."
                })

        return cleaned_results

    def _clean_grouped_mode_to_effect_result(
        self,
        parsed: dict,
        grouped_payload: List[dict]
    ) -> List[dict]:
        valid_lookup = {
            item["mode"]: {e["effect_text"] for e in item.get("effect_candidates", [])}
            for item in grouped_payload
        }

        cleaned_results = []
        for item in parsed.get("results", []):
            mode = item.get("mode", "")
            valid_effects = valid_lookup.get(mode, set())

            cleaned_effects = []
            for eff in item.get("selected_effects", []):
                if eff.get("effect_text") in valid_effects:
                    cleaned_effects.append(eff)

            cleaned_results.append({
                "mode": mode,
                "selected_effects": cleaned_effects,
                "global_reasoning": item.get("global_reasoning", "")
            })

        # 补齐没返回的 mode
        returned_modes = {r["mode"] for r in cleaned_results}
        for payload_item in grouped_payload:
            if payload_item["mode"] not in returned_modes:
                cleaned_results.append({
                    "mode": payload_item["mode"],
                    "selected_effects": [],
                    "global_reasoning": "Model did not return this mode."
                })

        return cleaned_results

    @traceable(
        run_type="chain",
        name="fmea_select_cause_to_mode_grouped",
        tags=["fmea", "selection", "cause_to_mode", "grouped"]
    )
    def select_cause_to_mode_grouped(
        self,
        simplified_result: dict,
        top_n_modes: int = 4,
    ) -> List[dict]:
        grouped_payload = self._build_grouped_cause_to_mode_payload(
            simplified_result=simplified_result,
            top_n_modes=top_n_modes,
        )

        if not grouped_payload:
            return []

        prompt = self._build_grouped_cause_to_mode_prompt(grouped_payload)

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = extract_json(content)
            return self._clean_grouped_cause_to_mode_result(parsed, grouped_payload)

        except Exception as e:
            print(f"[WARN] grouped cause->mode selection failed, fallback used: {e}")
            return self._fallback_grouped_cause_to_mode(grouped_payload)

    @traceable(
        run_type="chain",
        name="fmea_select_mode_to_effect_grouped",
        tags=["fmea", "selection", "mode_to_effect", "grouped"]
    )
    def select_mode_to_effect_grouped(
        self,
        simplified_result: dict,
        selected_modes: List[str],
        max_effects_per_mode: int = 3,
    ) -> List[dict]:
        grouped_payload = self._build_grouped_mode_to_effect_payload(
            simplified_result=simplified_result,
            selected_modes=selected_modes,
            max_effects_per_mode=max_effects_per_mode,
        )

        if not grouped_payload:
            return []

        prompt = self._build_grouped_mode_to_effect_prompt(grouped_payload)

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = extract_json(content)
            return self._clean_grouped_mode_to_effect_result(parsed, grouped_payload)

        except Exception as e:
            print(f"[WARN] grouped mode->effect selection failed, fallback used: {e}")
            return self._fallback_grouped_mode_to_effect(grouped_payload)

    @traceable(
        run_type="chain",
        name="fmea_select_all_grouped",
        tags=["fmea", "selection", "pipeline", "grouped"]
    )
    def select_all(
        self,
        simplified_result: dict,
        top_n_modes: int = 2,
        max_effects_per_mode: int = 3,
    ) -> dict:
        # Step 1: grouped cause -> mode
        cause_mode_results = self.select_cause_to_mode_grouped(
            simplified_result=simplified_result,
            top_n_modes=top_n_modes,
        )

        selected_mode_set = set()
        for item in cause_mode_results:
            for mode_item in item.get("selected_modes", []):
                if mode_item.get("selected", True):
                    selected_mode_set.add(mode_item["mode_text"])

        # Step 2: grouped mode -> effect
        mode_effect_list = self.select_mode_to_effect_grouped(
            simplified_result=simplified_result,
            selected_modes=sorted(selected_mode_set),
            max_effects_per_mode=max_effects_per_mode,
        )

        mode_effect_results = {
            item["mode"]: item
            for item in mode_effect_list
        }
        mode_context_lookup = {
            item.get("query_mode_text", ""): {
                "element_text": item.get("element_text", ""),
                "function_text": item.get("function_text", "")
            }
            for item in simplified_result.get("mode_to_effect", [])
        }

        # Step 3: join
        chains = []
        for cm in cause_mode_results:
            cause = cm["cause"]

            for mode_item in cm.get("selected_modes", []):
                if not mode_item.get("selected", True):
                    continue

                mode = mode_item["mode_text"]
                eff_result = mode_effect_results.get(mode, {})

                for eff in eff_result.get("selected_effects", []):
                    if not eff.get("selected", True):
                        continue

                    effect = eff["effect_text"]
                    ctx = mode_context_lookup.get(mode, {})
                    element_text = ctx.get("element_text", "")
                    function_text = ctx.get("function_text", "")

                    chains.append({
                        "element": element_text,
                        "function": function_text,
                        "cause": cause,
                        "mode": mode,
                        "effect": effect,
                        "chain_text": f"[{element_text} | {function_text}] {cause} -> {mode} -> {effect}"
                    })

        return {
            "cause_to_mode_selection": cause_mode_results,
            "mode_to_effect_selection": mode_effect_results,
            "final_chains": chains
        }


def pretty_print_selection_result(result: dict):
    print("\n" + "=" * 80)
    print("FMEA SELECTION RESULT")
    print("=" * 80)

    print("\n[1] Cause -> Mode selection")
    for item in result.get("cause_to_mode_selection", []):
        print(f"\nCause: {item.get('cause', '')}")
        for mode in item.get("selected_modes", []):
            print(f"  - Mode: {mode.get('mode_text', '')}")
            print(f"    Selected  : {mode.get('selected', True)}")
            print(f"    Reason    : {mode.get('reason', '')}")
            print(f"    Confidence: {mode.get('confidence', '')}")
        print(f"  Summary: {item.get('global_reasoning', '')}")

    print("\n[2] Mode -> Effect selection")
    for mode_text, item in result.get("mode_to_effect_selection", {}).items():
        print(f"\nMode: {mode_text}")
        for eff in item.get("selected_effects", []):
            print(f"  - Effect: {eff.get('effect_text', '')}")
            print(f"    Selected  : {eff.get('selected', True)}")
            print(f"    Reason    : {eff.get('reason', '')}")
            print(f"    Confidence: {eff.get('confidence', '')}")
        print(f"  Summary: {item.get('global_reasoning', '')}")

    print("\n[3] Final chains")
    for i, chain in enumerate(result.get("final_chains", []), 1):
        print(f"  [{i}] {chain['chain_text']}")

    print("=" * 80)