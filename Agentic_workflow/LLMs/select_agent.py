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
            temperature=0.2,
            json_mode=True,
        )

    def _build_mode_to_effect_lookup(self, simplified_result: dict) -> Dict[str, List[dict]]:
        return {
            item.get("query_mode_text", ""): item.get("predicted_effects", [])
            for item in simplified_result.get("mode_to_effect", [])
        }

    def _get_cause_candidates(
        self,
        simplified_result: dict,
        query_cause_text: str,
        top_n_modes: int = 4,
    ) -> List[dict]:
        for item in simplified_result.get("cause_to_mode", []):
            if item.get("query_cause_text") == query_cause_text:
                return item.get("predicted_modes", [])[:top_n_modes]
        return []

    def _build_cause_to_mode_prompt(
        self,
        query_cause_text: str,
        mode_candidates: List[dict],
    ) -> str:
        payload = [
            {
                "mode_text": item.get("mode_query_text", ""),
                "mode_rank": item.get("rank")
            }
            for item in mode_candidates
        ]

        schema = {
            "cause": query_cause_text,
            "selected_modes": [
                {
                    "mode_text": "string",
                    "selected": True,
                    "reason": "why this mode is plausible for the cause",
                    "confidence": 0.0
                }
            ],
            "global_reasoning": "short summary"
        }

        return f"""
You are an FMEA reasoning agent.

Task:
1. Choose the most suitable failure mode or modes for the given cause.
2. Use engineering semantic logic first.
3. Use rank only as supporting evidence:
   - rank 1 is stronger than rank 2
   - smaller rank means stronger retrieval preference
4. Do not invent any mode outside the provided candidates.
5. Normally select only 1 mode unless another is also clearly plausible.

Cause:
{query_cause_text}

Mode candidates:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}
""".strip()

    def _build_mode_to_effect_prompt(
        self,
        query_mode_text: str,
        effect_candidates: List[dict],
    ) -> str:
        payload = [
            {
                "effect_text": item.get("effect_query_text", ""),
                "effect_rank": item.get("rank")
            }
            for item in effect_candidates
        ]

        schema = {
            "mode": query_mode_text,
            "selected_effects": [
                {
                    "effect_text": "string",
                    "selected": True,
                    "reason": "why this effect is plausible for the mode",
                    "confidence": 0.0
                }
            ],
            "global_reasoning": "short summary"
        }

        return f"""
You are an FMEA reasoning agent.

Task:
1. Choose the most suitable failure effect or effects for the given failure mode.
2. Use engineering semantic logic first.
3. Use rank only as supporting evidence:
   - rank 1 is stronger than rank 2
   - smaller rank means stronger retrieval preference
4. Do not invent any effect outside the provided candidates.
5. Normally select only 1 effect unless another is also clearly plausible.

Failure mode:
{query_mode_text}

Effect candidates:
{json.dumps(payload, indent=2, ensure_ascii=False)}

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2, ensure_ascii=False)}
""".strip()

    @traceable(
        run_type="chain",
        name="fmea_select_cause_to_mode",
        tags=["fmea", "selection", "cause_to_mode"]
    )
    def select_cause_to_mode(
        self,
        simplified_result: dict,
        query_cause_text: str,
        top_n_modes: int = 4,
    ) -> dict:
        mode_candidates = self._get_cause_candidates(
            simplified_result=simplified_result,
            query_cause_text=query_cause_text,
            top_n_modes=top_n_modes,
        )

        if not mode_candidates:
            return {
                "cause": query_cause_text,
                "selected_modes": [],
                "global_reasoning": "No mode candidates found."
            }

        prompt = self._build_cause_to_mode_prompt(query_cause_text, mode_candidates)

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = extract_json(content)

            valid_modes = {item.get("mode_query_text") for item in mode_candidates}
            cleaned_modes = []
            for item in parsed.get("selected_modes", []):
                if item.get("mode_text") in valid_modes:
                    cleaned_modes.append(item)

            return {
                "cause": query_cause_text,
                "selected_modes": cleaned_modes,
                "global_reasoning": parsed.get("global_reasoning", "")
            }

        except Exception as e:
            print(f"[WARN] cause->mode selection failed, fallback used: {e}")
            best = sorted(mode_candidates, key=lambda x: x.get("rank", 999))[0]
            return {
                "cause": query_cause_text,
                "selected_modes": [
                    {
                        "mode_text": best["mode_query_text"],
                        "selected": True,
                        "reason": "Fallback selected best-ranked mode.",
                        "confidence": 1.0 / max(best.get("rank", 999), 1),
                    }
                ],
                "global_reasoning": "Fallback used best-ranked cause->mode candidate."
            }

    @traceable(
        run_type="chain",
        name="fmea_select_mode_to_effect",
        tags=["fmea", "selection", "mode_to_effect"]
    )
    def select_mode_to_effect(
        self,
        simplified_result: dict,
        query_mode_text: str,
        max_effects_per_mode: int = 2,
    ) -> dict:
        lookup = self._build_mode_to_effect_lookup(simplified_result)
        effect_candidates = lookup.get(query_mode_text, [])[:max_effects_per_mode]

        if not effect_candidates:
            return {
                "mode": query_mode_text,
                "selected_effects": [],
                "global_reasoning": "No effect candidates found."
            }

        prompt = self._build_mode_to_effect_prompt(query_mode_text, effect_candidates)

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = extract_json(content)

            valid_effects = {item.get("effect_query_text") for item in effect_candidates}
            cleaned_effects = []
            for item in parsed.get("selected_effects", []):
                if item.get("effect_text") in valid_effects:
                    cleaned_effects.append(item)

            return {
                "mode": query_mode_text,
                "selected_effects": cleaned_effects,
                "global_reasoning": parsed.get("global_reasoning", "")
            }

        except Exception as e:
            print(f"[WARN] mode->effect selection failed, fallback used: {e}")
            best = sorted(effect_candidates, key=lambda x: x.get("rank", 999))[0]
            return {
                "mode": query_mode_text,
                "selected_effects": [
                    {
                        "effect_text": best["effect_query_text"],
                        "selected": True,
                        "reason": "Fallback selected best-ranked effect.",
                        "confidence": 1.0 / max(best.get("rank", 999), 1),
                    }
                ],
                "global_reasoning": "Fallback used best-ranked mode->effect candidate."
            }

    @traceable(
        run_type="chain",
        name="fmea_select_all",
        tags=["fmea", "selection", "pipeline"]
    )
    def select_all(
        self,
        simplified_result: dict,
        top_n_modes: int = 2,
        max_effects_per_mode: int = 3,
    ) -> dict:
        cause_mode_results = []
        selected_mode_set = set()

        # Step 1: cause -> mode
        for cause_item in simplified_result.get("cause_to_mode", []):
            query_cause_text = cause_item.get("query_cause_text", "")
            result = self.select_cause_to_mode(
                simplified_result=simplified_result,
                query_cause_text=query_cause_text,
                top_n_modes=top_n_modes,
            )
            cause_mode_results.append(result)

            for mode_item in result.get("selected_modes", []):
                if mode_item.get("selected", True):
                    selected_mode_set.add(mode_item["mode_text"])

        # Step 2: unique mode -> effect
        mode_effect_results = {}
        for mode_text in selected_mode_set:
            result = self.select_mode_to_effect(
                simplified_result=simplified_result,
                query_mode_text=mode_text,
                max_effects_per_mode=max_effects_per_mode,
            )
            mode_effect_results[mode_text] = result

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
                    chains.append({
                        "cause": cause,
                        "mode": mode,
                        "effect": effect,
                        "chain_text": f"{cause} -> {mode} -> {effect}"
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