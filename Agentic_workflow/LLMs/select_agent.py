import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None
from .llm_init import get_llm_backend

# =========================
# DATA MODELS
# =========================
@dataclass
class EffectCandidate:
    text: str
    score: float


@dataclass
class ModePrediction:
    mode_text: str
    effects: List[EffectCandidate] = field(default_factory=list)


@dataclass
class CausePrediction:
    cause_text: str
    modes: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class SelectedChain:
    cause: str
    selected_modes: List[dict]
    global_reasoning: str

# =========================
# LLM AGENT
# =========================
class FMEASelectionAgent:
    def __init__(self, backend: str = "openai", model: Optional[str] = None):
        self.llm = get_llm_backend(
            backend=backend,
            model=model,
            temperature=0,
            json_mode=True,
        )

    def _build_prompt(
        self,
        cause_prediction: CausePrediction,
        mode_to_effects: Dict[str, ModePrediction],
        top_n_modes: int = 3,
        max_effects_per_mode: int = 2,
    ) -> str:
        mode_candidates = cause_prediction.modes[:top_n_modes]

        mode_effect_payload = []
        for mode_text, mode_score in mode_candidates:
            effects = mode_to_effects.get(mode_text)
            effect_candidates = []
            if effects:
                effect_candidates = [
                    {"effect_text": e.text, "effect_score": e.score}
                    for e in effects.effects
                ]
            mode_effect_payload.append({
                "mode_text": mode_text,
                "mode_score_from_cause": mode_score,
                "effect_candidates": effect_candidates,
            })

        schema = {
            "cause": cause_prediction.cause_text,
            "selected_modes": [
                {
                    "mode_text": "string",
                    "selected": True,
                    "reason": "why this mode is plausible for the cause",
                    "confidence": 0.0,
                    "selected_effects": [
                        {
                            "effect_text": "string",
                            "reason": "why this effect is most suitable for the mode",
                            "confidence": 0.0
                        }
                    ]
                }
            ],
            "global_reasoning": "short summary of why the final chain is best"
        }

        return f"""
You are an FMEA reasoning agent.

Task:
1. Choose the most suitable failure mode or modes for the given cause.
2. For each selected mode, choose the most suitable effect(s).
3. Use semantic engineering logic first, and use scores as supporting evidence.
4. Prefer concise, physically plausible FMEA chains.
5. Do not invent effects outside the provided candidates.
6. Select at most {top_n_modes} modes, but normally 1 mode unless another is clearly also plausible.
7. Select at most {max_effects_per_mode} effects per chosen mode.

Input:
{json.dumps(mode_effect_payload, indent=2)}

Cause:
{cause_prediction.cause_text}

Return ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}
""".strip()

    def select_chain(
        self,
        cause_prediction: CausePrediction,
        mode_to_effects: Dict[str, ModePrediction],
        top_n_modes: int = 3,
        max_effects_per_mode: int = 2,
    ) -> SelectedChain:
        prompt = self._build_prompt(
            cause_prediction=cause_prediction,
            mode_to_effects=mode_to_effects,
            top_n_modes=top_n_modes,
            max_effects_per_mode=max_effects_per_mode,
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            parsed = extract_json(content)

            selected_modes = parsed.get("selected_modes", [])
            global_reasoning = parsed.get("global_reasoning", "")

            # Light validation / cleanup
            valid_mode_names = {m for m, _ in cause_prediction.modes}
            for mode_item in selected_modes:
                if mode_item.get("mode_text") not in valid_mode_names:
                    mode_item["selected"] = False
                    mode_item["reason"] = "Rejected because mode is not in provided candidate list."
                    mode_item["selected_effects"] = []

                candidate_effects = {
                    e.text
                    for e in mode_to_effects.get(mode_item.get("mode_text", ""), ModePrediction("", [])).effects
                }
                cleaned_effects = []
                for eff in mode_item.get("selected_effects", []):
                    if eff.get("effect_text") in candidate_effects:
                        cleaned_effects.append(eff)
                mode_item["selected_effects"] = cleaned_effects[:max_effects_per_mode]

            return SelectedChain(
                cause=cause_prediction.cause_text,
                selected_modes=selected_modes,
                global_reasoning=global_reasoning,
            )

        except Exception as e:
            print(f"[WARN] LLM selection failed, using score-based fallback: {e}")
            return self._fallback_selection(cause_prediction, mode_to_effects, max_effects_per_mode)

    def _fallback_selection(
        self,
        cause_prediction: CausePrediction,
        mode_to_effects: Dict[str, ModePrediction],
        max_effects_per_mode: int = 2,
    ) -> SelectedChain:
        """
        Simple deterministic fallback:
        - choose top cause->mode candidate
        - choose top 1~2 effects for that mode
        """
        if not cause_prediction.modes:
            return SelectedChain(
                cause=cause_prediction.cause_text,
                selected_modes=[],
                global_reasoning="No mode candidates available.",
            )

        best_mode, best_mode_score = cause_prediction.modes[0]
        selected_effects = []
        if best_mode in mode_to_effects:
            sorted_effects = sorted(
                mode_to_effects[best_mode].effects,
                key=lambda x: x.score,
                reverse=True
            )[:max_effects_per_mode]
            selected_effects = [
                {
                    "effect_text": e.text,
                    "reason": "Selected by highest prediction score in fallback mode.",
                    "confidence": e.score,
                }
                for e in sorted_effects
            ]

        return SelectedChain(
            cause=cause_prediction.cause_text,
            selected_modes=[
                {
                    "mode_text": best_mode,
                    "selected": True,
                    "reason": "Selected by highest cause-to-mode prediction score in fallback mode.",
                    "confidence": best_mode_score,
                    "selected_effects": selected_effects,
                }
            ],
            global_reasoning="Fallback used highest-scoring cause→mode and mode→effect links.",
        )