from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class ChainPPLEvaluator_v1:
    def __init__(self, model_name="gpt2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def compute_ppl(self, text: str):
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
            ppl = torch.exp(loss)

        return ppl.item()

    def evaluate_chain(self, cause: str, mode: str, effect: str):
        # 正向句
        forward_text = (
            f"Because {cause}, this leads to {mode}, "
            f"resulting in {effect}."
        )

        # 反向句（打乱）
        reverse_text = (
            f"Because {effect}, this leads to {cause}, "
            f"resulting in {mode}."
        )

        forward_ppl = self.compute_ppl(forward_text)
        reverse_ppl = self.compute_ppl(reverse_text)

        delta_ppl = reverse_ppl - forward_ppl

        return {
            "forward_text": forward_text,
            "reverse_text": reverse_text,
            "forward_ppl": forward_ppl,
            "reverse_ppl": reverse_ppl,
            "delta_ppl": delta_ppl,
            "coherence_score": delta_ppl  # 越大越好
        }

class ChainPPLEvaluator_v2:
    def __init__(self, model_name="gpt2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def compute_nll(self, text: str) -> float:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, labels=enc["input_ids"])
            # out.loss is mean token NLL
            return float(out.loss.item())

    def _template(self, a: str, b: str, c: str) -> str:
        # 尽量中性/对称，减少“because”偏差
        # 你也可以换成中文模板，但要匹配模型语料；GPT2 英文更合适
        return f"{a} -> {b} -> {c}."

    def evaluate_chain(self, cause: str, mode: str, effect: str):
        forward = self._template(cause, mode, effect)

        # 多个负例（局部打乱）
        negatives = [
            self._template(effect, mode, cause),
            self._template(mode, cause, effect),
            self._template(cause, effect, mode),
            self._template(mode, effect, cause),
            self._template(effect, cause, mode),
        ]

        f_nll = self.compute_nll(forward)
        neg_nlls = [self.compute_nll(t) for t in negatives]

        # coherence: forward 比负例更“顺”，则 neg_nll - f_nll 为正
        avg_neg = float(np.mean(neg_nlls))
        coherence = avg_neg - f_nll

        return {
            "forward_text": forward,
            "neg_texts": negatives,
            "forward_nll": f_nll,
            "neg_nlls": neg_nlls,
            "avg_neg_nll": avg_neg,
            "coherence_score": coherence,
        }



@dataclass
class RankedChainGeneric:
    failure_cause: str
    failure_mode: str
    failure_effect: str
    discipline: Optional[str] = None   # NEW

    # universal
    coherence_score: Optional[float] = None

    # v1 optional
    forward_ppl: Optional[float] = None
    reverse_ppl: Optional[float] = None
    delta_ppl: Optional[float] = None

    # v2 optional
    forward_nll: Optional[float] = None
    avg_neg_nll: Optional[float] = None
    neg_nlls: Optional[List[float]] = None

    # keep raw texts for debugging
    forward_text: Optional[str] = None
    reverse_text: Optional[str] = None
    neg_texts: Optional[List[str]] = None

def _get_forward_metric(result: Dict[str, Any]) -> Tuple[str, float]:
    """
    Return (metric_name, metric_value) for tie-break (smaller is better).
    Supports:
      - forward_ppl (v1)
      - forward_nll (v2)
    """
    if "forward_ppl" in result and result["forward_ppl"] is not None:
        return ("forward_ppl", float(result["forward_ppl"]))
    if "forward_nll" in result and result["forward_nll"] is not None:
        return ("forward_nll", float(result["forward_nll"]))
    # fallback: no metric
    return ("", float("inf"))

def rank_effects_with_lm(
    evaluator,
    structure_input: Dict[str, Any],
    *,
    fixed_mode: str,
    fixed_cause: str,
    node_index: int = 0,
    top_k: int = 10,
    require_effect_in_sa: bool = True,
) -> List[RankedChainGeneric]:
    """
    Generic ranker for both ChainPPLEvaluator_v1 and ChainPPLEvaluator_v2.

    Ranking:
      1) coherence_score desc (bigger is better)
      2) forward metric asc (smaller is better), where forward metric is:
         - forward_ppl for v1
         - forward_nll for v2
    """
    nodes = structure_input.get("nodes") or []
    if not nodes:
        return []

    if node_index < 0 or node_index >= len(nodes):
        raise IndexError(f"node_index {node_index} out of range (nodes={len(nodes)})")

    node = nodes[node_index]
    effects = [str(e).strip() for e in (node.get("effects") or []) if str(e).strip()]
    if require_effect_in_sa and not effects:
        return []

    ranked: List[RankedChainGeneric] = []

    for eff in effects:
        r = evaluator.evaluate_chain(cause=fixed_cause, mode=fixed_mode, effect=eff)

        # must have coherence_score for ranking; if missing, skip
        if "coherence_score" not in r or r["coherence_score"] is None:
            continue

        item = RankedChainGeneric(
            failure_cause=fixed_cause,
            failure_mode=fixed_mode,
            failure_effect=eff,
            coherence_score=float(r["coherence_score"]),
            # v1 fields
            forward_ppl=float(r["forward_ppl"]) if "forward_ppl" in r and r["forward_ppl"] is not None else None,
            reverse_ppl=float(r["reverse_ppl"]) if "reverse_ppl" in r and r["reverse_ppl"] is not None else None,
            delta_ppl=float(r["delta_ppl"]) if "delta_ppl" in r and r["delta_ppl"] is not None else None,
            # v2 fields
            forward_nll=float(r["forward_nll"]) if "forward_nll" in r and r["forward_nll"] is not None else None,
            avg_neg_nll=float(r["avg_neg_nll"]) if "avg_neg_nll" in r and r["avg_neg_nll"] is not None else None,
            neg_nlls=[float(x) for x in r["neg_nlls"]] if "neg_nlls" in r and r["neg_nlls"] is not None else None,
            # texts
            forward_text=r.get("forward_text"),
            reverse_text=r.get("reverse_text"),
            neg_texts=r.get("neg_texts"),
        )
        ranked.append(item)

    # sort with dynamic forward metric
    def sort_key(x: RankedChainGeneric):
        # coherence higher better => -coh
        # forward metric smaller better
        # choose forward_ppl if exists else forward_nll else inf
        f = x.forward_ppl if x.forward_ppl is not None else (x.forward_nll if x.forward_nll is not None else float("inf"))
        return (-x.coherence_score, f)

    ranked.sort(key=sort_key)
    return ranked[:top_k]

def rank_causes_with_lm(
    evaluator,
    structure_input: Dict[str, Any],
    *,
    fixed_mode: str,
    fixed_effect: str,
    node_index: int = 0,
    top_k: int = 10,
    require_cause_in_sa: bool = True,
) -> List[RankedChainGeneric]:

    nodes = structure_input.get("nodes") or []
    if not nodes:
        return []

    if node_index < 0 or node_index >= len(nodes):
        raise IndexError(f"node_index {node_index} out of range (nodes={len(nodes)})")

    node = nodes[node_index]

    # -------------------------
    # NEW: extract (discipline, cause)
    # -------------------------
    causes: List[Tuple[str, str]] = []

    causes_dict = node.get("causes") or {}

    for discipline, cause_list in causes_dict.items():
        for c in cause_list:
            c = str(c).strip()
            if c:
                causes.append((discipline, c))

    if require_cause_in_sa and not causes:
        return []

    ranked: List[RankedChainGeneric] = []

    # -------------------------
    # ranking loop
    # -------------------------
    for discipline, cause in causes:

        r = evaluator.evaluate_chain(
            cause=cause,
            mode=fixed_mode,
            effect=fixed_effect
        )

        if "coherence_score" not in r or r["coherence_score"] is None:
            continue

        ranked.append(
            RankedChainGeneric(
                failure_cause=cause,
                discipline=discipline,   # NEW
                failure_mode=fixed_mode,
                failure_effect=fixed_effect,

                coherence_score=float(r["coherence_score"]),

                # v1 fields
                forward_ppl=float(r["forward_ppl"])
                if "forward_ppl" in r and r["forward_ppl"] is not None else None,

                reverse_ppl=float(r["reverse_ppl"])
                if "reverse_ppl" in r and r["reverse_ppl"] is not None else None,

                delta_ppl=float(r["delta_ppl"])
                if "delta_ppl" in r and r["delta_ppl"] is not None else None,

                # v2 fields
                forward_nll=float(r["forward_nll"])
                if "forward_nll" in r and r["forward_nll"] is not None else None,

                avg_neg_nll=float(r["avg_neg_nll"])
                if "avg_neg_nll" in r and r["avg_neg_nll"] is not None else None,

                neg_nlls=[float(x) for x in r["neg_nlls"]]
                if "neg_nlls" in r and r["neg_nlls"] is not None else None,

                # texts
                forward_text=r.get("forward_text"),
                reverse_text=r.get("reverse_text"),
                neg_texts=r.get("neg_texts"),
            )
        )

    # -------------------------
    # ranking rule
    # -------------------------
    def sort_key(x: RankedChainGeneric):

        forward_metric = (
            x.forward_ppl
            if x.forward_ppl is not None
            else (
                x.forward_nll
                if x.forward_nll is not None
                else float("inf")
            )
        )

        return (-x.coherence_score, forward_metric)

    ranked.sort(key=sort_key)

    return ranked[:top_k]

if  __name__ == "__main__":
    
    evaluator = ChainPPLEvaluator_v2("gpt2")


    structure_input_powertrain = {
        "product_domain": "motor_drives",
        "product_pnID": 133427,
        "nodes": [
                {
                "element_id": "E1",
                "failure_element": "Power train",
                "modes": [
                    "Incorrect",
                    "No pulses seen",
                    "No voltage applied",
                    "Incorrect torque applied",
                    "Not enough torque",
                    "Motor breaks/overheats (e.g. resulting in demagnetisation)",
                    "Unstable regulation",
                    "High loss in torque transfer",
                    "Gear train breaks/wears out",
                    "Transmission ratio drifts",
                    "creates too much noise"
                ],
                "causes": {
                    "mechanics": [
                    "Gears loose on motor shaft (slips)",
                    "External force on spline",
                    "Motor can not provide enough torque",
                    "Too much friction in gear train",
                    "Gears material/design choice",
                    "Manufacturing tolerances of gears",
                    "Lubrication choice (e.g. degradation)",
                    "Motor design (temperature spec, actuation length/duty cycle)"
                    ],
                    "hardware": [
                    "Encoder circuit crosstalk",
                    "HW cannot supply enough power",
                    "ADC measurements incorrect (incl. bandwidth)",
                    "Wrong motor driver dimension (current rating etc.)",
                    "Overcurrent detection incorrect (threshold etc.)",
                    "Incorrect control loop (bandwidth)",
                    "Motor not shorted while device is not powered"
                    ],
                    "software": [
                    "Control parameters incorrect",
                    "Thermal protection fails (e.g. I2T)"
                    ]
                },
                "effects": [
                    "Does not shift gear",
                    "Incorrect gear shift",
                    "Incorrect cadence (offset)",
                    "Unstable cadence setting",
                    "Incorrect cadence (fixed gear ratio)",
                    "Incorrect ratio (offset)",
                    "Unstable ratio setting",
                    "Does not enter limp home mode",
                    "Sets wrong gear ratio",
                    "Gear ratio drifts when battery is empty",
                    "Firmware update not possible/fails",
                    "Device bricked",
                    "Update takes too much time (>5 minutes)",
                    "Too much noise",
                ]
            }
        ]
    }

    # ------- MC -> E ---------------
    fixed_mode = "Incorrect torque applied"
    fixed_cause = "ADC measurements incorrect (incl. bandwidth)"

    top = rank_effects_with_lm(
        evaluator,
        structure_input_powertrain,
        fixed_mode=fixed_mode,
        fixed_cause=fixed_cause,
        node_index=0,
        top_k=10
    )

    for i, c in enumerate(top, 1):
        print(i, c.failure_effect, "coh=", round(c.coherence_score, 4), "f_nll=", None if c.forward_nll is None else round(c.forward_nll, 4))

    # ME -> Cause
    fixed_mode = "Not enough torque"
    fixed_effect = "Incorrect gear shift"

    top_causes = rank_causes_with_lm(
        evaluator,
        structure_input_powertrain,
        fixed_mode=fixed_mode,
        fixed_effect=fixed_effect,
        node_index=0,
        top_k=10
    )

    for i, c in enumerate(top_causes, 1):
        print(i, c.failure_cause, c.discipline,"coh=", round(c.coherence_score, 4),
            "f_nll=", None if c.forward_nll is None else round(c.forward_nll, 4))



    #=========== Single test ==============
    # chain = {
    #         "failure_mode": "Motor breaks/overheats (e.g. resulting in demagnetisation)",
    #         "failure_effect": "Incorrect gear shift",
    #         "failure_cause": "Thermal protection fails (e.g. I2T)",
    # }

    # result = evaluator.evaluate_chain(
    #     cause=chain["failure_cause"],
    #     mode=chain["failure_mode"],
    #     effect=chain["failure_effect"]
    # )

    # print(result)


