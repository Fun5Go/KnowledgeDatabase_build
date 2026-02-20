import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

class ChainPPLEvaluator:
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
    
evaluator = ChainPPLEvaluator("gpt2")

# chain = {
#         "failure_mode": "Transmission ratio drifts",
#         "failure_effect": "Gear ratio drifts when battery is empty",
#         "failure_cause": "HW cannot supply enough power",
# }

# result = evaluator.evaluate_chain(
#     cause=chain["failure_cause"],
#     mode=chain["failure_mode"],
#     effect=chain["failure_effect"]
# )

# print(result)


