import json
import argparse
from typing import List
from pathlib import Path
from Information_extraction_8D.Schemas.eightD_sentence_schema_V2 import SelectedSentence
from Information_extraction_8D.Evaluation.evaluation_tool import evaluate_iter1, summarize_eval

BASE_DIR = Path(__file__).resolve().parent

RAW_TEXT_PATH = Path(
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_test\failure_identification\8D6298110111R01.json"
)

SELECTED_SENTENCES_PATH = Path(
     r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_test\sentence_selected\8D6298110111R01_sentences.json" 
)



# OUTPUT_PATH = Path(
#     r"iter1_evaluation_result_8D6298190081R02.json"
# )

OUTPUT_PATH = BASE_DIR / "sentence_evaluation_8D6298110111R01.json"
# =========================================================
# IO helpers
# =========================================================

def load_raw_text_d2_d3_d4(path: Path):
    """
    Expected structure:
    {
      "sections": {
        "D2": {"raw_context": "..."},
        "D3": {"raw_context": "..."},
        "D4": {"raw_context": "..."}
      }
    }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sections = data.get("sections", {})

    d2 = sections.get("D2", {}).get("raw_context", "")
    d3 = sections.get("D3", {}).get("raw_context", "")
    d4 = sections.get("D4", {}).get("raw_context", "")

    return d2, d3, d4


def load_selected_sentences(path: Path) -> List[SelectedSentence]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_list = data.get("selected_sentences", data)

    sentences = []
    for item in raw_list:
        sentences.append(SelectedSentence(**item))

    return sentences


# =========================================================
# Main
# =========================================================

def main():
    print("Loading raw D2/D3/D4 text...")
    d2, d3, d4 = load_raw_text_d2_d3_d4(RAW_TEXT_PATH)

    print("Loading selected sentences (Iter1)...")
    sentences = load_selected_sentences(SELECTED_SENTENCES_PATH)

    print(f"   → {len(sentences)} sentences loaded")

    print(" Running Iter1 evaluation...")
    eval_result = evaluate_iter1(
        sentences=sentences,
        d2=d2,
        d3=d3,
        d4=d4,
        fuzzy_threshold=90,
        include_per_sentence=True,
    )

    print("\n========== ITER1 EVALUATION SUMMARY ==========")
    print(summarize_eval(eval_result))

    print("\n💾 Saving detailed evaluation result...")
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Result saved to:\n{OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()