from typing import List, Dict, Any
from rapidfuzz import fuzz
import re
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]  # Information_extraction_8D
sys.path.insert(0, str(ROOT))

from Schemas.eightD_sentence_schema_V2 import SelectedSentence


# =========================================================
# Text utils
# =========================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def tokenize_lemmas(text: str) -> List[str]:
    return tokenize(text)


def token_coverage(a: List[str], b: List[str]) -> float:
    if not a:
        return 0.0
    b_set = set(b)
    return sum(1 for t in a if t in b_set) / len(a)


def split_into_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]


# =========================================================
# Similarity metrics
# =========================================================

def sentence_similarity_metrics(
    sentence: str,
    reference_text: str,
) -> Dict[str, float]:
    sent_norm = normalize_text(sentence)
    ref_norm = normalize_text(reference_text)

    sent_tokens = tokenize_lemmas(sentence)
    ref_tokens = tokenize_lemmas(reference_text)

    fuzzy_score = fuzz.token_set_ratio(sent_norm, ref_norm) / 100.0
    coverage = token_coverage(sent_tokens, ref_tokens)

    return {
        "fuzzy_similarity": fuzzy_score,
        "token_coverage": coverage,
        "max_similarity": max(fuzzy_score, coverage),
    }


# =========================================================
# Coverage & Faithfulness
# =========================================================

def compute_source_coverage(
    source_text: str,
    summary_sentences: List[str],
) -> Dict[str, Any]:
    """
    Compute source-to-summary coverage by aligning each source unit
    against EACH summary sentence and taking the maximum similarity.
    """
    source_units = split_into_sentences(source_text)

    unit_scores = []
    unit_details = []

    for unit in source_units:
        per_sentence_scores = []

        for sent in summary_sentences:
            metrics = sentence_similarity_metrics(unit, sent)
            per_sentence_scores.append(metrics["max_similarity"])

        best_score = max(per_sentence_scores) if per_sentence_scores else 0.0
        unit_scores.append(best_score)

        unit_details.append({
            "source_unit": unit,
            "best_score": best_score,
            "per_summary_scores": per_sentence_scores,
        })

    return {
        "unit_count": len(source_units),
        "mean_coverage": sum(unit_scores) / max(len(unit_scores), 1),
        "coverage_scores": unit_scores,
        "unit_details": unit_details,   # ← 可用于诊断 / 可视化
    }

def compute_summary_faithfulness(
    summary_sentences: List[str],
    source_text: str,
) -> Dict[str, Any]:
    per_sentence = []
    scores = []

    for sent in summary_sentences:
        metrics = sentence_similarity_metrics(sent, source_text)
        scores.append(metrics["max_similarity"])
        per_sentence.append({
            "sentence": sent,
            **metrics,
        })

    return {
        "sentence_count": len(summary_sentences),
        "mean_faithfulness": sum(scores) / max(len(scores), 1),
        "sentence_scores": per_sentence,
    }


# =========================================================
# Compression
# =========================================================

def compute_compression_metrics(
    source_text: str,
    summary_text: str,
) -> Dict[str, float]:
    src_tokens = tokenize_lemmas(source_text)
    sum_tokens = tokenize_lemmas(summary_text)

    return {
        "source_token_count": len(src_tokens),
        "summary_token_count": len(sum_tokens),
        "compression_ratio": len(sum_tokens) / max(len(src_tokens), 1),
    }


def compute_information_density(
    source_coverage: float,
    compression_ratio: float,
) -> float:
    return source_coverage / max(compression_ratio, 1e-9)


# =========================================================
# Main evaluation API
# =========================================================

def evaluate_text_compression(
    source_text: str,
    summary_sentences: List[str],
) -> Dict[str, Any]:
    summary_text = " ".join(summary_sentences)

    source_cov = compute_source_coverage(
    source_text,
    summary_sentences,
)
    summary_faith = compute_summary_faithfulness(summary_sentences, source_text)
    compression = compute_compression_metrics(source_text, summary_text)

    info_density = compute_information_density(
        source_cov["mean_coverage"],
        compression["compression_ratio"],
    )

    return {
        "source_coverage": source_cov,
        "summary_faithfulness": summary_faith,
        "compression": compression,
        "information_density": info_density,
    }


# =========================================================
# IO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# RAW_TEXT_PATH = Path(
#     r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_test\failure_identification\8D6318110147R01.json"
# )

# SELECTED_SENTENCES_PATH = Path(
#     r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\JSON\8D_test\sentence_selected\8D6318110147R01_sentences.json"
# )

RAW_TEXT_PATH = Path(
    r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\eightD_json_raw\8D620721025401.json"
)

SELECTED_SENTENCES_PATH = Path(
 r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\eightD_json_raw\8D620721025401_iter1.json"
)

OUTPUT_PATH = BASE_DIR / "coverage_evaluation_8D620721025401_ini.json"


def load_raw_text_d2_d3_d4(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sections = data.get("sections", {})
    return (
        sections.get("D2", {}).get("raw_context", ""),
        sections.get("D3", {}).get("raw_context", ""),
        sections.get("D4", {}).get("raw_context", ""),
    )


def load_selected_sentences(path: Path) -> List[SelectedSentence]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_list = data.get("selected_sentences", data)
    return [SelectedSentence(**item) for item in raw_list]


# =========================================================
# Run
# =========================================================

def main():
    print("Loading raw D2/D3/D4 text...")
    d2, d3, d4 = load_raw_text_d2_d3_d4(RAW_TEXT_PATH)
    source_text = "\n".join([d2, d3, d4])

    print("Loading selected sentences...")
    selected = load_selected_sentences(SELECTED_SENTENCES_PATH)
    summary_sentences = [s.text for s in selected]

    print(f"→ {len(summary_sentences)} sentences loaded")

    print("Running compression evaluation...")
    eval_result = evaluate_text_compression(
        source_text=source_text,
        summary_sentences=summary_sentences,
    )

    print("Saving result...")
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)

    print(f"✅ Done: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
