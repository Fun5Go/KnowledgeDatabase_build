import os
import textwrap
from contextlib import contextmanager
from pathlib import Path

import langextract as lx
from dotenv import load_dotenv
from langextract.factory import ModelConfig
from langsmith.wrappers import wrap_openai
import json
from typing import Any
import re

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional dependency
    def traceable(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator

dotenv_path = Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=dotenv_path, override=False)
lx.providers.load_builtins_once()

LANGSMITH_PROJECT_NAME = os.getenv("LANGSMITH_PROJECT", "DocumentProcess")



# -----------------------------
# LangSmith
# -----------------------------
def configure_langsmith(project_name: str | None = None) -> str:
    tracing_value = os.getenv("LANGSMITH_TRACING_V2") or os.getenv("LANGSMITH_TRACING")
    if tracing_value and not os.getenv("LANGSMITH_TRACING_V2"):
        os.environ["LANGSMITH_TRACING_V2"] = tracing_value

    resolved_project = project_name or "langextract_fmea_demo"
    os.environ["LANGSMITH_PROJECT"] = resolved_project
    return resolved_project


LANGSMITH_PROJECT_NAME = configure_langsmith()


# -----------------------------
# Prompts
# -----------------------------
SPAN_PROMPT = textwrap.dedent("""\
Extract exact spans from technical specification text.

Classes:
1. element: a device, module, algorithm, signal, or named subject
2. sub_element: a terminal, interface, phase, gate, relay, terminal, or internal part of an element
3. function: a verb phrase describing behavior, capability, requirement, or action
4. condition: a trigger or context introduced by words such as when, if, while, after, before, unless, since
5. object: an applicability, scope, or operating-target phrase introduced by words such as on, for, in, across, between, individually on

Rules:
- Use exact text spans only.
- Do not paraphrase.
- Return spans in order of appearance.
- Do not overlap spans.
- A sentence may contain zero, one, or many spans of each class.
- Prefer the smallest exact span that still preserves meaning.
- Do not resolve pronouns in this pass.
- Do not force all classes to appear.
""")

COREF_PROMPT = textwrap.dedent("""\
Extract exact pronoun or reference spans from technical specification text.

Classes:
1. pronoun: a pronoun or short reference phrase such as it, this, that, these, those, one another

Attributes:
- referent: the exact earlier span in the same sentence or immediately preceding sentence that this reference points to

Rules:
- Use exact text spans only.
- Do not paraphrase.
- Only extract a pronoun/reference if the referent is clear.
- Return spans in order of appearance.
- Do not overlap spans.
""")


# -----------------------------
# Examples: pass 1 (span extraction)
# -----------------------------
SPAN_EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "The Safety module will report an SAFETY_ERROR_OVERVOLTAGE error "
            "when the measured rms input voltage is below the configurable "
            "parameter voltage_in - 15% for more than 10 seconds."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="element",
                extraction_text="Safety module",
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="will report an SAFETY_ERROR_OVERVOLTAGE error",
            ),
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text=(
                    "when the measured rms input voltage is below the configurable "
                    "parameter voltage_in - 15% for more than 10 seconds"
                ),
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The startSoftStartAlgorithm function will function individually on "
            "phases U and W."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="element",
                extraction_text="startSoftStartAlgorithm function",
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="will function",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="individually on phases U and W",
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "For each of these phases it will have a Zero Crossing Detection Input "
            "that will be active high when a zero crossing is detected."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="sub_element",
                extraction_text="Zero Crossing Detection Input",
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="will be active high",
            ),
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text="when a zero crossing is detected",
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "By applying a pulse at the gate, the TRIAC can conduct on both ways, "
            "either positive or negative waveform."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="sub_element",
                extraction_text="gate",
            ),
            lx.data.Extraction(
                extraction_class="element",
                extraction_text="TRIAC",
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="can conduct",
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="on both ways",
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "When the current reaches zero, the TRIAC is supposed to be disabled."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text="When the current reaches zero",
            ),
            lx.data.Extraction(
                extraction_class="element",
                extraction_text="TRIAC",
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="is supposed to be disabled",
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The Motor Control Module will measure the input frequency based on the "
            "ZCD inputs using a low pass filter with a 0.95 smoothing factor."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="element",
                extraction_text="Motor Control Module",
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="will measure the input frequency",
            ),
            lx.data.Extraction(
                extraction_class="sub_element",
                extraction_text="ZCD inputs",
            ),
            lx.data.Extraction(
                extraction_class="sub_element",
                extraction_text="low pass filter",
            ),
        ],
    ),
]


# -----------------------------
# Examples: pass 2 (coreference)
# -----------------------------
COREF_EXAMPLES = [
    lx.data.ExampleData(
        text=(
            "The startSoftStartAlgorithm function will function individually on "
            "phases U and W, for each of these phases it will have a Zero Crossing "
            "Detection Input."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="pronoun",
                extraction_text="these",
                attributes={"referent": "phases U and W"},
            ),
            lx.data.Extraction(
                extraction_class="pronoun",
                extraction_text="it",
                attributes={"referent": "startSoftStartAlgorithm function"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The TRIAC can be spuriously turned on. Theoretically when the current "
            "reaches zero, the TRIAC is supposed to be disabled, but since the load "
            "is highly inductive, this is not always true."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="pronoun",
                extraction_text="this",
                attributes={"referent": "the TRIAC is supposed to be disabled"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The currents and voltages are not in phase with one another."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="pronoun",
                extraction_text="one another",
                attributes={"referent": "currents and voltages"},
            ),
        ],
    ),
]


# -----------------------------
# Input
# -----------------------------
INPUT_TEXT = """
On the other side, the TRIACS can be simply considered as 2 SCRs connected together in a back to back configuration.
Therefore by applying a pulse at the gate, the TRIAC can conduct on both ways, either positive or negative waveform.
The TRIACS have a difficulty on turning off inductive loads.
Considering that in an inductive load the currents and voltages are not in phase with one another (the more inductive the load, the more this is true) the triacs can be spuriously turned on.
Theoretically when the current reaches zero, the TRIAC is supposed to be disabled, but since the load is highly inductive, the voltage at that stage is not equal to zero.
Thus the TRIAC will observe a high dV/dt between anode and cathode which might trigger a spurious turn on even if the gate is not triggered.
Furthermore the TRIACS have way lower breakdown voltage.
"""


# -----------------------------
# Model setup
# -----------------------------
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
if not api_key:
    raise RuntimeError("Missing API key. Set OPENAI_API_KEY or LANGEXTRACT_API_KEY.")

llm_model = os.getenv("LLM_MODEL", "azure/gpt-4.1")
api_base = (
    os.getenv("OPENAI_API_BASE")
    or os.getenv("OPENAI_BASE_URL")
    or "http://litellm.ame.local/v1"
)

print("[LangExtract Demo] Debug config:")
print(f"  LLM_MODEL       = {llm_model}")
print(f"  OPENAI_API_BASE = {api_base}")
print("  provider        = OpenAILanguageModel")
print(f"  api_key         = {'set' if api_key else 'missing'}")


@contextmanager
def traced_openai_client():
    import openai

    original_openai = openai.OpenAI

    def wrapped_openai(*args, **kwargs):
        return wrap_openai(original_openai(*args, **kwargs))

    openai.OpenAI = wrapped_openai
    try:
        yield
    finally:
        openai.OpenAI = original_openai


def build_model_config() -> ModelConfig:
    return ModelConfig(
        model_id=llm_model,
        provider="OpenAILanguageModel",
        provider_kwargs={
            "api_key": api_key,
            "base_url": api_base,
        },
    )


# -----------------------------
# Helpers
# -----------------------------
def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_plain_data(value.model_dump())
    if hasattr(value, "dict"):
        return to_plain_data(value.dict())
    if hasattr(value, "value") and not callable(getattr(value, "value")):
        return to_plain_data(value.value)
    if hasattr(value, "__dict__"):
        return to_plain_data(
            {key: item for key, item in vars(value).items() if not key.startswith("_")}
        )
    return str(value)


def to_extraction(value: Any) -> lx.data.Extraction:
    if isinstance(value, lx.data.Extraction):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif hasattr(value, "dict"):
        value = value.dict()
    elif hasattr(value, "__dict__"):
        value = vars(value)

    if not isinstance(value, dict):
        raise TypeError(f"Unsupported extraction structure: {type(value)!r}")

    payload = dict(value)
    token_interval = payload.pop("token_interval", payload.pop("_token_interval", None))
    char_interval = payload.pop("char_interval", None)
    alignment_status = payload.pop("alignment_status", None)
    extraction_index = payload.pop("extraction_index", None)
    group_index = payload.pop("group_index", None)
    description = payload.pop("description", None)
    attributes = payload.pop("attributes", None)

    extraction_class = payload.pop("extraction_class", None)
    extraction_text = payload.pop("extraction_text", None)
    if extraction_class is None or extraction_text is None:
        raise TypeError(f"Missing extraction fields in payload: {value!r}")

    return lx.data.Extraction(
        extraction_class=extraction_class,
        extraction_text=extraction_text,
        token_interval=token_interval,
        char_interval=char_interval,
        alignment_status=alignment_status,
        extraction_index=extraction_index,
        group_index=group_index,
        description=description,
        attributes=attributes,
    )


def normalize_extractions(extractions: list[Any]) -> list[lx.data.Extraction]:
    return [to_extraction(item) for item in extractions]


def extract_extractions(result: Any) -> list[Any]:
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif hasattr(result, "dict"):
        result = result.dict()

    if isinstance(result, dict):
        if "extractions" in result:
            return result.get("extractions", [])
        if "annotated_documents" in result and result["annotated_documents"]:
            first_doc = result["annotated_documents"][0]
            if isinstance(first_doc, dict):
                return first_doc.get("extractions", [])

    if hasattr(result, "extractions"):
        return getattr(result, "extractions") or []

    if hasattr(result, "annotated_documents"):
        docs = getattr(result, "annotated_documents") or []
        if docs:
            first_doc = docs[0]
            if hasattr(first_doc, "model_dump"):
                first_doc = first_doc.model_dump()
            elif hasattr(first_doc, "dict"):
                first_doc = first_doc.dict()
            elif hasattr(first_doc, "__dict__"):
                first_doc = vars(first_doc)
            if isinstance(first_doc, dict):
                return first_doc.get("extractions", [])

    raise TypeError(f"Unsupported result structure: {type(result)!r}")


def merge_pass_outputs(
    sentence: str,
    span_result: Any,
    coref_result: Any,
    document_id: str,
) -> lx.data.AnnotatedDocument:
    span_extractions = normalize_extractions(extract_extractions(span_result))
    coref_extractions = normalize_extractions(extract_extractions(coref_result))

    return lx.data.AnnotatedDocument(
        document_id=document_id,
        text=sentence,
        extractions=span_extractions + coref_extractions,
    )


def build_visualization_document(
    input_text: str,
    sentence_documents: list[lx.data.AnnotatedDocument],
) -> lx.data.AnnotatedDocument:
    normalized_input = re.sub(r"\s+", " ", input_text).strip()
    all_extractions: list[lx.data.Extraction] = []
    for doc in sentence_documents:
        all_extractions.extend(normalize_extractions(doc.extractions or []))

    return lx.data.AnnotatedDocument(
        document_id="input-001",
        text=normalized_input,
        extractions=all_extractions,
    )


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(to_plain_data(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -----------------------------
# Core extraction functions
# -----------------------------
def extract_spans_for_sentence(sentence: str, model_config: ModelConfig):
    return lx.extract(
        text_or_documents=sentence,
        prompt_description=SPAN_PROMPT,
        examples=SPAN_EXAMPLES,
        config=model_config,
        extraction_passes=3,
        max_char_buffer=500,
        temperature=0.0,
        prompt_validation_strict=True,
        show_progress=False,
    )


def extract_coref_for_sentence(sentence: str, model_config: ModelConfig):
    return lx.extract(
        text_or_documents=sentence,
        prompt_description=COREF_PROMPT,
        examples=COREF_EXAMPLES,
        config=model_config,
        extraction_passes=2,
        max_char_buffer=500,
        temperature=0.0,
        prompt_validation_strict=True,
        show_progress=False,
    )


@traceable(
    run_type="chain",
    name="langextract_demo_v2",
    tags=["langextract", "demo", "two_pass"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def main(text_or_documents: str) -> dict[str, str]:
    model_config = build_model_config()
    output_dir = Path("outputs") / "langextract_specification_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = split_sentences(text_or_documents)
    merged_results: list[lx.data.AnnotatedDocument] = []

    with traced_openai_client():
        for index, sentence in enumerate(sentences, start=1):
            span_result = extract_spans_for_sentence(sentence, model_config)
            coref_result = extract_coref_for_sentence(sentence, model_config)
            merged_results.append(
                merge_pass_outputs(
                    sentence,
                    span_result,
                    coref_result,
                    document_id=f"sentence-{index:03d}",
                )
            )

    # 保存原始合并 JSON
    visualization_document = build_visualization_document(
        text_or_documents,
        merged_results,
    )

    raw_json_path = output_dir / "merged_results.json"
    save_json(raw_json_path, merged_results)

    # 保存成 JSONL，方便后续看
    jsonl_path = output_dir / "merged_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in merged_results:
            f.write(json.dumps(to_plain_data(item), ensure_ascii=False) + "\n")

    # 也尝试保存 LangExtract 可视化格式
    # 这里把每句当一个 document 存
    try:
        lx.io.save_annotated_documents(
            [visualization_document],
            output_dir=output_dir,
            output_name="annotated_document_full_input.jsonl",
        )
        html_content = lx.visualize(
            str(output_dir / "annotated_document_full_input.jsonl")
        )
        with open(output_dir / "visualization.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        visualization_file = str(output_dir / "visualization.html")
    except Exception as e:
        visualization_file = f"visualization failed: {e}"

    return {
        "raw_json": str(raw_json_path),
        "jsonl_file": str(jsonl_path),
        "visualization_file": visualization_file,
    }


if __name__ == "__main__":
    artifacts = main(INPUT_TEXT)
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))
