import os
import textwrap
from contextlib import contextmanager
from pathlib import Path

import langextract as lx
from dotenv import load_dotenv
from langextract.factory import ModelConfig
from langsmith.wrappers import wrap_openai

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


def configure_langsmith(project_name: str | None = None) -> str:
    """Normalize LangSmith environment variables for this demo."""
    tracing_value = os.getenv("LANGSMITH_TRACING_V2") or os.getenv("LANGSMITH_TRACING")
    if tracing_value and not os.getenv("LANGSMITH_TRACING_V2"):
        os.environ["LANGSMITH_TRACING_V2"] = tracing_value

    resolved_project = project_name or LANGSMITH_PROJECT_NAME
    os.environ["LANGSMITH_PROJECT"] = resolved_project
    return resolved_project


LANGSMITH_PROJECT_NAME = configure_langsmith()

prompt = textwrap.dedent("""\
Extract entities from the specification text into these exact-span labels:

1. component: the equipment, module, function block, system part, or associated component being specified
2. function: the action, capability, behavior, or required operation
3. object: the target, item, signal, error, current, voltage, input, output, or thing involved in the function
4. constraint: the condition, scope, limit, duration, trigger, range, threshold, or context introduced by words like "when", "if", "while", "for", "during", "after", "before", or similar phrases

Rules:
- Use exact text spans only. Do not paraphrase.
- Return spans in order of appearance.
- Do not overlap spans unless necessary.
- If the same phrase appears again and is meaningful, extract it again.
""")

examples = [
    lx.data.ExampleData(
        text=(
            "The electronic soft-starter shall support up to 15 start/stop per hour"
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="component",
                extraction_text="electronic soft-starter",
                attributes={"role": "specified system element"},
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="shall support",
                attributes={"role": "requirement action"},
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="15 start/stop",
                attributes={"role": "operational quantity"},
            ),
            lx.data.Extraction(
                extraction_class="constraint",
                extraction_text="up to 15 start/stop per hour",
                attributes={"role": "performance limit"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The Safety module will report an SAFETY_ERROR_OVERVOLTAGE error when the measured rms input voltage is below the configurable parameter voltage_in - 15% for more than 10 seconds."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="component",
                extraction_text="Safety module",
                attributes={"role": "specified system element"},
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="will report",
                attributes={"role": "requirement action"},
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="an SAFETY_ERROR_OVERVOLTAGE error",
                attributes={"role": "reported object"},
            ),
            lx.data.Extraction(
                extraction_class="constraint",
                extraction_text="when the measured rms input voltage is below the configurable parameter voltage_in - 15% for more than 10 seconds",
                attributes={"role": "triggering condition"},
            ),
        ],
    ),
    lx.data.ExampleData(
        text=(
            "The startSoftStartAlgorithm function will function individually on phases U and W, for each of these phases it will have an Zero Crossing Detection Input that will be active high when a zero crossing is detected."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="component",
                extraction_text="startSoftStartAlgorithm function",
                attributes={"role": "specified function element"},
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="will function",
                attributes={"role": "requirement action"},
            ),
            lx.data.Extraction(
                extraction_class="constraint",
                extraction_text="individually on phases U and W",
                attributes={"role": "operating scope"},
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="will have",
                attributes={"role": "association action"},
            ),
            lx.data.Extraction(
                extraction_class="object",
                extraction_text="Zero Crossing Detection Input",
                attributes={"role": "associated input"},
            ),
            lx.data.Extraction(
                extraction_class="function",
                extraction_text="will be active high",
                attributes={"role": "behavior requirement"},
            ),
            lx.data.Extraction(
                extraction_class="constraint",
                extraction_text="when a zero crossing is detected",
                attributes={"role": "triggering condition"},
            ),
        ],
    )
]

input_text = (
    "During the soft start the thyristors shall handle peak currents up to 100A. "
    "The thyristors should handle that current for the whole duration of the soft start."
)

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing API key. Set OPENAI_API_KEY (preferred) or LANGEXTRACT_API_KEY."
    )

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
    """Patch OpenAI client construction so LangSmith can trace LangExtract calls."""
    import openai

    original_openai = openai.OpenAI

    def wrapped_openai(*args, **kwargs):
        return wrap_openai(original_openai(*args, **kwargs))

    openai.OpenAI = wrapped_openai
    try:
        yield
    finally:
        openai.OpenAI = original_openai


@traceable(
    run_type="chain",
    name="langextract_demo",
    tags=["langextract", "demo", "ner-comparison"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def main(text_or_documents: str, prompt_description: str):
    """Run LangExtract and print simplified entities for NER comparison."""

    model_config = ModelConfig(
        model_id=llm_model,
        provider="OpenAILanguageModel",
        provider_kwargs={
            "api_key": api_key,
            "base_url": api_base,
        },
    )

    with traced_openai_client():
        result = lx.extract(
            text_or_documents=text_or_documents,
            prompt_description=prompt_description,
            examples=examples,
            config=model_config,
        )

    print("\n=== LangExtract Result ===")
    for extraction in result.extractions:
        print({
            "text": extraction.extraction_text,
            "label": extraction.extraction_class,
            "attributes": extraction.attributes,
        })

    return result


if __name__ == "__main__":
    main(input_text, prompt)