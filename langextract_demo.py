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
Extract characters, emotions, and relationships in order of appearance.
Use exact text for extractions. Do not paraphrase or overlap entities.
Provide meaningful attributes for each entity to add context.
""")

examples = [
    lx.data.ExampleData(
        text=(
            "The electronic soft-starter shall support up to 15 start/stop per hour"
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="Element",
                extraction_text="electronic soft-starter",
                attributes={"emotional_state": "wonder"},
            ),
            lx.data.Extraction(
                extraction_class="requirement",
                extraction_text="shall support",
                attributes={"feeling": "gentle awe"},
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"},
            ),
        ],
    )
]

input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

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
    tags=["langextract", "demo"],
    project_name=LANGSMITH_PROJECT_NAME,
)
def main(text_or_documents: str, prompt_description: str) -> dict[str, str]:
    """Run the LangExtract demo and save artifacts."""

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

    output_dir = Path("outputs") / "langextract_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "extraction_results.jsonl"
    lx.io.save_annotated_documents(
        [result],
        output_dir=output_dir,
        output_name=output_file.name,
    )

    html_content = lx.visualize(str(output_file))
    with open(output_dir / "visualization.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    return {
        "output_file": str(output_file),
        "visualization_file": str(output_dir / "visualization.html"),
    }


if __name__ == "__main__":
    main(input_text, prompt)
