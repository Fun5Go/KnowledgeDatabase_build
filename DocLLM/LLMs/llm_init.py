from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - optional dependency
    ChatOllama = None

load_dotenv()


DOC_LLM_LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "DocumentProcess")


def configure_langsmith(project_name: str | None = None) -> str:
    """
    Normalize LangSmith environment variables for DocLLM runs.

    Why this exists:
    - Existing repo code relies on environment-based LangSmith tracing.
    - Some LangSmith versions expect `LANGSMITH_TRACING_V2=true`.
    - We want DocLLM traces to land in a predictable project.
    """

    tracing_value = os.getenv("LANGSMITH_TRACING_V2") or os.getenv("LANGSMITH_TRACING")
    if tracing_value and not os.getenv("LANGSMITH_TRACING_V2"):
        os.environ["LANGSMITH_TRACING_V2"] = tracing_value

    resolved_project = project_name or DOC_LLM_LANGSMITH_PROJECT
    os.environ["LANGSMITH_PROJECT"] = resolved_project
    return resolved_project


def get_llm_backend(
    backend: str = "openai",
    model: str | None = None,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> Any:
    """
    Returns an LLM instance based on selected backend.

    Mirrors the backend-selection structure already used in `Agentic_workflow/LLMs/llm_init.py`.
    """

    backend = backend or os.getenv("LLM_BACKEND", "openai")

    if backend == "local":
        if ChatOllama is None:
            raise RuntimeError("ChatOllama is not installed, but backend='local' was requested.")
        return ChatOllama(
            model=model or os.getenv("LLM_MODEL", "llama3.1:8b"),
            temperature=temperature,
        )

    if ChatOpenAI is None:
        raise RuntimeError("langchain_openai is not installed, but backend='openai' was requested.")

    model_kwargs: dict[str, Any] = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "azure/gpt-4.1"),
        temperature=temperature,
        openai_api_base=os.getenv("OPENAI_API_BASE", "http://litellm.ame.local/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs=model_kwargs,
    )
