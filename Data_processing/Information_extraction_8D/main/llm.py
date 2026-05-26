from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

# llm = ChatOpenAI(
#     model="azure/gpt-4.1",
#     temperature=0,
#     openai_api_base="http://litellm.ame.local/v1",
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
# )


# llm = ChatOllama(
#     model="llama3.1:8b",   # or "llama3", "mistral", “qwen2.5-coder”
#     temperature=0,
# )

def get_llm_backend(
    backend: str = "openai",
    model: str | None = None,
    temperature: float = 0,
    json_mode: bool = False
):
    """
    Returns an LLM instance based on selected backend.
    Backend controlled by environment: LLM_BACKEND=openai|azure|local
    """
    backend = backend or os.getenv("LLM_BACKEND", "openai")

    if backend == "local":
        return ChatOllama(
            model=model or os.getenv("LLM_MODEL", "llama3.1:8b"),
            temperature=temperature,
        )

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "azure/gpt-4.1"),
        temperature=temperature,
        openai_api_base=os.getenv("OPENAI_API_BASE", "http://litellm.ame.local/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs=model_kwargs,
    )