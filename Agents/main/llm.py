from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOpenAI(
    model="azure/gpt-4.1",
    temperature=0,
    openai_api_base="http://litellm.ame.local/v1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# llm = ChatOllama(
#     model="llama3.1:8b",   # or "llama3", "mistral", “qwen2.5-coder”
#     temperature=0,
# )
