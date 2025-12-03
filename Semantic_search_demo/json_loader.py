from langchain_community.document_loaders import JSONLoader
import getpass
import os
from dotenv import load_dotenv

# Read key from .env
load_dotenv()
# Manual input
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")

os.environ["LANGSMITH_TRACING"] = "true"

def load_json(file_path:str):
    if file_path.lower().endswith(".json"):
        loader = JSONLoader(
            file_path= file_path,
            jq_schema= ".[]",
            content_key="text",
            text_content=False
        )
    else:
        raise ValueError("Unsupported file type. Use JSON.")
    
    docs =  loader.load()
    print(f"[INFO] Loaded from {file_path}")
    return docs

def main():
    file_path = "../dfmea_effect_flat.json"
    docs = load_json(file_path)
    print(docs[1])
    print(docs[1].metadata)

if __name__ == "__main__":
    main()