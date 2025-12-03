from langchain_community.document_loaders import JSONLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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
            text_content=True
        )
    else:
        raise ValueError("Unsupported file type. Use JSON.")
    
    docs =  loader.load()
    print(f"[INFO] Loaded {len(docs)} FMEA entries from {file_path}")
    return docs

# To be finished
def create_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )
    print("[INFO] Embeddings model initialized.")
    return embeddings

def create_vector_store(embeddings, persist_dir="../../../DATA/chroma_langchain_fmea_db"):
    vector_store = Chroma(
        collection_name="fmea_example_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    print("[INFO] Vector store initialized.")
    return vector_store

def add_failureblock_to_vector_store(vector_store, docs):
    """Add FMEA Documents to vector store."""
    ids = vector_store.add_documents(docs)
    print(f"[INFO] Added {len(ids)} FMEA entries to the vector store.")
    return ids

def main():
    file_path = "../dfmea_effect_flat.json"
    docs = load_json(file_path)
    # print(docs[1])
    # print(docs[1].metadata)

    # Create embedding model
    embeddings = create_embeddings()

    # Init vector store
    vstore = create_vector_store(embeddings)

    # Add all FMEA rows
    add_failureblock_to_vector_store(vstore, docs)

    query = "gateway backend incorrect state"
    results = vstore.similarity_search(query, k=3)

    for i, r in enumerate(results):
        print(f"\n--- RESULT {i+1} ---")
        print("CONTENT:", r.page_content)
        print("METADATA:", r.metadata)


if __name__ == "__main__":
    main()