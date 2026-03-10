#### ---------------------------------------------
#### Imports
#### ---------------------------------------------
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


#### ---------------------------------------------
#### 1. Load documents
#### ---------------------------------------------
def load_document(file_path: str):
    """Load PDF or DOCX document and return list of Document objects."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or DOCX.")
    
    docs = loader.load()
    print(f"[INFO] Loaded {len(docs)} pages from {file_path}")
    return docs


#### ---------------------------------------------
#### 2. Split text into chunks
#### ---------------------------------------------
def split_documents(docs, chunk_size=600, overlap=120):
    """Split loaded documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks.")
    return chunks


#### ---------------------------------------------
#### 3. Create embeddings model
#### ---------------------------------------------
def create_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Initialize HuggingFace sentence-transformer embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}  # change to cuda if GPU
    )
    print("[INFO] Embeddings model initialized.")
    return embeddings


#### ---------------------------------------------
#### 4. Create vector store
#### ---------------------------------------------
def create_vector_store(embeddings, persist_dir="./DATA/chroma_langchain_db"):
    """Initialize Chroma vector store."""
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    print("[INFO] Vector store initialized.")
    return vector_store


#### ---------------------------------------------
#### 5. Add documents to vector store
#### ---------------------------------------------
def add_chunks_to_vector_store(vector_store, chunks):
    """Add document chunks to the vector store."""
    ids = vector_store.add_documents(chunks)
    print(f"[INFO] Added {len(ids)} chunks to the vector store.")
    return ids


#### ---------------------------------------------
#### 6. Query vector store
#### ---------------------------------------------
def query_vector_store(vector_store, query_text,top_n):
    """Perform similarity search in the vector store."""
    results = vector_store.similarity_search(query_text, k=top_n)
    print("[INFO] Query completed.\n")
    return results



def load_vector_store(
    embeddings,
    persist_dir="./DATA/chroma_langchain_db",
    collection_name="example_collection"
):
    """Load an existing Chroma vector store from disk."""
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    print(f"[INFO] Loaded vector store from: {persist_dir}")
    return vector_store


#### ---------------------------------------------
#### MAIN WORKFLOW
#### ---------------------------------------------
def main():
    file_path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\FS6782170087R34.pdf"

    # Step 1: Load
    docs = load_document(file_path)

    # Step 2: Split
    chunks = split_documents(docs)

    # Step 3: Embedding model
    embeddings = create_embeddings()

    # Step 4: Vector Store
    vector_store = create_vector_store(embeddings)

    # Step 5: Add chunks
    add_chunks_to_vector_store(vector_store, chunks)

    # Step 6: Query
    query = "Motor not shorted while device is not powered"
    results = query_vector_store(vector_store, query,top_n=3)

    # Print the best result
    print("------ Top Result ------")
    print(results[0].page_content)


#### ---------------------------------------------
#### RUN
#### ---------------------------------------------
if __name__ == "__main__":
    # main()
    persist_dir = "./DATA/chroma_langchain_db"
    collection_name = "example_collection"
    embeddings = create_embeddings()
    vector_store = load_vector_store(
        embeddings,
        persist_dir=persist_dir,
        collection_name=collection_name
    )

    query = "Motor not shorted while device is not powered"
    results = query_vector_store(vector_store, query, top_n=4)

    for r in results:
        print(r.page_content)
        print("-" * 80)
