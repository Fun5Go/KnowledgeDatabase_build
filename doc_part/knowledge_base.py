#### ---------------------------------------------
#### Imports
#### ---------------------------------------------
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re
from langchain_core.documents import Document
import pdfplumber

REQ_PATTERN = r"(REQ_\d+|DRQ_\d+|LIM_\d+|CHO_\d+|LIM_\d)"
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

def split_documents(docs):
    """
    Split documents by REQ/DRQ requirement IDs.
    """

    chunks = []

    for doc in docs:

        text = doc.page_content
        metadata = doc.metadata

        parts = re.split(REQ_PATTERN, text)

        for i in range(1, len(parts), 2):

            req_id = parts[i]
            body = parts[i+1].strip()

            chunk_text = f"{req_id}\n{body}"

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": metadata.get("source"),
                        "page": metadata.get("page"),
                        "requirement_id": req_id,
                        "type": "FS"
                    }
                )
            )

    print(f"[INFO] Extracted {len(chunks)} requirement chunks")

    return chunks

def extract_tables_from_pdf(file_path):
    """
    Extract tables and convert each row to semantic text.
    """

    table_chunks = []

    with pdfplumber.open(file_path) as pdf:

        for page_idx, page in enumerate(pdf.pages):

            tables = page.extract_tables()

            for t_index, table in enumerate(tables):

                if not table:
                    continue

                headers = table[0]

                for row_index, row in enumerate(table[1:]):

                    parts = []

                    for col, val in zip(headers, row):

                        if val:
                            parts.append(f"{col}: {val}")

                    row_text = ". ".join(parts)

                    table_chunks.append(
                        Document(
                            page_content=row_text,
                            metadata={
                                "page": page_idx,
                                "table_id": t_index,
                                "row_id": row_index,
                                "type": "table_row"
                            }
                        )
                    )

    print(f"[INFO] Extracted {len(table_chunks)} table rows")

    return table_chunks

def prepare_chunks(file_path):

    docs = load_document(file_path)

    req_chunks = split_documents(docs)

    table_chunks = []

    if file_path.endswith(".pdf"):
        table_chunks = extract_tables_from_pdf(file_path)

    # all_chunks = req_chunks + table_chunks

    print(f"[INFO] Total chunks: {len(req_chunks)}")

    return req_chunks

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

def main():
    file_path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\doc_part\FS6782170087R34.pdf"

    chunks = prepare_chunks(file_path)

    # Step 3: Embedding model
    embeddings = create_embeddings()

    # Step 4: Vector Store
    vector_store = create_vector_store(embeddings)

    # Step 5: Add chunks
    add_chunks_to_vector_store(vector_store, chunks)



#### ---------------------------------------------
#### RUN
#### ---------------------------------------------
if __name__ == "__main__":
    main()