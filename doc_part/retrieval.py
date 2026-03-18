from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

def create_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Initialize HuggingFace sentence-transformer embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}  # change to cuda if GPU
    )
    print("[INFO] Embeddings model initialized.")
    return embeddings

def query_vector_store(vector_store, query_text, top_n=5, doc_type=None):
    """
    Perform similarity search with optional document type filter.

    doc_type:
        - "table_row"
        - "requirement"
        - None   -> search all
    """
    metadata_filter = {"type": doc_type} if doc_type else None

    results = vector_store.similarity_search(
        query_text,
        k=top_n,
        filter=metadata_filter
    )

    print(f"[INFO] Query completed. doc_type={doc_type}")
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

def create_cross_encoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Initialize CrossEncoder reranker."""
    model = CrossEncoder(model_name)
    print("[INFO] CrossEncoder initialized.")
    return model

def rerank_results(query, docs, cross_encoder, top_n=5):
    """
    Rerank retrieved documents using CrossEncoder.
    """

    pairs = [(query, doc.page_content) for doc in docs]

    scores = cross_encoder.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked = [doc for doc, score in scored_docs[:top_n]]

    return reranked


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
    cross_encoder = create_cross_encoder()

    query = "Motor design (temperature spec, actuation length/duty cycle)"

    # Step 1: vector search
    initial_results = query_vector_store(
        vector_store,
        query,
        top_n=20,
        doc_type="FS"
    )

    # Step 2: rerank
    reranked_results = rerank_results(
        query,
        initial_results,
        cross_encoder,
        top_n=10
    )

    print("\n==== FINAL RESULTS ====\n")

    for r in reranked_results:
        print(r.page_content)
        print("-" * 80)