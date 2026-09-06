import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter #chunks the document into smaller pieces for better processing
from langchain_huggingface import HuggingFaceEmbeddings #converts text into vector embeddings
from langchain_community.vectorstores import FAISS #stores the vector embeddings for efficient retrieval

DATA_PATH="Data/" #path to the pdfs directory
DB_FAISS_PATH="vectorstore/db_faiss" #path to store the FAISS index

#Step 1: Load RAW Pdfs
def load_pdfs(pdf_directory):
    loader = DirectoryLoader(pdf_directory, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    return documents

#Step 2: Split Documents into Chunks
def create_chunks(documents, chunk_size=500, chunk_overlap=50):  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

#Step 3:Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

#Step 4: Execute the pipeline (Refactored into a callable function)
def build_vector_store():
    """Reads PDFs from Data folder, chunks them, embeds them, and saves to FAISS."""
    # Ensure directories exist to prevent crashes
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    
    documents = load_pdfs(DATA_PATH)
    if not documents:
        return False # No documents found to process
        
    chunks = create_chunks(documents)
    embedding_model = get_embedding_model()
    
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return True

# If the script is run directly from the terminal, execute the pipeline
if __name__ == "__main__":
    print("Building vector store...")
    success = build_vector_store()
    if success:
        print("Vector store built successfully!")
    else:
        print("No documents found in the Data directory.")