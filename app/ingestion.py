import os
import weaviate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

# Configuration
# We use os.getenv to allow Docker to inject the correct hostnames
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"

def clean_metadata(docs):
    """
    Weaviate does not allow dots in property names (e.g., 'ptex.fullbanner').
    This function replaces dots with underscores in all metadata keys.
    """
    for doc in docs:
        new_metadata = {}
        for key, value in doc.metadata.items():
            # Replace '.' with '_' to make Weaviate happy
            clean_key = key.replace(".", "_")
            
            # Ensure value is a simple type (str, int, float, bool) or list of them
            if isinstance(value, (str, int, float, bool, list)):
                new_metadata[clean_key] = value
            else:
                # If it's something complex, convert to string
                new_metadata[clean_key] = str(value)
                
        doc.metadata = new_metadata
    return docs

def ingest_docs(pdf_path: str):
    """
    Ingests a specific PDF file into Weaviate.
    """
    print(f"--- Processing File: {pdf_path} ---")
    
    if not os.path.exists(pdf_path):
        return {"status": "error", "message": "File not found"}

    # 1. Load the PDF
    try:
        loader = PyPDFLoader(pdf_path)
        raw_documents = loader.load()
    except Exception as e:
        return {"status": "error", "message": f"Failed to load PDF: {str(e)}"}

    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(raw_documents)
    
    # --- IMPORTANT: Clean Metadata ---
    chunks = clean_metadata(chunks)
    
    print(f"Split into {len(chunks)} chunks.")

    # 3. Embed and Upload
    # We pass the base_url so it works inside Docker
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )
    
    # Connect using custom settings for Docker compatibility
    print(f"Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT}...")
    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=50051,
        grpc_secure=False,
    )

    try:
        WeaviateVectorStore.from_documents(
            client=client,
            documents=chunks,
            embedding=embeddings,
            index_name="DocumentConfig"
        )
        print(f"✅ Successfully ingested {pdf_path}")
        return {"status": "success", "chunks": len(chunks)}
    except Exception as e:
        print(f"❌ Error uploading to Weaviate: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        client.close()

if __name__ == "__main__":
    # Test with default if run directly
    ingest_docs("../data/sample.pdf")