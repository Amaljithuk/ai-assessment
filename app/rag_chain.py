import os
import time
import weaviate
from weaviate.exceptions import WeaviateConnectionError
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# 1. Helper to connect to Database
def get_weaviate_client():
    print(f"--- Connecting to Weaviate at {WEAVIATE_HOST}:{WEAVIATE_PORT} ---")
    while True:
        try:
            client = weaviate.connect_to_custom(
                http_host=WEAVIATE_HOST,
                http_port=WEAVIATE_PORT,
                http_secure=False,
                grpc_host=WEAVIATE_HOST,
                grpc_port=50051,
                grpc_secure=False,
            )
            if not client.is_connected():
                client.connect()
            print("✅ Connected to Weaviate!")
            return client
        except (WeaviateConnectionError, Exception) as e:
            print(f"⏳ Weaviate not ready yet... waiting 5s. (Error: {str(e)})")
            time.sleep(5)

# 2. Helper to get just the Retriever
def get_retriever():
    client = get_weaviate_client()
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="DocumentConfig",
        text_key="text",
        embedding=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Main function to get the RAG Chain
def get_rag_chain():
    # Use the helper above
    retriever = get_retriever()
    
    llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
    
    template = """You are an AI assistant for Question Answering.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. 
    Keep the answer concise.

    Context: {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain