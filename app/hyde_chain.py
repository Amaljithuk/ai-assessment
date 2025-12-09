import os
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda # <--- NEW IMPORT
from langchain_core.output_parsers import StrOutputParser

# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

def get_hyde_chain():
    # 1. Connect to Weaviate
    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=WEAVIATE_PORT,
        http_secure=False,
        grpc_host=WEAVIATE_HOST,
        grpc_port=50051,
        grpc_secure=False,
    )

    # 2. Define Models
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

    # 3. Create the HyDe Generator
    hyde_template = """Please write a brief, scientific passage that answers the question. 
    Do not verify facts, just generate a plausible-sounding answer.
    
    Question: {question}
    Passage:"""
    
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
    hyde_generator = hyde_prompt | llm | StrOutputParser()

    # 4. Create the Retriever
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="DocumentConfig",
        text_key="text",
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 5. The HyDe Retrieval Logic
    def retrieve_with_hyde(question):
        print(f"--- Generating HyDe Document for: {question} ---")
        hypothetical_doc = hyde_generator.invoke({"question": question})
        print(f"Generated HyDe Doc: {hypothetical_doc[:100]}...") 
        return retriever.invoke(hypothetical_doc)

    # 6. Final Answer Generation
    final_template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    final_prompt = ChatPromptTemplate.from_template(final_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7. Build the Chain (Using RunnableLambda to fix the Error)
    chain = (
        {
            # We wrap the functions so the '|' operator works
            "context": RunnableLambda(lambda x: x["question"]) | RunnableLambda(retrieve_with_hyde) | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda x: x["question"])
        }
        | final_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain