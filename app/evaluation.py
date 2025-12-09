import os
import pandas as pd
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig # <--- NEW IMPORT
from langchain_ollama import ChatOllama, OllamaEmbeddings
from app.rag_chain import get_rag_chain, get_retriever

# 1. Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"--- Setting up Evaluator using {OLLAMA_BASE_URL} ---")

# Increase timeout significantly to prevent crashes
ollama_model = ChatOllama(
    model="llama3", 
    temperature=0,
    base_url=OLLAMA_BASE_URL,
    request_timeout=360.0 # 6 minutes timeout per request
)

ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_BASE_URL
)

evaluator_llm = LangchainLLMWrapper(ollama_model)

# 2. Test Data
data_samples = {
    'question': [
        "What is the main topic of this document?", 
        "List one key detail mentioned in the text."
    ],
    'ground_truth': [
        "The document is about Photosynthesis.", 
        "It mentions sunlight converting to chemical energy."
    ]
}

def run_eval():
    print("--- 1. Generating Answers ---")
    chain = get_rag_chain()
    retriever = get_retriever()
    
    answers = []
    contexts = []
    
    for q in data_samples['question']:
        print(f"Asking: {q}")
        try:
            response_text = chain.invoke(q)
            answers.append(response_text)
            docs = retriever.invoke(q)
            context_text = [doc.page_content for doc in docs]
            contexts.append(context_text)
        except Exception as e:
            print(f"Error processing question '{q}': {e}")
            return

    # 3. Prepare Dataset
    data_samples['answer'] = answers
    data_samples['contexts'] = contexts
    dataset = Dataset.from_dict(data_samples)

    print("--- 2. Running Ragas Evaluation (Sequential Mode) ---")
    
    # Configure Ragas to run SLOWLY (1 by 1) so Ollama doesn't crash
    my_run_config = RunConfig(
        max_workers=1,  # Only 1 query at a time
        timeout=360     # Allow plenty of time
    )

    results = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=evaluator_llm, 
        embeddings=ollama_embeddings,
        run_config=my_run_config # <--- APPLY CONFIG HERE
    )

    print("\n\n========== EVALUATION REPORT ==========")
    print(results)
    
    output_path = "data/evaluation_report.csv"
    df = results.to_pandas()
    df.to_csv(output_path, index=False)
    print(f"✅ Report saved to: {output_path}")

if __name__ == "__main__":
    run_eval()