import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.rag_chain import get_rag_chain
from app.ingestion import ingest_docs
# 1. NEW: Import the HyDe chain factory
from app.hyde_chain import get_hyde_chain

app = FastAPI(title="AI Assessment RAG API")

# 2. Load Chains at startup
print("--- Loading Standard RAG Chain ---")
rag_chain = get_rag_chain()

print("--- Loading HyDe Chain ---")
hyde_chain = get_hyde_chain()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"status": "running", "message": "Go to /docs to test the API"}

# Standard RAG Endpoint
@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    try:
        # Standard chain expects a string or simple input depending on implementation
        # Our rag_chain.py expects just the string usually, but let's be safe
        result = rag_chain.invoke(request.question)
        return {"answer": result}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 3. NEW: HyDe Endpoint
@app.post("/hyde-chat", response_model=QueryResponse)
def hyde_chat(request: QueryRequest):
    try:
        print(f"HyDe Request: {request.question}")
        # Our hyde_chain specifically expects a dictionary input {"question": ...}
        result = hyde_chain.invoke({"question": request.question})
        return {"answer": result}
    except Exception as e:
        print(f"Error in HyDe: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Upload Endpoint
@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    file_location = f"data/{file.filename}"
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
        
    print(f"File saved to {file_location}. Starting ingestion...")
    result = ingest_docs(file_location)
    
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
        
    return {"message": "File uploaded and ingested successfully", "details": result}