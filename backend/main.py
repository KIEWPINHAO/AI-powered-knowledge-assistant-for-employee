import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from .retrieval import answer_employee_question
from .ingestion import ingest_pdf  # Import your ingestion function!

app = FastAPI(title="Company Policy RAG API", version="1.0")

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        result = answer_employee_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW UPLOAD ENDPOINT ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # 1. Ensure the data folder exists
        os.makedirs("./data", exist_ok=True)
        
        # 2. Save the uploaded file locally
        file_path = f"./data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Trigger your ingestion script to chunk and embed it!
        ingest_pdf(file_path)
        
        return {"message": f"Successfully uploaded and vectorized {file.filename}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "HR RAG API is running!"}