from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
from agents.reader_agent import process_case_study
from agents.categorize_agent import categorize_case_study
from agents.validation_agent import validate_case_study
from typing import List

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_files(files: List[UploadFile]):
    all_metadata = []
    all_validation = []

    for file in files:
        file_bytes = await file.read()
        case_text, _ = process_case_study(file.filename, file_bytes)

        categorization = categorize_case_study(case_text)
        validation = validate_case_study(
            categorization.get("category", ""),
            categorization.get("domain", ""),
            categorization.get("technology", "")
        )

        metadata = {
            "file_name": file.filename,
            "summary": categorization.get("summary", ""),
            "category": categorization.get("category", ""),
            "domain": categorization.get("domain", ""),
            "technology": categorization.get("technology", "")
        }
        all_metadata.append(metadata)
        all_validation.append({"file_name": file.filename, **validation})

    return {"metadata": all_metadata, "validation": all_validation}

@app.post("/chat")
async def chat_with_metadata(query: str, metadata: list):
    # Chatbot logic here
    return {"answer": "This would be AI's response"}

@app.get("/search")
async def search(category: str = None, domain: str = None):
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    filtered = metadata
    if category:
        filtered = [m for m in filtered if m.get("category") == category]
    if domain:
        filtered = [m for m in filtered if m.get("domain") == domain]
    return {"results": filtered}
