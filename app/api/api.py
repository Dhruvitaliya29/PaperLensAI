"""
FastAPI Backend for PaperLens AI
"""

from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.rag.rag_pipeline import RAGPipeline


app = FastAPI(
    title="PaperLens AI",
    version="1.0"
)

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

VECTOR_FOLDER = Path("data/vectorstore")
VECTOR_FOLDER.mkdir(parents=True, exist_ok=True)

pipeline = RAGPipeline()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():

    return {
        "message": "Welcome to PaperLens AI API"
    }


@app.get("/health")
def health():

    return {
        "status": "Running"
    }


@app.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...)
):

    uploaded = []

    for file in files:

        save_path = UPLOAD_FOLDER / file.filename

        with open(save_path, "wb") as f:
            f.write(await file.read())

        uploaded.append(file.filename)

    return {
        "uploaded_files": uploaded
    }


@app.post("/index")
def index_documents():

    result = pipeline.index_documents(
        UPLOAD_FOLDER
    )

    pipeline.save_vector_store(
        VECTOR_FOLDER
    )

    return {
        "message": "Documents indexed successfully.",
        "documents": result["documents"],
        "chunks": result["chunks"]
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    pipeline.load_vector_store(
        VECTOR_FOLDER
    )

    response = pipeline.ask(
        request.question
    )

    return response