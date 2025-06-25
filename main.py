# Main File where all the routes and fastapi exists

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import List
import os
import tempfile
import shutil

from services import RAGService

app = FastAPI(
    title="PDF RAG API",
    description="A RAG system for PDFs using LlamaParse, LangChain, and Groq",
    version="1.0.0"
)

# Initialize the RAG service
rag_service = RAGService()


@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload and ingest multiple PDF files into the RAG system.

    Args:
        files: List of PDF files to upload and process

    Returns:
        JSON response with ingestion status
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Temporary: File type validation by checking filename ending characters.
    pdf_files = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a PDF. Only PDF files are supported."
            )
        pdf_files.append(file)

    # Recording a file path list to be ingested
    # Storing temp files because LlamaParse accepts locally stored files and their paths.
    temp_file_paths = []
    try:
        # Temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()

        # Saving uploaded files to temporary location
        for file in pdf_files:
            temp_path = os.path.join(temp_dir, file.filename)
            temp_file_paths.append(temp_path)

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Processing and Ingesting PDFs
        result = await rag_service.ingest_pdfs(temp_file_paths)

        return JSONResponse(
            status_code=200 if result["status"] == "success" else 500,
            content={
                "status": result["status"],
                "message": result["message"],
                "files_processed": result["files_processed"],
                "file_names": [file.filename for file in pdf_files]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

    finally:
        # Deleting temporary files stored locally while parsing.
        for temp_path in temp_file_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_path}: {str(e)}")

        # Removing temporary directory
        try:
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temp directory {temp_dir}: {str(e)}")


@app.post("/query")
async def query_documents(question: str = Form(...)):
    """
    Query the RAG system with a question to retrieve relevant answers.

    Args:
        question: The question to ask the RAG system

    Returns:
        JSON response with the answer and source information
    """
    if not question or question.strip() == "":
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = rag_service.ask_question(question)

        return JSONResponse(
            status_code=200 if result["status"] == "success" else 400,
            content={
                "status": result["status"],
                "message": result["message"],
                "query": question,
                "answer": result["answer"],
                "sources": result["sources"]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)