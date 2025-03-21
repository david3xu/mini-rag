"""
Documents API module for the Mini RAG application.

This module provides endpoints for uploading and processing documents,
including chunking, embedding generation, and vector storage.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid
import logging
import psutil

from app.services.document_processor import document_processor
from app.services.embeddings import embeddings_service
from app.services.vectorstore import vector_store
from config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Model definitions
class DocumentResult(BaseModel):
    """Result of document processing."""
    filename: str
    chunks_processed: Optional[int] = None
    status: str
    error: Optional[str] = None

class UploadResponse(BaseModel):
    """Response to document upload request."""
    status: str
    files_processed: int
    results: List[DocumentResult]

async def process_document_background(file_path: str, original_filename: str, results: List[DocumentResult]):
    """
    Process a document in the background to avoid blocking the API response.
    
    Args:
        file_path: Path to temporary file
        original_filename: Original name of the uploaded file
        results: List to update with processing results
    """
    result = DocumentResult(filename=original_filename, status="processing")
    
    # Find the result to update
    for i, r in enumerate(results):
        if r.filename == original_filename:
            results[i] = result
            break
    else:
        results.append(result)
    
    try:
        # Process the file
        chunks = document_processor.process_file(file_path)
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Process in batches to manage memory
        batch_size = settings.DEFAULT_BATCH_SIZE
        
        # Check available memory and adjust batch size if needed
        mem = psutil.virtual_memory()
        if mem.available < settings.MEMORY_SAFETY_MARGIN_MB * 1024 * 1024:
            # Reduce batch size if memory is constrained
            batch_size = max(1, batch_size // 2)
            logger.warning(f"Memory constrained. Reducing batch size to {batch_size}")
        
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            batch_texts = texts[i:batch_end]
            batch_ids = ids[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            
            # Generate embeddings
            batch_embeddings = embeddings_service.generate_embeddings(batch_texts)
            
            # Add to vector store
            vector_store.add_documents(
                documents=batch_texts,
                embeddings=batch_embeddings,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
        
        # Update result with success
        for i, r in enumerate(results):
            if r.filename == original_filename:
                results[i] = DocumentResult(
                    filename=original_filename,
                    chunks_processed=len(chunks),
                    status="success"
                )
                break
                
    except Exception as e:
        logger.exception(f"Error processing document {original_filename}: {str(e)}")
        
        # Update result with error
        for i, r in enumerate(results):
            if r.filename == original_filename:
                results[i] = DocumentResult(
                    filename=original_filename,
                    status="error",
                    error=str(e)
                )
                break
                
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Upload and process documents for the RAG system.
    
    Handles multiple file uploads, processes them into chunks, generates embeddings,
    and stores them in the vector database.
    
    Args:
        background_tasks: FastAPI background tasks handler
        files: List of files to upload and process
        
    Returns:
        Upload response with status and results
        
    Raises:
        HTTPException: If processing fails or files are invalid
    """
    try:
        results = []
        temp_files = []
        
        for file in files:
            # Validate file extension
            filename = file.filename
            ext = os.path.splitext(filename)[1].lower()
            
            if ext not in ['.pdf', '.txt', '.md', '.json']:
                results.append(DocumentResult(
                    filename=filename,
                    status="error",
                    error=f"Unsupported file type: {ext}. Supported types: PDF, TXT, MD, JSON."
                ))
                continue
            
            # Check file size before reading content
            file_size = 0
            chunk_size = 1024 * 1024  # 1MB chunks for memory efficiency
            content = bytearray()
            
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                content.extend(chunk)
                file_size += len(chunk)
                
                if file_size > settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
                    await file.close()
                    results.append(DocumentResult(
                        filename=filename,
                        status="error",
                        error=f"File too large. Maximum size is {settings.MAX_DOCUMENT_SIZE_MB}MB"
                    ))
                    break
            
            if file_size > settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
                continue
            
            # Save file temporarily
            file_id = str(uuid.uuid4())
            temp_dir = os.path.join("data", "uploads")
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, f"{file_id}_{filename}")
            
            with open(temp_file_path, "wb") as f:
                f.write(content)
            
            # Add to results as processing
            results.append(DocumentResult(
                filename=filename,
                status="pending"
            ))
            
            # Process in background
            background_tasks.add_task(
                process_document_background,
                temp_file_path,
                filename,
                results
            )
            
            temp_files.append(temp_file_path)
        
        return UploadResponse(
            status="processing",
            files_processed=len(files),
            results=results
        )
        
    except Exception as e:
        logger.exception(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=UploadResponse)
async def get_document_status(task_id: str):
    """
    Check the status of document processing.
    
    Args:
        task_id: Task identifier for the document processing job
        
    Returns:
        Current status of document processing
    """
    # This is a placeholder for a more sophisticated status tracking system
    # In a production implementation, this would query a task queue or database
    
    return UploadResponse(
        status="unknown",
        files_processed=0,
        results=[]
    )