# backend/process_uploads.py
import os
import sys
sys.path.append('.')
import logging
import time
import re

from app.services.document_processor import document_processor
from app.services.embeddings import embeddings_service
from app.services.vectorstore import vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_directory_files():
    """Process original documents in the uploads directory."""
    uploads_dir = "data/uploads"
    start_time = time.time()
    
    # Get all files
    all_files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
    
    # Filter for original files (no UUID prefix)
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_')
    original_files = [f for f in all_files if not uuid_pattern.match(f)]
    
    logger.info(f"Found {len(original_files)} original files to process")
    logger.info(f"Files: {original_files}")
    
    # Process each original file
    success_count = 0
    total_chunks = 0
    
    for filename in original_files:
        file_path = os.path.join(uploads_dir, filename)
        logger.info(f"Processing file: {filename}")
        
        try:
            # Process document to chunks
            chunks = document_processor.process_file(file_path)
            logger.info(f"Generated {len(chunks)} chunks from {filename}")
            
            # Extract component data
            texts = [chunk["text"] for chunk in chunks]
            ids = [chunk["id"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            embeddings = embeddings_service.generate_embeddings(texts)
            
            # Add to vector store
            logger.info(f"Adding documents to vector store")
            vector_store.add_documents(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            # Verify addition
            count = vector_store.collection.count()
            logger.info(f"Vector store now contains {count} documents")
            
            success_count += 1
            total_chunks += len(chunks)
            logger.info(f"✓ Successfully processed {filename}")
            
        except Exception as e:
            logger.error(f"✗ Error processing {filename}: {str(e)}")
            logger.exception(e)  # Log full stack trace
    
    # Process summary
    elapsed_time = time.time() - start_time
    logger.info(f"Processing complete in {elapsed_time:.2f} seconds")
    logger.info(f"Successfully processed {success_count}/{len(original_files)} files")
    logger.info(f"Total chunks added: {total_chunks}")
    
    return success_count

if __name__ == "__main__":
    process_directory_files()