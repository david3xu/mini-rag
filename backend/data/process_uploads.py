# Enhanced process_uploads.py with standardized path resolution

import os
import sys
import logging
import time
import re
import shutil
import gc
from pathlib import Path

# Configure logging with standardized format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_absolute_path(relative_path):
    """
    Convert relative paths to absolute paths with correct base directory reference.
    
    Args:
        relative_path: The path to resolve
        
    Returns:
        Absolute path with proper structure
    """
    # Get base directory (script location or parent if inside duplicate structure)
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    # Determine correct base directory
    if os.path.basename(current_dir) == 'backend':
        BASE_DIR = current_dir
    else:
        BASE_DIR = os.path.join(os.path.dirname(current_dir), 'backend')
    
    # Normalize path separators for cross-platform compatibility
    normalized_path = relative_path.replace('\\', '/')
    
    # Strip redundant prefixes
    if normalized_path.startswith("./backend/"):
        path = normalized_path[10:]
    elif normalized_path.startswith("backend/"):
        path = normalized_path[8:]
    else:
        path = normalized_path
    
    # Join with base directory and normalize
    return os.path.normpath(os.path.join(BASE_DIR, path))

def repair_directory_structure():
    """
    Identify and repair duplicate directory structures.
    
    Returns:
        Boolean indicating if repairs were made
    """
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    logger.info(f"Script execution directory: {current_dir}")
    
    # Check for duplicate backend directory
    duplicate_path = os.path.join(current_dir, "backend")
    if os.path.exists(duplicate_path) and os.path.basename(current_dir) == 'backend':
        logger.warning(f"Detected duplicate directory structure: {duplicate_path}")
        
        # Process duplicated data directory
        duplicate_data = os.path.join(duplicate_path, "data")
        if os.path.exists(duplicate_data):
            target_data = os.path.join(current_dir, "data")
            os.makedirs(target_data, exist_ok=True)
            
            # Copy content to correct location
            for item in os.listdir(duplicate_data):
                src = os.path.join(duplicate_data, item)
                dst = os.path.join(target_data, item)
                
                if not os.path.exists(dst):
                    if os.path.isdir(src):
                        logger.info(f"Migrating directory: {src} → {dst}")
                        shutil.copytree(src, dst)
                    else:
                        logger.info(f"Migrating file: {src} → {dst}")
                        shutil.copy2(src, dst)
            
            # Remove duplicate structure after migration
            logger.info(f"Removing duplicate structure: {duplicate_path}")
            shutil.rmtree(duplicate_path)
            return True
    
    return False

def resolve_uploads_directory():
    """
    Determine correct uploads directory with structure validation.
    
    Returns:
        Absolute path to the uploads directory
    """
    # First repair any duplicate structures
    repair_directory_structure()
    
    # Get the uploads directory using standardized path function
    uploads_path = get_absolute_path("data/uploads")
    logger.info(f"Resolved uploads directory: {uploads_path}")
    
    # Ensure directory exists
    os.makedirs(uploads_path, exist_ok=True)
    
    return uploads_path

def process_directory_files():
    """
    Process original documents in the uploads directory with memory optimization.
    
    Returns:
        Number of successfully processed files
    """
    # Resolve correct uploads directory
    uploads_dir = resolve_uploads_directory()
    start_time = time.time()
    
    # Add correct import paths
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import after directory structure is fixed and paths are set
    try:
        from app.services.document_processor import document_processor
        from app.services.embeddings import embeddings_service
        from app.services.vectorstore import vector_store
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Check Python path and directory structure")
        return 0
    
    # Get all files
    all_files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
    
    # Filter for original files (no UUID prefix and not .gitkeep)
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_')
    original_files = [f for f in all_files if not uuid_pattern.match(f) and f != '.gitkeep']
    
    logger.info(f"Found {len(original_files)} original files to process")
    if original_files:
        logger.info(f"Files to process: {', '.join(original_files)}")
    
    # Process each original file
    success_count = 0
    total_chunks = 0
    
    for filename in original_files:
        file_path = os.path.join(uploads_dir, filename)
        logger.info(f"Processing file: {filename}")
        
        try:
            # Process document into chunks
            chunks = document_processor.process_file(file_path)
            chunk_count = len(chunks)
            logger.info(f"Generated {chunk_count} chunks from {filename}")
            
            # Extract component data
            texts = [chunk["text"] for chunk in chunks]
            ids = [chunk["id"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Generate embeddings with memory optimization
            logger.info(f"Generating embeddings for {chunk_count} chunks")
            embeddings = embeddings_service.generate_embeddings(texts)
            
            # Memory optimization between operations
            gc.collect()
            
            # Add to vector store with individual processing for reliable error handling
            logger.info(f"Adding documents to vector store")
            vector_store.add_documents(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
                use_individual_processing=True
            )
            
            # Verify operation completion
            current_count = vector_store.collection.count()
            logger.info(f"Vector store now contains {current_count} documents")
            
            success_count += 1
            total_chunks += chunk_count
            logger.info(f"✓ Successfully processed {filename}")
            
            # Memory cleanup after each file
            del chunks, texts, embeddings, ids, metadatas
            gc.collect()
            
        except Exception as e:
            logger.error(f"✗ Error processing {filename}: {str(e)}")
            logger.exception(e)  # Log full stack trace
    
    # Process summary
    elapsed_time = time.time() - start_time
    logger.info(f"Processing operation complete in {elapsed_time:.2f} seconds")
    logger.info(f"Successfully processed {success_count}/{len(original_files)} files")
    logger.info(f"Total document chunks added: {total_chunks}")
    
    return success_count

if __name__ == "__main__":
    try:
        process_directory_files()
    except Exception as e:
        logger.critical(f"Unhandled exception in main process: {str(e)}")
        logger.exception(e)
        sys.exit(1)