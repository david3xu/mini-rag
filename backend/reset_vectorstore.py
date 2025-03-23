# backend/reset_vectorstore.py
import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_vector_database():
    """Reset the vector database to resolve schema compatibility issues."""
    # Vector store location
    vector_db_path = "./vector_db/chroma_db"
    
    logger.info(f"Preparing to reset vector database at: {os.path.abspath(vector_db_path)}")
    
    # Create backup directory
    backup_dir = f"{vector_db_path}_backup_{int(time.time())}"
    
    try:
        # Backup existing database if it exists
        if os.path.exists(vector_db_path):
            logger.info(f"Creating backup at: {backup_dir}")
            shutil.copytree(vector_db_path, backup_dir)
            logger.info("Backup created successfully")
            
            # Remove existing database
            logger.info("Removing incompatible database...")
            shutil.rmtree(vector_db_path)
            logger.info("Existing database removed")
        
        # Create fresh directory
        os.makedirs(vector_db_path, exist_ok=True)
        logger.info("Created fresh vector database directory")
        
        logger.info("Vector database reset complete")
        logger.info("You can now process documents into the new database")
        return True
    except Exception as e:
        logger.error(f"Error resetting vector database: {str(e)}")
        return False

if __name__ == "__main__":
    import time
    reset_vector_database()