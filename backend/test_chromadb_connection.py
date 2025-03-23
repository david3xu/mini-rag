# backend/test_chromadb_connection.py
import sys
sys.path.append('.')
import os
import logging

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def verify_chroma_environment():
    """Verify ChromaDB environment configuration."""
    print("=== ChromaDB Environment Verification ===")
    
    # Check vector DB path
    vector_db_path = "./vector_db/chroma_db"
    print(f"Checking vector DB path: {os.path.abspath(vector_db_path)}")
    
    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path, exist_ok=True)
        print(f"Created vector DB directory: {os.path.abspath(vector_db_path)}")
    
    # Check permissions
    try:
        test_file_path = os.path.join(vector_db_path, "test_permissions.txt")
        with open(test_file_path, "w") as f:
            f.write("test")
        os.remove(test_file_path)
        print("✓ Directory has proper write permissions")
    except Exception as e:
        print(f"✗ Permission error: {str(e)}")
    
    # Check ChromaDB version
    try:
        import chromadb
        print(f"ChromaDB version: {chromadb.__version__}")
        
        # Verify expected API
        client_class = chromadb.PersistentClient
        client = client_class(path=vector_db_path)
        print("✓ ChromaDB client initialized successfully")
        
        # Create test collection
        collection = client.get_or_create_collection(name="test_connection")
        print(f"✓ Test collection created: {collection.name}")
        
        # Clean up
        client.delete_collection("test_connection")
        print("✓ Test collection deleted")
    except Exception as e:
        print(f"✗ ChromaDB initialization error: {str(e)}")

if __name__ == "__main__":
    verify_chroma_environment()