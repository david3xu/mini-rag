# backend/debug_chromadb.py
import sys
import os
sys.path.append('.')

import chromadb
import logging
import json

# Configure verbose logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_chromadb_directly():
    """Test ChromaDB operations directly without application code."""
    print("=== ChromaDB Direct Testing ===")
    
    # Use temp directory for testing
    persist_path = "./temp_vector_test"
    os.makedirs(persist_path, exist_ok=True)
    
    try:
        # Initialize client directly
        print("Initializing ChromaDB client...")
        client = chromadb.PersistentClient(path=persist_path)
        
        # Create collection
        print("Creating test collection...")
        collection = client.get_or_create_collection(name="test_collection")
        
        # Add simple documents
        print("Adding documents...")
        collection.add(
            documents=["Test document 1", "Test document 2"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # Simple test embeddings
            ids=["test1", "test2"],
            metadatas=[{"source": "test1.txt"}, {"source": "test2.txt"}]
        )
        
        # Verify documents were added
        print("Checking document count...")
        count = collection.count()
        print(f"Collection contains {count} documents")
        
        # Simple search
        print("Testing search...")
        results = collection.query(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=2
        )
        
        print("Search results:")
        print(json.dumps(results, indent=2))
        
        print("✓ All ChromaDB operations completed successfully")
        return True
    except Exception as e:
        print(f"✗ ChromaDB error: {str(e)}")
        return False
    finally:
        # Cleanup
        if os.path.exists(persist_path):
            import shutil
            shutil.rmtree(persist_path)

if __name__ == "__main__":
    test_chromadb_directly()