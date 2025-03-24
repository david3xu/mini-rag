"""
Minimal script that directly calls ChromaDB to test if search works without our custom timeout.
"""
import sys
import os
import time
import chromadb
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import embeddings service to generate a test embedding
sys.path.append('.')
from app.services.embeddings import embeddings_service
from config import settings

def test_direct_chromadb_search():
    """Execute a search directly using ChromaDB client without our custom timeout mechanisms."""
    print("=== Direct ChromaDB Search Test ===")
    
    # Vector store location
    vector_db_path = settings.VECTOR_DB_PATH
    collection_name = settings.VECTOR_DB_COLLECTION
    
    try:
        # Generate test embedding
        print("Generating test embedding...")
        query = "test query"
        query_embedding = embeddings_service.generate_embedding(query)
        
        # Create direct ChromaDB client
        print(f"Creating direct ChromaDB client at {vector_db_path}...")
        client = chromadb.PersistentClient(path=vector_db_path)
        
        # Get collection
        print(f"Getting collection '{collection_name}'...")
        collection = client.get_collection(name=collection_name)
        
        # Get document count
        count = collection.count()
        print(f"Collection contains {count} documents")
        
        # Execute direct search
        print("Executing direct search...")
        start_time = time.time()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=['metadatas', 'documents', 'distances']
        )
        
        elapsed = time.time() - start_time
        print(f"Search completed in {elapsed:.2f}s")
        
        # Print results
        if results and 'ids' in results and results['ids'] and len(results['ids'][0]) > 0:
            print(f"Found {len(results['ids'][0])} results:")
            for i, doc_id in enumerate(results['ids'][0]):
                doc_content = results['documents'][0][i][:100] + "..." if len(results['documents'][0][i]) > 100 else results['documents'][0][i]
                print(f"  Result {i+1}: [ID: {doc_id}] {doc_content}")
        else:
            print("No results found")
        
        return True
    except Exception as e:
        print(f"Error in direct search: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    test_direct_chromadb_search() 