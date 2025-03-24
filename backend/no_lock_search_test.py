#!/usr/bin/env python
"""
Search test that accesses ChromaDB directly with no locks.
"""
import sys
import time
import logging
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append('.')

# Import services and config, but use ChromaDB directly
from app.services.embeddings import embeddings_service
from config import settings

def run_direct_search():
    """Run search directly without vectorstore service."""
    print("\n=== Direct ChromaDB Search (No Locks) ===\n")
    
    # Basic query
    query = "test query"
    print(f"Generating embedding for: '{query}'")
    start_time = time.time()
    query_embedding = embeddings_service.generate_embedding(query)
    print(f"Embedding generated in {time.time() - start_time:.2f}s")
    
    # Vector store location
    vector_db_path = settings.VECTOR_DB_PATH
    collection_name = settings.VECTOR_DB_COLLECTION
    
    # Create new ChromaDB client
    print(f"Creating new ChromaDB client at: {vector_db_path}")
    client = chromadb.PersistentClient(path=vector_db_path)
    
    # Get collection
    print(f"Getting collection: {collection_name}")
    collection = client.get_collection(name=collection_name)
    count = collection.count()
    print(f"Collection contains {count} documents")
    
    # Run search
    print("\nRunning direct search...")
    start_time = time.time()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=['metadatas', 'documents', 'distances']
    )
    
    elapsed = time.time() - start_time
    print(f"Search completed in {elapsed:.2f}s")
    
    # Display results
    if results and 'ids' in results and results['ids'] and len(results['ids'][0]) > 0:
        print(f"Found {len(results['ids'][0])} results:")
        for i, doc_id in enumerate(results['ids'][0]):
            doc_content = results['documents'][0][i][:100] + "..." if len(results['documents'][0][i]) > 100 else results['documents'][0][i]
            print(f"  Result {i+1}: [ID: {doc_id}] {doc_content}")
    else:
        print("No results found")
    
    print("\nSearch test complete")
    
    # Close client
    del collection
    del client

if __name__ == "__main__":
    run_direct_search() 