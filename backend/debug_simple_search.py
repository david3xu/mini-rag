"""
Debug script that uses quick_search without signal-based timeout mechanism.
"""
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
sys.path.append('.')

from app.services.vectorstore import vector_store
from app.services.embeddings import embeddings_service

def test_simple_search():
    """Execute quick_search which doesn't use signal-based timeout."""
    print("=== Simple Search Debug Test ===")
    
    try:
        # Basic query
        query = "test query"
        print(f"Generating embedding for: '{query}'")
        start_time = time.time()
        query_embedding = embeddings_service.generate_embedding(query)
        print(f"Embedding generated in {time.time() - start_time:.2f}s")
        
        # Document count
        print(f"Vector store document count: {vector_store.collection.count()}")
        
        # Execute quick search (simpler implementation without signal timeout)
        print("\nExecuting quick_search...")
        start_time = time.time()
        
        # Use quick_search which has a simpler implementation
        results = vector_store.quick_search(query_embedding, k=1)
        
        elapsed = time.time() - start_time
        print(f"Search completed in {elapsed:.2f}s, returned {len(results)} results")
        
        # Print results if any
        if results:
            for i, doc in enumerate(results):
                print(f"Result {i+1}: {doc['content'][:100]}...")
        
        return True
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    test_simple_search() 