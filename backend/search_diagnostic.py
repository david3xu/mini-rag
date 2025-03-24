# backend/search_diagnostic.py
#!/usr/bin/env python
"""
Minimal diagnostic script for vector search operations.
"""
import sys
import os
import time
import logging
import psutil

logging.basicConfig(level=logging.INFO)
sys.path.append('.')

from app.services.vectorstore import vector_store
from app.services.embeddings import embeddings_service
from config import settings

def test_minimal_search():
    """Execute minimal search operation with diagnostics."""
    print("=== Vector Search Diagnostic ===")
    
    try:
        # Basic query
        query = "test query"
        print(f"Generating embedding for: '{query}'")
        start_time = time.time()
        query_embedding = embeddings_service.generate_embedding(query)
        print(f"Embedding generated in {time.time() - start_time:.2f}s")
        
        # Document count
        print(f"Vector store document count: {vector_store.collection.count()}")
        
        # Memory diagnostics
        mem = psutil.virtual_memory()
        print(f"Memory before search: {mem.percent}% used, {mem.available/(1024**3):.2f}GB available")
        
        # Execute search with timeout from config
        timeout_ms = settings.QUERY_TIMEOUT_MS
        print(f"\nExecuting minimal search (timeout: {timeout_ms/1000}s)...")
        start_time = time.time()
        
        # Limit to 1 result
        results = vector_store.search(query_embedding, k=1, timeout_ms=timeout_ms)
        
        elapsed = time.time() - start_time
        print(f"Search completed in {elapsed:.2f}s, returned {len(results)} results")
        
        return True
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    test_minimal_search()