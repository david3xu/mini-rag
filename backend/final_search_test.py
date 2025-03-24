#!/usr/bin/env python
"""
Final test script combining our fixes to verify they work.
"""
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append('.')

# Import services
from app.services.vectorstore import vector_store
from app.services.embeddings import embeddings_service

def run_search():
    """Run search with our fix and measure performance."""
    print("\n=== Final Search Test ===\n")
    
    # Basic query
    query = "test query"
    print(f"Generating embedding for: '{query}'")
    start_time = time.time()
    query_embedding = embeddings_service.generate_embedding(query)
    print(f"Embedding generated in {time.time() - start_time:.2f}s")
    
    # Get document count
    count = vector_store.collection.count()
    print(f"Vector store document count: {count}")
    
    # Run search with our fix
    print("\nRunning search...")
    start_time = time.time()
    results = vector_store.search(query_embedding, k=3)
    elapsed = time.time() - start_time
    print(f"Search completed in {elapsed:.2f}s")
    
    # Display results
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  Result {i+1}: [ID: {result['id']}] {result['content'][:100]}...")
    
    print("\nSearch test complete")

if __name__ == "__main__":
    run_search() 