# backend/minimal_chromadb_test.py
"""
Minimal test for ChromaDB functionality.
"""
import os
import chromadb
import time
import numpy as np

# Use a temporary test database
DB_PATH = "./test_vector_db"
os.makedirs(DB_PATH, exist_ok=True)

print(f"Testing ChromaDB version: {chromadb.__version__}")

try:
    # Create a client
    print("Creating ChromaDB client...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Create a collection
    print("Creating collection...")
    collection = client.get_or_create_collection(name="test_collection")
    
    # Add documents
    print("Adding documents...")
    collection.add(
        documents=["This is a test document", "Another test document"],
        embeddings=[[0.1 for _ in range(10)], [0.2 for _ in range(10)]],
        ids=["id1", "id2"]
    )
    
    # Check count
    print(f"Collection contains {collection.count()} documents")
    
    # Perform a search
    print("Running search...")
    start_time = time.time()
    results = collection.query(
        query_embeddings=[[0.1 for _ in range(10)]],
        n_results=1
    )
    elapsed = time.time() - start_time
    
    print(f"Search completed in {elapsed:.2f}s")
    print(f"Found {len(results['ids'][0])} results")
    print(f"Results: {results}")
    
    print("Test completed successfully")
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
finally:
    # Clean up
    import shutil
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"Removed test database: {DB_PATH}") 