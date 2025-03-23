# backend/check_vectorstore.py
import os
import sys
sys.path.append('.')

from app.services.vectorstore import vector_store
from app.services.embeddings import embeddings_service

def diagnose_vector_store():
    """Check vector store status and health."""
    print("=== Vector Store Diagnostic ===")
    
    try:
        # Check if collection exists
        collection = vector_store.collection
        print(f"✓ Collection initialized: {vector_store.collection_name}")
        
        # Check document count
        count = collection.count()
        print(f"✓ Document count: {count}")
        
        if count == 0:
            print("! Warning: No documents in vector store")
            print("  You need to add documents before search will work")
        
        # Test a basic query if documents exist
        if count > 0:
            query = "test query"
            print(f"\nTesting search with query: '{query}'")
            
            # Generate embedding
            embedding = embeddings_service.generate_embedding(query)
            print(f"✓ Generated query embedding (dim={len(embedding)})")
            
            # Perform search
            results = vector_store.search(embedding, k=2)
            print(f"✓ Search returned {len(results)} results")
            
            # Show results
            for i, doc in enumerate(results):
                print(f"  Result {i+1}: {doc['content'][:50]}...")
        
        return count > 0
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    diagnose_vector_store()