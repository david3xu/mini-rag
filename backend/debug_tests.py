"""
Debug script to examine vector database content.

This script helps diagnose issues with document retrieval by examining
the contents of the vector database directly.
"""

import sys
import os
import json
from pathlib import Path

# Ensure the backend directory is in the path
backend_dir = Path(__file__).parent.absolute()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Import our modules
from app.services.vectorstore import vector_store
from app.services.embeddings import embeddings_service
from config import settings

def examine_vectordb_contents():
    """Examine all contents of the vector database."""
    print(f"\n=== Vector Database Contents ===")
    print(f"Database path: {vector_store.persist_directory}")
    print(f"Collection name: {vector_store.collection_name}")
    
    # Get collection info
    try:
        info = vector_store.get_collection_info()
        print(f"Collection info: {json.dumps(info, indent=2)}")
    except Exception as e:
        print(f"Error getting collection info: {e}")
    
    # Try to get all documents (may be limited)
    try:
        print("\nAttempting to retrieve all documents...")
        collection = vector_store.collection
        all_docs = collection.get(include=['metadatas', 'documents', 'embeddings'])
        
        if all_docs and 'ids' in all_docs and len(all_docs['ids']) > 0:
            print(f"Found {len(all_docs['ids'])} documents")
            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs.get('metadatas', [{}])[i] if i < len(all_docs.get('metadatas', [])) else {}
                content = all_docs.get('documents', [""])[i] if i < len(all_docs.get('documents', [])) else ""
                print(f"\nDocument {i+1}:")
                print(f"  ID: {doc_id}")
                print(f"  Metadata: {metadata}")
                print(f"  Content: {content[:100]}...")
        else:
            print("No documents found in collection")
    except Exception as e:
        print(f"Error examining vector store: {e}")

def test_search_query(query="What are the memory optimization techniques in Mini-RAG?"):
    """Test search with a specific query."""
    print(f"\n=== Testing Search Query ===")
    print(f"Query: {query}")
    
    # Generate embedding
    try:
        print("Generating query embedding...")
        query_embedding = embeddings_service.generate_embedding(query)
        print(f"Embedding generated (length: {len(query_embedding)})")
        
        # Search with embedding
        print("Searching vector store...")
        results = vector_store.search(query_embedding)
        
        if results:
            print(f"Found {len(results)} results")
            for i, doc in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  ID: {doc.get('id')}")
                print(f"  Metadata: {doc.get('metadata', {})}")
                print(f"  Distance: {doc.get('distance')}")
                print(f"  Content: {doc.get('content', '')[:100]}...")
        else:
            print("No results found")
    except Exception as e:
        print(f"Error during search test: {e}")

if __name__ == "__main__":
    print("=== Vector Store Debug ===")
    examine_vectordb_contents()
    test_search_query() 