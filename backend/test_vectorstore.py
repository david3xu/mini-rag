#!/usr/bin/env python
"""
Test script to verify vector store operations.
"""

import os
import sys
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure we can import from the right location
sys.path.append('.')

from app.services.vectorstore import VectorStoreService, vector_store
from app.services.embeddings import embeddings_service
from app.api.models import DocumentChunk

def test_vectorstore():
    """Test basic vector store operations."""
    print("\n=== Testing Vector Store Service ===")
    
    # Test 1: Create temporary store
    print("\nTest 1: Creating temporary vector store...")
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Initialize service
        service = VectorStoreService(persist_directory=temp_dir)
        print(f"Service created with persist_directory: {service.persist_directory}")
        
        # Access collection
        collection = service.collection
        
        if collection is not None:
            print("✓ Collection initialized successfully")
            collection_name = collection.name
            print(f"  Collection name: {collection_name}")
        else:
            print("✗ Failed to initialize collection")
            return False
    
    # Test 2: Use singleton instance
    print("\nTest 2: Using singleton vector store instance...")
    try:
        count = vector_store.collection.count()
        print(f"✓ Accessed singleton vector store. Document count: {count}")
    except Exception as e:
        print(f"✗ Error accessing singleton: {str(e)}")
        return False
    
    # Test 3: Add and search documents
    print("\nTest 3: Adding and searching documents...")
    
    # Create test documents
    chunks = [
        DocumentChunk(id="test1", text="This is a test document about vector databases.", metadata={"source": "test.txt"}),
        DocumentChunk(id="test2", text="Embeddings are vector representations of text.", metadata={"source": "test.txt"}),
        DocumentChunk(id="test3", text="Mini-RAG uses a vector store for semantic search.", metadata={"source": "test.txt"})
    ]
    
    # First clear any existing documents with these IDs
    try:
        print("Clearing any existing test documents...")
        ids_to_clear = [chunk.id for chunk in chunks]
        vector_store.collection.delete(ids=ids_to_clear)
        print(f"✓ Cleared {len(ids_to_clear)} documents if they existed")
    except Exception as e:
        print(f"Note: Could not clear existing documents: {str(e)}")
    
    try:
        # Generate embeddings
        print("Generating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = embeddings_service.generate_embeddings(texts)
        print(f"✓ Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
        
        # Add to vector store
        print("Adding documents to vector store...")
        vector_store.add_documents(chunks, embeddings)
        
        # Verify count
        count_after = vector_store.collection.count()
        print(f"Document count after adding: {count_after}")
        
        # Perform search
        print("\nPerforming similarity search...")
        query = "vector embeddings"
        query_embedding = embeddings_service.generate_embedding(query)
        
        results = vector_store.similarity_search(query, query_embedding, k=3)
        
        # Print results
        print(f"Search returned {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  Result {i+1}: ID={result['id']}, Text='{result['content'][:50]}...'")
            
        print("✓ Search completed successfully")
        
        # Clean up
        print("\nCleaning up test documents...")
        vector_store.collection.delete(ids=[chunk.id for chunk in chunks])
        count_after_cleanup = vector_store.collection.count()
        print(f"Document count after cleanup: {count_after_cleanup}")
        
        return True
    except Exception as e:
        print(f"✗ Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Vector Store Test ===")
    print(f"Current directory: {os.getcwd()}")
    
    # Run tests
    success = test_vectorstore()
    
    # Print result
    if success:
        print("\n=== All Tests Passed! ===")
        sys.exit(0)
    else:
        print("\n=== Tests Failed! ===")
        sys.exit(1) 