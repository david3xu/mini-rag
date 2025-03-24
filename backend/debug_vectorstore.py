#!/usr/bin/env python
"""
Simple script to debug the vector store operations.
This script demonstrates how the vector store works with detailed output.
"""

import sys
import os
import tempfile

# Make sure we can import from the correct paths
sys.path.append('.')

# Import the services and models
from app.services.vectorstore import VectorStoreService
from app.services.embeddings import embeddings_service
from app.api.models import DocumentChunk
from config import settings

# Adjust paths to be absolute
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models/embeddings/all-MiniLM-L6-v2")

def test_initialization():
    """Test vector store initialization."""
    print("\n=== Testing VectorStoreService initialization ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating vector store in temporary directory: {temp_dir}")
        
        # Create service instance
        service = VectorStoreService(persist_directory=temp_dir)
        
        # Access the collection to initialize it
        print("Initializing the collection...")
        collection = service.collection
        
        if collection is not None:
            print("✓ Collection created successfully")
        else:
            print("✗ Failed to create collection")

def test_add_documents():
    """Test adding documents to the vector store."""
    print("\n=== Testing adding documents to vector store ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating vector store in temporary directory: {temp_dir}")
        
        # Create service instance
        service = VectorStoreService(persist_directory=temp_dir)
        
        # Create test document chunks
        print("\nCreating test document chunks...")
        chunks = [
            DocumentChunk(id="1", text="Test content 1", metadata={"source": "test1.txt"}),
            DocumentChunk(id="2", text="Test content 2", metadata={"source": "test2.txt"}),
            DocumentChunk(id="3", text="Test content 3", metadata={"source": "test3.txt"})
        ]
        
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: ID={chunk.id}, Text='{chunk.text}', Metadata={chunk.metadata}")
        
        # Temporarily override embedding model path
        original_embedding_model = settings.EMBEDDING_MODEL
        settings.EMBEDDING_MODEL = EMBEDDING_MODEL_PATH
        print(f"\nTemporarily using embedding model at: {settings.EMBEDDING_MODEL}")
        
        try:
            # Generate embeddings
            print("\nGenerating embeddings for document chunks...")
            texts = [chunk.text for chunk in chunks]
            embeddings = embeddings_service.generate_embeddings(texts)
            print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
            
            # Add documents
            print("\nAdding documents to vector store...")
            service.add_documents(chunks, embeddings)
            
            # Verify documents were added
            print("\nVerifying documents were added correctly...")
            try:
                count = service.collection.count()
                print(f"Document count in collection: {count}")
                if count == 3:
                    print("✓ Documents added successfully!")
                else:
                    print(f"✗ Expected 3 documents, but found {count}")
            except Exception as e:
                print(f"✗ Error counting documents: {str(e)}")
        finally:
            # Restore original embedding model path
            settings.EMBEDDING_MODEL = original_embedding_model

def test_search():
    """Test performing similarity search."""
    print("\n=== Testing similarity search in vector store ===")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating vector store in temporary directory: {temp_dir}")
        
        # Create service instance
        service = VectorStoreService(persist_directory=temp_dir)
        
        # Create test document chunks
        print("\nCreating test document chunks...")
        chunks = [
            DocumentChunk(id="1", text="Mini-RAG is a lightweight system", metadata={"source": "test1.txt"}),
            DocumentChunk(id="2", text="It uses vector embeddings for search", metadata={"source": "test2.txt"}),
            DocumentChunk(id="3", text="Memory optimization is important", metadata={"source": "test3.txt"})
        ]
        
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: ID={chunk.id}, Text='{chunk.text}', Metadata={chunk.metadata}")
        
        # Temporarily override embedding model path
        original_embedding_model = settings.EMBEDDING_MODEL
        settings.EMBEDDING_MODEL = EMBEDDING_MODEL_PATH
        print(f"\nTemporarily using embedding model at: {settings.EMBEDDING_MODEL}")
        
        try:
            # Generate embeddings
            print("\nGenerating embeddings for document chunks...")
            texts = [chunk.text for chunk in chunks]
            embeddings = embeddings_service.generate_embeddings(texts)
            print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
            
            # Add documents
            print("\nAdding documents to vector store...")
            service.add_documents(chunks, embeddings)
            
            # Perform search
            query = "system architecture"
            print(f"\nCreating embedding for query: '{query}'")
            query_embedding = embeddings_service.generate_embedding(query)
            
            print("\nPerforming similarity search...")
            results = service.search(query_embedding, k=3)
            
            # Print results
            print(f"\nSearch returned {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: ID={result['id']}, Distance={result.get('distance', 'N/A')}")
                print(f"    Content: '{result['content']}'")
                print(f"    Metadata: {result['metadata']}")
            
            if results:
                print("\n✓ Similarity search completed successfully!")
            else:
                print("\n✗ Similarity search returned no results")
        finally:
            # Restore original embedding model path
            settings.EMBEDDING_MODEL = original_embedding_model

if __name__ == "__main__":
    print("=== Running Mini-RAG Vector Store Debug Tests ===")
    print("These tests verify the vector store operations with detailed output")
    print(f"Using embedding model: {EMBEDDING_MODEL_PATH}")
    
    # Run all tests
    test_initialization()
    test_add_documents()
    test_search()
    
    print("\n=== All tests completed ===") 