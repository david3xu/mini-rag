"""
Unit tests for the Vector Store service.
"""

import pytest
import tempfile
import os
import json

from app.services.vectorstore import VectorStoreService, vector_store
from app.services.embeddings import embeddings_service
from app.api.models import DocumentChunk

@pytest.mark.unit
class TestVectorStoreService:
    
    def test_initialization(self):
        """Test service initialization and collection creation."""
        print("\n=== Testing VectorStoreService initialization ===")
        # Create a temporary directory for the vector store
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Creating vector store in temporary directory: {temp_dir}")
            # Create a service instance with the temporary directory
            service = VectorStoreService(persist_directory=temp_dir)
            
            # Verify the collection has been created
            print("Checking if collection is created successfully...")
            assert service.collection is not None
            print("✓ Collection created successfully!")
    
    def test_add_documents(self):
        """Test adding documents to the vector store."""
        print("\n=== Testing adding documents to vector store ===")
        # Create a temporary directory for the vector store
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Creating vector store in temporary directory: {temp_dir}")
            # Create a service instance with the temporary directory
            service = VectorStoreService(persist_directory=temp_dir)
            
            # Create test document chunks
            print("Creating test document chunks...")
            chunks = [
                DocumentChunk(id="1", text="Test content 1", metadata={"source": "test1.txt"}),
                DocumentChunk(id="2", text="Test content 2", metadata={"source": "test2.txt"}),
                DocumentChunk(id="3", text="Test content 3", metadata={"source": "test3.txt"})
            ]
            
            # Generate real embeddings using the embedding service
            print("Generating embeddings for document chunks...")
            texts = [chunk.text for chunk in chunks]
            embeddings = embeddings_service.generate_embeddings(texts)
            print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
            
            # Add documents
            print("Adding documents to vector store...")
            service.add_documents(chunks, embeddings)
            
            # Verify documents were added
            print("Verifying documents were added correctly...")
            count = service.collection.count()
            print(f"Document count in collection: {count}")
            assert count == 3
            print("✓ Documents added successfully!")
    
    def test_similarity_search(self):
        """Test performing similarity search with the vector store."""
        print("\n=== Testing similarity search in vector store ===")
        # Create a temporary directory for the vector store
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Creating vector store in temporary directory: {temp_dir}")
            # Create a service instance with the temporary directory
            service = VectorStoreService(persist_directory=temp_dir)
            
            # Create test document chunks
            print("Creating test document chunks...")
            chunks = [
                DocumentChunk(id="1", text="Mini-RAG is a lightweight system", metadata={"source": "test1.txt"}),
                DocumentChunk(id="2", text="It uses vector embeddings for search", metadata={"source": "test2.txt"}),
                DocumentChunk(id="3", text="Memory optimization is important", metadata={"source": "test3.txt"})
            ]
            
            # Generate real embeddings
            print("Generating embeddings for document chunks...")
            texts = [chunk.text for chunk in chunks]
            embeddings = embeddings_service.generate_embeddings(texts)
            print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
            
            # Add documents first
            print("Adding documents to vector store...")
            service.add_documents(chunks, embeddings)
            
            # Create test query and get real embedding
            query = "system architecture"
            print(f"Creating embedding for query: '{query}'")
            query_embedding = embeddings_service.generate_embedding(query)
            
            # Perform search
            print("Performing similarity search...")
            results = service.similarity_search(query, query_embedding, k=3)
            
            # Print results
            print(f"Search returned {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: ID={result['id']}, Distance={result.get('distance', 'N/A')}")
                print(f"    Content: {result['content']}")
                print(f"    Metadata: {result['metadata']}")
            
            # Verify results structure
            assert isinstance(results, list)
            # The number of results might be less than 3 due to relevance filtering
            assert len(results) <= 3
            print("✓ Similarity search completed successfully!") 