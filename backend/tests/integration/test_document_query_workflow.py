"""
Integration test for the document processing and query workflow.

This module provides a comprehensive test for the end-to-end document
processing and query workflow, with better debugging capabilities.
"""

import pytest
import tempfile
import os
import time
import uuid
import sys
from fastapi.testclient import TestClient

from main import app
from app.api.chat import QueryRequest
from app.services.vectorstore import vector_store
from app.services.embeddings import embeddings_service
from config import settings

@pytest.mark.integration
class TestDocumentQueryWorkflow:
    """Test the complete document processing and query workflow."""
    
    def setup_method(self):
        """Set up temporary test environment."""
        # Generate a unique test ID for this test run
        self.test_id = str(uuid.uuid4())[:8]
        
        # Clean vector store of previous test data
        self._clean_vector_store()
        
        # Create a temporary file with test content and unique identifiers
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        self.temp_file.write(f"""
        Mini-RAG Test Document {self.test_id} - A lightweight Retrieval-Augmented Generation system designed for 
        resource-efficient operation. It provides document processing, embedding generation, 
        semantic search and LLM-enhanced responses with minimal computational requirements.
        
        The system implements memory optimization techniques including:
        - Lazy loading of ML models (unique test identifier: {self.test_id})
        - Batch processing with configurable sizes
        - Explicit memory cleanup mechanisms
        - Document chunking with controlled overlap
        
        Key features include:
        1. Efficient document processing with memory-optimized chunking
        2. Semantic search using vector embeddings
        3. Local LLM integration with phi-2.gguf
        4. Resource monitoring and optimization
        """.encode('utf-8'))
        self.temp_file.flush()
        self.document_path = self.temp_file.name
        print(f"Created test document with ID {self.test_id} at {self.document_path}")
    
    def _clean_vector_store(self):
        """Clean vector store of previous test data."""
        try:
            print("Cleaning vector store of previous test data...")
            # Clean up any previous test documents
            # Metadata filter not used to avoid complexity
            collection = vector_store.collection
            collection.delete()
            print("Vector store cleaned")
        except Exception as e:
            print(f"Error cleaning vector store: {e}")
    
    def teardown_method(self):
        """Clean up after test."""
        # Remove the temporary file
        if os.path.exists(self.document_path):
            os.remove(self.document_path)
            print(f"Removed test document {self.document_path}")
        
        # Attempt to clean up vectors for this document
        try:
            vector_store.delete_by_metadata({"source": os.path.basename(self.document_path)})
            print(f"Cleaned up vectors for {os.path.basename(self.document_path)}")
        except:
            # Don't fail tests if cleanup fails
            pass
    
    def test_document_processing_and_query(self):
        """Test end-to-end document processing and querying."""
        print("\n=== Testing document processing and query workflow ===")
        
        # Create test client
        client = TestClient(app)
        
        # Step 1: Process a document
        print(f"Processing document: {self.document_path}")
        print(f"Document content sample: {self.temp_file.read(200).decode()}...")
        self.temp_file.seek(0)  # Reset file position
        
        with open(self.document_path, "rb") as file:
            response = client.post(
                "/api/documents/upload",
                files={"files": (os.path.basename(self.document_path), file, "text/plain")}
            )
        
        print(f"Document processing response status: {response.status_code}")
        assert response.status_code == 200
        process_result = response.json()
        print(f"Document upload response: {process_result}")
        
        # Verify document was properly processed into chunks
        from app.services.document_processor import document_processor
        chunks = document_processor.process_file(self.document_path)
        print(f"Document was processed into {len(chunks)} chunks")
        
        # Print chunk content to verify memory optimization is included
        for i, chunk in enumerate(chunks[:2]):  # Just show first 2 chunks
            print(f"Chunk {i+1}: {chunk['text'][:100]}...")
            print(f"  Metadata: {chunk['metadata']}")
        
        # Get document count in vector store 
        count = vector_store.collection.count()
        print(f"Vector store contains {count} documents")
        
        # Wait longer for background processing
        time.sleep(5)
        
        # Step 2: Try direct semantic match first
        test_query = "memory optimization techniques"
        print(f"\nTrying direct semantic search for: '{test_query}'")
        query_embedding = embeddings_service.generate_embedding(test_query)
        
        # Search with higher limit and longer timeout
        direct_results = vector_store.search(
            query_embedding, 
            k=5,         # Retrieve more results
            timeout_ms=15000  # Longer timeout
        )
        print(f"Direct search returned {len(direct_results)} results")
        
        # Print results to see if our content is there
        for i, result in enumerate(direct_results):
            print(f"Result {i+1}: {result['content'][:100]}...")
            print(f"  Source: {result.get('metadata', {}).get('source', 'N/A')}")
        
        # Step 3: Query through the API
        query_text = f"What are the memory optimization techniques in Mini-RAG with the identifier {self.test_id}?"
        print(f"\nQuerying with: '{query_text}'")
        query_request = QueryRequest(query=query_text)
        response = client.post("/api/chat", json=query_request.dict())
        
        print(f"Query response status: {response.status_code}")
        assert response.status_code == 200
        query_result = response.json()
        
        # Verify response structure
        assert "answer" in query_result
        assert "sources" in query_result
        print(f"Query answer: {query_result['answer']}")
        print(f"Sources: {len(query_result['sources'])} found")
        
        # Print all sources for debugging
        for i, source in enumerate(query_result['sources']):
            print(f"Source {i+1}: {source['content'][:100]}...")
            print(f"  Metadata: {source['metadata']}")
        
        # Verify content
        assert len(query_result['sources']) > 0
        assert "memory" in query_result['answer'].lower()
        assert "optimization" in query_result['answer'].lower()
        assert self.test_id in query_result['answer'], f"Expected to find unique test ID {self.test_id} in answer"
        
        # At least one source should be from our document
        document_basename = os.path.basename(self.document_path)
        print(f"\nLooking for document with basename: {document_basename}")
        
        source_matched = False
        for source in query_result['sources']:
            source_path = source.get('metadata', {}).get('source', '')
            print(f"Checking source: {source_path}")
            
            # Check if source contains either the full path or the basename
            if (document_basename in source_path or 
                self.document_path in source_path or
                os.path.basename(source_path) == document_basename):
                source_matched = True
                print("✓ Match found!")
                break
        
        assert source_matched, f"Query didn't return sources from our document (looking for {document_basename})"
        print("✓ Document query workflow test passed") 