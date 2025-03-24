"""
Integration test for the document processing and query workflow.

This test verifies the end-to-end functionality of the document processing
and question answering features of the Mini-RAG system.
"""

import pytest
import tempfile
import os
from fastapi.testclient import TestClient

from main import app
from app.api.chat import QueryRequest
from app.services.vectorstore import vector_store
from config import settings

@pytest.mark.integration
class TestDocumentQueryWorkflow:
    """Test the complete document processing and query workflow."""
    
    def setup_method(self):
        """Set up temporary test environment."""
        # Create a temporary file with test content
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        self.temp_file.write(b"""
        Mini-RAG is a lightweight Retrieval-Augmented Generation system designed for 
        resource-efficient operation. It provides document processing, embedding generation, 
        semantic search and LLM-enhanced responses with minimal computational requirements.
        
        The system implements memory optimization techniques including:
        - Lazy loading of ML models
        - Batch processing with configurable sizes
        - Explicit memory cleanup mechanisms
        - Document chunking with controlled overlap
        
        Key features include:
        1. Efficient document processing with memory-optimized chunking
        2. Semantic search using vector embeddings
        3. Local LLM integration with phi-2.gguf
        4. Resource monitoring and optimization
        """)
        self.temp_file.flush()
        self.document_path = self.temp_file.name
    
    def teardown_method(self):
        """Clean up after test."""
        # Remove the temporary file
        if os.path.exists(self.document_path):
            os.remove(self.document_path)
        
        # Attempt to clean up vectors for this document
        try:
            vector_store.delete_by_metadata({"source": os.path.basename(self.document_path)})
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
        with open(self.document_path, "rb") as file:
            response = client.post(
                "/api/documents/upload",
                files={"files": (os.path.basename(self.document_path), file, "text/plain")}
            )
        
        print(f"Document processing response status: {response.status_code}")
        assert response.status_code == 200
        process_result = response.json()
        print(f"Document upload response: {process_result}")
        
        # Step 2: Query the document
        query_text = "What are the memory optimization techniques in Mini-RAG?"
        print(f"Querying with: '{query_text}'")
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
        
        # Verify content
        assert len(query_result['sources']) > 0
        assert "memory" in query_result['answer'].lower()
        assert "optimization" in query_result['answer'].lower()
        
        # At least one source should be from our document
        source_matched = False
        for source in query_result['sources']:
            if os.path.basename(self.document_path) in source.get('metadata', {}).get('source', ''):
                source_matched = True
                break
        
        assert source_matched, "Query didn't return sources from our document"
        print("✓ Document query workflow test passed") 