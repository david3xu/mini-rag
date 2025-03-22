"""
Unit tests for the LLM service.
"""

import pytest
import time
import gc

from app.services.llm import LLMService, llm_service

@pytest.mark.unit
class TestLLMService:
    
    def test_lazy_loading(self):
        """Test that the model is only loaded when accessed."""
        service = LLMService()
        
        # Model should not be loaded initially
        assert service._llm is None
        
        # Access model property to trigger loading
        _ = service.llm
        
        # Model should now be loaded
        assert service._llm is not None
        
        # Manually clean up for testing
        service._llm = None
        gc.collect()
    
    def test_generate_text(self):
        """Test generating text from the LLM."""
        service = LLMService()
        
        # Create test prompt
        prompt = "Explain what a RAG system is"
        
        # Generate response
        response = service.generate_text(prompt, max_tokens=20)
        
        # Verify results
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Manually clean up for testing
        service._llm = None
        gc.collect()
    
    def test_generate_streaming_text(self):
        """Test generating a streaming response from the LLM."""
        service = LLMService()
        
        # Create test prompt
        prompt = "Explain what a RAG system is"
        
        # Generate streaming response
        response_stream = service.generate_text(prompt, max_tokens=20, stream=True)
        
        # Collect chunks
        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)
        
        # Verify results
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        # Manually clean up for testing
        service._llm = None
        gc.collect() 