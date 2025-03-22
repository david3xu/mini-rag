"""
Unit tests for the Document Processor service.
"""

import pytest
import os
import tempfile

from app.services.document_processor import DocumentProcessor, document_processor

@pytest.mark.unit
class TestDocumentProcessor:
    
    def test_split_text_into_chunks(self):
        """Test the text chunking functionality."""
        processor = DocumentProcessor()
        
        # Create a long repeating text
        text = "This is a test sentence. " * 20
        
        # Set small chunk size and overlap for testing
        chunk_size = 50
        overlap = 10
        
        # Split into chunks
        chunks = processor._split_text_into_chunks(text, chunk_size, overlap)
        
        # Verify results
        assert len(chunks) > 1
        
        # Verify chunk sizes
        for chunk in chunks:
            assert len(chunk) <= chunk_size
            
        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Check for overlap
            end_of_chunk = chunk[-overlap:] if len(chunk) > overlap else chunk
            start_of_next = next_chunk[:overlap] if len(next_chunk) > overlap else next_chunk
            
            assert end_of_chunk in start_of_next or start_of_next in end_of_chunk 