"""
Test configuration and fixtures for the Mini-RAG backend tests.
"""

import pytest
import os
import tempfile
import json

from config import settings, get_path

# Override settings for testing to use smaller models and memory footprint
@pytest.fixture(scope="session", autouse=True)
def configure_test_settings():
    """Configure test-specific settings to optimize resource usage during testing."""
    # Get a reference to app modules after setting the correct paths
    from app.services.embeddings import embeddings_service
    from app.services.vectorstore import vector_store
    
    # Configure smaller model sizes and batch sizes for testing
    settings.CHUNK_SIZE = 100  # Smaller chunks for testing
    settings.CHUNK_OVERLAP = 10
    settings.DEFAULT_BATCH_SIZE = 2  # Small batch size to test batching logic
    settings.MODEL_N_CTX = 512  # Smaller context window for testing
    settings.MEMORY_SAFETY_MARGIN_MB = 128  # Smaller safety margin for testing
    
    # Fix paths for testing using the get_path function to prevent duplicate backend directories
    settings.MODEL_PATH = get_path("models/phi-2.gguf")
    settings.EMBEDDING_MODEL = get_path("models/embeddings/all-MiniLM-L6-v2")
    settings.VECTOR_DB_PATH = get_path("vector_db/chroma_db")
    
    # Update the embedding service model name
    embeddings_service.model_name = settings.EMBEDDING_MODEL
    
    # Reset the model to ensure it's reloaded with the new path
    embeddings_service._model = None
    vector_store._client = None
    vector_store._collection = None
    
    yield
    
    # No need to reset since we're using real paths

@pytest.fixture
def sample_text_content():
    """Sample text content for testing document processing."""
    return """
    Mini-RAG is a lightweight Retrieval-Augmented Generation system designed for 
    resource-efficient operation. It provides document processing, embedding generation, 
    semantic search and LLM-enhanced responses with minimal computational requirements.
    
    The system implements memory optimization techniques including:
    - Lazy loading of ML models
    - Batch processing with configurable sizes
    - Explicit memory cleanup mechanisms
    - Document chunking with controlled overlap
    """

@pytest.fixture
def sample_text_file(sample_text_content):
    """Create a temporary text file with sample content."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
        temp.write(sample_text_content.encode('utf-8'))
        temp.flush()
        temp_path = temp.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path) 