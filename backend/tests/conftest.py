"""
Configuration file for pytest.
"""
import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app


@pytest.fixture
def test_client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def test_document_content():
    """Provide sample document content for tests."""
    return """
    This is a test document.
    It contains multiple lines of text.
    This is used for testing document processing and vector embedding.
    """


@pytest.fixture
def test_document_file(tmp_path, test_document_content):
    """Create a temporary text file with test content."""
    file_path = tmp_path / "test_document.txt"
    file_path.write_text(test_document_content)
    return file_path


@pytest.fixture
def mock_embedding():
    """Provide a mock embedding vector for tests."""
    # Simple 4-dimensional embedding for testing
    return [0.1, 0.2, 0.3, 0.4] 