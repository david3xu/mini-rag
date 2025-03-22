import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import the main app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        """Test that the health check endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestDocumentAPI:
    def test_list_documents_empty(self, client):
        """Test listing documents when none exist."""
        # This assumes the documents endpoint exists
        response = client.get("/api/documents")
        assert response.status_code == 200
        assert response.json() == []
    
    @pytest.mark.skip(reason="Implement when document upload is available")
    def test_upload_document(self, client):
        """Test uploading a document."""
        # This will be implemented when the document upload endpoint is available
        pass


class TestSearchAPI:
    @pytest.mark.skip(reason="Implement when search is available")
    def test_search_documents(self, client):
        """Test searching documents."""
        # This will be implemented when the search endpoint is available
        pass 