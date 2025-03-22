import os
import sys
import pytest
import tempfile
import shutil
import numpy as np

# Conditionally import chromadb if available
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Mark all tests in this file as requiring chromadb
pytestmark = pytest.mark.skipif(not CHROMA_AVAILABLE, reason="ChromaDB not installed")


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the vector database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def chroma_client(temp_db_path):
    """Create a ChromaDB client with a temporary persistence directory."""
    client = chromadb.PersistentClient(
        path=temp_db_path,
        settings=ChromaSettings(
            anonymized_telemetry=False
        )
    )
    return client


class TestChromaDB:
    def test_collection_creation(self, chroma_client):
        """Test creating a collection in ChromaDB."""
        collection_name = "test_collection"
        collection = chroma_client.create_collection(name=collection_name)
        
        assert collection.name == collection_name
        assert collection.count() == 0
    
    def test_add_and_query_embeddings(self, chroma_client):
        """Test adding embeddings and querying them."""
        collection = chroma_client.create_collection(name="test_embeddings")
        
        # Create some test embeddings (2 documents, 4-dimensional vectors)
        embeddings = [
            [0.1, 0.2, 0.3, 0.4],  # Document 1
            [0.5, 0.6, 0.7, 0.8],  # Document 2
        ]
        
        # Add embeddings to the collection
        collection.add(
            embeddings=embeddings,
            documents=["This is document 1", "This is document 2"],
            ids=["doc1", "doc2"],
            metadatas=[
                {"source": "test", "page": 1},
                {"source": "test", "page": 2}
            ]
        )
        
        # Verify count
        assert collection.count() == 2
        
        # Query the collection
        results = collection.query(
            query_embeddings=[0.1, 0.2, 0.3, 0.4],
            n_results=2
        )
        
        # Verify results
        assert len(results["ids"][0]) == 2
        assert "doc1" in results["ids"][0]
        
        # Get by id
        result = collection.get(ids=["doc1"])
        assert len(result["ids"]) == 1
        assert result["documents"][0] == "This is document 1"
        assert result["metadatas"][0]["page"] == 1 