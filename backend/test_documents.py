# backend/test_documents.py
import sys
sys.path.append('.')
import logging

from app.services.vectorstore import vector_store
from app.api.models import DocumentChunk

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_individual_document():
    """Test adding a single document with precise error reporting."""
    print("=== Testing Single Document Addition ===")
    
    # Create minimal test document
    doc = DocumentChunk(
        id="test_single_doc",
        text="This is a simple test document for vector store debugging.",
        metadata={"source": "test_single.txt"}
    )
    
    # Create simple embedding (all zeros except one value)
    # This avoids embedding model complexity
    embedding_dim = 384  # Standard dimension for all-MiniLM-L6-v2
    embedding = [0.0] * embedding_dim
    embedding[0] = 1.0  # Set first dimension to 1.0
    
    try:
        # Attempt to add document directly to collection
        print(f"Adding document with ID: {doc.id}")
        
        # Get collection first to ensure it's initialized
        collection = vector_store.collection
        print(f"Collection initialized: {collection.name}")
        
        # Add document directly with collection API
        collection.add(
            documents=[doc.text],
            embeddings=[embedding],
            ids=[doc.id],
            metadatas=[doc.metadata]
        )
        
        # Verify document was added
        count = collection.count()
        print(f"Collection now contains {count} documents")
        
        # Try to retrieve the document
        result = collection.get(ids=[doc.id])
        if result and result["ids"]:
            print(f"✓ Successfully retrieved document: {result['ids'][0]}")
            return True
        else:
            print("✗ Document not found after addition")
            return False
    
    except Exception as e:
        print(f"✗ Error adding document: {str(e)}")
        return False

if __name__ == "__main__":
    test_individual_document()