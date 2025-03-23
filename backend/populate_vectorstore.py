# backend/populate_vectorstore.py
import os
import sys
sys.path.append('.')

from app.services.vectorstore import vector_store
from app.services.embeddings import embeddings_service
from app.api.models import DocumentChunk

def add_test_documents():
    """Add test documents to vector store."""
    print("=== Adding Test Documents to Vector Store ===")
    
    # Create test documents
    documents = [
        DocumentChunk(
            id="doc1",
            text="Mini-RAG is a lightweight Retrieval-Augmented Generation system designed for resource-efficient operation.",
            metadata={"source": "test_doc1.txt"}
        ),
        DocumentChunk(
            id="doc2",
            text="Vector search uses embeddings to find semantically similar documents.",
            metadata={"source": "test_doc2.txt"}
        ),
        DocumentChunk(
            id="doc3",
            text="The system implements memory optimization techniques including lazy loading and batch processing.",
            metadata={"source": "test_doc3.txt"}
        ),
        DocumentChunk(
            id="doc4",
            text="ChromaDB provides efficient vector storage with disk-based persistence.",
            metadata={"source": "test_doc4.txt"}
        )
    ]
    
    try:
        # Generate embeddings for documents
        print("Generating embeddings...")
        texts = [doc.text for doc in documents]
        embeddings = embeddings_service.generate_embeddings(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        # Add documents to vector store
        print("Adding documents to vector store...")
        vector_store.add_documents(documents, embeddings)
        
        # Verify documents were added
        count = vector_store.collection.count()
        print(f"✓ Vector store now contains {count} documents")
        
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    add_test_documents()