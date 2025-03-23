#!/usr/bin/env python
"""
Comprehensive debug script for the Mini-RAG system.
This script tests both health endpoints and vector store operations.
"""

import sys
import os
import tempfile
import time
import json
import psutil

# Make sure we can import from the correct paths
sys.path.append('.')
sys.path.append('./backend')  # Add the backend directory to the path

# Imports for test client
try:
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.models import DocumentChunk
    from app.api.health import ResourceMetrics  # Fixed import for ResourceMetrics
    from app.services.vectorstore import VectorStoreService
    from app.services.embeddings import embeddings_service
    from config import settings
except ImportError as e:
    print(f"Import error: {e}")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path)
    print("\nTrying alternate imports...")
    
    # Try with relative imports
    from fastapi.testclient import TestClient
    from main import app
    from app.api.models import DocumentChunk
    from app.api.health import ResourceMetrics  # Fixed import for ResourceMetrics
    from app.services.vectorstore import VectorStoreService
    from app.services.embeddings import embeddings_service
    from config import settings

# Create test client
client = TestClient(app)

def print_separator(title):
    """Print a section separator with a title."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_health_endpoint():
    """Test the main health endpoint."""
    print_separator("TESTING HEALTH ENDPOINT")
    
    # Send request to health endpoint
    print("Sending GET request to /api/health...")
    response = client.get("/api/health")
    
    # Print response details
    print(f"Response status code: {response.status_code}")
    print(f"Response content type: {response.headers.get('content-type', 'unknown')}")
    print("Response data:")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
        
        # Verify response structure
        if 'status' in data:
            print("✓ Response contains status field")
        else:
            print("✗ Response missing status field")
            
        if 'model_loaded' in data:
            print(f"✓ Model loaded: {data['model_loaded']}")
        
        if 'vector_store_available' in data:
            print(f"✓ Vector store available: {data['vector_store_available']}")
            
        if 'document_count' in data:
            print(f"✓ Document count: {data['document_count']}")
            
        if 'resources' in data:
            print("✓ Resources metrics included")
            print(f"  CPU: {data['resources'].get('cpu_percent')}%")
            print(f"  Memory: {data['resources'].get('memory_percent')}%")
    except Exception as e:
        print(f"Error parsing response: {str(e)}")

def test_readiness_endpoint():
    """Test the readiness endpoint."""
    print_separator("TESTING READINESS ENDPOINT")
    
    # Send request to readiness endpoint
    print("Sending GET request to /api/health/readiness...")
    response = client.get("/api/health/readiness")
    
    # Print response details
    print(f"Response status code: {response.status_code}")
    print("Response data:")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
        
        # Verify response structure
        if data.get('status') == 'ready':
            print("✓ System is ready")
        else:
            print(f"✗ System is not ready, status: {data.get('status')}")
    except Exception as e:
        print(f"Error parsing response: {str(e)}")

def test_liveness_endpoint():
    """Test the liveness endpoint."""
    print_separator("TESTING LIVENESS ENDPOINT")
    
    # Send request to liveness endpoint
    print("Sending GET request to /api/health/liveness...")
    response = client.get("/api/health/liveness")
    
    # Print response details
    print(f"Response status code: {response.status_code}")
    print("Response data:")
    try:
        data = response.json()
        print(json.dumps(data, indent=2))
        
        # Verify response structure
        if data.get('status') == 'alive':
            print("✓ System is alive")
        else:
            print(f"✗ System is not alive, status: {data.get('status')}")
    except Exception as e:
        print(f"Error parsing response: {str(e)}")

def test_resource_metrics():
    """Test resource metrics generation."""
    print_separator("TESTING RESOURCE METRICS")
    
    print("Gathering system metrics...")
    
    # Get CPU and memory usage
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory.percent}%")
    print(f"Memory Total: {memory.total / (1024**3):.2f} GB")
    print(f"Memory Available: {memory.available / (1024**3):.2f} GB")
    
    # Create ResourceMetrics object
    metrics = ResourceMetrics(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_available=memory.available,
        memory_total=memory.total
    )
    
    print("\nResourceMetrics object:")
    print(f"  CPU: {metrics.cpu_percent}%")
    print(f"  Memory: {metrics.memory_percent}%")
    print(f"  Memory Available: {metrics.memory_available / (1024**3):.2f} GB")
    print(f"  Memory Total: {metrics.memory_total / (1024**3):.2f} GB")
    
    # Test that metrics are within reasonable ranges
    if 0 <= metrics.cpu_percent <= 100:
        print("✓ CPU percentage is valid")
    else:
        print(f"✗ CPU percentage {metrics.cpu_percent} is invalid")
        
    if 0 <= metrics.memory_percent <= 100:
        print("✓ Memory percentage is valid")
    else:
        print(f"✗ Memory percentage {metrics.memory_percent} is invalid")
        
    if metrics.memory_available <= metrics.memory_total:
        print("✓ Memory available is less than or equal to total memory")
    else:
        print("✗ Memory available exceeds total memory")

def test_vectorstore_initialization():
    """Test vector store initialization."""
    print_separator("TESTING VECTORSTORE INITIALIZATION")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating vector store in temporary directory: {temp_dir}")
        
        # Create service instance
        service = VectorStoreService(persist_directory=temp_dir)
        
        # Access the collection to initialize it
        print("Initializing the collection...")
        collection = service.collection
        
        if collection is not None:
            print("✓ Collection created successfully")
        else:
            print("✗ Failed to create collection")

def test_vectorstore_add_documents():
    """Test adding documents to the vector store."""
    print_separator("TESTING ADDING DOCUMENTS TO VECTORSTORE")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating vector store in temporary directory: {temp_dir}")
        
        # Create service instance
        service = VectorStoreService(persist_directory=temp_dir)
        
        # Create test document chunks
        print("\nCreating test document chunks...")
        chunks = [
            DocumentChunk(id="1", text="Test content 1", metadata={"source": "test1.txt"}),
            DocumentChunk(id="2", text="Test content 2", metadata={"source": "test2.txt"}),
            DocumentChunk(id="3", text="Test content 3", metadata={"source": "test3.txt"})
        ]
        
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: ID={chunk.id}, Text='{chunk.text}', Metadata={chunk.metadata}")
        
        try:
            # Generate embeddings
            print("\nGenerating embeddings for document chunks...")
            texts = [chunk.text for chunk in chunks]
            embeddings = embeddings_service.generate_embeddings(texts)
            print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
            
            # Add documents
            print("\nAdding documents to vector store...")
            service.add_documents(chunks, embeddings)
            
            # Verify documents were added
            print("\nVerifying documents were added correctly...")
            try:
                count = service.collection.count()
                print(f"Document count in collection: {count}")
                if count == 3:
                    print("✓ Documents added successfully!")
                else:
                    print(f"✗ Expected 3 documents, but found {count}")
            except Exception as e:
                print(f"✗ Error counting documents: {str(e)}")
        except Exception as e:
            print(f"✗ Error in test: {str(e)}")

def test_vectorstore_similarity_search():
    """Test performing similarity search."""
    print_separator("TESTING SIMILARITY SEARCH IN VECTORSTORE")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating vector store in temporary directory: {temp_dir}")
        
        # Create service instance
        service = VectorStoreService(persist_directory=temp_dir)
        
        # Create test document chunks
        print("\nCreating test document chunks...")
        chunks = [
            DocumentChunk(id="1", text="Mini-RAG is a lightweight system", metadata={"source": "test1.txt"}),
            DocumentChunk(id="2", text="It uses vector embeddings for search", metadata={"source": "test2.txt"}),
            DocumentChunk(id="3", text="Memory optimization is important", metadata={"source": "test3.txt"})
        ]
        
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: ID={chunk.id}, Text='{chunk.text}', Metadata={chunk.metadata}")
        
        try:
            # Generate embeddings
            print("\nGenerating embeddings for document chunks...")
            texts = [chunk.text for chunk in chunks]
            embeddings = embeddings_service.generate_embeddings(texts)
            print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0]) if embeddings else 0}")
            
            # Add documents
            print("\nAdding documents to vector store...")
            service.add_documents(chunks, embeddings)
            
            # Perform search
            query = "system architecture"
            print(f"\nCreating embedding for query: '{query}'")
            query_embedding = embeddings_service.generate_embedding(query)
            
            print("\nPerforming similarity search...")
            results = service.similarity_search(query, query_embedding, k=3)
            
            # Print results
            print(f"\nSearch returned {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: ID={result['id']}, Distance={result.get('distance', 'N/A')}")
                print(f"    Content: '{result['content']}'")
                print(f"    Metadata: {result['metadata']}")
            
            if results:
                print("\n✓ Similarity search completed successfully!")
            else:
                print("\n✗ Similarity search returned no results")
        except Exception as e:
            print(f"✗ Error in test: {str(e)}")

def run_all_tests():
    """Run all tests in sequence."""
    start_time = time.time()
    
    # Run health API tests
    test_health_endpoint()
    test_readiness_endpoint()
    test_liveness_endpoint()
    test_resource_metrics()
    
    # Run vector store tests
    test_vectorstore_initialization()
    test_vectorstore_add_documents()
    test_vectorstore_similarity_search()
    
    # Print summary
    elapsed_time = time.time() - start_time
    print_separator(f"ALL TESTS COMPLETED IN {elapsed_time:.2f} SECONDS")

if __name__ == "__main__":
    print("=== Mini-RAG System Debug Tool ===")
    print("Running comprehensive system tests...")
    print(f"Using embedding model: {settings.EMBEDDING_MODEL}")
    print(f"Using LLM model: {settings.MODEL_PATH}")
    print(f"Vector DB path: {settings.VECTOR_DB_PATH}\n")
    
    # Run all tests
    run_all_tests() 