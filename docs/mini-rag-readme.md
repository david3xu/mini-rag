# Mini-RAG Implementation Guide

## System Configuration & Initialization

### Backend Execution Commands

Deploy the Mini-RAG backend using the following standardized commands:

```bash
# Navigate to project root directory
cd path/to/mini-rag

# Install dependencies
pip install -r backend/requirements.txt

# Run FastAPI application with development server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Alternative execution from backend directory:

```bash
# Navigate to backend directory
cd path/to/mini-rag/backend

# Install dependencies
pip install -r requirements.txt

# Run application server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Model Initialization Requirements

Critical model components must be configured before application execution:

```bash
# Create models directory structure
mkdir -p backend/models

# Retrieve optimized quantized model
wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf -O backend/models/phi-2.gguf

# Verify embedding model path configuration
python backend/debug_paths.py
```

## API Interaction Protocol

### Health Monitoring & Diagnostics

```bash
# Verify system health status
curl -X GET http://localhost:8000/api/health

# Check service readiness state
curl -X GET http://localhost:8000/api/health/readiness

# Confirm system liveness
curl -X GET http://localhost:8000/api/health/liveness
```

### Document Management Operations

```bash
# Upload document for processing
curl -X POST http://localhost:8000/api/documents/upload \
  -F "files=@/path/to/your/document.pdf" \
  -H "Content-Type: multipart/form-data"

# Upload multiple documents simultaneously
curl -X POST http://localhost:8000/api/documents/upload \
  -F "files=@/path/to/first.pdf" \
  -F "files=@/path/to/second.txt" \
  -H "Content-Type: multipart/form-data"
```

### Query Processing Interface

```bash
# Submit standard query to retrieve information
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'

# Process query with streaming response
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the key features of the Mini-RAG system."}'
```

### OpenAI-Compatible Interface

```bash
# Submit query using OpenAI-compatible format
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What are the memory optimization techniques in Mini-RAG?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'

# OpenAI-compatible streaming request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain document chunking strategies."}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": true
  }'
```

## Response Structure Specifications

### Health Endpoint Response Example

```json
{
  "status": "ok",
  "timestamp": "2025-03-22T14:30:45.123456",
  "services": {
    "vector_store": {
      "status": "ok",
      "details": "Vector store available. Document count: 12"
    },
    "language_model": {
      "status": "ok",
      "details": "Model file available. Size: 1500.5MB"
    }
  },
  "resources": {
    "cpu_percent": 12.5,
    "memory_used_mb": 1024.3,
    "memory_available_mb": 4096.7,
    "memory_percent": 32.5
  },
  "config": {
    "APP_NAME": "Mini RAG",
    "MODEL_PATH": "/path/to/models/phi-2.gguf",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "CHUNK_SIZE": 1000,
    "DEFAULT_BATCH_SIZE": 8
  }
}
```

### Query Response Structure Example

```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval-based and generation-based approaches for producing relevant and accurate responses. The Mini-RAG system implements this through document processing, vector embeddings, and LLM integration with optimized resource utilization.",
  "sources": [
    {
      "id": "doc_1",
      "content": "Mini-RAG is a lightweight Retrieval-Augmented Generation system designed for resource-efficient operation.",
      "metadata": {
        "source": "sample.txt",
        "chunk": 2
      }
    },
    {
      "id": "doc_2",
      "content": "The system implements memory optimization techniques including lazy loading of ML models and batch processing.",
      "metadata": {
        "source": "technical_doc.pdf",
        "page": 3,
        "chunk": 1
      }
    }
  ]
}
```

## System Architecture & Components

### Architecture Overview

- **Frontend**: React + TypeScript application
- **Backend**: FastAPI Python service
- **Vector Database**: ChromaDB for efficient vector storage
- **LLM Integration**: Support for local models like phi-2.gguf
- **Embedding Service**: Sentence-transformers with memory-optimized batch processing
- **Azure Integration**: Optional migration pathway to Azure services

### Core Functionality

- Document processing with efficient chunking strategies
- Optimized embedding generation with batch processing
- Vector-based semantic search with relevance ranking
- LLM-enhanced response generation with source attribution
- Memory-efficient implementation for resource-constrained environments
- Azure service compatibility for cloud deployment

## Operational Guidelines

### Document Processing Parameters

- Supported file formats: PDF, TXT, MD, JSON
- Maximum file size: 10MB per document
- Optimal batch upload: 5-10 documents concurrently
- Recommended chunking strategy: 1000-character chunks with 100-character overlap

### Query Optimization Strategies

- Specific queries yield more precise results
- Include relevant context in queries for improved retrieval
- Limit query length to 200-300 characters for optimal performance
- Use streaming responses for extended content generation

### Resource Management Protocol

- Monitor system health during high-volume operations
- Implement appropriate pauses between large document uploads
- Process resource-intensive operations during low-demand periods
- Configure batch sizes based on available system memory

## Deployment & Integration

### Docker Deployment Configuration

```bash
# Deploy with standard configuration
docker-compose up -d

# Deploy with Azure integration
docker-compose -f azure-docker-compose.yml up -d
```

### Testing & Verification

```bash
# Execute backend test suite
cd backend
python -m pytest

# Run frontend tests
cd frontend
npm test

# Verify system health
python backend/debug_tests.py

# Test vector store functionality
python backend/test_vectorstore.py
```

### Troubleshooting Procedures

```bash
# Fix duplicate directory issues
python backend/clean_duplicate_dirs.py

# Debug path issues
python backend/debug_paths.py

# Check embedding model configuration
python backend/simple_test.py
```

## Implementation Considerations

### Memory Optimization Techniques

- Lazy loading of ML models reduces startup resource requirements
- Batch processing with configurable chunk sizes manages memory consumption
- Explicit memory cleanup after operations prevents resource leaks
- Adaptive resource allocation adjusts to system capabilities
- Disk-based vector storage minimizes memory utilization

### Performance Tuning Parameters

- MODEL_N_CTX: Controls context window size for LLM
- DEFAULT_BATCH_SIZE: Configures document processing batch size
- CHUNK_SIZE: Determines document segmentation granularity
- CHUNK_OVERLAP: Defines overlap between adjacent chunks
- MEMORY_SAFETY_MARGIN_MB: Establishes threshold for adaptive processing

This implementation guide provides comprehensive instructions for deploying, configuring, and interacting with the Mini-RAG system. Adjust parameters based on specific operational requirements and system constraints.
