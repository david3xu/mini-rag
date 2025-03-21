# Mini-RAG Project

A lightweight Retrieval-Augmented Generation (RAG) system with a TypeScript frontend and Python backend, designed for resource-efficient operation.

## Overview

Mini-RAG allows users to upload documents, process them into vector embeddings, and perform semantic searches with LLM-enhanced responses. The system is designed to operate with minimal computational resources while maintaining high-quality retrieval and generation capabilities.

## Architecture

- **Frontend**: React + TypeScript application
- **Backend**: FastAPI Python service
- **Vector Database**: ChromaDB for efficient vector storage
- **LLM Integration**: Support for local models like phi-2.gguf

## Directory Structure

```
mini-rag/
├── frontend/                       # TypeScript React frontend
│   ├── src/                        # Frontend source code
│   ├── public/                     # Static assets
│   ├── package.json                # Frontend dependencies
│   ├── tsconfig.json               # TypeScript configuration
│   └── .env files                  # Environment configurations
│
├── backend/                        # Python FastAPI backend
│   ├── app/                        # Backend application code
│   │   ├── api/                    # API endpoints
│   │   └── services/               # Service implementations
│   ├── data/                       # Document storage
│   │   ├── uploads/                # Uploaded documents
│   │   └── processed/              # Processed documents
│   ├── models/                     # LLM models (gitignored)
│   ├── vector_db/                  # Vector database storage (gitignored)
│   ├── tests/                      # Test suite
│   │   ├── unit/                   # Unit tests
│   │   └── integration/            # Integration tests
│   ├── main.py                     # Application entry point
│   ├── config.py                   # Configuration settings
│   ├── debug_tests.py              # Debug utilities
│   └── TESTING.md                  # Testing documentation
│
├── docs/                           # Documentation
├── .env.local                      # Environment variables
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Development configuration
└── azure-docker-compose.yml        # Azure deployment configuration
```

## Features

- Document uploading and processing
- Efficient text chunking and embedding generation
- Semantic search with relevance ranking
- LLM-enhanced responses using retrieved context
- Resource-optimized implementation suitable for constrained environments
- Docker-based deployment for both development and production
- Comprehensive test suite for backend components

## Development

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.10+ (for local backend development)

### Getting Started

1. Clone the repository
2. Configure environment variables in `.env`
3. Start the development environment:
   ```bash
   docker-compose up -d
   ```

### Testing

The backend includes a comprehensive test suite using pytest:

```bash
# Run all backend tests
cd backend && PYTHONPATH=$PYTHONPATH:$PWD python -m pytest

# Run specific test categories
python -m pytest tests/unit/api/

# Run tests with coverage
python -m pytest --cov=app
```

See [backend/TESTING.md](backend/TESTING.md) for detailed testing documentation.

## Path Management

The system uses a path resolution system to prevent duplicate directory issues:

```python
from config import get_path

# Resolves to the correct absolute path regardless of execution context
correct_path = get_path("vector_db/chroma_db")
```

## Maintenance Utilities

- **Cleanup Script**: `backend/clean_duplicate_dirs.py` fixes duplicate directory issues
  ```bash
  cd backend && python clean_duplicate_dirs.py
  ```

## Azure Deployment

The system is designed to be easily migrated to Azure with the following integration points:

- Document Storage: Azure Blob Storage
- Vector Database: Azure Cognitive Search
- LLM Service: Azure OpenAI
- Embedding Generation: Azure OpenAI Embeddings

## License

[Specify the license here]
