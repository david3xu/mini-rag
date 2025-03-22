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
│   ├── main.py                     # Application entry point
│   └── config.py                   # Configuration settings
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

## Testing

The Mini-RAG project implements a comprehensive testing strategy across all components:

### Backend Testing

- **Unit Tests**: Located in `backend/tests/unit/`
  - Test individual functions and classes in isolation
  - Mock external dependencies (vector DB, models)
  - Run with pytest: `cd backend && python -m pytest tests/unit/`

- **Integration Tests**: Located in `backend/tests/integration/`
  - Test API endpoints with the TestClient
  - Test service interactions
  - Run with pytest: `cd backend && python -m pytest tests/integration/`

- **Vector Storage Tests**: Located in `backend/tests/vector_db/`
  - Test ChromaDB interactions with smaller test embeddings
  - Validate persistence and retrieval mechanisms
  - Run with pytest: `cd backend && python -m pytest tests/vector_db/`

### Frontend Testing

- **Component Tests**: Located in `frontend/src/__tests__/components/`
  - Test React components with React Testing Library
  - Verify rendering and user interactions
  - Run with Jest: `cd frontend && npm test`

- **Hook Tests**: Located in `frontend/src/__tests__/hooks/`
  - Test custom React hooks
  - Run with Jest: `cd frontend && npm test`

- **API Client Tests**: Located in `frontend/src/__tests__/api/`
  - Test API client functions with mocked responses
  - Run with Jest: `cd frontend && npm test`

### End-to-End Testing

- **System Tests**: Located in `e2e/`
  - Test complete user workflows
  - Uses Cypress to interact with the running application
  - Run with: `cd e2e && npm test`

### Test Coverage

Generate test coverage reports:
- Backend: `cd backend && python -m pytest --cov=app tests/`
- Frontend: `cd frontend && npm test -- --coverage`

## Azure Deployment

The system is designed to be easily migrated to Azure with the following integration points:

- Document Storage: Azure Blob Storage
- Vector Database: Azure Cognitive Search
- LLM Service: Azure OpenAI
- Embedding Generation: Azure OpenAI Embeddings

## License

[Specify the license here]
