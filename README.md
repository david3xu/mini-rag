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
├── frontend/             # TypeScript React frontend
├── backend/              # Python FastAPI backend
│   ├── app/              # Backend application code
│   ├── models/           # LLM models
│   └── vector_db/        # Vector database storage
├── data/                 # Document storage
├── docs/                 # Documentation
└── docker-compose.yml    # Development and deployment configuration
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

## Azure Deployment

The system is designed to be easily migrated to Azure with the following integration points:

- Document Storage: Azure Blob Storage
- Vector Database: Azure Cognitive Search
- LLM Service: Azure OpenAI
- Embedding Generation: Azure OpenAI Embeddings

## License

[Specify the license here]
