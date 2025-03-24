# Mini-RAG Backend Testing Guide

This document describes the testing approach, structure, and utilities for the Mini-RAG backend system.

## Test Organization

The backend tests are organized into the following categories:

```
backend/tests/
├── unit/                       # Unit tests for individual components
│   ├── api/                    # API endpoint tests
│   │   └── test_health.py      # Tests for health endpoints
│   └── services/               # Service implementation tests
│       ├── test_document_processor.py # Tests for document processing
│       ├── test_embeddings.py         # Tests for embedding service
│       ├── test_llm.py                # Tests for LLM service
│       └── test_vectorstore.py        # Tests for vector store
├── integration/                # Integration tests between components
│   └── test_document_query_workflow.py # End-to-end document processing and query
├── examples/                   # Example scripts and utilities
│   ├── clean_duplicate_dirs.py # Utility to fix duplicate directories
│   └── debug_tests.py          # Script for interactive debugging
├── data/                       # Test data files
├── fixtures/                   # Test data and fixtures
└── conftest.py                 # Test fixtures and configurations
```

## Testing Architecture

The Mini-RAG testing architecture uses pytest with FastAPI's TestClient for API testing. Key features include:

1. **Automatic Test Configuration**: The `conftest.py` uses session-scoped fixtures to configure test-specific settings
2. **Resource Optimization**: Tests use smaller models and optimized settings to reduce resource usage
3. **Path Management**: The system uses `get_path()` to prevent duplicate directory issues during testing
4. **Isolated Vector Store**: Tests use isolated vector store instances to prevent test interference

## Key Test Fixtures

### General Fixtures

- `configure_test_settings`: Configures optimized settings for testing (smaller models, reduced resource usage)
- `sample_text_content`: Provides standard text content for document processing tests
- `sample_text_file`: Creates a temporary file with sample content for file-based tests

### API Testing Fixtures

- TestClient instances for isolated API testing
- Mock service responses for predictable testing

## Health Endpoint Tests

The health endpoint tests (`tests/unit/api/test_health.py`) verify system health monitoring:

1. **General Health (`/api/health`)**: Tests overall system health, including:

   - Vector store availability
   - Language model status
   - Resource metrics (CPU, memory)

2. **Readiness (`/api/health/readiness`)**: Tests if the system is ready to receive requests

3. **Liveness (`/api/health/liveness`)**: Tests if the system is alive and functioning

4. **Resource Metrics**: Tests accurate measurement of system resources

## Service Tests

Service tests validate the core functionality of backend services:

1. **Vector Store Service** (`tests/unit/services/test_vectorstore.py`):

   - Initialization and collection creation
   - Document addition
   - Similarity search

2. **Document Processor** (`tests/unit/services/test_document_processor.py`):

   - Text splitting
   - Metadata handling
   - Chunking strategies

3. **Embedding Service** (`tests/unit/services/test_embeddings.py`):

   - Embedding generation
   - Batch processing
   - Model handling

4. **LLM Service** (`tests/unit/services/test_llm.py`):
   - Text generation
   - Context handling
   - Response formatting

## Integration Tests

Integration tests validate the interaction between components:

1. **Document Query Workflow** (`tests/integration/test_document_query_workflow.py`):
   - End-to-end testing of document processing and querying
   - Verifies document ingestion, embedding, storage, and retrieval
   - Tests the complete RAG pipeline

## Running Tests

### Running All Tests

```bash
cd backend
PYTHONPATH=$PYTHONPATH:$PWD python -m pytest
```

### Running Specific Test Categories

```bash
# Run only API tests
python -m pytest tests/unit/api/

# Run only integration tests
python -m pytest tests/integration/

# Run a specific test file
python -m pytest tests/unit/api/test_health.py

# Run a specific test function
python -m pytest tests/unit/api/test_health.py::TestHealthEndpoint::test_liveness_endpoint
```

### Test with Coverage

```bash
python -m pytest --cov=app
```

## Debugging Tests

For debugging tests, several approaches are available:

1. **Verbose Mode**: Add `-v` to see more detailed test output
2. **Print Statements**: Use `-s` to see print output in tests
3. **Debug Script**: Use `examples/debug_tests.py` for interactive debugging

## Test Maintenance

### Path Management

The tests use the `get_path()` function from `config.py` to prevent duplicate directory issues:

```python
# Good practice (prevents duplication)
settings.MODEL_PATH = get_path("models/phi-2.gguf")

# Avoid relative paths that might cause duplication
# settings.MODEL_PATH = "./models/phi-2.gguf"  # Problematic
```

### Cleanup Utilities

If duplicate directories are created during testing, the `examples/clean_duplicate_dirs.py` script can fix the issue:

```bash
# Standard cleanup (safe)
python tests/examples/clean_duplicate_dirs.py

# Force cleanup (when standard fails)
python tests/examples/clean_duplicate_dirs.py --force
```

## Best Practices

1. Always reset service singletons before/after tests
2. Use isolated paths for test data when possible
3. Clean up test artifacts after tests complete
4. Keep tests independent and idempotent
5. Avoid hardcoded paths; use the `get_path()` function
