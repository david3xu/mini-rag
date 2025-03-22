# Mini-RAG Testing Framework: Technical Implementation Guide

## Project Overview

Mini-RAG is a lightweight Retrieval-Augmented Generation system designed for resource-efficient operation in local environments with an Azure migration pathway. This document outlines the comprehensive testing architecture, implementation strategies, and validation protocols to ensure system integrity, performance optimization, and migration readiness.

## Testing Architecture

### Directory Structure Implementation

```
mini-rag/
├── backend/
│   ├── tests/
│   │   ├── conftest.py              # Test configuration and fixtures
│   │   ├── unit/                    # Unit tests for isolated components
│   │   │   ├── services/            # Service-level tests
│   │   │   ├── api/                 # API endpoint tests
│   │   ├── integration/             # Component interaction tests
│   │   ├── performance/             # Resource optimization tests
│   │   ├── azure/                   # Migration pathway tests
│   │   ├── data/                    # Test data resources
│   │   └── mocks/                   # Mock implementations for testing
├── frontend/
│   ├── src/
│   │   ├── __tests__/               # Jest test directory
│   │   │   ├── components/          # Component tests
│   │   │   ├── hooks/               # Hook tests
│   │   │   └── utils/               # Utility function tests
│   ├── cypress/                     # E2E testing
│   │   ├── integration/             # Test specifications
│   │   └── fixtures/                # Test data
└── .github/
    └── workflows/                   # CI pipeline configuration
```

### Testing Tool Selection Matrix

| Component | Primary Tools | Supporting Tools | Purpose |
|-----------|--------------|------------------|---------|
| Backend Unit Testing | pytest, pytest-asyncio | pytest-cov, unittest.mock | Component validation, service isolation |
| Backend Integration | pytest | TestClient, httpx | API workflow validation |
| Frontend Unit Testing | Jest, React Testing Library | MSW | Component and utility testing |
| Frontend Integration | Cypress | MSW | User workflow validation |
| Performance Testing | locust, pytest-benchmark | psutil | Resource utilization validation |
| Migration Testing | pytest | Azure SDK mocks | Service compatibility verification |

## Backend Testing Implementation

### Core Testing Domains

- **Service Component Testing**: Validate individual service modules with controlled dependencies
- **API Interface Testing**: Verify endpoint behavior, response structure, and error handling
- **Resource Optimization Testing**: Confirm memory management, efficient processing, and cleanup mechanisms
- **Migration Compatibility Testing**: Ensure seamless transition between local and Azure components

### Implementation Guidelines

#### Service Testing Framework

```python
# tests/unit/services/test_embeddings_service.py
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from app.services.embeddings import EmbeddingService, embeddings_service

@pytest.fixture
def mock_sentence_transformer():
    """Mock the SentenceTransformer class."""
    with patch('app.services.embeddings.SentenceTransformer') as mock:
        model = MagicMock()
        # Configure mock model to return predictable embeddings
        model.encode.return_value = np.random.rand(3, 384)
        mock.return_value = model
        yield mock

def test_embedding_generation_batching(mock_sentence_transformer):
    """Test embedding generation with batch processing."""
    # Initialize service with clean state
    service = EmbeddingService()
    
    # Mock texts to embed
    texts = ["Document 1", "Document 2", "Document 3"]
    
    # Use smaller batch size to test batching
    embeddings = service.generate_embeddings(texts, batch_size=2)
    
    # Verify results
    assert len(embeddings) == len(texts)
    assert all(len(emb) == service.dimension for emb in embeddings)
    
    # Verify batching behavior
    calls = service.model.encode.call_args_list
    assert len(calls) == 2  # Should call twice with batch size 2
    assert list(calls[0][0][0]) == texts[:2]  # First batch
    assert list(calls[1][0][0]) == texts[2:]  # Second batch
```

#### Memory Management Testing

```python
# tests/performance/test_resource_management.py
import pytest
import time
import psutil
import os
from app.services.llm import llm_service

def test_model_unloading_after_inactivity():
    """Test that LLM model is unloaded after inactivity period."""
    # Record initial memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Access model property to load it
    _ = llm_service.llm
    assert llm_service._llm is not None
    
    # Record memory after model loading
    loaded_memory = process.memory_info().rss
    
    # Modify last_used_time to simulate inactivity
    llm_service.last_used_time = time.time() - 7200  # 2 hours ago
    
    # Trigger cleanup
    llm_service.unload_model_if_inactive(threshold_seconds=60)
    
    # Check model is unloaded
    assert llm_service._llm is None
    
    # Verify memory reduction
    final_memory = process.memory_info().rss
    assert final_memory < loaded_memory
```

#### Document Processing Testing

```python
# tests/unit/services/test_document_processor.py
import pytest
import os
import tempfile
from app.services.document_processor import document_processor

@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
        # Generate repeating content to ensure multi-chunk processing
        content = "This is a test document with content that will be processed. " * 50
        temp.write(content.encode('utf-8'))
        temp.flush()
        temp_path = temp.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_text_file_processing(sample_text_file):
    """Test processing of text files into chunks."""
    # Process the sample file
    chunks = document_processor.process_file(sample_text_file)
    
    # Verify chunk generation
    assert len(chunks) > 1
    assert all(isinstance(chunk, dict) for chunk in chunks)
    assert all("id" in chunk for chunk in chunks)
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)
    
    # Verify metadata
    for chunk in chunks:
        assert "source" in chunk["metadata"]
        assert os.path.basename(sample_text_file) in chunk["metadata"]["source"]
        assert "chunk" in chunk["metadata"]
```

## Frontend Testing Implementation

### Component Testing Strategy

- **Isolated Component Testing**: Validate rendering, props handling, and state management
- **Interaction Testing**: Verify user input handling and event propagation
- **UI State Validation**: Confirm loading states, error handling, and conditional rendering
- **Custom Hook Testing**: Ensure hook behavior and state management

### Implementation Guidelines

#### React Component Testing

```javascript
// src/__tests__/components/ChatInput.test.tsx
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatInput from '../../components/ChatInput';

describe('ChatInput Component', () => {
  test('renders with placeholder text', () => {
    render(<ChatInput 
      onSubmit={jest.fn()} 
      isLoading={false} 
      placeholder="Test placeholder" 
    />);
    
    expect(screen.getByPlaceholderText('Test placeholder')).toBeInTheDocument();
  });
  
  test('handles user input and submission', async () => {
    const mockSubmit = jest.fn();
    render(<ChatInput onSubmit={mockSubmit} isLoading={false} />);
    
    // Type in the textarea
    const textarea = screen.getByRole('textbox');
    await userEvent.type(textarea, 'test query');
    
    // Submit the form
    const submitButton = screen.getByRole('button', { name: /send/i });
    await userEvent.click(submitButton);
    
    // Verify submission
    expect(mockSubmit).toHaveBeenCalledWith('test query');
    
    // Verify textarea is cleared after submission
    expect(textarea).toHaveValue('');
  });
  
  test('disables form elements when loading', () => {
    render(<ChatInput onSubmit={jest.fn()} isLoading={true} />);
    
    expect(screen.getByRole('textbox')).toBeDisabled();
    expect(screen.getByRole('button', { name: /processing/i })).toBeDisabled();
  });
  
  test('handles keyboard shortcut submission', async () => {
    const mockSubmit = jest.fn();
    render(<ChatInput onSubmit={mockSubmit} isLoading={false} />);
    
    // Type in the textarea
    const textarea = screen.getByRole('textbox');
    await userEvent.type(textarea, 'test query');
    
    // Use Ctrl+Enter to submit
    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true });
    
    // Verify submission
    expect(mockSubmit).toHaveBeenCalledWith('test query');
  });
});
```

#### Custom Hook Testing

```javascript
// src/__tests__/hooks/useApi.test.ts
import { renderHook, act } from '@testing-library/react-hooks';
import axios from 'axios';
import { useApi } from '../../hooks/useApi';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('useApi Hook', () => {
  test('initializes with default state', () => {
    const apiFunction = jest.fn();
    const { result } = renderHook(() => useApi(apiFunction));
    
    expect(result.current.state).toEqual({
      data: null,
      isLoading: false,
      error: null
    });
  });
  
  test('handles successful API calls', async () => {
    // Mock response
    const mockResponse = { data: { result: 'success' } };
    const apiFunction = jest.fn().mockResolvedValue(mockResponse);
    
    // Render hook
    const { result, waitForNextUpdate } = renderHook(() => useApi(apiFunction));
    
    // Execute request
    act(() => {
      result.current.executeRequest({ id: '123' });
    });
    
    // Verify loading state
    expect(result.current.state.isLoading).toBe(true);
    
    // Wait for resolution
    await waitForNextUpdate();
    
    // Verify final state
    expect(result.current.state).toEqual({
      data: mockResponse.data,
      isLoading: false,
      error: null
    });
    
    // Verify API call
    expect(apiFunction).toHaveBeenCalledWith({ id: '123' });
  });
  
  test('handles API errors', async () => {
    // Mock error
    const errorMessage = 'API error occurred';
    const apiFunction = jest.fn().mockRejectedValue(new Error(errorMessage));
    
    // Render hook
    const { result, waitForNextUpdate } = renderHook(() => useApi(apiFunction));
    
    // Execute request
    act(() => {
      result.current.executeRequest({ id: '123' });
    });
    
    // Wait for rejection
    await waitForNextUpdate();
    
    // Verify error state
    expect(result.current.state).toEqual({
      data: null,
      isLoading: false,
      error: errorMessage
    });
  });
});
```

## Integration Testing Framework

### API Workflow Testing

```python
# tests/integration/test_document_query_workflow.py
import pytest
from fastapi.testclient import TestClient
import os
import tempfile
import time
from main import app

client = TestClient(app)

@pytest.fixture
def test_document():
    """Create a test document with known content."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
        content = "Mini-RAG is a system for retrieval-augmented generation that operates efficiently on local resources."
        temp.write(content.encode('utf-8'))
        temp.flush()
        temp_path = temp.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

def test_upload_and_query_workflow(test_document):
    """Test complete document upload and query workflow."""
    # 1. Upload document
    with open(test_document, "rb") as f:
        filename = os.path.basename(test_document)
        response = client.post(
            "/api/documents/upload",
            files={"files": (filename, f, "text/plain")}
        )
    
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    
    # 2. Allow time for background processing
    time.sleep(5)
    
    # 3. Query the system with related question
    query_response = client.post(
        "/api/chat",
        json={"query": "What is Mini-RAG?"}
    )
    
    assert query_response.status_code == 200
    
    # 4. Validate response contains relevant information
    response_data = query_response.json()
    assert "answer" in response_data
    assert "sources" in response_data
    assert len(response_data["sources"]) > 0
    
    # 5. Verify answer relevance
    assert "retrieval" in response_data["answer"].lower() or "rag" in response_data["answer"].lower()
    
    # 6. Verify source attribution
    assert filename in response_data["sources"][0]["metadata"]["source"]
```

### End-to-End Testing (Cypress)

```javascript
// cypress/integration/document_upload_spec.js
describe('Document Upload and Query Flow', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  it('allows document upload and querying', () => {
    // 1. Upload document
    cy.fixture('test-document.txt').then(fileContent => {
      cy.get('input[type="file"]').attachFile({
        fileContent,
        fileName: 'test-document.txt',
        mimeType: 'text/plain'
      });
    });
    
    cy.get('button').contains('Upload').click();
    
    // 2. Verify upload success
    cy.contains('Successfully processed', { timeout: 10000 });
    
    // 3. Enter query
    cy.get('textarea[placeholder*="Ask a question"]').type('What information is in the document?');
    cy.get('button').contains('Send').click();
    
    // 4. Verify response
    cy.contains('This document contains information about', { timeout: 10000 });
    
    // 5. Check source attribution
    cy.contains('Show sources').click();
    cy.contains('test-document.txt');
  });
  
  it('handles errors appropriately', () => {
    // 1. Try to submit without documents
    cy.get('textarea[placeholder*="Ask a question"]').type('What is in the document?');
    cy.get('button').contains('Send').click();
    
    // 2. Verify appropriate error message
    cy.contains('I don\'t have enough information', { timeout: 10000 });
  });
});
```

## Performance Testing Strategy

### Memory Optimization Validation

```python
# tests/performance/test_batch_processing.py
import pytest
import psutil
import os
import numpy as np
from app.services.embeddings import embeddings_service

def measure_memory_usage():
    """Helper function to measure current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_batch_size_memory_impact(batch_size):
    """Test memory impact of different batch sizes."""
    # Generate test data
    num_documents = 100
    documents = [f"This is test document {i}" * 10 for i in range(num_documents)]
    
    # Record baseline memory
    baseline_memory = measure_memory_usage()
    
    # Process with specified batch size
    embeddings = embeddings_service.generate_embeddings(documents, batch_size=batch_size)
    
    # Verify expected output
    assert len(embeddings) == num_documents
    
    # Record peak memory
    peak_memory = measure_memory_usage()
    memory_increase = peak_memory - baseline_memory
    
    print(f"Batch size {batch_size}: Memory increase {memory_increase:.2f} MB")
    
    # Smaller batches should use less memory at peak
    # This is an approximate test as memory behavior can vary by system
    if batch_size <= 4:  # For smaller batch sizes
        assert memory_increase < 500  # Example threshold
```

### Response Time Benchmarking

```python
# tests/performance/test_query_performance.py
import pytest
import time
from app.services.vectorstore import vector_store
from app.services.llm import llm_service
from app.services.embeddings import embeddings_service

@pytest.mark.benchmark
def test_query_response_time(benchmark):
    """Benchmark response time for queries."""
    def query_process():
        # Generate embedding for test query
        query = "What is retrieval-augmented generation?"
        query_embedding = embeddings_service.generate_embedding(query)
        
        # Retrieve documents
        documents = vector_store.search(query_embedding, k=3)
        
        # Generate response
        prompt = llm_service.format_rag_prompt(query, documents)
        response = llm_service.generate_text(prompt, max_tokens=100)
        
        return response
    
    # Execute benchmark
    result = benchmark(query_process)
    
    # Verify response and timing
    assert result is not None
    assert len(result) > 0
    
    # Timing assertions (adjust thresholds based on system capabilities)
    assert benchmark.stats.stats.mean < 5.0  # Average time below 5 seconds
```

## Azure Migration Testing Framework

### Cloud Service Compatibility

```python
# tests/azure/test_service_migration.py
import pytest
from unittest.mock import patch, MagicMock
import json
from app.services.azure_openai import azure_openai_service
from app.services.azure_search import azure_search_service

@pytest.fixture
def mock_azure_openai():
    with patch('app.services.azure_openai.openai') as mock_openai:
        # Configure mock responses
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Azure OpenAI response"
        
        mock_openai.ChatCompletion.create.return_value = mock_response
        yield mock_openai

def test_azure_openai_integration(mock_azure_openai):
    """Test Azure OpenAI service compatibility."""
    # Test text generation
    response = azure_openai_service.generate_text("Test prompt")
    
    # Verify response
    assert response == "Azure OpenAI response"
    
    # Verify correct API usage
    mock_azure_openai.ChatCompletion.create.assert_called_once()
    args = mock_azure_openai.ChatCompletion.create.call_args
    assert args[1]["engine"] == azure_openai_service.deployment_name
    assert len(args[1]["messages"]) == 2
    assert args[1]["messages"][1]["content"] == "Test prompt"

@pytest.fixture
def mock_azure_search():
    with patch('app.services.azure_search.SearchClient') as mock_client:
        # Configure mock search results
        mock_client.return_value.search.return_value = [
            {
                "id": "doc1",
                "content": "Azure Search document content",
                "metadata": json.dumps({"source": "test.txt"})
            }
        ]
        yield mock_client

def test_azure_search_integration(mock_azure_search):
    """Test Azure Search service compatibility."""
    # Test document search
    results = azure_search_service.search([0.1] * 384, k=1)
    
    # Verify results
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert results[0]["content"] == "Azure Search document content"
    assert results[0]["metadata"]["source"] == "test.txt"
```

## Continuous Integration Pipeline

```yaml
# .github/workflows/test.yml
name: Mini-RAG Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r backend/requirements.txt
          pip install pytest pytest-asyncio pytest-cov pytest-benchmark
          
      - name: Run unit tests
        run: |
          cd backend
          python -m pytest tests/unit -v --cov=app
          
      - name: Run integration tests
        run: |
          cd backend
          python -m pytest tests/integration -v
          
      - name: Run performance tests
        run: |
          cd backend
          python -m pytest tests/performance -v --benchmark-disable  # Disable actual benchmarking in CI
          
      - name: Upload coverage report
        uses: codecov/codecov-action@v3
        with:
          directory: ./backend

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          
      - name: Install dependencies
        run: |
          cd frontend
          npm ci
          
      - name: Run Jest tests
        run: |
          cd frontend
          npm test -- --coverage
          
      - name: Run Cypress tests
        uses: cypress-io/github-action@v5
        with:
          working-directory: frontend
          start: npm start
          wait-on: 'http://localhost:3000'
          wait-on-timeout: 120
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- Establish directory structure and test configurations
- Implement basic unit tests for critical backend services
- Set up frontend component testing framework
- Create test data fixtures and mock configurations

### Phase 2: Component Testing (Week 3-4)

- Complete unit test coverage for all backend services
- Implement API endpoint validation tests
- Create comprehensive frontend component tests
- Set up CI pipeline with automated test execution

### Phase 3: Integration & Performance (Week 5-6)

- Implement backend workflow integration tests
- Create end-to-end frontend tests with Cypress
- Develop memory and resource optimization tests
- Implement response time benchmarking

### Phase 4: Migration Testing (Week 7-8)

- Create Azure service compatibility tests
- Implement migration validation framework
- Develop comparative performance tests
- Complete comprehensive test documentation

## Technical Implementation Considerations

### Test Environment Configuration

- Create isolated test environments with controlled data
- Implement testing-specific configuration profiles
- Use smaller model variants for faster test execution
- Configure resource monitoring for performance testing

### Mock Implementation Strategy

- Develop consistent mocking patterns for external dependencies
- Create realistic data generators for test scenarios
- Implement service virtualization for Azure components
- Use factory patterns for test object creation

### CI/CD Optimization

- Implement parallel test execution for faster feedback
- Configure test caching to reduce redundant execution
- Establish test prioritization for critical components
- Implement selective testing based on code changes

This technical implementation guide provides a comprehensive framework for establishing robust testing practices for the Mini-RAG system, focusing on resource optimization validation and Azure migration readiness.
