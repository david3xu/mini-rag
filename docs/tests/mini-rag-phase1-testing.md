# Mini-RAG System: Phase 1 Testing Implementation

## Overview

This implementation plan establishes the foundation for comprehensive testing of the Mini-RAG system, focusing on critical components while maintaining resource efficiency. Phase 1 testing will validate core functionality, memory optimization patterns, and establish the framework for subsequent testing phases.

## Directory Structure Implementation

```
mini-rag/
├── backend/
│   ├── tests/
│   │   ├── conftest.py              # Test configuration and fixtures
│   │   ├── unit/
│   │   │   ├── services/            # Service-level test modules
│   │   │   │   ├── test_embeddings.py
│   │   │   │   ├── test_document_processor.py
│   │   │   │   ├── test_llm.py
│   │   │   │   └── test_vectorstore.py
│   │   │   └── api/
│   │   │       └── test_health.py   # Initial API endpoint test
│   │   └── data/                    # Test data resources
│   │       ├── documents/
│   │       │   ├── sample.txt
│   │       │   └── sample.pdf
├── frontend/
│   ├── src/
│   │   ├── __tests__/
│   │   │   ├── components/
│   │   │   │   ├── ChatInput.test.tsx
│   │   │   │   ├── MessageList.test.tsx
│   │   │   │   └── DocumentUpload.test.tsx
│   │   │   └── utils/
│   │   │       └── text.test.ts
│   ├── jest.config.js
│   └── setupTests.js
```

## Backend Test Configuration

### Install Test Dependencies

```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-benchmark
```

### Configure pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --cov=app --cov-report=term-missing
markers =
    unit: unit tests
    integration: integration tests
    performance: performance and resource optimization tests
    memory: memory usage tests
```

### Implement Resource-Optimized Fixtures

```python
"""
Test configuration and fixtures for the Mini-RAG backend tests.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import numpy as np
import json

from config import settings

# Override settings for testing to use smaller models and memory footprint
@pytest.fixture(scope="session", autouse=True)
def configure_test_settings():
    """Configure test-specific settings to optimize resource usage during testing."""
    settings.CHUNK_SIZE = 100  # Smaller chunks for testing
    settings.CHUNK_OVERLAP = 10
    settings.DEFAULT_BATCH_SIZE = 2  # Small batch size to test batching logic
    settings.MODEL_N_CTX = 512  # Smaller context window for testing
    settings.MEMORY_SAFETY_MARGIN_MB = 128  # Smaller safety margin for testing
    
    # Create and use temporary directories for test data
    original_vector_db_path = settings.VECTOR_DB_PATH
    original_model_path = settings.MODEL_PATH
    
    with tempfile.TemporaryDirectory() as vector_temp_dir:
        settings.VECTOR_DB_PATH = vector_temp_dir
        yield
        
    # Reset settings after tests
    settings.VECTOR_DB_PATH = original_vector_db_path
    settings.MODEL_PATH = original_model_path

@pytest.fixture
def sample_text_content():
    """Sample text content for testing document processing."""
    return """
    Mini-RAG is a lightweight Retrieval-Augmented Generation system designed for 
    resource-efficient operation. It provides document processing, embedding generation, 
    semantic search and LLM-enhanced responses with minimal computational requirements.
    
    The system implements memory optimization techniques including:
    - Lazy loading of ML models
    - Batch processing with configurable sizes
    - Explicit memory cleanup mechanisms
    - Document chunking with controlled overlap
    """

@pytest.fixture
def sample_text_file(sample_text_content):
    """Create a temporary text file with sample content."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
        temp.write(sample_text_content.encode('utf-8'))
        temp.flush()
        temp_path = temp.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

# Additional fixtures for service mocking
@pytest.fixture
def mock_sentence_transformer():
    """Mock the SentenceTransformer class for testing embedding service."""
    with patch('app.services.embeddings.SentenceTransformer') as mock:
        model = MagicMock()
        # Generate deterministic fake embeddings for consistent testing
        model.encode.return_value = np.random.RandomState(42).rand(3, 384)
        mock.return_value = model
        yield mock

@pytest.fixture
def mock_llama_cpp():
    """Mock the llama_cpp module for testing LLM service."""
    with patch('app.services.llm.llama_cpp') as mock:
        # Configure mock to return predictable responses
        llama_instance = MagicMock()
        llama_instance.return_value = {
            "choices": [{"text": "This is a mock LLM response for testing."}]
        }
        mock.Llama.return_value = llama_instance
        
        # Configure streaming response
        def streaming_generator(*args, **kwargs):
            responses = [
                {"choices": [{"text": "This "}]},
                {"choices": [{"text": "is "}]},
                {"choices": [{"text": "a mock "}]},
                {"choices": [{"text": "LLM "}]},
                {"choices": [{"text": "response."}]}
            ]
            for r in responses:
                yield r
                
        llama_instance.side_effect = streaming_generator
        
        yield mock

@pytest.fixture
def mock_chromadb():
    """Mock the ChromaDB client for testing vector store."""
    with patch('app.services.vectorstore.chromadb') as mock:
        # Configure mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2', 'doc3']],
            'documents': [['document content 1', 'document content 2', 'document content 3']],
            'metadatas': [[json.dumps({'source': 'test.txt'}), 
                          json.dumps({'source': 'test.pdf'}),
                          json.dumps({'source': 'test.md'})]]
        }
        
        # Configure mock client
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock.Client.return_value = mock_client
        
        yield mock, mock_collection
```

## Backend Unit Tests Implementation

### Core Testing Domains

- **Service Component Testing**: Validates individual service modules with controlled dependencies
- **API Interface Testing**: Verifies endpoint behavior, response structure, and error handling
- **Resource Optimization Testing**: Confirms memory management, efficient processing, and cleanup mechanisms

### Implementation Examples

#### Embedding Service Tests

```python
"""
Unit tests for the Embedding service.
"""

import pytest
import time
import gc
from unittest.mock import patch, MagicMock
import numpy as np

from app.services.embeddings import EmbeddingService, embeddings_service

@pytest.mark.unit
class TestEmbeddingService:
    
    def test_lazy_loading(self, mock_sentence_transformer):
        """Test that the model is only loaded when accessed."""
        service = EmbeddingService()
        
        # Model should not be loaded initially
        assert service._model is None
        assert mock_sentence_transformer.call_count == 0
        
        # Access model property to trigger loading
        _ = service.model
        
        # Model should now be loaded
        assert service._model is not None
        assert mock_sentence_transformer.call_count == 1
        
        # Second access shouldn't reload
        _ = service.model
        assert mock_sentence_transformer.call_count == 1
    
    def test_unload_model_if_inactive(self, mock_sentence_transformer):
        """Test model unloading after inactivity period."""
        service = EmbeddingService()
        
        # Load model
        _ = service.model
        assert service._model is not None
        
        # Set last used time to past
        service.last_used_time = time.time() - 4000  # 4000 seconds ago
        
        # Unload model
        service.unload_model_if_inactive(threshold_seconds=3600)  # 1 hour threshold
        
        # Model should be unloaded
        assert service._model is None
    
    def test_generate_embeddings_batching(self, mock_sentence_transformer):
        """Test that generate_embeddings uses batching correctly."""
        service = EmbeddingService()
        
        # Generate test texts
        texts = [f"Document {i}" for i in range(5)]
        
        # Process with batch size 2
        embeddings = service.generate_embeddings(texts, batch_size=2)
        
        # Verify results
        assert len(embeddings) == 5
        model = service.model  # Access model property
        
        # Should be called 3 times with batch sizes [2, 2, 1]
        assert model.encode.call_count == 3
```

#### Document Processor Tests

```python
"""
Unit tests for the Document Processor service.
"""

import pytest
import os
import tempfile
from unittest.mock import patch

from app.services.document_processor import DocumentProcessor, document_processor

@pytest.mark.unit
class TestDocumentProcessor:
    
    def test_split_text_into_chunks(self):
        """Test the text chunking functionality."""
        processor = DocumentProcessor()
        
        # Create a long repeating text
        text = "This is a test sentence. " * 20
        
        # Set small chunk size and overlap for testing
        chunk_size = 50
        overlap = 10
        
        # Split into chunks
        chunks = processor._split_text_into_chunks(text, chunk_size, overlap)
        
        # Verify results
        assert len(chunks) > 1
        
        # Verify chunk sizes
        for chunk in chunks:
            assert len(chunk) <= chunk_size
            
        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Check for overlap
            end_of_chunk = chunk[-overlap:] if len(chunk) > overlap else chunk
            start_of_next = next_chunk[:overlap] if len(next_chunk) > overlap else next_chunk
            
            assert end_of_chunk in start_of_next or start_of_next in end_of_chunk
```

## Frontend Test Configuration

### Jest Configuration (jest.config.js)

```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  roots: ['<rootDir>/src'],
  transform: {
    '^.+\\.tsx?$': 'ts-jest',
  },
  testRegex: '(/__tests__/.*|(\\.|/)(test|spec))\\.tsx?$',
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  moduleNameMapper: {
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  setupFilesAfterEnv: ['<rootDir>/setupTests.js'],
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/index.tsx',
  ],
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 70,
      lines: 70,
      statements: 70,
    },
  },
};
```

### Setup File Configuration (setupTests.js)

```javascript
// Setup file for Jest tests
import '@testing-library/jest-dom';

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Mock matchMedia for responsive design testing
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Suppress console errors during tests to keep output clean
jest.spyOn(console, 'error').mockImplementation(() => {});
```

## Frontend Component Tests

### Testing Strategies

- **Isolated Component Testing**: Validates rendering, props handling, and state management
- **Interaction Testing**: Verifies user input handling and event propagation
- **UI State Validation**: Confirms loading states, error handling, and conditional rendering

### Implementation Examples

#### ChatInput Component Tests

```tsx
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatInput from '../../components/ChatInput';

describe('ChatInput Component', () => {
  const mockSubmit = jest.fn();
  
  beforeEach(() => {
    mockSubmit.mockClear();
  });
  
  test('renders with default placeholder', () => {
    render(<ChatInput onSubmit={mockSubmit} isLoading={false} />);
    
    const textarea = screen.getByRole('textbox', { name: /message input/i });
    expect(textarea).toBeInTheDocument();
    expect(textarea).toHaveAttribute('placeholder', 'Ask a question about your documents...');
  });
  
  test('handles text input', async () => {
    render(<ChatInput onSubmit={mockSubmit} isLoading={false} />);
    
    const textarea = screen.getByRole('textbox');
    await userEvent.type(textarea, 'test question');
    
    expect(textarea).toHaveValue('test question');
  });
  
  test('handles form submission', async () => {
    render(<ChatInput onSubmit={mockSubmit} isLoading={false} />);
    
    const textarea = screen.getByRole('textbox');
    await userEvent.type(textarea, 'test query');
    
    const button = screen.getByRole('button', { name: /send/i });
    await userEvent.click(button);
    
    expect(mockSubmit).toHaveBeenCalledWith('test query');
    expect(textarea).toHaveValue('');
  });
  
  test('disables input and button when loading', () => {
    render(<ChatInput onSubmit={mockSubmit} isLoading={true} />);
    
    expect(screen.getByRole('textbox')).toBeDisabled();
    expect(screen.getByRole('button')).toBeDisabled();
    expect(screen.getByText(/processing/i)).toBeInTheDocument();
  });
```

#### Text Utilities Tests

```typescript
import { truncateText, highlightSearchTerms, getFileExtension, formatFileSize } from '../../utils/text';

describe('Text Utility Functions', () => {
  describe('truncateText', () => {
    test('returns original text when shorter than maxLength', () => {
      const text = 'Short text';
      expect(truncateText(text, 20)).toBe(text);
    });
    
    test('truncates text when longer than maxLength', () => {
      const text = 'This is a long text that should be truncated';
      expect(truncateText(text, 20)).toBe('This is a long text...');
    });
    
    test('uses default maxLength when not specified', () => {
      const text = 'A'.repeat(250);
      expect(truncateText(text)).toBe('A'.repeat(200) + '...');
    });
  });
  
  describe('formatFileSize', () => {
    test('formats bytes correctly', () => {
      expect(formatFileSize(0)).toBe('0 Bytes');
      expect(formatFileSize(500)).toBe('500 Bytes');
    });
    
    test('formats kilobytes correctly', () => {
      expect(formatFileSize(1024)).toBe('1 KB');
      expect(formatFileSize(2048)).toBe('2 KB');
    });
    
    test('formats megabytes correctly', () => {
      expect(formatFileSize(1048576)).toBe('1 MB');
      expect(formatFileSize(5242880)).toBe('5 MB');
    });
  });
});
```

## Test Data Resources

### Sample Text Document (tests/data/documents/sample.txt)

```
Mini-RAG System: Technical Overview

Mini-RAG is a lightweight Retrieval-Augmented Generation (RAG) system designed for resource-efficient operation. It combines document processing, semantic search, and language model generation with optimized memory usage.

Key Features:
- Document chunking with configurable sizes and overlaps
- Embedding generation using sentence-transformers
- Vector storage with ChromaDB for efficient retrieval
- Local LLM integration with phi-2.gguf
- Memory-optimized processing pipeline
- TypeScript/React frontend with responsive design
- Python/FastAPI backend with resource monitoring

The system implements several memory optimization techniques:
1. Lazy loading of model resources
2. Batch processing with adaptive sizes
3. Explicit memory cleanup after operations
4. Resource monitoring and throttling
5. Disk-based persistence for vector data

Mini-RAG provides a migration pathway to Azure cloud services while maintaining local execution capabilities for development and testing.
```

## Implementation Instructions

### Directory Structure Setup

```bash
mkdir -p backend/tests/{unit/{services,api},data/documents}
mkdir -p frontend/src/__tests__/{components,utils}
```

### Backend Implementation

1. Create test configuration files
   - pytest.ini in backend directory
   - conftest.py in backend/tests directory

2. Create unit test modules
   - test_embeddings.py in backend/tests/unit/services
   - test_document_processor.py in backend/tests/unit/services
   - test_llm.py in backend/tests/unit/services
   - test_vectorstore.py in backend/tests/unit/services
   - test_health.py in backend/tests/unit/api

3. Create sample data files
   - sample.txt in backend/tests/data/documents

4. Update requirements.txt
   ```
   pytest==7.3.1
   pytest-asyncio==0.21.0
   pytest-cov==4.1.0
   pytest-mock==3.10.0
   pytest-benchmark==4.0.0
   ```

### Frontend Implementation

1. Create test configuration files
   - jest.config.js in frontend directory
   - setupTests.js in frontend directory

2. Create component test files
   - ChatInput.test.tsx in frontend/src/__tests__/components
   - MessageList.test.tsx in frontend/src/__tests__/components
   - DocumentUpload.test.tsx in frontend/src/__tests__/components

3. Create utility test files
   - text.test.ts in frontend/src/__tests__/utils

4. Update package.json
   ```json
   "scripts": {
     "test": "react-scripts test",
     "test:coverage": "react-scripts test --coverage --watchAll=false"
   },
   "devDependencies": {
     "@testing-library/jest-dom": "^5.16.5",
     "@testing-library/react": "^13.4.0",
     "@testing-library/user-event": "^14.4.3",
     "identity-obj-proxy": "^3.0.0",
     "ts-jest": "^29.1.0"
   }
   ```

## Execution Instructions

### Backend Tests

```bash
# Run all unit tests
cd backend
python -m pytest tests/unit -v

# Run with coverage report
python -m pytest tests/unit --cov=app --cov-report=term-missing
```

### Frontend Tests

```bash
# Run all tests with watch mode
cd frontend
npm run test

# Run with coverage report
npm run test:coverage
```

## Technical Implementation Considerations

### Memory Optimization

- Test fixtures use temporary directories for vector storage
- Configuration overrides reduce model sizes during testing
- Batch sizes are reduced to ensure memory efficiency
- Resource monitoring is validated in health endpoint tests

### Resource Efficiency

- Mock implementations avoid loading actual models
- Fixed random seeds ensure deterministic testing
- Targeted test scope minimizes execution time
- Component isolation prevents unnecessary resource usage

## Phase 1 Testing Benefits

- Validates memory optimization techniques in core services
- Establishes patterns for resource-efficient testing
- Provides immediate feedback on critical components
- Creates foundation for more comprehensive test suites
- Ensures alignment with architectural principles

This implementation plan establishes a robust foundation for Mini-RAG system testing, with emphasis on validating resource optimization and memory management capabilities across all components.
