# Mini-RAG System Performance Optimization

## Executive Summary

The Mini-RAG system is currently experiencing performance issues, including slow response times and timeouts during operation. Analysis reveals that memory management, resource utilization, and concurrent operations are the primary bottlenecks. This document outlines a comprehensive optimization strategy focused on improving performance while maintaining the existing architecture.

## Identified Issues

1. **LLM Loading and Inference Latency**: The Phi-2 model is experiencing long initialization times and slow inference.
2. **Vector Search Performance**: ChromaDB operations are inefficient with larger document collections.
3. **Resource Constraints**: The system exceeds available memory during concurrent operations.
4. **Request Timeouts**: Long-running operations trigger timeouts before completion.

## Recommended Optimizations

### 1. LLM Service Optimization

The LLM service should be enhanced with:

- Memory check before model loading
- Automatic model unloading after inactivity
- Thread and CPU count optimization
- Timeout handling for long-running generations
- More efficient prompt handling with token limits
- Background cleanup thread

### 2. Vector Store Performance Optimization

The vector store service should include:

- Memory checks before operations
- Dynamic batch size based on available memory
- Timeout handling for queries
- Metadata caching for faster retrieval
- Automatic resource cleanup after inactivity
- Quick search method for testing
- Process monitoring with timing logs

### 3. Testing and Diagnostic Endpoints

Add lightweight endpoints for isolated component testing:

- LLM-only testing endpoint
- Vector store-only testing endpoint
- Combined lightweight testing
- Simple ping test for connectivity
- Detailed timing information

### 4. Configuration Adjustments

Optimize configuration settings:

- Reduce context window size (512 tokens instead of default)
- Optimize batch processing parameters
- Implement memory safety margins
- Configure timeout thresholds
- Limit thread count

## Implementation Plan

### Phase 1: Add Testing Endpoints

1. Create `backend/app/api/test.py` with the diagnostic endpoints
2. Update `backend/main.py` to include the test router
3. Test the endpoints to isolate problem components

### Phase 2: Optimize LLM Service

1. Update `backend/app/services/llm.py` with the optimized implementation
2. Enable the model unloading mechanism
3. Add memory checks and timeout handling

### Phase 3: Optimize Vector Store

1. Update `backend/app/services/vectorstore.py` with memory-optimized implementation
2. Implement dynamic batch sizing
3. Add metadata caching and quick search methods

### Phase 4: Update Configuration

1. Update `.env` file with optimized settings
2. Reduce context window and batch parameters
3. Configure memory safety margins

## Testing Procedure

Use the provided test script to evaluate system performance:

```bash
#!/bin/bash
# Run tests in sequence from simplest to most complex

# Test 1: Simple ping to verify server responsiveness
curl -s http://localhost:8000/healthz | jq

# Test 2: Health check for system components
curl -s http://localhost:8000/api/health | jq

# Test 3: LLM-only endpoint (no vector search)
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "mode": "llm"}' \
  http://localhost:8000/api/test/simple | jq

# Test 4: Vector search only (no LLM)
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "memory optimization", "mode": "vector"}' \
  http://localhost:8000/api/test/simple | jq

# Test 5: Simple combined test (lightweight RAG)
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about Mini-RAG", "mode": "combined"}' \
  http://localhost:8000/api/test/simple | jq
```

## Expected Outcomes

After implementing these optimizations:

1. **Response Times**: Faster responses with reduced latency
2. **Memory Usage**: Lower peak memory consumption
3. **Stability**: Fewer timeouts and errors
4. **Scalability**: Better handling of concurrent requests
5. **Resource Efficiency**: Optimized use of CPU and memory

## Key Optimization Patterns

The implementation follows these core optimization patterns:

1. **Lazy Loading**: Only load resources when needed
2. **Resource Cleanup**: Automatically unload inactive components
3. **Batch Processing**: Process documents in memory-efficient batches
4. **Memory Monitoring**: Check available memory before operations
5. **Timeout Handling**: Prevent long-running operations
6. **Dynamic Resource Allocation**: Adjust parameters based on system state
7. **Component Isolation**: Allow testing individual components

## Compatibility with llama-cpp-python

The optimizations align with llama-cpp-python capabilities:

1. **Context Window Control**: Reducing `n_ctx` to 512 (from default)
2. **Thread Management**: Limiting thread count with `n_threads`
3. **Memory Options**: Using `use_mlock=False` and `use_mmap=True`
4. **Batching Support**: Implementing efficient batch processing
5. **Resource Management**: Proper cleanup and initialization

## Conclusion

These optimizations address core performance issues while maintaining compatibility with the existing codebase. The focus on memory efficiency and component isolation should significantly improve response times and system stability.

Start by implementing the test endpoints to isolate issues, then proceed with the core service optimizations. Use the streaming endpoints for better user experience during long-running operations.
