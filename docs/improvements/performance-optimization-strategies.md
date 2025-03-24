# Mini-RAG Performance Optimization Guide

This document outlines strategies to optimize the Mini-RAG system response time under CPU constraints without changing the underlying model. These optimizations focus on system architecture, resource management, and implementation techniques.

## Table of Contents

1. [Computational Optimizations](#computational-optimizations)
2. [Caching Strategies](#caching-strategies)
3. [Memory Management](#memory-management)
4. [Thread Optimization](#thread-optimization)
5. [Context Window Tuning](#context-window-tuning)
6. [Prompt Engineering](#prompt-engineering)
7. [Implementation Roadmap](#implementation-roadmap)

## Computational Optimizations

### Enable BLAS Optimizations for llama-cpp-python

BLAS (Basic Linear Algebra Subprograms) can significantly accelerate matrix operations critical for inference:

```bash
# Uninstall current version
pip uninstall -y llama-cpp-python

# Reinstall with BLAS optimizations
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
```

This optimization leverages highly optimized linear algebra routines which can improve matrix multiplication speed by 2-3x on CPU.

### Leverage CPU-Specific Instructions

Enable processor-specific optimizations based on your hardware:

```bash
# For systems with AVX2 instruction support
CMAKE_ARGS="-DLLAMA_AVX2=ON -DLLAMA_F16C=ON -DLLAMA_FMA=ON" pip install llama-cpp-python

# For newer CPUs with AVX-512 support
CMAKE_ARGS="-DLLAMA_AVX512=ON" pip install llama-cpp-python
```

### Implement KV Cache Reuse

Add key-value cache reuse to `backend/app/services/llm.py` for similar or follow-up queries:

```python
def __init__(self):
    # Existing code...
    self._kv_cache = None
    self._last_prompt = None

def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                  stream: bool = False, timeout_seconds: int = 30, reuse_cache: bool = True):
    # Existing code...
    
    # Check if we can reuse KV cache
    cache_hit = False
    if reuse_cache and self._kv_cache is not None and self._last_prompt is not None:
        # Reuse cache if the new prompt starts with the previous prompt
        if prompt.startswith(self._last_prompt) and len(prompt) - len(self._last_prompt) < 20:
            cache_hit = True
            logger.info("Reusing KV cache for similar prompt")
    
    try:
        # Generate response
        if not stream:
            # Memory optimization: before large operation
            gc.collect()
            
            # Set cache parameters
            cache_params = {}
            if cache_hit:
                cache_params = {"cache": self._kv_cache}
                
            response = self.llm(
                formatted_prompt,
                max_tokens=actual_max_tokens,
                temperature=temperature,
                echo=False,
                stop=["</s>", "<s>"],
                **cache_params
            )
            
            # Store KV cache and prompt for potential reuse
            if not cache_hit:
                self._kv_cache = response.get("cache", None)
                self._last_prompt = prompt
            
            return response["choices"][0]["text"].strip()
```

## Caching Strategies

### Response Caching

Implement a simple LRU cache for common queries in `backend/app/services/cache.py`:

```python
import functools
from typing import Dict, Any, Optional, Tuple
import time

# Simple LRU cache implementation
class ResponseCache:
    def __init__(self, max_size=100, ttl=3600):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response if it exists and hasn't expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Add or update cache entry."""
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())

# Create singleton instance
response_cache = ResponseCache()
```

Then integrate in `backend/app/api/chat.py`:

```python
from app.services.cache import response_cache
import hashlib

@router.post("", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a chat query using RAG."""
    try:
        # Generate cache key from query
        cache_key = hashlib.md5(request.query.encode()).hexdigest()
        
        # Check cache first
        cached_response = response_cache.get(cache_key)
        if cached_response:
            print(f"[DEBUG] Cache hit for query: {request.query}")
            return cached_response
            
        # Continue with normal processing...
        # [existing code]
        
        # Store in cache before returning
        response = QueryResponse(
            answer=answer,
            sources=sources
        )
        response_cache.set(cache_key, response)
        
        return response
    except Exception as e:
        # [error handling code]
```

### Vector Search Results Caching

Cache common vector search results to avoid repeated embedding and search operations:

```python
# In backend/app/services/vectorstore.py

def __init__(self):
    # Existing initialization code...
    self._search_cache = {}  # Simple cache for search results
    self._search_cache_size = 50  # Maximum number of cached searches

def search(self, query_embedding, k=3, filter_metadata=None, timeout_ms=3000):
    # Generate cache key
    cache_key = str(query_embedding[:5]) + str(k) + str(filter_metadata)
    
    # Check cache
    if cache_key in self._search_cache:
        logger.info("Using cached vector search results")
        return self._search_cache[cache_key]
    
    # Existing search code...
    
    # Cache results
    if len(self._search_cache) >= self._search_cache_size:
        # Remove a random key to avoid computation overhead
        if self._search_cache:
            self._search_cache.pop(next(iter(self._search_cache)))
    
    self._search_cache[cache_key] = documents
    return documents
```

### Pre-Computed Responses for Common Queries

Add a dictionary of pre-computed responses for frequently asked questions:

```python
# In backend/app/api/chat.py

# Initialize with pre-computed responses for common queries
PRECOMPUTED_RESPONSES = {
    "what is mini-rag": "Mini-RAG is a lightweight Retrieval-Augmented Generation system designed for resource-efficient operation. It combines document processing, semantic search, and language model generation with optimized memory usage.",
    "how does rag work": "Retrieval-Augmented Generation works by retrieving relevant information from a vector database based on semantic similarity to the user's query, then using this information as context for an LLM to generate an informed response.",
    "memory optimization": "Mini-RAG implements several memory optimization techniques including lazy loading of models, batch processing with adaptive sizes, explicit memory cleanup, and resource monitoring with throttling."
}

@router.post("", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a chat query using RAG."""
    try:
        # Check for pre-computed responses
        normalized_query = request.query.lower().strip()
        for key, response in PRECOMPUTED_RESPONSES.items():
            if normalized_query in key or key in normalized_query:
                print(f"[DEBUG] Using pre-computed response for query: {request.query}")
                return QueryResponse(
                    answer=response,
                    sources=[]  # No sources for pre-computed responses
                )
        
        # Continue with normal processing...
```

## Memory Management

### Implement Gradual Resource Release

Add timed cleanup for loaded resources to release memory when not actively serving requests:

```python
# In backend/app/services/llm.py

def __init__(self):
    # Existing initialization...
    self._last_access_time = time.time()
    self._cleanup_threshold = 300  # 5 minutes

def generate_text(self, prompt, *args, **kwargs):
    # Update last access time
    self._last_access_time = time.time()
    
    # Check if model needs to be loaded
    if self._llm is None:
        # Load model code...
    
    # Existing generation code...

# Add periodic cleanup check
def cleanup_resources_if_inactive(self):
    """Release resources if inactive for threshold period."""
    current_time = time.time()
    if (self._llm is not None and 
        current_time - self._last_access_time > self._cleanup_threshold):
        logger.info("Releasing LLM resources due to inactivity")
        self._llm = None
        gc.collect()
```

### Optimize Memory Allocation

Ensure precise memory allocation and deallocation:

```python
# In backend/app/services/embeddings.py

def generate_embeddings(self, texts, batch_size=None):
    # Existing code...
    
    try:
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            # Process batch
            batch = texts[i:min(i + batch_size, len(texts))]
            batch_embeddings = self.model.encode(batch)
            
            # Convert to list and append
            all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
            
            # Explicitly delete batch data
            del batch
            del batch_embeddings
            
            # Force garbage collection between batches
            if i % (batch_size * 3) == 0:
                gc.collect()
        
        return all_embeddings
    finally:
        # Ensure cleanup happens
        gc.collect()
```

## Thread Optimization

### Dynamic Thread Allocation

Adjust thread count based on query complexity for optimal performance:

```python
# In config.py
def get_optimal_threads(prompt_length, available_cpus=None):
    """Determine optimal thread count based on prompt length and system resources."""
    if available_cpus is None:
        available_cpus = os.cpu_count() or 1
    
    # For very short prompts, minimal threads
    if prompt_length < 100:
        return 1
    # For medium prompts, balanced approach
    elif prompt_length < 500:
        return min(2, available_cpus)
    # For long prompts, more threads but still constrained
    else:
        return min(4, available_cpus)
```

Then in `llm.py`:

```python
def generate_text(self, prompt, max_tokens=512, temperature=0.7, stream=False, timeout_seconds=30):
    # Existing code...
    
    # Dynamically set thread count based on query complexity
    from config import get_optimal_threads
    thread_count = get_optimal_threads(len(prompt))
    
    # Generate with optimized thread count
    response = self.llm(
        formatted_prompt,
        max_tokens=actual_max_tokens,
        temperature=temperature,
        echo=False,
        stop=["</s>", "<s>"],
        n_threads=thread_count  # Apply dynamic thread allocation
    )
```

### Asynchronous Processing

Improve handling of concurrent requests with proper async programming:

```python
# In backend/app/api/chat.py
import asyncio

@router.post("", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a chat query using RAG with improved concurrency."""
    try:
        # Create a semaphore to limit concurrent LLM operations
        # This should be a module-level variable in practice
        sem = asyncio.Semaphore(2)  # Maximum 2 concurrent LLM operations
        
        # Generate query embedding - this can happen without the semaphore
        query_embedding = embeddings_service.generate_embedding(request.query)
        
        # Retrieve documents - this can also happen without the semaphore
        documents = vector_store.search(query_embedding)
        
        # Format prompt
        prompt = llm_service.format_rag_prompt(request.query, documents)
        
        # LLM generation needs the semaphore
        async with sem:
            # Run in thread pool to avoid blocking
            answer = await asyncio.to_thread(
                llm_service.generate_text,
                prompt,
                max_tokens=512,
                temperature=0.7
            )
        
        # Format and return response
        sources = []
        for doc in documents:
            # Format source documents
            # [existing code]
            
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        # [error handling code]
```

## Context Window Tuning

### Reduce Context Window Size

Adjust the context window to reduce computational load:

```python
# In config.py
MODEL_N_CTX: int = 256  # Reduced from default for faster inference
```

### Limit Document Content

Optimize the amount of context provided to the model:

```python
# In llm.py, in format_rag_prompt method
def format_rag_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
    """Format a prompt for RAG."""
    # Limit to fewer documents for speed
    max_docs = min(2, len(documents))
    limited_docs = documents[:max_docs]
    
    # Limit context size further per document
    max_content_length = min(150, self._n_ctx // (max_docs * 2))
    context_parts = []
    
    for i, doc in enumerate(limited_docs):
        content = doc['content']
        if len(content) > max_content_length:
            # Extract only the most relevant portion
            # Simple approach: first N characters
            content = content[:max_content_length] + "..."
            
            # Advanced approach: extract sentences until limit
            # sentences = content.split('. ')
            # truncated = []
            # length = 0
            # for sentence in sentences:
            #     if length + len(sentence) <= max_content_length:
            #         truncated.append(sentence)
            #         length += len(sentence) + 2  # Account for ". "
            #     else:
            #         break
            # content = '. '.join(truncated) + ('...' if truncated else '')
            
        context_parts.append(f"Document {i+1}:\n{content}")
    
    context = "\n\n".join(context_parts)
    
    # Create more concise prompt
    prompt = f"<|user|>\nAnswer based on these documents:\n\n{context}\n\nQuestion: {query}\n<|assistant|>"
    
    return prompt
```

## Prompt Engineering

### Implement Tiered Response Approach

Create a new endpoint for quick responses with minimal context:

```python
# In backend/app/api/chat.py
@router.post("/quick")
async def process_quick_query(request: QueryRequest):
    """Quick response endpoint with reduced context."""
    try:
        # Generate query embedding
        query_embedding = embeddings_service.generate_embedding(request.query)
        
        # Get only the single best document match
        documents = vector_store.quick_search(query_embedding, k=1)
        
        if not documents:
            return {"answer": "I don't have enough information to answer that question.", "is_complete": False}
        
        # Use a simplified prompt
        doc_content = documents[0]['content']
        # Take just first 100 characters of context
        short_context = doc_content[:100] + ("..." if len(doc_content) > 100 else "")
        
        prompt = f"Question: {request.query}\nContext: {short_context}\nAnswer (briefly):"
        
        # Generate with minimal settings
        answer = llm_service.generate_text(
            prompt,
            max_tokens=50,  # Very limited tokens
            temperature=0.7
        )
        
        return {"answer": answer, "is_complete": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Optimize Prompt Structure

Implement more efficient prompt formats that reduce token usage:

```python
def format_efficient_rag_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
    """Create a token-efficient prompt for RAG."""
    # Extract key sentences only
    key_sentences = []
    for doc in documents[:2]:  # Limit to 2 documents
        content = doc['content']
        # Split into sentences
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
        # Take only first 2-3 most relevant sentences
        key_sentences.extend(sentences[:3])
    
    # Join key sentences
    context = '. '.join(key_sentences)
    
    # Very concise prompt format
    prompt = f"Q:{query} C:{context} A:"
    
    return prompt
```

## Implementation Roadmap

For best results, implement these optimizations in the following order:

1. **Immediate Gains**
   - Enable BLAS optimizations
   - Reduce context window and document content

2. **Short-term Improvements**
   - Implement response caching
   - Add dynamic thread allocation
   - Add pre-computed responses for common queries

3. **Medium-term Enhancements**
   - Implement KV cache reuse
   - Add vector search results caching
   - Create quick response endpoint

4. **Advanced Optimizations**
   - Implement asyncio-based processing
   - Add CPU-specific instruction optimizations
   - Implement tiered response approach

## Performance Monitoring

Track the impact of optimizations using these metrics:

```python
# Add to main.py or a separate monitoring.py
import time
from typing import Dict, List, Deque
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.response_times: Deque[float] = deque(maxlen=window_size)
        self.embedding_times: Deque[float] = deque(maxlen=window_size)
        self.search_times: Deque[float] = deque(maxlen=window_size)
        self.llm_times: Deque[float] = deque(maxlen=window_size)
    
    def record_response_time(self, elapsed: float):
        self.response_times.append(elapsed)
    
    def record_embedding_time(self, elapsed: float):
        self.embedding_times.append(elapsed)
    
    def record_search_time(self, elapsed: float):
        self.search_times.append(elapsed)
    
    def record_llm_time(self, elapsed: float):
        self.llm_times.append(elapsed)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        def calc_stats(times):
            if not times:
                return {"avg": 0, "min": 0, "max": 0, "p95": 0}
            
            times_list = list(times)
            times_list.sort()
            p95_idx = int(len(times_list) * 0.95)
            
            return {
                "avg": sum(times_list) / len(times_list),
                "min": min(times_list),
                "max": max(times_list),
                "p95": times_list[p95_idx] if times_list else 0
            }
        
        return {
            "response": calc_stats(self.response_times),
            "embedding": calc_stats(self.embedding_times),
            "search": calc_stats(self.search_times),
            "llm": calc_stats(self.llm_times)
        }

# Create singleton instance
performance_monitor = PerformanceMonitor()
```

Then add to your endpoints:

```python
@router.post("", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a chat query using RAG."""
    start_time = time.time()
    try:
        # Record embedding time
        embedding_start = time.time()
        query_embedding = embeddings_service.generate_embedding(request.query)
        performance_monitor.record_embedding_time(time.time() - embedding_start)
        
        # Record search time
        search_start = time.time()
        documents = vector_store.search(query_embedding)
        performance_monitor.record_search_time(time.time() - search_start)
        
        # Record LLM time
        llm_start = time.time()
        prompt = llm_service.format_rag_prompt(request.query, documents)
        answer = llm_service.generate_text(prompt)
        performance_monitor.record_llm_time(time.time() - llm_start)
        
        # [rest of the function]
        
        # Record overall response time
        performance_monitor.record_response_time(time.time() - start_time)
        
        return response
        
    except Exception as e:
        # [error handling]
```

Add a performance endpoint:

```python
@router.get("/performance")
async def get_performance_stats():
    """Get system performance statistics."""
    return performance_monitor.get_stats()
```

These optimizations, when implemented systematically, should significantly improve your Mini-RAG system's response time under CPU constraints without changing the underlying model.
