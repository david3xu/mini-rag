# Debugging the Mini-RAG Document Query Workflow Test

## The Problem

The integration test `test_document_processing_and_query` is failing with an assertion error. When querying for "What are the memory optimization techniques in Mini-RAG?", the test expects to find sources from the uploaded test document, but the assertion fails:

```python
source_matched = False
for source in query_result['sources']:
    if os.path.basename(self.document_path) in source.get('metadata', {}).get('source', ''):
        source_matched = True
        break

assert source_matched, "Query didn't return sources from our document"
```

This suggests that the vector search is not returning the expected document, or the metadata format doesn't match what the test is looking for.

## Root Causes and Solutions

### 1. Vector Store Search Method Silently Fails

**Issue:** In `backend/app/services/vectorstore.py`, the `search` method catches all exceptions and returns an empty list instead of raising errors:

```python
except Exception as e:
    logger.error(f"Error searching vector store: {str(e)}")
    logger.error(f"Stack trace: {traceback.format_exc()}")
    return []  # Return empty results rather than failing completely
```

This hides potential errors that could explain why documents aren't found.

**Solution:**

```python
def search(self, query_embedding, k=3, filter_metadata=None, timeout_ms=3000):
    try:
        # Existing search code...
        
    except Exception as e:
        logger.error(f"Error searching vector store: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        logger.error(f"Search parameters: k={k}, filter={filter_metadata}")
        
        # During testing, we should raise the exception to debug
        if 'pytest' in sys.modules:
            raise
            
        return []  # In production, still return empty to avoid breaking the app
```

### 2. Metadata Source Path Format Mismatch

**Issue:** The test looks for `os.path.basename(self.document_path)` in the source field, but the document processor might store full paths or use a different format.

**Solution:**

```python
source_matched = False
for source in query_result['sources']:
    source_value = source.get('metadata', {}).get('source', '')
    # Check if the source contains either the full path or just the basename
    if (self.document_path in source_value or 
        os.path.basename(self.document_path) in source_value):
        source_matched = True
        break
```

Adding debug output helps identify the exact format:

```python
# Add before the assertion
print("\nDebug information:")
print(f"Document path: {self.document_path}")
print(f"Document basename: {os.path.basename(self.document_path)}")
print("Sources returned:")
for i, source in enumerate(query_result['sources']):
    print(f"Source {i+1} metadata: {source.get('metadata', {})}")
```

### 3. ChromaDB Collection Query Timeout

**Issue:** The `timeout_ms` parameter might be too short for the search to complete, particularly on slower systems.

**Solution:**

```python
# Increase timeout in backend/app/services/vectorstore.py
def search(self, query_embedding, k=3, filter_metadata=None, timeout_ms=10000):  # Increased from 3000
    # Existing code...
```

### 4. Document Processing and Chunking Issues

**Issue:** The document might be getting chunked in a way that separates the key phrases needed to match the query.

**Solution:**

```python
# Add debug code to inspect chunks during the test
print("\nDocument chunks created:")
from app.services.document_processor import document_processor
chunks = document_processor.process_file(self.document_path)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk['text'][:100]}...")
    print(f"  ID: {chunk['id']}")
    print(f"  Metadata: {chunk['metadata']}")
```

### 5. Collection Lock Preventing Access

**Issue:** The `collection_lock` in vectorstore.py might be preventing concurrent access during testing.

**Solution:**

```python
# Modify search to use collection direct access for tests
def search(self, query_embedding, k=3, filter_metadata=None, timeout_ms=3000):
    try:
        # Access collection without lock to ensure we have it initialized
        collection = self.client.get_collection(name=self.collection_name)
        
        # Rest of the search code...
```

### 6. JSON Handling in Metadata

**Issue:** ChromaDB might be storing metadata as JSON strings instead of Python dictionaries.

**Solution:**

```python
# Ensure proper metadata handling in search result processing
metadata = {}
if results.get('metadatas') and results['metadatas'][0][i]:
    # Handle both string and dictionary metadata
    meta_value = results['metadatas'][0][i]
    if isinstance(meta_value, str):
        try:
            metadata = json.loads(meta_value)
        except json.JSONDecodeError:
            metadata = {"source": meta_value}
    else:
        metadata = meta_value
```

### 7. Missing Embedding Model During Testing

**Issue:** The embeddings of the query and document might not be similar enough using the test environment's model.

**Solution:**

```python
# Add a direct semantic match test before the real query
def test_document_processing_and_query(self):
    # After processing the document, before querying
    
    # Test direct semantic match
    test_query = "memory optimization techniques"
    query_embedding = embeddings_service.generate_embedding(test_query)
    
    # Search directly with this embedding
    direct_results = vector_store.search(query_embedding, k=5)  # Get more results
    print(f"\nDirect semantic search results for '{test_query}':")
    for i, result in enumerate(direct_results):
        print(f"Result {i+1}: {result['content'][:100]}...")
        
    # Then continue with the regular query test
```

## Complete Integration Test Fix

Here's a comprehensive fix that combines several solutions:

```python
def test_document_processing_and_query(self):
    """Test end-to-end document processing and querying."""
    print("\n=== Testing document processing and query workflow ===")
    
    # Create test client
    client = TestClient(app)
    
    # Step 1: Process a document
    print(f"Processing document: {self.document_path}")
    print(f"Document content sample: {self.temp_file.read(200).decode()}...")
    self.temp_file.seek(0)  # Reset file position
    
    with open(self.document_path, "rb") as file:
        response = client.post(
            "/api/documents/upload",
            files={"files": (os.path.basename(self.document_path), file, "text/plain")}
        )
    
    print(f"Document processing response status: {response.status_code}")
    assert response.status_code == 200
    process_result = response.json()
    print(f"Document upload response: {process_result}")
    
    # Verify document was properly processed into chunks
    from app.services.document_processor import document_processor
    chunks = document_processor.process_file(self.document_path)
    print(f"Document was processed into {len(chunks)} chunks")
    
    # Print chunk content to verify memory optimization is included
    for i, chunk in enumerate(chunks[:2]):  # Just show first 2 chunks
        print(f"Chunk {i+1}: {chunk['text'][:100]}...")
        print(f"  Metadata: {chunk['metadata']}")
    
    # Get document count in vector store 
    from app.services.vectorstore import vector_store
    count = vector_store.collection.count()
    print(f"Vector store contains {count} documents")
    
    # Step 2: Try direct semantic match first
    from app.services.embeddings import embeddings_service
    test_query = "memory optimization techniques"
    print(f"\nTrying direct semantic search for: '{test_query}'")
    query_embedding = embeddings_service.generate_embedding(test_query)
    
    # Search with higher limit and longer timeout
    direct_results = vector_store.search(
        query_embedding, 
        k=5,         # Retrieve more results
        timeout_ms=15000  # Longer timeout
    )
    print(f"Direct search returned {len(direct_results)} results")
    
    # Print results to see if our content is there
    for i, result in enumerate(direct_results):
        print(f"Result {i+1}: {result['content'][:100]}...")
        print(f"  Source: {result.get('metadata', {}).get('source', 'N/A')}")
    
    # Step 3: Query through the API
    query_text = "What are the memory optimization techniques in Mini-RAG?"
    print(f"\nQuerying with: '{query_text}'")
    query_request = QueryRequest(query=query_text)
    response = client.post("/api/chat", json=query_request.dict())
    
    print(f"Query response status: {response.status_code}")
    assert response.status_code == 200
    query_result = response.json()
    
    # Verify response structure
    assert "answer" in query_result
    assert "sources" in query_result
    print(f"Query answer: {query_result['answer']}")
    print(f"Sources: {len(query_result['sources'])} found")
    
    # Print all sources for debugging
    for i, source in enumerate(query_result['sources']):
        print(f"Source {i+1}: {source['content'][:100]}...")
        print(f"  Metadata: {source['metadata']}")
    
    # Verify content
    assert len(query_result['sources']) > 0
    assert "memory" in query_result['answer'].lower()
    assert "optimization" in query_result['answer'].lower()
    
    # At least one source should be from our document
    document_basename = os.path.basename(self.document_path)
    print(f"\nLooking for document with basename: {document_basename}")
    
    source_matched = False
    for source in query_result['sources']:
        source_path = source.get('metadata', {}).get('source', '')
        print(f"Checking source: {source_path}")
        
        # Check if source contains either the full path or the basename
        if document_basename in source_path or self.document_path in source_path:
            source_matched = True
            print("✓ Match found!")
            break
    
    assert source_matched, "Query didn't return sources from our document"
    print("✓ Document query workflow test passed")
```

## Improved Vector Store Implementation

To address the underlying issues in the vector store implementation:

```python
# In backend/app/services/vectorstore.py

def search(self, query_embedding, k=3, filter_metadata=None, timeout_ms=10000):
    """Search for similar documents with robust timeout handling."""
    if not query_embedding:
        logger.warning("Empty query embedding provided for search")
        raise ValueError("Query embedding cannot be empty")
    
    # Enforce reasonable limits
    k = min(k, 10)  # Cap at 10 results for resource efficiency
    
    # Memory optimization before search
    self._check_available_memory()
    gc.collect()  # Force garbage collection to optimize memory
    
    try:
        # Access collection without lock to reduce contention
        collection = self.client.get_collection(name=self.collection_name)
        
        # Check if collection is empty
        count = collection.count()
        if count == 0:
            logger.warning("Vector store is empty - no documents to search")
            return []
                
        logger.info(f"Searching for top {k} documents among {count} total documents")
        
        # Prepare metadata filter
        filter_dict = filter_metadata if filter_metadata else None
        
        # Direct collection query
        start_time = time.time()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict,
            include=['metadatas', 'documents', 'distances']
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Search completed in {elapsed_ms:.2f}ms")
        
        # Format results
        documents = []
        if results and 'ids' in results and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                try:
                    # Extract metadata with better error handling
                    metadata = {}
                    if results.get('metadatas') and results['metadatas'][0][i]:
                        meta_val = results['metadatas'][0][i]
                        # Handle string or dict metadata
                        if isinstance(meta_val, str):
                            try:
                                metadata = json.loads(meta_val)
                            except json.JSONDecodeError:
                                metadata = {"source": meta_val}
                        else:
                            metadata = meta_val
                    
                    # Build document result
                    documents.append({
                        "id": doc_id,
                        "content": results['documents'][0][i],
                        "metadata": metadata,
                        "distance": results.get('distances', [[0]])[0][i] if results.get('distances') else None
                    })
                    
                    # Debug log the source for investigation
                    logger.debug(f"Source metadata: {metadata}")
                    
                except Exception as item_error:
                    logger.error(f"Error processing search result {i}: {str(item_error)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
        
        logger.info(f"Search returned {len(documents)} documents")
        return documents
            
    except Exception as e:
        logger.error(f"Error searching vector store: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # For testing environments, raise the exception
        if 'pytest' in sys.modules:
            raise
            
        return []  # Return empty results in production
```

## Additional Recommendations

1. **Explicit Document Tagging**: Add a unique test tag to documents uploaded during testing:
   ```python
   metadata = {"source": file_path, "chunk": i, "test_tag": "integration_test_document"}
   ```

2. **Increase Test Wait Time**: Allow more time for document processing to complete:
   ```python
   # Wait longer for background processing to complete
   time.sleep(10)  # Increased from 5
   ```

3. **Force Synchronous Processing**: For testing purposes, use synchronous document processing:
   ```python
   # Modify the upload endpoint for test-only
   if 'pytest' in sys.modules:
       # Process synchronously for tests
       background_tasks = None
   ```

4. **Improve Test Environment Isolation**: Create a separate test database instance:
   ```python
   # In conftest.py
   @pytest.fixture(scope="function")
   def test_vector_store():
       """Create an isolated vector store for testing."""
       with tempfile.TemporaryDirectory() as temp_dir:
           vs = VectorStoreService(persist_directory=temp_dir)
           yield vs
   ```

5. **Logging Enhancement**: Add more detailed logging during test execution:
   ```python
   # Set logging level higher during tests
   logging.basicConfig(
       level=logging.DEBUG,
       format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   )
   ```

## Conclusion

The test failure is likely caused by one or more of the issues outlined above. The most probable causes are:

1. Metadata format mismatches between what's stored and what the test is checking
2. Search timeout or error handling issues that hide failures
3. Document chunking that separates key information
4. Concurrent access issues with the collection lock

Implementing the suggested fixes should help identify and resolve the underlying issue. The comprehensive debug test will provide visibility into what's happening and help pinpoint the exact problem.
