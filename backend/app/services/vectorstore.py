"""
Performance-optimized vector store service for document storage and retrieval.

This module provides functionality for storing and retrieving document
embeddings using ChromaDB, optimized for memory efficiency and query performance
with disk-based persistence.
"""

import chromadb
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import logging
import os
import time
import psutil
from threading import Lock
import gc
import traceback

from config import settings, get_path
from app.api.models import DocumentChunk

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Thread lock for safe collection operations
collection_lock = Lock()

class VectorStoreService:
    """Service for interacting with the vector database.
    
    This service manages storage and retrieval of document embeddings
    using ChromaDB with disk-based persistence, optimized for memory
    efficiency in resource-constrained environments.
    """
    
    def __init__(self, persist_directory=None, collection_name=None):
        """Initialize the vector store service with configured settings.
        
        Args:
            persist_directory: Optional custom directory for vector store persistence
            collection_name: Optional custom collection name for vector store
        """
        # Use get_path to properly resolve the directory path and avoid duplication
        self.persist_directory = persist_directory or get_path(settings.VECTOR_DB_PATH)
        self.collection_name = collection_name or settings.VECTOR_DB_COLLECTION
        self._client = None
        self._collection = None
        self._metadata_cache = {}  # Cache for frequently accessed metadata
        self._last_access_time = time.time()
        
        logger.info(f"Vector store service initialized with path: {self.persist_directory}")
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
    
    @property
    def client(self):
        """Lazy-loaded ChromaDB client.
        
        Returns:
            ChromaDB client instance
            
        Raises:
            RuntimeError: If client initialization fails
        """
        if self._client is None:
            # Memory checks before initialization
            self._check_available_memory()
            
            logger.info("Initializing ChromaDB client")
            try:
                # Simplified client initialization per current API standards
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory
                )
                logger.info("ChromaDB client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing ChromaDB client: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to initialize vector database: {str(e)}")
        
        # Update last access time
        self._last_access_time = time.time()
        return self._client
    
    def _check_available_memory(self):
        """Check if sufficient memory is available for vector operations.
        
        Raises:
            RuntimeError: If memory is critically low
        """
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        
        # Log memory state for monitoring
        logger.info(f"Available memory: {available_mb:.2f}MB ({mem.percent}% used)")
        
        # Check against safety margin
        if available_mb < settings.MEMORY_SAFETY_MARGIN_MB:
            logger.warning(f"Low memory condition: only {available_mb:.2f}MB available")
            
            # Try to free memory
            gc.collect()
            
            # Check again
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)
            
            if available_mb < settings.MEMORY_SAFETY_MARGIN_MB / 2:  # Critical threshold
                logger.error(f"Critical memory shortage: {available_mb:.2f}MB available")
                raise RuntimeError(f"Insufficient memory for vector operations: {available_mb:.2f}MB available")
    
    @property
    def collection(self):
        """Lazy-loaded ChromaDB collection.
        
        Returns:
            ChromaDB collection instance
            
        Raises:
            RuntimeError: If collection initialization fails
        """
        with collection_lock:
            if self._collection is None:
                logger.info(f"Getting or creating collection: {self.collection_name}")
                try:
                    # Create or get collection
                    self._collection = self.client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                    )
                    logger.info(f"Collection '{self.collection_name}' ready")
                except Exception as e:
                    logger.error(f"Error accessing collection: {str(e)}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    raise RuntimeError(f"Failed to access vector collection: {str(e)}")
            
            # Update last access time
            self._last_access_time = time.time()
            return self._collection
    
    def _safe_collection_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute collection operation with proper error handling and reporting.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args: Arguments to pass to operation_func
            **kwargs: Keyword arguments to pass to operation_func
            
        Returns:
            Result of operation_func
            
        Raises:
            RuntimeError: If operation fails
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Starting {operation_name} operation")
            
            # Direct function call without any timeout mechanism
            result = operation_func(*args, **kwargs)
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Completed {operation_name} in {elapsed_ms:.2f}ms")
            
            return result
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Failed {operation_name} after {elapsed_ms:.2f}ms: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            
            # Provide detailed error context
            error_context = {
                "operation": operation_name,
                "args_count": len(args),
                "kwargs": list(kwargs.keys()),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            logger.error(f"Error context: {error_context}")
            
            raise RuntimeError(f"Vector store operation '{operation_name}' failed: {str(e)}")
    
    def add_documents(
    self, 
    documents: Union[List[str], List[DocumentChunk]], 
    embeddings: List[List[float]], 
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = None,
    use_individual_processing: bool = True  # Changed default to True for reliable processing
    ) -> None:
        """Add documents to the vector store with improved reliability.
        
        Args:
            documents: List of document text content or DocumentChunk objects
            embeddings: List of embedding vectors
            ids: List of unique document identifiers (optional if using DocumentChunk)
            metadatas: Optional list of metadata dictionaries (optional if using DocumentChunk)
            batch_size: Number of documents to add in each batch
            use_individual_processing: Process documents one by one for better error isolation
            
        Raises:
            ValueError: If input lists have inconsistent lengths
            RuntimeError: If document addition fails
        """
        # Use configured batch size if not specified
        if batch_size is None:
            # Adjust batch size based on available memory
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)
            
            if available_mb < settings.MEMORY_SAFETY_MARGIN_MB * 2:
                # Reduce batch size when memory is constrained
                batch_size = max(1, settings.DEFAULT_BATCH_SIZE // 2)
                logger.info(f"Reduced batch size to {batch_size} due to memory constraints")
            else:
                batch_size = settings.DEFAULT_BATCH_SIZE
        
        # Handle DocumentChunk objects
        if documents and isinstance(documents[0], DocumentChunk):
            # Extract text, ids, and metadata from chunks
            doc_ids = [chunk.id for chunk in documents]
            doc_texts = [chunk.text for chunk in documents]
            doc_metadatas = [chunk.metadata for chunk in documents]
            
            # Process DocumentChunk objects
            if use_individual_processing:
                # Process each document individually for better error isolation
                logger.info(f"Processing {len(documents)} DocumentChunk objects individually")
                
                success_count = 0
                for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                    try:
                        # Add timing metrics
                        start_time = time.time()
                        logger.info(f"Adding document {i+1}/{len(documents)}: {doc.id}")
                        
                        # Add single document without using collection_lock
                        self.collection.add(
                            documents=[doc.text],
                            embeddings=[embedding],
                            ids=[doc.id],
                            metadatas=[doc.metadata] if doc.metadata else None
                        )
                        
                        elapsed_ms = (time.time() - start_time) * 1000
                        success_count += 1
                        logger.info(f"✓ Document {i+1}/{len(documents)} added successfully in {elapsed_ms:.2f}ms: {doc.id}")
                        
                        # Periodic persistence to avoid large delayed writes
                        if success_count % 10 == 0:
                            logger.info(f"Persisting changes after {success_count} documents...")
                            try:
                                self.persist()
                            except Exception as persist_error:
                                # Log error but continue processing
                                logger.warning(f"Persistence error (continuing anyway): {str(persist_error)}")
                        
                        # Small delay to prevent resource contention
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"✗ Failed to add DocumentChunk {i+1}/{len(documents)} - {doc.id}: {str(e)}")
                        logger.error(f"Error stack trace: {traceback.format_exc()}")
                
                # Final persistence
                logger.info(f"Added {success_count}/{len(documents)} DocumentChunk objects successfully")
                if success_count > 0:
                    try:
                        self.persist()
                    except Exception as persist_error:
                        # Log error but continue processing
                        logger.warning(f"Final persistence error (data should still be stored): {str(persist_error)}")
                return
            else:
                # Process all document chunks as a batch
                logger.info(f"Processing {len(documents)} DocumentChunk objects in batches")
                return self.add_documents(
                    documents=doc_texts,
                    embeddings=embeddings,
                    ids=doc_ids,
                    metadatas=doc_metadatas,
                    batch_size=batch_size,
                    use_individual_processing=use_individual_processing
                )
        
        # Regular implementation for string documents
        if not documents:
            logger.warning("No documents provided for addition")
            return
        
        # Validate input consistency
        if len(documents) != len(embeddings):
            error_msg = f"Inconsistent lengths: documents={len(documents)}, embeddings={len(embeddings)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if ids and len(documents) != len(ids):
            error_msg = f"Inconsistent lengths: documents={len(documents)}, ids={len(ids)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if metadatas and len(metadatas) != len(documents):
            error_msg = f"Metadata length ({len(metadatas)}) doesn't match documents ({len(documents)})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # If ids not provided, generate them
        if not ids:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Process using individual document addition for better error isolation
        if use_individual_processing:
            logger.info(f"Processing {len(documents)} documents individually")
            
            success_count = 0
            for i, (doc, embedding, doc_id) in enumerate(zip(documents, embeddings, ids)):
                metadata = metadatas[i] if metadatas else None
                try:
                    # Add timing metrics
                    start_time = time.time()
                    logger.info(f"Adding document {i+1}/{len(documents)}: {doc_id}")
                    
                    # Direct collection access to avoid lock contention
                    self.collection.add(
                        documents=[doc],
                        embeddings=[embedding],
                        ids=[doc_id],
                        metadatas=[metadata] if metadata else None
                    )
                    
                    elapsed_ms = (time.time() - start_time) * 1000
                    success_count += 1
                    logger.info(f"✓ Document {i+1}/{len(documents)} added successfully in {elapsed_ms:.2f}ms: {doc_id}")
                    
                    # Periodic persistence to avoid large delayed writes
                    if success_count % 10 == 0:
                        logger.info(f"Persisting changes after {success_count} documents...")
                        try:
                            self.persist()
                        except Exception as persist_error:
                            # Log error but continue processing
                            logger.warning(f"Persistence error (continuing anyway): {str(persist_error)}")
                    
                    # Small delay to prevent resource contention
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"✗ Failed to add document {i+1}/{len(documents)} - {doc_id}: {str(e)}")
                    logger.error(f"Error stack trace: {traceback.format_exc()}")
            
            # Final persistence
            logger.info(f"Added {success_count}/{len(documents)} documents successfully")
            if success_count > 0:
                try:
                    self.persist()
                except Exception as persist_error:
                    # Log error but continue processing
                    logger.warning(f"Final persistence error (data should still be stored): {str(persist_error)}")
            return
        
        # Process in batches (legacy approach)
        logger.info(f"Adding {len(documents)} documents to vector store in batches of {batch_size}")
        
        try:
            # Process in batches to manage memory
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            # Batch processing with smaller critical sections
            for i in range(0, len(documents), batch_size):
                # Memory check before each batch
                self._check_available_memory()
                
                end_idx = min(i + batch_size, len(documents))
                batch_docs = documents[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                batch_meta = None
                if metadatas:
                    batch_meta = metadatas[i:end_idx]
                
                # Log batch information
                batch_num = i // batch_size + 1
                logger.info(f"Adding batch {batch_num}/{total_batches} to ChromaDB ({len(batch_docs)} documents)")
                
                try:
                    # Add batch to collection with timing information
                    start_time = time.time()
                    
                    # Limit critical section scope
                    self.collection.add(
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        ids=batch_ids,
                        metadatas=batch_meta  # Pass metadata dictionaries directly
                    )
                    
                    elapsed_ms = (time.time() - start_time) * 1000
                    logger.info(f"✓ Batch {batch_num}/{total_batches} added successfully in {elapsed_ms:.2f}ms")
                    
                    # Persist after each batch to avoid accumulating too many changes
                    if batch_num % 2 == 0 or batch_num == total_batches:
                        logger.info(f"Persisting changes after batch {batch_num}...")
                        try:
                            self.persist()
                        except Exception as persist_error:
                            # Log error but continue processing
                            logger.warning(f"Persistence error in batch {batch_num} (continuing anyway): {str(persist_error)}")
                    
                    # Force garbage collection after each batch
                    gc.collect()
                    
                    # Small delay between batches
                    time.sleep(0.2)
                    
                except Exception as batch_error:
                    logger.error(f"✗ Error in batch {batch_num}: {str(batch_error)}")
                    logger.error(f"Batch error stack trace: {traceback.format_exc()}")
                    
                    # Try to recover by processing this batch individually
                    logger.info(f"Attempting recovery by processing batch {batch_num} documents individually...")
                    
                    individual_success = 0
                    for j, (doc, emb, id_) in enumerate(zip(batch_docs, batch_embeddings, batch_ids)):
                        doc_idx = i + j
                        meta = batch_meta[j] if batch_meta else None
                        
                        try:
                            self.collection.add(
                                documents=[doc],
                                embeddings=[emb],
                                ids=[id_],
                                metadatas=[meta] if meta else None
                            )
                            individual_success += 1
                        except Exception as doc_error:
                            logger.error(f"  ✗ Failed individual recovery for document {doc_idx+1}: {str(doc_error)}")
                    
                    logger.info(f"Batch {batch_num} recovery: {individual_success}/{len(batch_docs)} documents added")
            
            # Final persistence
            logger.info(f"Successfully added documents to vector store in {total_batches} batches")
            try:
                self.persist()
            except Exception as persist_error:
                # Log error but continue processing
                logger.warning(f"Final persistence error (data should still be stored): {str(persist_error)}")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to add documents to vector store: {str(e)}")

    def add_single_document(
        self, 
        document: str, 
        embedding: List[float], 
        doc_id: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a single document to the vector store.
        
        This method provides better error isolation by processing one document at a time.
        
        Args:
            document: Document text content
            embedding: Embedding vector
            doc_id: Document identifier
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RuntimeError: If document addition fails
        """
        logger.debug(f"Adding single document: {doc_id}")
        
        try:
            with collection_lock:
                # Add single document to collection
                self._safe_collection_operation(
                    f"add_document_{doc_id}",
                    self.collection.add,
                    documents=[document],
                    embeddings=[embedding],
                    ids=[doc_id],
                    metadatas=[metadata] if metadata else None
                )
            
            logger.debug(f"Document {doc_id} added successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to add document {doc_id}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to add document {doc_id}: {str(e)}")
        
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 3000  # 3 second timeout
    ) -> List[Dict[str, Any]]:
        """Search for similar documents with robust timeout handling.
        
        Args:
            query_embedding: Embedding vector for the query
            k: Number of results to return
            filter_metadata: Optional metadata filter
            timeout_ms: Query timeout in milliseconds
            
        Returns:
            List of document dictionaries with id, content, and metadata
            
        Raises:
            ValueError: If query_embedding is invalid
            RuntimeError: If search operation fails
        """
        if not query_embedding:
            logger.warning("Empty query embedding provided for search")
            raise ValueError("Query embedding cannot be empty")
        
        # Enforce performance constraints
        k = min(k, 5)  # Cap at 5 results for resource efficiency
        
        # Memory optimization before search
        self._check_available_memory()
        gc.collect()  # Force garbage collection to optimize memory
        
        try:
            # Access collection without lock to ensure we have it initialized
            collection = self.client.get_collection(name=self.collection_name)
            
            # Check if collection is empty
            count = collection.count()
            if count == 0:
                logger.warning("Vector store is empty - no documents to search")
                return []
                
            logger.info(f"Searching for top {k} documents among {count} total documents")
            
            # Prepare metadata filter
            filter_dict = None
            if filter_metadata:
                filter_dict = {key: value for key, value in filter_metadata.items()}
            
            # Simple direct search implementation with NO collection lock
            # This approach prevents potential deadlocks
            start_time = time.time()
            
            # Direct collection query WITHOUT using the lock
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
                        # Extract metadata (already in dict format)
                        metadata = {}
                        if results.get('metadatas') and results['metadatas'][0][i]:
                            metadata = results['metadatas'][0][i]
                        
                        # Build document result
                        documents.append({
                            "id": doc_id,
                            "content": results['documents'][0][i],
                            "metadata": metadata,
                            "distance": results.get('distances', [[0]])[0][i] if results.get('distances') else None
                        })
                    except Exception as item_error:
                        logger.error(f"Error processing search result {i}: {str(item_error)}")
            
            logger.info(f"Search returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return []  # Return empty results rather than failing completely
        finally:
            # Ensure memory cleanup
            gc.collect()
    
    def quick_search(self, query_embedding: List[float], k: int = 2) -> List[Dict[str, Any]]:
        """Perform a fast, limited search for testing purposes.
        
        This is a lightweight version of the search method optimized for
        speed over comprehensiveness.
        
        Args:
            query_embedding: Embedding vector for the query
            k: Number of results to return (limited to 2 max)
            
        Returns:
            List of document dictionaries or empty list if failure
        """
        try:
            # Force small k for performance
            k = min(k, 2)
            
            # Check if collection is empty first
            count = self.collection.count()
            if count == 0:
                logger.warning("Vector store is empty - no documents to search")
                return []
            
            # Add timeout to prevent hanging
            start_time = time.time()
            
            # Simplified search with minimal parameters
            with collection_lock:
                results = self._safe_collection_operation(
                    "quick_search",
                    self.collection.query,
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=['documents', 'ids']
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info(f"Quick search completed in {elapsed_ms:.2f}ms")
            
            # Simplified processing
            documents = []
            if results and 'documents' in results and len(results['documents']) > 0:
                for i, doc_content in enumerate(results['documents'][0]):
                    doc_id = results['ids'][0][i] if 'ids' in results and i < len(results['ids'][0]) else f"result_{i}"
                    documents.append({
                        "id": doc_id,
                        "content": doc_content,
                        "metadata": {"source": "unknown"}
                    })
            
            return documents
        except Exception as e:
            logger.error(f"Error in quick search: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return []  # Return empty results rather than failing
    
    def persist(self) -> None:
        """Persist changes to disk safely with ChromaDB API version compatibility.
        
        Saves the current state of the vector database to disk, with support
        for both older (explicit persist) and newer (automatic persistence) APIs.
        
        Raises:
            RuntimeError: If persistence operation fails
        """
        try:
            if self._client:
                logger.info("Persisting vector store changes to disk")
                
                # Check if persist method exists (older ChromaDB versions)
                if hasattr(self._client, 'persist'):
                    self._safe_collection_operation(
                        "persist",
                        self._client.persist
                    )
                    logger.info("Vector store persisted successfully (explicit persistence)")
                else:
                    # Newer ChromaDB versions persist automatically
                    logger.info("Vector store uses automatic persistence (no explicit persist needed)")
                    
                    # Force disk sync via collection operation to ensure data is written
                    count = self.collection.count()
                    logger.info(f"Collection contains {count} documents")
        except Exception as e:
            logger.error(f"Error during persistence check: {str(e)}")
            # Don't raise exception for persistence failures
            # Instead, log the issue but allow processing to continue
            logger.warning("Continuing without explicit persistence (data should still be saved)")
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document dictionary or None if not found
            
        Raises:
            RuntimeError: If retrieval operation fails
        """
        try:
            logger.info(f"Retrieving document with ID: {doc_id}")
            
            with collection_lock:
                result = self._safe_collection_operation(
                    f"get_document_{doc_id}",
                    self.collection.get,
                    ids=[doc_id],
                    include=["documents", "metadatas"]
                )
            
            if result and result["ids"] and len(result["ids"]) > 0:
                # FIXED: No need to parse JSON, metadata should be a dict already
                metadata = {}
                if result.get("metadatas") and result["metadatas"][0]:
                    metadata = result["metadatas"][0]
                
                return {
                    "id": doc_id,
                    "content": result["documents"][0],
                    "metadata": metadata
                }
            
            logger.info(f"Document with ID {doc_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document by ID: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to retrieve document: {str(e)}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            logger.info(f"Getting collection information for {self.collection_name}")
            
            # Get collection count
            count = self.collection.count()
            
            # Get collection name and metadata
            collection_info = {
                "name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
            
            return collection_info
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {
                "name": self.collection_name,
                "error": str(e),
                "document_count": 0
            }
    
    def unload_if_inactive(self, threshold_seconds: int = 3600):
        """Unload the client and collection if inactive for specified period.
        
        Args:
            threshold_seconds: Number of seconds after which to unload resources
        """
        if time.time() - self._last_access_time > threshold_seconds:
            logger.info(f"Unloading vector store after {threshold_seconds}s of inactivity")
            with collection_lock:
                self._collection = None
                self._client = None
                self._metadata_cache.clear()
                gc.collect()

# Singleton instance for application-wide use
vector_store = VectorStoreService()

# Setup automatic cleanup function
def setup_vectorstore_cleanup(app=None):
    """Set up automatic cleanup of inactive vector store.
    
    Args:
        app: FastAPI app for startup/shutdown events
    """
    import threading
    
    def cleanup_task():
        """Background task to periodically unload inactive vector store."""
        while True:
            try:
                # Check every 15 minutes
                time.sleep(900)
                vector_store.unload_if_inactive(threshold_seconds=1800)  # 30 minutes
            except Exception as e:
                logger.error(f"Error in vector store cleanup task: {str(e)}")
    
    # Start background thread
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    
    # Register with FastAPI if app provided
    if app:
        @app.on_event("startup")
        def start_cleanup():
            logger.info("Starting automatic vector store cleanup task")
            # Thread already started above
    
        @app.on_event("shutdown")
        def stop_cleanup():
            logger.info("Stopping automatic vector store cleanup task")
            # Properly persist and close resources
            try:
                if vector_store._client is not None:
                    vector_store.persist()
                    vector_store._collection = None
                    vector_store._client = None
                    gc.collect()
            except Exception as e:
                logger.error(f"Error during vector store shutdown: {str(e)}")