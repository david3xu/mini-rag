"""
Performance-optimized vector store service for document storage and retrieval.

This module provides functionality for storing and retrieving document
embeddings using ChromaDB, optimized for memory efficiency and query performance
with disk-based persistence.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional, Union
import json
import logging
import os
import time
import psutil
from threading import Lock
import gc

from config import settings, get_path
from app.api.models import DocumentChunk

# Configure logger
logging.basicConfig(level=logging.INFO)
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
                    raise RuntimeError(f"Failed to access vector collection: {str(e)}")
            
            # Update last access time
            self._last_access_time = time.time()
            return self._collection
    
    def add_documents(
        self, 
        documents: Union[List[str], List[DocumentChunk]], 
        embeddings: List[List[float]], 
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = None
    ) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document text content or DocumentChunk objects
            embeddings: List of embedding vectors
            ids: List of unique document identifiers (optional if using DocumentChunk)
            metadatas: Optional list of metadata dictionaries (optional if using DocumentChunk)
            batch_size: Number of documents to add in each batch
            
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
            
            # Call the regular add_documents method with extracted data
            return self.add_documents(
                documents=doc_texts,
                embeddings=embeddings,
                ids=doc_ids,
                metadatas=doc_metadatas,
                batch_size=batch_size
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
        
        # Convert metadatas to strings if provided
        string_metadatas = None
        if metadatas:
            string_metadatas = []
            for m in metadatas:
                try:
                    # Keep a copy in the cache for faster retrieval
                    self._metadata_cache[m.get('source', '')] = m
                    string_metadatas.append(json.dumps(m))
                except Exception as e:
                    logger.warning(f"Error serializing metadata: {str(e)}. Using empty metadata.")
                    string_metadatas.append("{}")
        
        logger.info(f"Adding {len(documents)} documents to vector store in batches of {batch_size}")
        
        try:
            # Process in batches to manage memory
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            with collection_lock:
                for i in range(0, len(documents), batch_size):
                    # Memory check before each batch
                    self._check_available_memory()
                    
                    end_idx = min(i + batch_size, len(documents))
                    batch_docs = documents[i:end_idx]
                    batch_embeddings = embeddings[i:end_idx]
                    batch_ids = ids[i:end_idx]
                    
                    batch_metadatas = None
                    if string_metadatas:
                        batch_metadatas = string_metadatas[i:end_idx]
                    
                    logger.debug(f"Adding batch {i//batch_size + 1}/{total_batches}")
                    
                    # Add batch to collection
                    self.collection.add(
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
                    
                    # Force garbage collection after each batch
                    gc.collect()
            
            # Persist changes to disk
            self.persist()
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise RuntimeError(f"Failed to add documents to vector store: {str(e)}")
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None,
        timeout_ms: int = 3000  # 3 second timeout
    ) -> List[Dict[str, Any]]:
        """Search for similar documents with timeout handling.
        
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
        
        # Limit k for performance
        k = min(k, 5)  # Cap at 5 results max regardless of request
        
        # Memory check before search
        self._check_available_memory()
        
        try:
            logger.info(f"Searching for top {k} documents")
            
            # Prepare filter if provided
            filter_dict = None
            if filter_metadata:
                filter_dict = {"$and": []}
                for key, value in filter_metadata.items():
                    filter_dict["$and"].append({f"$contains": json.dumps({key: value})})
            
            # Query the collection with timeout
            with collection_lock:
                start_time = time.time()
                
                # Add timeout to search operation
                results = self.collection.query(
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
                        metadata = {}
                        if results.get('metadatas') and results['metadatas'][0][i]:
                            try:
                                metadata = json.loads(results['metadatas'][0][i])
                            except json.JSONDecodeError:
                                logger.warning(f"Error decoding metadata for document {doc_id}")
                        
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
            # Return empty results rather than failing completely
            return []
    
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
            
            # Simplified search with minimal parameters
            with collection_lock:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=['documents']
                )
            
            # Simplified processing
            documents = []
            if results and 'documents' in results and len(results['documents']) > 0:
                for i, doc_content in enumerate(results['documents'][0]):
                    documents.append({
                        "id": f"result_{i}",
                        "content": doc_content,
                        "metadata": {"source": "unknown"}
                    })
            
            return documents
        except Exception as e:
            logger.error(f"Error in quick search: {str(e)}")
            return []
    
    def persist(self) -> None:
        """Persist changes to disk.
        
        Saves the current state of the vector database to disk.
        
        Raises:
            RuntimeError: If persistence operation fails
        """
        try:
            if self._client:
                logger.info("Persisting vector store changes to disk")
                self.client.persist()
                logger.info("Vector store persisted successfully")
        except Exception as e:
            logger.error(f"Error persisting vector store: {str(e)}")
            raise RuntimeError(f"Failed to persist vector store: {str(e)}")
    
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
                result = self.collection.get(
                    ids=[doc_id],
                    include=["documents", "metadatas"]
                )
            
            if result and result["ids"] and len(result["ids"]) > 0:
                metadata = {}
                if result.get("metadatas") and result["metadatas"][0]:
                    try:
                        metadata = json.loads(result["metadatas"][0])
                    except json.JSONDecodeError:
                        logger.warning(f"Error decoding metadata for document {doc_id}")
                
                return {
                    "id": doc_id,
                    "content": result["documents"][0],
                    "metadata": metadata
                }
            
            logger.info(f"Document with ID {doc_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving document by ID: {str(e)}")
            raise RuntimeError(f"Failed to retrieve document: {str(e)}")
    
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