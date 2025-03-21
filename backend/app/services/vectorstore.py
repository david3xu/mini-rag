"""
Vector store service for document storage and retrieval.

This module provides functionality for storing and retrieving document
embeddings using ChromaDB, optimized for memory efficiency with disk-based
persistence.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional, Tuple
import json
import logging
import os
import time
from threading import Lock

from config import settings

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
    
    def __init__(self):
        """Initialize the vector store service with configured settings."""
        self.persist_directory = settings.VECTOR_DB_PATH
        self.collection_name = settings.VECTOR_DB_COLLECTION
        self._client = None
        self._collection = None
        
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
            logger.info("Initializing ChromaDB client")
            try:
                self._client = chromadb.Client(ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_directory,
                    anonymized_telemetry=False
                ))
                logger.info("ChromaDB client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing ChromaDB client: {str(e)}")
                raise RuntimeError(f"Failed to initialize vector database: {str(e)}")
        
        return self._client
    
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
            
            return self._collection
    
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of document text content
            embeddings: List of embedding vectors
            ids: List of unique document identifiers
            metadatas: Optional list of metadata dictionaries
            batch_size: Number of documents to add in each batch
            
        Raises:
            ValueError: If input lists have inconsistent lengths
            RuntimeError: If document addition fails
        """
        if not documents:
            logger.warning("No documents provided for addition")
            return
        
        # Validate input consistency
        if len(documents) != len(embeddings) or len(documents) != len(ids):
            error_msg = f"Inconsistent lengths: documents={len(documents)}, embeddings={len(embeddings)}, ids={len(ids)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if metadatas and len(metadatas) != len(documents):
            error_msg = f"Metadata length ({len(metadatas)}) doesn't match documents ({len(documents)})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert metadatas to strings if provided
        string_metadatas = None
        if metadatas:
            string_metadatas = []
            for m in metadatas:
                try:
                    string_metadatas.append(json.dumps(m))
                except Exception as e:
                    logger.warning(f"Error serializing metadata: {str(e)}. Using empty metadata.")
                    string_metadatas.append("{}")
        
        logger.info(f"Adding {len(documents)} documents to vector store in batches of {batch_size}")
        
        try:
            # Process in batches to manage memory
            with collection_lock:
                for i in range(0, len(documents), batch_size):
                    end_idx = min(i + batch_size, len(documents))
                    batch_docs = documents[i:end_idx]
                    batch_embeddings = embeddings[i:end_idx]
                    batch_ids = ids[i:end_idx]
                    
                    batch_metadatas = None
                    if string_metadatas:
                        batch_metadatas = string_metadatas[i:end_idx]
                    
                    logger.debug(f"Adding batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                    
                    # Add batch to collection
                    self.collection.add(
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
            
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
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query_embedding: Embedding vector for the query
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of document dictionaries with id, content, and metadata
            
        Raises:
            ValueError: If query_embedding is invalid
            RuntimeError: If search operation fails
        """
        if not query_embedding:
            logger.warning("Empty query embedding provided for search")
            raise ValueError("Query embedding cannot be empty")
        
        try:
            logger.info(f"Searching for top {k} documents")
            
            # Prepare filter if provided
            filter_dict = None
            if filter_metadata:
                filter_dict = {"$and": []}
                for key, value in filter_metadata.items():
                    filter_dict["$and"].append({f"$contains": json.dumps({key: value})})
            
            # Query the collection
            with collection_lock:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    where=filter_dict
                )
            
            # Format results
            documents = []
            if results and 'ids' in results and len(results['ids']) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
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
            
            logger.info(f"Search returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise RuntimeError(f"Failed to search vector store: {str(e)}")
    
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
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if document was deleted, False if not found
            
        Raises:
            RuntimeError: If deletion operation fails
        """
        try:
            logger.info(f"Deleting document with ID: {doc_id}")
            
            # Check if document exists
            doc = self.get_document_by_id(doc_id)
            if not doc:
                logger.warning(f"Document with ID {doc_id} not found for deletion")
                return False
            
            # Delete the document
            with collection_lock:
                self.collection.delete(ids=[doc_id])
            
            # Persist changes
            self.persist()
            logger.info(f"Document with ID {doc_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise RuntimeError(f"Failed to delete document: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection.
        
        Returns:
            Dictionary with collection statistics
            
        Raises:
            RuntimeError: If operation fails
        """
        try:
            logger.info("Retrieving collection statistics")
            
            with collection_lock:
                count = self.collection.count()
            
            return {
                "name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error retrieving collection stats: {str(e)}")
            raise RuntimeError(f"Failed to get collection statistics: {str(e)}")

# Singleton instance for application-wide use
vector_store = VectorStoreService()