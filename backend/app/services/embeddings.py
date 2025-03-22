"""
Embedding service for generating vector representations of text.

This module provides functionality for converting text into embedding vectors
using the sentence-transformers library, optimized for memory-efficient operation.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
import os
import gc
import time
import logging

from config import settings

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Build absolute path to backend directory
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
logger.info(f"Backend directory: {BACKEND_DIR}")

class EmbeddingService:
    """Service for generating embeddings from text.
    
    This service handles the creation of vector embeddings from text chunks,
    optimized for CPU usage with the all-MiniLM-L6-v2 model. It supports
    batch processing to manage memory efficiently.
    """
    
    def __init__(self):
        """Initialize the embedding service with the configured model."""
        # Lazy loading - model will be initialized only when needed
        self._model = None
        
        # Keep the original model name for reference
        self.model_name = settings.EMBEDDING_MODEL
        
        # Create an absolute path for the actual model location
        if settings.EMBEDDING_MODEL.startswith("./backend/"):
            # If path starts with ./backend, replace with absolute backend path
            relative_path = settings.EMBEDDING_MODEL[10:]  # Remove ./backend/
            self.model_path = os.path.join(BACKEND_DIR, relative_path)
        else:
            # Otherwise use as is (might be absolute already or a HF model name)
            self.model_path = settings.EMBEDDING_MODEL
            
        # Check if the path exists at init time and log
        if os.path.exists(self.model_path):
            logger.info(f"Embedding model path verified: {self.model_path}")
        else:
            logger.warning(f"Embedding model path not found: {self.model_path}")
            logger.warning("Will attempt to load from HuggingFace hub if needed.")
        
        self.dimension = settings.EMBEDDING_DIMENSION
        self.last_used_time = 0
        logger.info(f"Embedding service initialized with model: {self.model_name}")
    
    def _resolve_model_path(self, model_path):
        """Resolve the model path to ensure it exists.
        
        This handles different path formats (relative to current dir,
        relative to backend dir, absolute paths) to find the actual model.
        
        Args:
            model_path: The original model path from settings
            
        Returns:
            The resolved path that exists, or the original path if no resolution found
        """
        # Check if the original path exists
        if os.path.exists(model_path):
            return model_path
        
        # Try resolving as an absolute path
        abs_path = os.path.abspath(model_path)
        if os.path.exists(abs_path):
            return abs_path
        
        # Handle relative paths that include "backend" in the path
        if "backend" in model_path:
            # Try removing the ./backend/ prefix if we're already in backend dir
            if model_path.startswith("./backend/") and os.getcwd().endswith("backend"):
                without_prefix = model_path[10:]  # Remove ./backend/
                if os.path.exists(without_prefix):
                    return without_prefix
                
            # Try using path relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate up from app/services to backend
            backend_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            
            # If path starts with ./backend, remove that part
            if model_path.startswith("./backend/"):
                path_without_backend = model_path[10:]
            elif model_path.startswith("backend/"):
                path_without_backend = model_path[8:]
            else:
                path_without_backend = model_path
                
            # Construct path relative to backend directory
            rel_path = os.path.join(backend_dir, path_without_backend)
            if os.path.exists(rel_path):
                return rel_path
        
        # If all else fails, return the original path
        # (SentenceTransformer will try to download it or fail)
        logger.warning(f"Could not resolve model path: {model_path}")
        return model_path
    
    @property
    def model(self):
        """Lazy-loaded model property to minimize startup resources.
        
        Returns:
            SentenceTransformer: The sentence transformer model
        """
        if self._model is None:
            # Use the predetermined model path from init
            logger.info(f"Loading embedding model: {self.model_path}")
            
            # Initialize the model when first needed
            try:
                # Try using the resolved path first
                self._model = SentenceTransformer(
                    self.model_path, 
                    device="cpu"
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                # Try a fallback to the HuggingFace model if path resolution failed
                if not os.path.exists(self.model_path):
                    try:
                        logger.info("Attempting to load from HuggingFace hub: all-MiniLM-L6-v2")
                        self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                        logger.info("Successfully loaded model from HuggingFace hub")
                    except Exception as fallback_error:
                        logger.error(f"Fallback to HuggingFace hub failed: {str(fallback_error)}")
                        raise
                else:
                    raise
        
        # Update last used time
        self.last_used_time = time.time()
        return self._model
    
    # def unload_model_if_inactive(self, threshold_seconds: int = 3600):
    #     """Unload the model if it hasn't been used for a specified time period.
        
    #     Args:
    #         threshold_seconds: Number of seconds after which to unload the model
    #     """
    #     if self._model is not None and time.time() - self.last_used_time > threshold_seconds:
    #         logger.info("Unloading embedding model due to inactivity")
    #         self._model = None
    #         # Force garbage collection to release memory
    #         gc.collect()
    
    def generate_embeddings(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process together for memory efficiency.
                        If None, uses the default batch size from settings.
        
        Returns:
            List of embedding vectors (as float lists)
        
        Raises:
            ValueError: If texts is empty
            RuntimeError: If embedding generation fails
        """
        if not texts:
            logger.warning("Empty text list provided for embedding generation")
            return []
        
        try:
            # Use configured batch size if not specified
            if batch_size is None:
                batch_size = settings.DEFAULT_BATCH_SIZE
            
            logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
            
            # Process in batches to manage memory
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:min(i + batch_size, len(texts))]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                # Generate embeddings for the current batch
                batch_embeddings = self.model.encode(batch)
                all_embeddings.extend(batch_embeddings.tolist())
                
                # Explicitly release memory after each batch
                if i + batch_size < len(texts):
                    del batch_embeddings
                    gc.collect()
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text string.
        
        Args:
            text: Text string to embed
        
        Returns:
            Embedding vector as a list of floats
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails
        """
        if not text.strip():
            logger.warning("Empty text provided for embedding generation")
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            # Generate embedding for single text (no need for batching)
            embedding = self.model.encode([text])[0].tolist()
            return embedding
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
            
        Raises:
            ValueError: If embeddings have different dimensions
        """
        if len(embedding1) != len(embedding2):
            raise ValueError(f"Embedding dimensions don't match: {len(embedding1)} vs {len(embedding2)}")
        
        # Convert to numpy arrays for efficient computation
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Prevent division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

# Singleton instance for application-wide use
embeddings_service = EmbeddingService()