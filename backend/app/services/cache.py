"""
Cache service for response and embedding caching.

This module provides caching mechanisms to improve response time
by storing frequently accessed query results and embeddings.
"""

import functools
from typing import Dict, Any, Optional, Tuple, List
import time
import logging
import gc
import threading

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread lock for cache operations
cache_lock = threading.Lock()

class ResponseCache:
    """LRU cache for query responses.
    
    This cache stores responses to common queries to avoid
    regenerating answers for identical or very similar queries.
    """
    
    def __init__(self, max_size=100, ttl=3600):
        """Initialize response cache.
        
        Args:
            max_size: Maximum number of items to store in cache
            ttl: Time-to-live in seconds for cache entries
        """
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        logger.info(f"Response cache initialized with size {max_size} and TTL {ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response if it exists and hasn't expired.
        
        Args:
            key: Cache key to look up
            
        Returns:
            Cached value or None if not found or expired
        """
        with cache_lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    logger.info(f"Cache hit for key: {key[:20]}...")
                    return value
                else:
                    # Remove expired entry
                    logger.debug(f"Removing expired cache entry: {key[:20]}...")
                    del self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Add or update cache entry.
        
        Args:
            key: Cache key
            value: Value to store
        """
        with cache_lock:
            # If cache is full, remove oldest entry
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                logger.debug(f"Cache full, removing oldest entry: {oldest_key[:20]}...")
                del self.cache[oldest_key]
                # Force memory cleanup
                gc.collect()
            
            logger.debug(f"Adding cache entry: {key[:20]}...")
            self.cache[key] = (value, time.time())
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with cache_lock:
            self.cache.clear()
            gc.collect()
            logger.info("Cache cleared")
    
    def remove(self, key: str) -> bool:
        """Remove a specific cache entry.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if entry was removed, False if not found
        """
        with cache_lock:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Removed cache entry: {key[:20]}...")
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with cache_lock:
            stats = {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "memory_usage_mb": self._estimate_memory_usage() / (1024 * 1024)
            }
            return stats
    
    def _estimate_memory_usage(self) -> float:
        """Roughly estimate memory usage of the cache.
        
        Returns:
            Estimated memory usage in bytes
        """
        # Very rough estimation based on key length and value size
        total_bytes = 0
        for key, (value, _) in self.cache.items():
            # Estimate key size
            key_size = len(key) * 2  # Unicode strings in Python use ~2 bytes per char
            
            # Estimate value size
            value_size = 0
            if isinstance(value, dict):
                # For dictionaries, estimate size based on keys and values
                for k, v in value.items():
                    value_size += len(k) * 2
                    if isinstance(v, str):
                        value_size += len(v) * 2
                    elif isinstance(v, list):
                        value_size += len(v) * 8  # Rough approximation
                    else:
                        value_size += 8  # Default size for primitive values
            elif isinstance(value, str):
                value_size = len(value) * 2
            else:
                value_size = 100  # Default for unknown types
            
            total_bytes += key_size + value_size
        
        return total_bytes


class EmbeddingCache:
    """Cache for text embeddings.
    
    This cache stores embeddings for text chunks to avoid
    regenerating embeddings for the same or similar text.
    """
    
    def __init__(self, max_size=500):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to store
        """
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
        self.last_used: Dict[str, float] = {}
        logger.info(f"Embedding cache initialized with size {max_size}")
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text if it exists.
        
        Args:
            text: Text to look up
            
        Returns:
            Embedding vector or None if not found
        """
        with cache_lock:
            # Use normalized text as key
            key = self._normalize_text(text)
            if key in self.cache:
                self.last_used[key] = time.time()
                logger.debug(f"Embedding cache hit for text: {text[:30]}...")
                return self.cache[key]
            return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """Add or update embedding in cache.
        
        Args:
            text: Text to cache
            embedding: Embedding vector to store
        """
        with cache_lock:
            key = self._normalize_text(text)
            
            # If cache is full, remove least recently used
            if len(self.cache) >= self.max_size:
                lru_key = min(self.last_used.keys(), key=lambda k: self.last_used[k])
                logger.debug(f"Embedding cache full, removing LRU entry")
                del self.cache[lru_key]
                del self.last_used[lru_key]
            
            self.cache[key] = embedding
            self.last_used[key] = time.time()
            logger.debug(f"Added embedding to cache for text: {text[:30]}...")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for use as cache key.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Simple normalization: lowercase and strip whitespace
        return text.lower().strip()
    
    def clear(self) -> None:
        """Clear all embeddings from cache."""
        with cache_lock:
            self.cache.clear()
            self.last_used.clear()
            gc.collect()
            logger.info("Embedding cache cleared")


# Create singleton instances
response_cache = ResponseCache()
embedding_cache = EmbeddingCache() 