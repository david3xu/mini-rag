"""
Configuration module for the Mini RAG backend application.

This module provides configuration settings for the application,
including model paths, API settings, and resource optimization parameters.
"""

import os
from pydantic import BaseModel


# Create absolute paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Helper function to resolve paths
def get_path(relative_path):
    """Convert relative paths to absolute paths correctly.
    
    This prevents duplicate 'backend' directories when the code
    is run from different locations.
    """
    # If path starts with ./backend/ or backend/, use just the portion after it
    if relative_path.startswith("./backend/"):
        path = relative_path[10:]  # Skip ./backend/
    elif relative_path.startswith("backend/"):
        path = relative_path[8:]   # Skip backend/
    else:
        path = relative_path
        
    return os.path.join(BASE_DIR, path)


class Settings(BaseModel):
    """Application settings for Mini RAG backend.
    
    This class defines configuration parameters for various components of the
    system, including API settings, model configurations, vector database options,
    and document processing parameters.
    
    Attributes:
        APP_NAME: Name of the application
        API_PREFIX: URL prefix for API routes
        DEBUG: Flag to enable debug mode
        
        MODEL_PATH: Path to the LLM model
        MODEL_N_CTX: Maximum context length for LLM
        MODEL_N_BATCH: Batch size for inference
        MODEL_N_GPU_LAYERS: Number of layers to offload to GPU (0 for CPU-only)
        
        EMBEDDING_MODEL: Name or path of embedding model
        EMBEDDING_DIMENSION: Vector dimension for embeddings
        
        VECTOR_DB_PATH: Path to vector database storage
        VECTOR_DB_COLLECTION: Name of vector database collection
        
        CHUNK_SIZE: Maximum size of document chunks
        CHUNK_OVERLAP: Overlap between adjacent chunks
        MAX_DOCUMENT_SIZE_MB: Maximum allowed document size
    """
    
    # App settings
    APP_NAME: str = "Mini RAG"
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    
    # LLM Settings
    MODEL_PATH: str = os.environ.get("MODEL_PATH", get_path("models/phi-2.gguf"))
    MODEL_N_CTX: int = 512  # Already optimized at 512, good!
    MODEL_N_BATCH: int = 8
    MODEL_N_GPU_LAYERS: int = 0  # 0 for CPU-only inference
    MODEL_N_THREADS: int = min(4, os.cpu_count() or 1)  # Limit thread count
    MODEL_UNLOAD_TIMEOUT: int = 1800  # Unload model after 30 minutes of inactivity
    
    # Embedding Settings
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", get_path("models/embeddings/all-MiniLM-L6-v2"))
    EMBEDDING_DIMENSION: int = 384
    
    # Vector DB Settings
    VECTOR_DB_PATH: str = os.environ.get("VECTOR_DB_PATH", get_path("vector_db/chroma_db"))
    VECTOR_DB_COLLECTION: str = "documents"
    VECTOR_DB_IMPL: str = "duckdb+parquet"  # Ensure disk-based implementation
    
    # Document Processing
    CHUNK_SIZE: int = 500  # Reduced from 1000 for memory efficiency
    CHUNK_OVERLAP: int = 50  # Reduced from 100 for efficiency
    MAX_DOCUMENT_SIZE_MB: int = 5  # Reduced from 10 for memory efficiency
    
    # Memory Optimization
    DEFAULT_BATCH_SIZE: int = 4  # Reduced from 8 for memory efficiency
    MEMORY_SAFETY_MARGIN_MB: int = 256  # Reduced from 512 for better balance
    
    # Request Optimization
    REQUEST_TIMEOUT_SECONDS: int = 10  # Timeout for requests
    QUERY_TIMEOUT_MS: int = 3000  # Timeout for vector queries (3 seconds)
    
    # Azure Configuration (used when migrating)
    AZURE_OPENAI_ENDPOINT: str = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.environ.get("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_DEPLOYMENT: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
    
    AZURE_SEARCH_ENDPOINT: str = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_API_KEY: str = os.environ.get("AZURE_SEARCH_API_KEY", "")
    AZURE_SEARCH_INDEX: str = os.environ.get("AZURE_SEARCH_INDEX", "documents")
    
    class Config:
        """Pydantic configuration class.
        
        Specifies behavior for the Settings class.
        """
        env_file = ".env"
        case_sensitive = True


# Create singleton settings instance
settings = Settings()