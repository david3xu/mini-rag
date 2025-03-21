"""
LLM service for generating text with local phi-2 model.

This module provides text generation capabilities using the phi-2.gguf model
via llama.cpp, optimized for CPU usage with efficient memory management.
"""

import llama_cpp
from typing import Generator, Dict, Any, Optional, List, Union
import time
import os
import gc
import logging
import threading

from config import settings

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread lock for model loading/unloading operations
model_lock = threading.Lock()

class LLMService:
    """Service for generating text with local LLM.
    
    This service manages the phi-2.gguf model for text generation,
    with optimizations for memory usage through lazy loading,
    context management, and resource cleanup.
    """
    
    def __init__(self):
        """Initialize the LLM service with configured settings."""
        # Use lazy loading to avoid loading the model at startup
        self._llm = None
        self._model_path = settings.MODEL_PATH
        self._n_ctx = settings.MODEL_N_CTX
        self._n_batch = settings.MODEL_N_BATCH
        self._n_gpu_layers = settings.MODEL_N_GPU_LAYERS
        self.last_used_time = 0
        
        logger.info(f"LLM service initialized with model path: {self._model_path}")
        
        # Check if model file exists
        if not os.path.exists(self._model_path):
            logger.warning(f"Model file not found at: {self._model_path}")
    
    @property
    def llm(self) -> llama_cpp.Llama:
        """Lazy-loaded LLM property to minimize startup resources.
        
        Returns:
            llama_cpp.Llama: The loaded LLM model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        with model_lock:
            if self._llm is None:
                logger.info(f"Loading LLM model from: {self._model_path}")
                
                if not os.path.exists(self._model_path):
                    logger.error(f"Model file not found: {self._model_path}")
                    raise FileNotFoundError(f"Model file not found: {self._model_path}")
                
                try:
                    self._llm = llama_cpp.Llama(
                        model_path=self._model_path,
                        n_ctx=self._n_ctx,
                        n_batch=self._n_batch,
                        n_gpu_layers=self._n_gpu_layers,
                        verbose=False,
                    )
                    logger.info("LLM model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading LLM model: {str(e)}")
                    raise RuntimeError(f"Failed to load LLM model: {str(e)}")
            
            # Update last used timestamp
            self.last_used_time = time.time()
            return self._llm
    
    def unload_model_if_inactive(self, threshold_seconds: int = 1800):
        """Unload the model if it hasn't been used for a specified time period.
        
        Args:
            threshold_seconds: Number of seconds after which to unload the model
        """
        with model_lock:
            if self._llm is not None and time.time() - self.last_used_time > threshold_seconds:
                logger.info("Unloading LLM model due to inactivity")
                self._llm = None
                # Force garbage collection to release memory
                gc.collect()
    
    def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """Generate text from the LLM.
        
        Args:
            prompt: Text prompt for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stream: Whether to stream the response
            
        Returns:
            Generated text or a generator yielding text chunks
            
        Raises:
            ValueError: If prompt is empty
            RuntimeError: If text generation fails
        """
        if not prompt.strip():
            logger.warning("Empty prompt provided for text generation")
            raise ValueError("Cannot generate text from empty prompt")
        
        # Format for phi-2 prompt
        phi_prompt = f"<s>Instruct: {prompt}\nOutput:"
        
        try:
            logger.info(f"Generating text with max_tokens={max_tokens}, temperature={temperature}")
            
            if not stream:
                # Generate complete response
                response = self.llm(
                    phi_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    echo=False
                )
                return response["choices"][0]["text"].strip()
            else:
                # Return generator for streaming response
                return self._generate_streaming_response(phi_prompt, max_tokens, temperature)
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise RuntimeError(f"Failed to generate text: {str(e)}")
    
    def _generate_streaming_response(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> Generator[str, None, None]:
        """Generate streaming text response.
        
        Args:
            prompt: Formatted prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            RuntimeError: If streaming generation fails
        """
        try:
            # Generate streaming response
            response_text = ""
            for token in self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                stream=True
            ):
                chunk = token["choices"][0]["text"]
                response_text += chunk
                yield chunk
            
            return response_text
        except Exception as e:
            logger.error(f"Error during streaming text generation: {str(e)}")
            raise RuntimeError(f"Failed to generate streaming text: {str(e)}")
    
    def format_rag_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Format a prompt for RAG.
        
        Creates a formatted prompt that includes context documents for
        retrieval-augmented generation.
        
        Args:
            query: User's query text
            documents: List of relevant documents with content and metadata
            
        Returns:
            Formatted prompt string
        """
        # Extract and format document content
        context = "\n\n".join([
            f"Document {i+1}:\n{doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        # Create RAG prompt with context and query
        prompt = f"""
        You are a helpful assistant that provides accurate and informative answers based on the provided documents.
        
        Answer the following query based ONLY on the information from these documents.
        If the documents don't contain relevant information to answer the query, say "I don't have enough information to answer this question."
        Do not make up or infer information that is not explicitly stated in the documents.
        
        Documents:
        {context}

        Query: {query}

        Answer:
        """
        return prompt

# Singleton instance for application-wide use
llm_service = LLMService()