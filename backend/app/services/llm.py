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
import psutil
import multiprocessing

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
        self._is_loading = False
        self._kv_cache = None
        self._last_prompt = None
        
        # Determine optimal thread count for generation
        try:
            # Get physical core count (non-hyperthreaded)
            physical_cores = psutil.cpu_count(logical=False) or 1
            self._n_threads = min(physical_cores, settings.MODEL_N_THREADS)
        except:
            # Fallback to standard approach
            self._n_threads = max(multiprocessing.cpu_count() // 2, 1)
            
        logger.info(f"LLM service initialized with model path: {self._model_path}")
        logger.info(f"Thread configuration: using {self._n_threads} threads for inference")
        
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
            # Check available memory before loading
            if self._llm is None and not self._is_loading:
                self._check_memory_before_loading()
                
                logger.info(f"Loading LLM model from: {self._model_path}")
                self._is_loading = True
                
                if not os.path.exists(self._model_path):
                    self._is_loading = False
                    logger.error(f"Model file not found: {self._model_path}")
                    raise FileNotFoundError(f"Model file not found: {self._model_path}")
                
                try:
                    # Use more efficient settings for memory constrained environments
                    self._llm = llama_cpp.Llama(
                        model_path=self._model_path,
                        n_ctx=self._n_ctx,
                        n_batch=self._n_batch,
                        n_gpu_layers=self._n_gpu_layers,
                        n_threads=self._n_threads,  # Set thread count here during initialization
                        verbose=False,
                        # Memory optimization parameters
                        use_mlock=False,  # Don't lock memory
                        use_mmap=True,  # Use memory mapping
                    )
                    logger.info(f"LLM model loaded successfully with {self._n_threads} threads")
                except Exception as e:
                    self._is_loading = False
                    logger.error(f"Error loading LLM model: {str(e)}")
                    raise RuntimeError(f"Failed to load LLM model: {str(e)}")
                finally:
                    self._is_loading = False
            
            # Update last used timestamp
            self.last_used_time = time.time()
            return self._llm
    
    def _check_memory_before_loading(self):
        """Check available memory before loading the model.
        
        Raises:
            RuntimeError: If not enough memory is available
        """
        # Get memory info
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        
        # Estimate model size (rough approximation based on file size)
        model_size_mb = os.path.getsize(self._model_path) / (1024 * 1024)
        needed_memory_mb = model_size_mb * 1.5  # 1.5x model size for safe loading
        
        logger.info(f"Available memory: {available_mb:.2f}MB, Estimated needed: {needed_memory_mb:.2f}MB")
        
        # Check if enough memory is available
        if available_mb < needed_memory_mb:
            logger.warning(f"Low memory condition: {available_mb:.2f}MB available, need ~{needed_memory_mb:.2f}MB")
            
            # Try to free memory
            gc.collect()
            
            # Check again
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)
            
            if available_mb < needed_memory_mb:
                logger.error(f"Insufficient memory to load model: {available_mb:.2f}MB available")
                raise RuntimeError(f"Insufficient memory to load model. Need ~{needed_memory_mb:.2f}MB but only {available_mb:.2f}MB available")
    
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
        stream: bool = False,
        timeout_seconds: int = 30,
        thread_count: Optional[int] = None,
        reuse_cache: bool = True
    ) -> Union[str, Generator[str, None, None]]:
        """Generate text from the LLM.
        
        Args:
            prompt: Text prompt for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            stream: Whether to stream the response
            timeout_seconds: Maximum time to wait for generation
            thread_count: Number of threads to use (None = use default)
            reuse_cache: Whether to reuse KV cache for similar prompts
            
        Returns:
            Generated text or a generator yielding text chunks
            
        Raises:
            ValueError: If prompt is empty
            RuntimeError: If text generation fails
        """
        if not prompt.strip():
            logger.warning("Empty prompt provided for text generation")
            raise ValueError("Cannot generate text from empty prompt")
        
        # Use the prompt as provided, don't reformat
        formatted_prompt = prompt
        
        try:
            logger.info(f"Generating text with max_tokens={max_tokens}, temperature={temperature}")
            
            # Apply shorter context for faster generation
            actual_max_tokens = min(max_tokens, self._n_ctx - len(formatted_prompt) // 4)
            
            # Log thread count info but don't pass it to generation call
            if thread_count is not None:
                logger.info(f"Thread count specified: {thread_count} (note: threads configured during model initialization)")
            
            # Check if we can reuse KV cache
            cache_hit = False
            if reuse_cache and hasattr(self, '_kv_cache') and self._kv_cache is not None and hasattr(self, '_last_prompt') and self._last_prompt is not None:
                # Reuse cache if the new prompt starts with the previous prompt
                if prompt.startswith(self._last_prompt) and len(prompt) - len(self._last_prompt) < 20:
                    cache_hit = True
                    logger.info("Reusing KV cache for similar prompt")
            
            if not stream:
                # Generate complete response with timeout handling
                start_time = time.time()
                
                # Memory optimization: before large operation
                gc.collect()
                
                # Set cache parameters
                cache_params = {}
                if cache_hit:
                    cache_params = {"cache": self._kv_cache}
                
                # Call LLM without n_threads parameter (configured during initialization)
                response = self.llm(
                    formatted_prompt,
                    max_tokens=actual_max_tokens,
                    temperature=temperature,
                    echo=False,
                    stop=["</s>", "<s>"],  # Stop at special tokens
                    **cache_params
                )
                
                # Store KV cache and prompt for potential reuse
                if not cache_hit and hasattr(response, "get"):
                    self._kv_cache = response.get("cache", None)
                    self._last_prompt = prompt
                
                elapsed = time.time() - start_time
                if elapsed > 5:  # Log slow generations
                    logger.warning(f"Slow text generation: {elapsed:.2f} seconds")
                
                return response["choices"][0]["text"].strip()
            else:
                # Return generator for streaming response
                return self._generate_streaming_response(formatted_prompt, actual_max_tokens, temperature, timeout_seconds)
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise RuntimeError(f"Failed to generate text: {str(e)}")
    
    def _generate_streaming_response(
        self, prompt: str, max_tokens: int, temperature: float, timeout_seconds: int
    ) -> Generator[str, None, None]:
        """Generate streaming text response.
        
        Args:
            prompt: Formatted prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout_seconds: Maximum generation time
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            RuntimeError: If streaming generation fails
        """
        try:
            # Generate streaming response
            start_time = time.time()
            response_text = ""
            
            # Memory optimization: before large operation
            gc.collect()
            
            # Call LLM without n_threads parameter (configured during initialization)
            for token in self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                stream=True,
                stop=["</s>", "<s>"]  # Stop at special tokens
            ):
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    logger.warning(f"Generation timeout after {timeout_seconds} seconds")
                    yield "\n\n[Generation timed out due to performance constraints]"
                    break
                    
                chunk = token["choices"][0]["text"]
                response_text += chunk
                yield chunk
            
            # Log generation time for monitoring
            elapsed = time.time() - start_time
            if elapsed > 5:  # Log slow generations
                logger.warning(f"Slow streaming generation: {elapsed:.2f} seconds")
            
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
        # Limit number of documents for memory efficiency
        max_docs = min(3, len(documents))
        limited_docs = documents[:max_docs]
        
        # Limit context size per document
        max_content_length = min(500, self._n_ctx // (max_docs * 2))
        context_parts = []
        
        for i, doc in enumerate(limited_docs):
            content = doc['content']
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            context_parts.append(f"Document {i+1}:\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Create RAG prompt with a format known to work with Phi-2
        prompt = f"<|user|>\nI have the following documents:\n\n{context}\n\nBased on these documents, answer this question: {query}\n<|assistant|>"
        
        return prompt
    
    def quick_generate(self, query: str) -> str:
        """Generate a quick response without full context loading.
        
        This is a lightweight method for testing when the main generation
        might be too resource-intensive.
        
        Args:
            query: User query text
            
        Returns:
            A simple generated response
        """
        try:
            # Try different prompt formats that may work with Phi-2
            prompt_formats = [
                f"<s>Instruct: {query}\nOutput:",
                f"<|user|>\n{query}\n<|assistant|>",
                f"Q: {query}\nA:"
            ]
            
            # Try each format until we get a non-empty response
            for prompt in prompt_formats:
                # Memory optimization before generation
                gc.collect()
                
                # Generate with minimal parameters
                response = self.llm(
                    prompt,
                    max_tokens=50,
                    temperature=0.7,
                    echo=False,
                    stop=["</s>", "<s>"]
                )
                
                result = response["choices"][0]["text"].strip()
                if result:
                    logger.info(f"Successful prompt format: {prompt.split(query)[0]}")
                    return result
            
            # If all formats failed, return a default message
            return "I'm having trouble generating a response right now."
        except Exception as e:
            logger.error(f"Error in quick generation: {str(e)}")
            return f"Error generating response: {str(e)}"

# Singleton instance for application-wide use
llm_service = LLMService()

# Automatic cleanup task
def setup_automatic_cleanup(app=None):
    """Set up automatic cleanup of inactive model.
    
    Args:
        app: FastAPI app for startup/shutdown events
    """
    import threading
    
    def cleanup_task():
        """Background task to periodically unload inactive models."""
        while True:
            try:
                # Check every 10 minutes
                time.sleep(600)
                llm_service.unload_model_if_inactive(threshold_seconds=1800)  # 30 minutes
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
    
    # Start background thread
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    
    # Register with FastAPI if app provided
    if app:
        @app.on_event("startup")
        def start_cleanup():
            logger.info("Starting automatic model cleanup task")
            # Thread already started above
    
        @app.on_event("shutdown")
        def stop_cleanup():
            logger.info("Stopping automatic model cleanup task")
            # Thread will be stopped automatically as daemon=True
            
            # Unload model to free memory
            with model_lock:
                if llm_service._llm is not None:
                    llm_service._llm = None
                    gc.collect()