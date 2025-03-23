# save as diagnostic.py in your backend directory
import time
import os
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embedding_model():
    """Test if the embedding model loads correctly."""
    logger.info("Testing embedding model...")
    try:
        from app.services.embeddings import embeddings_service
        logger.info("Embedding service imported successfully")
        
        start = time.time()
        embedding = embeddings_service.generate_embedding("Test sentence for embedding")
        elapsed = time.time() - start
        
        logger.info(f"Generated embedding of dimension {len(embedding)} in {elapsed:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Embedding model test failed: {str(e)}")
        return False

def test_llm_model():
    """Test if the LLM loads correctly."""
    logger.info("Testing LLM...")
    try:
        from app.services.llm import llm_service
        logger.info("LLM service imported successfully")
        
        start = time.time()
        response = llm_service.generate_text("Hello, world!", max_tokens=10)
        elapsed = time.time() - start
        
        logger.info(f"Generated text: '{response}' in {elapsed:.2f} seconds")
        return len(response.strip()) > 0
    except Exception as e:
        logger.error(f"LLM test failed: {str(e)}")
        return False

def test_llm_with_different_formats():
    """Test LLM with different prompt formats."""
    logger.info("\nTesting LLM with various prompt formats...")
    try:
        from app.services.llm import llm_service
        
        formats = [
            ("Basic", "Hello, world!"),
            ("Q&A", "Q: What is retrieval augmented generation? A:"),
            ("Phi-Instruct", "<s>Instruct: Explain what RAG means.\nOutput:"),
            ("Chat", "<|user|>\nExplain RAG\n<|assistant|>"),
            ("System", "<|system|>\nYou are a helpful AI assistant.\n<|user|>\nWhat is RAG?\n<|assistant|>")
        ]
        
        success = False
        for name, prompt in formats:
            logger.info(f"Testing format: {name} with prompt: '{prompt}'")
            start = time.time()
            response = llm_service.generate_text(prompt, max_tokens=50)
            elapsed = time.time() - start
            
            logger.info(f"  Response: '{response}' in {elapsed:.2f} seconds")
            if len(response.strip()) > 0:
                logger.info(f"✅ Format {name} produced non-empty response")
                success = True
            else:
                logger.info(f"❌ Format {name} produced empty response")
        
        return success
    except Exception as e:
        logger.error(f"LLM formats test failed: {str(e)}")
        return False

def test_extended_parameters():
    """Test LLM with extended parameters."""
    logger.info("\nTesting LLM with extended parameters...")
    try:
        from app.services.llm import llm_service
        
        # Try different parameter combinations
        param_sets = [
            ("Large tokens", {"prompt": "<s>Instruct: What is RAG?\nOutput:", "max_tokens": 100}),
            ("Higher temp", {"prompt": "<s>Instruct: What is RAG?\nOutput:", "max_tokens": 50, "temperature": 0.9}),
            ("Lower temp", {"prompt": "<s>Instruct: What is RAG?\nOutput:", "max_tokens": 50, "temperature": 0.4})
        ]
        
        success = False
        for name, params in param_sets:
            logger.info(f"Testing parameters: {name}")
            prompt = params.pop("prompt")
            start = time.time()
            response = llm_service.generate_text(prompt, **params)
            elapsed = time.time() - start
            
            logger.info(f"  Response: '{response}' in {elapsed:.2f} seconds")
            if len(response.strip()) > 0:
                logger.info(f"✅ Parameters {name} produced non-empty response")
                success = True
            else:
                logger.info(f"❌ Parameters {name} produced empty response")
        
        return success
    except Exception as e:
        logger.error(f"Extended parameters test failed: {str(e)}")
        return False

def test_vector_store():
    """Test vector store functionality."""
    logger.info("\nTesting vector store...")
    try:
        from app.services.vectorstore import vector_store
        from app.services.embeddings import embeddings_service
        
        # Test collection access
        collection = vector_store.collection
        count = collection.count()
        logger.info(f"Vector store contains {count} documents")
        
        # Try a search if documents exist
        if count > 0:
            query = "Test query for vector store"
            query_embedding = embeddings_service.generate_embedding(query)
            
            start = time.time()
            results = vector_store.search(query_embedding, k=2)
            elapsed = time.time() - start
            
            logger.info(f"Search returned {len(results)} results in {elapsed:.2f} seconds")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result['content'][:50]}...")
        
        return True
    except Exception as e:
        logger.error(f"Vector store test failed: {str(e)}")
        return False

def main():
    """Run diagnostics."""
    logger.info("Starting Mini-RAG diagnostics")
    
    # Test embedding model
    embedding_ok = test_embedding_model()
    
    # Test LLM basic functionality
    llm_ok = test_llm_model()
    
    # Test vector store
    vector_ok = test_vector_store()
    
    # If LLM returned empty response, try different formats and parameters
    format_ok = False
    params_ok = False
    
    if llm_ok:
        format_ok = test_llm_with_different_formats()
        params_ok = test_extended_parameters()
    
    # Report results
    logger.info("\n--- Diagnostic Results ---")
    logger.info(f"Embedding model: {'OK' if embedding_ok else 'FAILED'}")
    logger.info(f"LLM model loading: {'OK' if llm_ok else 'FAILED'}")
    logger.info(f"LLM response with formats: {'OK' if format_ok else 'NEEDS ATTENTION'}")
    logger.info(f"LLM response with parameters: {'OK' if params_ok else 'NEEDS ATTENTION'}")
    logger.info(f"Vector store: {'OK' if vector_ok else 'FAILED'}")
    
    # Recommendations
    logger.info("\n--- Recommendations ---")
    if embedding_ok and not (llm_ok and (format_ok or params_ok)):
        logger.info("1. Start with vector search-only test endpoints")
    
    if format_ok or params_ok:
        logger.info("2. Use the working prompt format in your implementation")
    else:
        logger.info("2. Try updating llama-cpp-python to a newer version")

if __name__ == "__main__":
    main()