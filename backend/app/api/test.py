"""
Test API endpoint module for the Mini RAG application.

This module provides simplified endpoints for testing system components
without engaging the full RAG pipeline.
"""

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import time

from app.services.embeddings import embeddings_service
from app.services.vectorstore import vector_store
from app.services.llm import llm_service

# Create router
router = APIRouter()

class TestQueryRequest(BaseModel):
    """Test query request with minimal parameters."""
    query: str
    mode: Optional[str] = "llm"  # "llm", "vector", "combined"

class TestResponse(BaseModel):
    """Simple test response with timing information."""
    result: str
    elapsed_ms: float
    mode: str

class PromptFormatRequest(BaseModel):
    """Request to test a specific prompt format."""
    query: str
    format: str = "chat"  # chat, instruct, qa, plain

@router.post("/llm-only", response_model=TestResponse)
async def test_llm_only(request: TestQueryRequest):
    """
    Test LLM generation only without vector store.
    
    This endpoint tests the LLM service in isolation to verify
    that text generation is working properly.
    
    Args:
        request: Query request with text to generate from
        
    Returns:
        Generated text with timing information
    """
    start_time = time.time()
    
    try:
        # Generate text with minimal parameters
        result = llm_service.quick_generate(request.query)
        
        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000
        
        return TestResponse(
            result=result,
            elapsed_ms=elapsed_ms,
            mode="llm-only"
        )
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=500, 
            detail=f"LLM generation failed after {elapsed_ms:.2f}ms: {str(e)}"
        )

@router.post("/vector-only", response_model=TestResponse)
async def test_vector_only(request: TestQueryRequest):
    """
    Test vector search only without LLM generation.
    
    This endpoint tests the vector store and embedding services
    in isolation to verify search functionality.
    
    Args:
        request: Query request with text to search for
        
    Returns:
        Search results summary with timing information
    """
    start_time = time.time()
    
    try:
        # Generate embedding for query
        query_embedding = embeddings_service.generate_embedding(request.query)
        
        # Perform quick search with limited results
        results = vector_store.quick_search(query_embedding, k=2)
        
        # Format results into text string
        if results:
            result_text = f"Found {len(results)} results:\n"
            for i, doc in enumerate(results):
                result_text += f"Result {i+1}: {doc['content'][:100]}...\n"
        else:
            result_text = "No matching documents found."
        
        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000
        
        return TestResponse(
            result=result_text,
            elapsed_ms=elapsed_ms,
            mode="vector-only"
        )
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed after {elapsed_ms:.2f}ms: {str(e)}"
        )

@router.post("/simple", response_model=TestResponse)
async def test_simple_request(request: TestQueryRequest):
    """
    Simple test endpoint that dispatches to appropriate test based on mode.
    
    Args:
        request: Query request with text and test mode
        
    Returns:
        Test results from the requested mode
    """
    if request.mode == "llm":
        return await test_llm_only(request)
    elif request.mode == "vector":
        return await test_vector_only(request)
    else:
        # Combined lightweight test
        start_time = time.time()
        
        try:
            # Generate embedding
            query_embedding = embeddings_service.generate_embedding(request.query)
            
            # Get limited results
            results = vector_store.quick_search(query_embedding, k=1)
            
            # Format simple prompt with minimal context
            context = results[0]['content'][:200] if results else "No context available."
            prompt = f"Query: {request.query}\nContext: {context}\nAnswer briefly:"
            
            # Generate minimal response
            result = llm_service.quick_generate(prompt)
            
            # Calculate elapsed time
            elapsed_ms = (time.time() - start_time) * 1000
            
            return TestResponse(
                result=result,
                elapsed_ms=elapsed_ms,
                mode="combined-light"
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            raise HTTPException(
                status_code=500,
                detail=f"Test failed after {elapsed_ms:.2f}ms: {str(e)}"
            )

@router.post("/prompt-format", response_model=TestResponse)
async def test_prompt_format(request: PromptFormatRequest):
    """
    Test different prompt formats for the LLM.
    
    Args:
        request: Query and format to test
        
    Returns:
        Generated text with timing information
    """
    start_time = time.time()
    
    try:
        # Format the prompt according to the requested format
        query = request.query
        
        if request.format == "chat":
            prompt = f"<|user|>\n{query}\n<|assistant|>"
        elif request.format == "instruct":
            prompt = f"<s>Instruct: {query}\nOutput:"
        elif request.format == "qa":
            prompt = f"Q: {query}\nA:"
        elif request.format == "llama":
            prompt = f"[INST] {query} [/INST]"
        elif request.format == "alpaca":
            prompt = f"### Instruction:\n{query}\n\n### Response:"
        elif request.format == "chatml":
            prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        else:  # plain
            prompt = query
        
        # Generate text directly with the formatted prompt
        result = llm_service.generate_text(prompt, max_tokens=100)
        
        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000
        
        return TestResponse(
            result=result or "No response generated",
            elapsed_ms=elapsed_ms,
            mode=f"format-{request.format}"
        )
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=500, 
            detail=f"Prompt format test failed after {elapsed_ms:.2f}ms: {str(e)}"
        )

@router.get("/ping")
async def ping_test():
    """
    Simple ping endpoint for basic connectivity testing.
    
    Returns:
        Simple response with timestamp
    """
    return {
        "status": "ok",
        "timestamp": time.time(),
        "message": "Mini-RAG API is responding"
    }