"""
Chat API module for the Mini RAG application.

This module provides endpoints for processing chat queries,
including retrieval-augmented generation and an OpenAI-compatible interface.
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import json

from app.services.embeddings import embeddings_service
from app.services.vectorstore import vector_store
from app.services.llm import llm_service
from config import settings

# Create routers
router = APIRouter()
openai_router = APIRouter()

# Model definitions
class SourceMetadata(BaseModel):
    """Metadata associated with a source document."""
    source: str
    page: Optional[int] = None
    chunk: Optional[int] = None
    sub_chunk: Optional[int] = None

class Source(BaseModel):
    """Source document information."""
    id: str
    content: str
    metadata: SourceMetadata

class QueryRequest(BaseModel):
    """User query request."""
    query: str

class QueryResponse(BaseModel):
    """Response to a user query."""
    answer: str
    sources: List[Source] = []

@router.post("", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a chat query using RAG.
    
    Retrieves relevant documents from the vector store based on the query,
    and generates a response using the language model with the retrieved context.
    
    Args:
        request: Query request containing the user's question
        
    Returns:
        Generated answer with source references
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Generate query embedding
        query_embedding = embeddings_service.generate_embedding(request.query)
        
        # Retrieve relevant documents
        documents = vector_store.search(query_embedding)
        
        # Generate prompt with context
        prompt = llm_service.format_rag_prompt(request.query, documents)
        
        # Generate answer
        answer = llm_service.generate_text(prompt)
        
        # Format response
        sources = []
        for doc in documents:
            content = doc["content"]
            # Truncate long content for display
            if len(content) > 200:
                content = content[:200] + "..."
                
            sources.append(Source(
                id=doc["id"],
                content=content,
                metadata=SourceMetadata(**doc["metadata"])
            ))
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def process_query_stream(request: QueryRequest):
    """
    Process a chat query with streaming response.
    
    Memory-efficient version of query processing that streams the response
    as it's generated by the language model.
    
    Args:
        request: Query request containing the user's question
        
    Returns:
        Streaming response with generated content
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Generate query embedding
        query_embedding = embeddings_service.generate_embedding(request.query)
        
        # Retrieve relevant documents
        documents = vector_store.search(query_embedding)
        
        # Generate prompt with context
        prompt = llm_service.format_rag_prompt(request.query, documents)
        
        # Create streaming response
        return StreamingResponse(
            llm_service.generate_text(prompt, stream=True),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@openai_router.post("/v1/chat/completions")
async def openai_compatible_chat(
    request: Dict[str, Any] = Body(...),
):
    """
    OpenAI-compatible endpoint for chat completions.
    
    This endpoint provides compatibility with the OpenAI Chat API format,
    facilitating migration to Azure OpenAI services.
    
    Args:
        request: OpenAI-formatted request with messages array
        
    Returns:
        OpenAI-formatted response
    """
    messages = request.get("messages", [])
    temperature = request.get("temperature", 0.7)
    max_tokens = request.get("max_tokens", 512)
    stream = request.get("stream", False)
    
    # Extract the query from messages (typically the last user message)
    query = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
    
    if not query:
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "phi-2-local",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I don't understand your query. Could you please provide more information?"
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    
    # Process in streaming mode if requested
    if stream:
        async def generate_stream():
            try:
                # Generate query embedding
                query_embedding = embeddings_service.generate_embedding(query)
                
                # Retrieve relevant documents
                documents = vector_store.search(query_embedding)
                
                # Generate prompt with context
                prompt = llm_service.format_rag_prompt(query, documents)
                
                # Stream the response in OpenAI format
                for chunk in llm_service.generate_text(prompt, temperature=temperature, 
                                                      max_tokens=max_tokens, stream=True):
                    response_data = {
                        'id': f'chatcmpl-{int(time.time())}',
                        'object': 'chat.completion.chunk',
                        'created': int(time.time()),
                        'model': 'phi-2-local',
                        'choices': [
                            {
                                'index': 0,
                                'delta': {'content': chunk},
                                'finish_reason': None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(response_data)}\n\n"
                            
                # Send the final chunk
                final_data = {
                    'id': f'chatcmpl-{int(time.time())}',
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': 'phi-2-local',
                    'choices': [
                        {
                            'index': 0,
                            'delta': {},
                            'finish_reason': 'stop'
                        }
                    ]
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
                
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    
    # Generate query embedding
    query_embedding = embeddings_service.generate_embedding(query)
    
    # Retrieve relevant documents
    documents = vector_store.search(query_embedding)
    
    # Generate prompt with context
    prompt = llm_service.format_rag_prompt(query, documents)
    
    # Generate answer
    answer = llm_service.generate_text(
        prompt, 
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Format response like OpenAI API
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "phi-2-local",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt) // 4,  # Rough estimation
            "completion_tokens": len(answer) // 4,  # Rough estimation
            "total_tokens": (len(prompt) + len(answer)) // 4  # Rough estimation
        }
    }